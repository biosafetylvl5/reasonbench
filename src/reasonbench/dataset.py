"""Loading a prompt's cases, and resolving the images they point at.

A dataset row becomes one ``Case``: its columns are Jinja variables, and any
image column is resolved to a data URL here so request building stays free of
file I/O. Payloads live on ``CaseImage`` and never reach the manifest.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import mimetypes
import random
from typing import TYPE_CHECKING, Any

from pydantic import Field

from reasonbench.config import Frozen
from reasonbench.errors import ConfigError

if TYPE_CHECKING:
    from pathlib import Path

    from reasonbench.config import DatasetSpec, ImageColumn, PromptSpec

MAX_TOTAL_INLINE_BYTES = 64 * 1024 * 1024


class CaseImage(Frozen):
    """One resolved image for one case."""

    column: str
    ref: str
    url: str
    media_type: str
    bytes_: int = Field(default=0, alias="bytes")
    sha256: str = ""
    attach_to: tuple[str, ...] | str = "all"

    def applies_to(self, variant_id: str) -> bool:
        """Return whether this image belongs on ``variant_id``."""
        return self.attach_to == "all" or variant_id in self.attach_to

    @property
    def is_pdf(self) -> bool:
        """Return whether this must be sent as a ``file`` part."""
        return self.media_type == "application/pdf"


class Case(Frozen):
    """One row of a dataset."""

    case_id: str
    index: int
    source_line: int | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    images: tuple[CaseImage, ...] = ()
    digest: str = ""


class CaseSet(Frozen):
    """The selected rows of one dataset file."""

    path: str
    sha256: str
    n_rows: int
    cases: tuple[Case, ...]
    total_image_bytes: int = 0


def _read_rows(path: Path, fmt: str) -> list[tuple[int, dict[str, Any]]]:
    if fmt == "auto":
        fmt = "csv" if path.suffix.lower() in {".csv", ".tsv"} else "jsonl"
    text = path.read_text(encoding="utf-8-sig")
    rows: list[tuple[int, dict[str, Any]]] = []
    if fmt == "jsonl":
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ConfigError(f"{path}:{lineno}: expected a JSON object")
            rows.append((lineno, obj))
        return rows
    reader = csv.DictReader(text.splitlines())
    for lineno, raw in enumerate(reader, start=2):
        # A cell starting with '[' is a JSON array; nothing else is reinterpreted.
        obj = {
            k: (_maybe_json(v) if isinstance(v, str) else v)
            for k, v in raw.items()
            if k is not None
        }
        rows.append((lineno, obj))
    return rows


def _maybe_json(value: str) -> Any:
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _resolve_image(
    spec: ImageColumn,
    cell: str,
    root: Path,
    max_bytes: int,
    case_id: str,
) -> CaseImage:
    if cell.startswith(("http://", "https://", "data:")):
        media = spec.media_type or mimetypes.guess_type(cell)[0] or "image/png"
        return CaseImage(
            column=spec.column,
            ref=cell,
            url=cell,
            media_type=media,
            attach_to=spec.attach_to,
        )
    path = (root / cell).resolve()
    if not path.is_file():
        raise ConfigError(f"case {case_id!r}: image not found: {path}")
    payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise ConfigError(
            f"case {case_id!r}: image {path.name} is {len(payload)} bytes, "
            f"over the {max_bytes} byte cap",
        )
    media = spec.media_type or mimetypes.guess_type(path.name)[0] or "image/png"
    return CaseImage(
        column=spec.column,
        ref=cell,
        url=f"data:{media};base64,{base64.b64encode(payload).decode('ascii')}",
        media_type=media,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        attach_to=spec.attach_to,
    )


def _select(
    rows: list[tuple[int, dict[str, Any]]],
    spec: DatasetSpec,
    ids: list[str],
    limit: int | None,
) -> list[int]:
    order = list(range(len(rows)))
    if spec.select:
        by_id = {cid: i for i, cid in enumerate(ids)}
        missing = [c for c in spec.select if c not in by_id]
        if missing:
            raise ConfigError(f"dataset has no case(s): {', '.join(missing)}")
        order = [by_id[c] for c in spec.select]
    if spec.sample is not None:
        rng = random.Random(spec.sample.seed)  # noqa: S311
        order = sorted(rng.sample(order, min(spec.sample.n, len(order))))
    cap = limit if limit is not None else spec.limit
    return order[:cap] if cap is not None else order


def load_cases(
    prompt: PromptSpec, prompt_path: Path, *, limit: int | None = None
) -> CaseSet | None:
    """Load and resolve the cases for one prompt, or ``None`` if it has no dataset."""
    spec = prompt.dataset
    if spec is None:
        return None

    base = prompt_path.parent
    path = (base / spec.path).resolve()
    if not path.is_file():
        raise ConfigError(f"dataset file not found: {path}")
    raw = path.read_bytes()
    rows = _read_rows(path, spec.format)
    if not rows:
        raise ConfigError(f"dataset is empty: {path}")

    ids = [
        str(obj.get(spec.id_column) or f"row-{i + 1:04d}")
        for i, (_, obj) in enumerate(rows)
    ]
    duplicates = {c for c in ids if ids.count(c) > 1}
    if duplicates:
        raise ConfigError(f"duplicate case ids: {', '.join(sorted(duplicates))}")

    root = (base / spec.image_root).resolve() if spec.image_root else path.parent
    total = 0
    cases: list[Case] = []
    for position in _select(rows, spec, ids, limit):
        lineno, obj = rows[position]
        case_id = ids[position]
        missing = [c for c in spec.required_columns if c not in obj]
        if missing:
            raise ConfigError(
                f"case {case_id!r} is missing column(s): {', '.join(missing)}",
            )
        images: list[CaseImage] = []
        for col in spec.images:
            cell = obj.get(col.column)
            refs = cell if isinstance(cell, list) else ([cell] if cell else [])
            if not refs and col.required:
                raise ConfigError(f"case {case_id!r}: column {col.column!r} is empty")
            for ref in refs:
                image = _resolve_image(
                    col, str(ref), root, spec.max_image_bytes, case_id
                )
                total += image.bytes_
                images.append(image)
        if total > MAX_TOTAL_INLINE_BYTES:
            raise ConfigError(
                f"selected cases inline {total} bytes of images, over the "
                f"{MAX_TOTAL_INLINE_BYTES} byte cap; narrow with dataset.limit",
            )
        variables = {
            k: v for k, v in obj.items() if k not in {c.column for c in spec.images}
        }
        payload = json.dumps(
            {"vars": variables, "images": [i.sha256 for i in images]},
            sort_keys=True,
            default=str,
        )
        cases.append(
            Case(
                case_id=case_id,
                index=position,
                source_line=lineno,
                variables=variables,
                images=tuple(images),
                digest=hashlib.sha256(payload.encode()).hexdigest()[:12],
            )
        )

    return CaseSet(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        n_rows=len(rows),
        cases=tuple(cases),
        total_image_bytes=total,
    )
