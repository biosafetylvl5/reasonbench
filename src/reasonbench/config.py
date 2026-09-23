"""Pydantic models for the two config files, and the YAML loaders.

``models.yaml`` is the run setup: slugs, sweep axes, limits, judge.
``prompts/<name>.yaml`` is the prompt, its variants, and the rubric.

Everything here is frozen, so the async workers can share one instance.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reasonbench.errors import ConfigError, MissingAPIKeyError

KEY_FILENAME = "openrouter.key"

# A cell value only reaches a regex through `re_escape`; without it a decimal
# point, an alternation or a `+` in the data silently rewrites the pattern.
_ENV = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701
_ENV.filters["re_escape"] = re.escape


def render_template(text: str, variables: dict[str, Any]) -> str:
    """Render one Jinja template with StrictUndefined."""
    return _ENV.from_string(text).render(**variables)


def looks_templated(text: str) -> bool:
    """Return whether a string contains Jinja syntax worth rendering."""
    return "{{" in text or "{%" in text or "{#" in text


class Frozen(BaseModel):
    """Base for every configuration model: immutable and strict about typos."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Settings(BaseSettings):
    """Runtime secrets, never read from the YAML files.

    Resolution order is environment variable, then ``.env``, then an
    ``openrouter.key`` file in the working directory.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = None

    @classmethod
    def resolve(cls, key_file: Path | None = None) -> str:
        """Return the API key, or raise :class:`ConfigError` if none is found."""
        settings = cls()
        if settings.openrouter_api_key:
            return settings.openrouter_api_key.strip()

        candidate = key_file or Path(KEY_FILENAME)
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
            if key:
                return key

        raise MissingAPIKeyError(
            "no OpenRouter API key",
            hint=(
                f"set OPENROUTER_API_KEY, add it to .env, or place it in {candidate}."
            ),
        )


class Target(StrEnum):
    """Which part of a response a criterion is applied to."""

    OUTPUT = "output"
    REASONING = "reasoning"
    BOTH = "both"


class TextScope(StrEnum):
    """Which slice of the target text a deterministic check reads."""

    FULL = "full"
    LAST_LINE = "last_line"
    FIRST_LINE = "first_line"


class Guidance(Frozen):
    """Instructions handed to the judge for a single criterion.

    Only ``summary`` is required. ``levels`` turns the criterion into an
    anchored rubric. A bare string in YAML becomes ``summary``.
    """

    summary: str
    levels: dict[int, str] = Field(default_factory=dict)
    positive_indicators: tuple[str, ...] = ()
    negative_indicators: tuple[str, ...] = ()
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, value: Any) -> Any:
        """Allow ``guidance: "some text"`` as shorthand for ``{summary: ...}``."""
        if isinstance(value, str):
            return {"summary": value}
        return value


class RegexCheck(Frozen):
    """Pass when ``pattern`` is found in the target text."""

    type: Literal["regex"]
    pattern: str
    ignore_case: bool = False


class ExactCheck(Frozen):
    """Pass when the target text equals ``expected``."""

    type: Literal["exact"]
    expected: str
    strip: bool = True
    ignore_case: bool = True


class ContainsCheck(Frozen):
    """Pass when ``expected`` appears anywhere in the target text."""

    type: Literal["contains"]
    expected: str
    ignore_case: bool = True


class NumericCheck(Frozen):
    """Pass when a number extracted from the target is within ``tol``."""

    type: Literal["numeric"]
    expected: float
    tol: float = 0.0
    extract: str | None = None


Check = Annotated[
    RegexCheck | ExactCheck | ContainsCheck | NumericCheck,
    Field(discriminator="type"),
]


class DeterministicCriterion(Frozen):
    """A criterion scored by a pure function."""

    id: str
    kind: Literal["deterministic"]
    target: Target = Target.OUTPUT
    scope: TextScope = TextScope.FULL
    weight: float = 1.0
    check: Check


class JudgeCriterion(Frozen):
    """A criterion scored by the judge model on an integer ``scale``."""

    id: str
    kind: Literal["judge"]
    target: Target = Target.OUTPUT
    weight: float = 1.0
    scale: tuple[int, int] = (0, 4)
    guidance: Guidance

    @model_validator(mode="after")
    def _check_scale(self) -> Self:
        low, high = self.scale
        if high <= low:
            raise ValueError(f"criterion {self.id!r}: scale must be increasing")
        unknown = set(self.guidance.levels) - set(range(low, high + 1))
        if unknown:
            raise ValueError(
                f"criterion {self.id!r}: guidance levels {sorted(unknown)} "
                f"fall outside scale {low}-{high}",
            )
        return self


Criterion = Annotated[
    DeterministicCriterion | JudgeCriterion,
    Field(discriminator="kind"),
]


class Rubric(Frozen):
    """A set of criteria plus task-specific framing for the judge."""

    judge_instructions: str | None = None
    criteria: tuple[Criterion, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [c.id for c in self.criteria]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate criterion ids: {sorted(duplicates)}")
        if not ids:
            raise ValueError("rubric must define at least one criterion")
        return self

    @property
    def judge_criteria(self) -> tuple[JudgeCriterion, ...]:
        """Return only the criteria that require a judge call."""
        return tuple(c for c in self.criteria if isinstance(c, JudgeCriterion))


class Attachment(Frozen):
    """An image or PDF sent alongside the prompt text.

    ``path`` is resolved relative to the prompt YAML and inlined as a data URL
    at load time, so request building stays free of file I/O.
    """

    id: str
    url: str
    media_type: str
    attach_to: tuple[str, ...] | Literal["all"] = "all"

    def applies_to(self, variant_id: str) -> bool:
        """Return whether this attachment belongs on ``variant_id``."""
        return self.attach_to == "all" or variant_id in self.attach_to

    @property
    def is_pdf(self) -> bool:
        """Return whether this attachment must be sent as a ``file`` part."""
        return self.media_type == "application/pdf"


class ImageColumn(Frozen):
    """A dataset column whose cells point at images to attach."""

    column: str
    media_type: str | None = None
    required: bool = True
    attach_to: tuple[str, ...] | Literal["all"] = "all"

    def applies_to(self, variant_id: str) -> bool:
        """Return whether this column's images belong on ``variant_id``."""
        return self.attach_to == "all" or variant_id in self.attach_to


class CaseSample(Frozen):
    """A deterministic subsample of the dataset."""

    n: int = Field(ge=1)
    seed: int = 0


class DatasetSpec(Frozen):
    """Where a prompt's cases live, and how to read them."""

    path: str
    format: Literal["auto", "jsonl", "csv"] = "auto"
    id_column: str = "case_id"
    required_columns: tuple[str, ...] = ()
    limit: int | None = Field(default=None, ge=1)
    select: tuple[str, ...] = ()
    sample: CaseSample | None = None
    image_root: str | None = None
    images: tuple[ImageColumn, ...] = ()
    max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1)


class Variant(Frozen):
    """One phrasing of the prompt; variants are a sweep axis."""

    id: str
    user: str


class PromptSpec(Frozen):
    """A prompt, its variants and attachments, and the rubric to grade it."""

    version: int = 1
    id: str
    title: str
    system: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    variants: tuple[Variant, ...]
    attachments: tuple[Attachment, ...] = ()
    dataset: DatasetSpec | None = None
    rubric: Rubric

    @model_validator(mode="after")
    def _unique_variant_ids(self) -> Self:
        ids = [v.id for v in self.variants]
        if not ids:
            raise ValueError("prompt must define at least one variant")
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate variant ids: {sorted(duplicates)}")
        for attachment in self.attachments:
            if attachment.attach_to == "all":
                continue
            unknown = set(attachment.attach_to) - set(ids)
            if unknown:
                raise ValueError(
                    f"attachment {attachment.id!r} references unknown "
                    f"variants: {sorted(unknown)}",
                )
        return self

    def render(
        self, variant: Variant, case_vars: dict[str, Any] | None = None
    ) -> tuple[str | None, str]:
        """Return ``(system, user)``, with case values overriding ``variables``."""
        merged = {**self.variables, **(case_vars or {})}
        render = lambda text: render_template(text, merged)
        return (
            render(self.system) if self.system else None,
            render(variant.user),
        )


class PriceEntry(Frozen):
    """USD per million tokens for one model."""

    prompt: float = Field(ge=0.0)
    completion: float = Field(ge=0.0)
    # Most servers count reasoning tokens inside completion_tokens, so pricing
    # them again doubles the bill and trips the cap at half the real spend.
    # Set this only where a provider bills them separately.
    reasoning: float | None = Field(default=None, ge=0.0)
    # Cached tokens are a subset of prompt_tokens, hence the subtraction in
    # price_usage rather than an addition.
    cached_prompt: float | None = Field(default=None, ge=0.0)


class Pricing(Frozen):
    """Local prices, used when the endpoint does not report a cost."""

    per_mtok: dict[str, PriceEntry] = Field(default_factory=dict)

    def entry_for(self, model: str) -> PriceEntry | None:
        """Return the price for ``model``, falling back to a ``*`` wildcard."""
        return self.per_mtok.get(model) or self.per_mtok.get("*")


class SweepAxes(Frozen):
    """The parameter grid. Cells are the cartesian product of every axis."""

    repeats: int = Field(default=1, ge=1)
    temperature: tuple[float, ...] = (0.0,)
    reasoning_effort: tuple[str | None, ...] = (None,)

    @model_validator(mode="after")
    def _validate_effort(self) -> Self:
        allowed = {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}
        unknown = set(self.reasoning_effort) - allowed
        if unknown:
            listed = sorted(str(value) for value in unknown)
            raise ValueError(f"unknown reasoning_effort values: {listed}")
        if not self.temperature:
            raise ValueError("temperature axis must not be empty")
        return self


class JudgeSettings(Frozen):
    """How the judge is invoked. Task-specific wording lives in the rubric."""

    model: str
    temperature: float = 0.0
    repeats: int = Field(default=1, ge=1)
    blind: bool = True
    max_chars: int = Field(default=20_000, ge=500)
    # Retries for a judge reply that does not parse into scores. Distinct from
    # HTTP retries: the call succeeded, the content was just unusable.
    parse_retries: int = Field(default=2, ge=0)
    persona: str = (
        "You are a strict, impartial grader assessing the quality of a language "
        "model's reasoning and final answer."
    )


class RunConfig(Frozen):
    """The whole of ``models.yaml``."""

    version: int = 1
    models: tuple[str, ...]
    sweep: SweepAxes = Field(default_factory=SweepAxes)
    max_tokens: int = Field(default=8000, ge=1)
    max_concurrency: int = Field(default=8, ge=1)
    max_retries: int = Field(default=4, ge=0)
    timeout_s: float = Field(default=300.0, gt=0)
    budget_usd: float = Field(default=2.0, gt=0)
    max_samples: int = Field(default=2000, ge=1)
    max_total_tokens: int | None = Field(default=None, gt=0)
    base_url: str | None = None
    # OpenRouter reports usage.cost; almost nothing else does. Leave this None
    # to infer it from the base URL.
    reports_cost: bool | None = None
    pricing: Pricing = Field(default_factory=Pricing)
    require_pricing: bool | None = None
    judge: JudgeSettings

    @model_validator(mode="after")
    def _require_models(self) -> Self:
        if not self.models:
            raise ValueError("models list must not be empty")
        duplicates = {m for m in self.models if self.models.count(m) > 1}
        if duplicates:
            raise ValueError(f"duplicate model slugs: {sorted(duplicates)}")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"No such config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    return data


def _inline_attachment(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve an attachment's ``path`` into an inline base64 data URL."""
    resolved = dict(raw)
    path_value = resolved.pop("path", None)
    if path_value is None:
        if "url" not in resolved:
            raise ConfigError(
                f"attachment {resolved.get('id', '?')!r} needs either path or url",
            )
        resolved.setdefault(
            "media_type",
            mimetypes.guess_type(resolved["url"])[0] or "image/png",
        )
        return resolved

    file_path = (base_dir / str(path_value)).resolve()
    if not file_path.is_file():
        raise ConfigError(f"attachment file not found: {file_path}")
    media_type = resolved.get("media_type") or (
        mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    )
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    resolved["media_type"] = media_type
    resolved["url"] = f"data:{media_type};base64,{payload}"
    resolved.setdefault("id", file_path.stem)
    return resolved


def load_run_config(path: Path) -> RunConfig:
    """Load and validate ``models.yaml``."""
    try:
        return RunConfig.model_validate(_read_yaml(path))
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_prompt(path: Path) -> PromptSpec:
    """Load and validate a prompt YAML, inlining any attachments."""
    raw = _read_yaml(path)
    base_dir = path.parent
    if raw.get("attachments"):
        raw = dict(raw)
        raw["attachments"] = [
            _inline_attachment(a, base_dir) for a in raw["attachments"]
        ]
    try:
        return PromptSpec.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
