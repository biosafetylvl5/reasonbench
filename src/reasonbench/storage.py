"""Run directories: a SQLite file plus the raw responses behind it.

Samples commit as they land, so an interrupted sweep resumes by skipping the ids
already present. Scores are a separate table and can be dropped and recomputed
without touching the samples.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

import yaml

from reasonbench.config import Frozen
from reasonbench.errors import RunDirError
from reasonbench.openrouter import ReasoningAvailability, SampleResult

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id              TEXT PRIMARY KEY,
    prompt_id              TEXT NOT NULL,
    variant_id             TEXT NOT NULL,
    model                  TEXT NOT NULL,
    temperature            REAL NOT NULL,
    reasoning_effort       TEXT,
    repeat                 INTEGER NOT NULL,
    ok                     INTEGER NOT NULL,
    output                 TEXT NOT NULL DEFAULT '',
    reasoning_availability TEXT NOT NULL,
    reasoning_text         TEXT NOT NULL DEFAULT '',
    reasoning_summary      TEXT NOT NULL DEFAULT '',
    reasoning_tokens       INTEGER NOT NULL DEFAULT 0,
    prompt_tokens          INTEGER NOT NULL DEFAULT 0,
    completion_tokens      INTEGER NOT NULL DEFAULT 0,
    total_tokens           INTEGER NOT NULL DEFAULT 0,
    cost                   REAL NOT NULL DEFAULT 0.0,
    latency_s              REAL NOT NULL DEFAULT 0.0,
    served_model           TEXT,
    provider               TEXT,
    finish_reason          TEXT,
    error                  TEXT,
    created_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    sample_id     TEXT NOT NULL,
    criterion_id  TEXT NOT NULL,
    judge_repeat  INTEGER NOT NULL DEFAULT 0,
    kind          TEXT NOT NULL,
    target        TEXT NOT NULL,
    weight        REAL NOT NULL,
    score         REAL,
    normalized    REAL,
    applicable    INTEGER NOT NULL,
    reason        TEXT,
    PRIMARY KEY (sample_id, criterion_id, judge_repeat)
);

CREATE INDEX IF NOT EXISTS idx_scores_sample ON scores (sample_id);
"""


class SampleRow(Frozen):
    """A sample read back out of the database."""

    sample_id: str
    prompt_id: str
    variant_id: str
    model: str
    temperature: float
    reasoning_effort: str | None
    repeat: int
    ok: bool
    output: str
    reasoning_availability: ReasoningAvailability
    reasoning_text: str
    reasoning_summary: str
    reasoning_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency_s: float
    served_model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    created_at: str | None = None

    @property
    def readable_reasoning(self) -> str | None:
        """Return gradeable trace text, or ``None`` when there is none."""
        if self.reasoning_availability is ReasoningAvailability.FULL_TEXT:
            return self.reasoning_text or None
        if self.reasoning_availability is ReasoningAvailability.SUMMARY_ONLY:
            return self.reasoning_summary or None
        return None


class ScoreRow(Frozen):
    """One criterion's score for one sample."""

    sample_id: str
    criterion_id: str
    judge_repeat: int
    kind: str
    target: str
    weight: float
    score: float | None
    normalized: float | None
    applicable: bool
    reason: str | None = None


def new_run_dir(
    root: Path, label: str | None = None, *, run_id: str | None = None
) -> Path:
    """Create and return a fresh timestamped run directory."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    suffix = run_id or secrets.token_hex(2)
    stem = f"{stamp}-{suffix}"
    name = f"{stem}_{label}" if label else stem
    run_dir = root / name
    (run_dir / "raw").mkdir(parents=True, exist_ok=False)
    return run_dir


class RunStore:
    """Owns the SQLite connection and raw-artifact directory for one run."""

    def __init__(self, run_dir: Path, *, create: bool = False) -> None:
        if not create and not (run_dir / "results.sqlite").is_file():
            raise RunDirError(
                f"not a run directory: {run_dir}",
                hint="list runs with `reasonbench ls`.",
            )
        self.run_dir = run_dir
        self.raw_dir = run_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(run_dir / "results.sqlite")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Commit and close the database connection."""
        self._conn.commit()
        self._conn.close()

    def write_manifest(self, payload: dict[str, Any]) -> None:
        """Freeze the resolved configuration alongside the results."""
        path = self.run_dir / "manifest.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def add_sample(self, result: SampleResult, raw: dict[str, Any]) -> None:
        """Persist one completed sample and its verbatim API response."""
        artifact = self.raw_dir / f"{result.sample.sample_id}.json"
        artifact.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        sample = result.sample
        self._conn.execute(
            """
            INSERT OR REPLACE INTO samples VALUES (
                :sample_id, :prompt_id, :variant_id, :model, :temperature,
                :reasoning_effort, :repeat, :ok, :output, :reasoning_availability,
                :reasoning_text, :reasoning_summary, :reasoning_tokens,
                :prompt_tokens, :completion_tokens, :total_tokens, :cost,
                :latency_s, :served_model, :provider, :finish_reason, :error,
                :created_at
            )
            """,
            {
                "sample_id": sample.sample_id,
                "prompt_id": sample.prompt_id,
                "variant_id": sample.variant_id,
                "model": sample.model,
                "temperature": sample.temperature,
                "reasoning_effort": sample.reasoning_effort,
                "repeat": sample.repeat,
                "ok": int(result.ok),
                "output": result.output,
                "reasoning_availability": str(result.reasoning.availability),
                "reasoning_text": result.reasoning.text,
                "reasoning_summary": result.reasoning.summary,
                "reasoning_tokens": result.usage.reasoning_tokens,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cost": result.usage.cost,
                "latency_s": result.latency_s,
                "served_model": result.served_model,
                "provider": result.provider,
                "finish_reason": result.finish_reason,
                "error": result.error,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        self._conn.commit()

    def save_judge_raw(
        self,
        sample_id: str,
        repeat: int,
        attempt: int,
        raw: dict[str, Any],
    ) -> None:
        """Persist a judge response so grading can be audited after the fact."""
        judge_dir = self.run_dir / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        name = f"{sample_id}_r{repeat}_a{attempt}.json"
        (judge_dir / name).write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def clear_scores(self, kind: str | None = None) -> None:
        """Drop stored scores, optionally only those of one kind."""
        if kind is None:
            self._conn.execute("DELETE FROM scores")
        else:
            self._conn.execute("DELETE FROM scores WHERE kind = ?", (kind,))
        self._conn.commit()

    def add_scores(self, rows: list[ScoreRow]) -> None:
        """Persist a batch of criterion scores."""
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO scores VALUES (
                :sample_id, :criterion_id, :judge_repeat, :kind, :target,
                :weight, :score, :normalized, :applicable, :reason
            )
            """,
            [r.model_dump() for r in rows],
        )
        self._conn.commit()

    def existing_sample_ids(self) -> set[str]:
        """Return ids of successful samples, which a resumed run can skip.

        Failures are excluded so a resume retries them; ``INSERT OR REPLACE``
        overwrites the old row.
        """
        cursor = self._conn.execute("SELECT sample_id FROM samples WHERE ok = 1")
        return {row["sample_id"] for row in cursor}

    def total_cost(self) -> float:
        """Return the summed cost of every stored sample."""
        cursor = self._conn.execute("SELECT COALESCE(SUM(cost), 0.0) AS c FROM samples")
        return float(cursor.fetchone()["c"])

    def samples(self) -> list[SampleRow]:
        """Return every stored sample."""
        cursor = self._conn.execute("SELECT * FROM samples ORDER BY model, variant_id")
        return [SampleRow.model_validate(dict(row)) for row in cursor]

    def scores(self) -> list[ScoreRow]:
        """Return every stored score."""
        cursor = self._conn.execute("SELECT * FROM scores")
        return [
            ScoreRow.model_validate(dict(row) | {"applicable": bool(row["applicable"])})
            for row in cursor
        ]
