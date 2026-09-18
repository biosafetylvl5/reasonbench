"""Threshold checks over an aggregated run, and the verdict CI reads.

An assertion whose observed value is ``None`` is skipped, not failed: a
criterion that could not be applied has not been measured, and failing on it
would rank a model down for not exposing its reasoning. Gradeability is
asserted separately, with ``min_coverage``, which is always measurable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import Field, model_validator

from reasonbench.config import Frozen
from reasonbench.errors import ConfigError

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

    from reasonbench.scoring import GroupSummary

Status = Literal["passed", "failed", "skipped"]


class OverallGate(Frozen):
    """Thresholds over the run as a whole."""

    min_samples: int | None = 1
    min_weighted_mean: float | None = None
    max_failure_rate: float | None = None
    max_cost_usd: float | None = None
    min_coverage: float | None = None


class CriterionGate(Frozen):
    """Thresholds for one criterion."""

    id: str
    min_mean: float | None = None
    max_mean: float | None = None
    min_coverage: float | None = None
    min_scored: int | None = None


class GroupGate(Frozen):
    """Thresholds applied to every group of a grouping."""

    by: tuple[str, ...] = ("model",)
    min_weighted_mean: float | None = None
    max_failure_rate: float | None = None
    require_all_groups: bool = True


class GateSpec(Frozen):
    """The whole of ``gate.yaml``."""

    version: int = 1
    on_unmeasurable: Literal["warn", "fail", "pass"] = "warn"
    overall: OverallGate = Field(default_factory=OverallGate)
    criteria: tuple[CriterionGate, ...] = ()
    groups: tuple[GroupGate, ...] = ()

    @model_validator(mode="after")
    def _unique_criteria(self) -> Self:
        ids = [c.id for c in self.criteria]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate criterion gates: {sorted(duplicates)}")
        return self


class Assertion(Frozen):
    """One threshold, and what the run actually did."""

    id: str
    scope: Literal["overall", "criterion", "group"]
    subject: str = ""
    metric: str
    comparator: Literal[">=", "<="]
    threshold: float
    observed: float | None
    status: Status
    reason: str


class GateResult(Frozen):
    """Every assertion, and whether the run passes."""

    configured: bool
    passed: bool
    assertions: tuple[Assertion, ...] = ()

    @property
    def n_failed(self) -> int:
        """Return how many assertions were breached."""
        return sum(1 for a in self.assertions if a.status == "failed")

    @property
    def n_skipped(self) -> int:
        """Return how many assertions had nothing to measure."""
        return sum(1 for a in self.assertions if a.status == "skipped")

    @property
    def n_passed(self) -> int:
        """Return how many assertions held."""
        return sum(1 for a in self.assertions if a.status == "passed")


def load_gate(path: Path) -> GateSpec:
    """Load and validate ``gate.yaml``."""
    if not path.is_file():
        raise ConfigError(f"no such gate file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    try:
        return GateSpec.model_validate(data)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _check(  # noqa: PLR0913, PLR0917
    ident: str,
    scope: Literal["overall", "criterion", "group"],
    subject: str,
    metric: str,
    comparator: Literal[">=", "<="],
    threshold: float,
    observed: float | None,
    *,
    unmeasurable_status: Status = "skipped",
    skip_reason: str = "nothing to measure",
) -> Assertion:
    if observed is None:
        return Assertion(
            id=ident,
            scope=scope,
            subject=subject,
            metric=metric,
            comparator=comparator,
            threshold=threshold,
            observed=None,
            status=unmeasurable_status,
            reason=skip_reason,
        )
    held = observed >= threshold if comparator == ">=" else observed <= threshold
    return Assertion(
        id=ident,
        scope=scope,
        subject=subject,
        metric=metric,
        comparator=comparator,
        threshold=threshold,
        observed=observed,
        status="passed" if held else "failed",
        reason=(
            f"{metric} {observed:.4g} {comparator} {threshold:.4g}"
            if held
            else f"{metric} {observed:.4g} breaches {comparator} {threshold:.4g}"
        ),
    )


def evaluate(  # noqa: PLR0912
    spec: GateSpec | None,
    overall: GroupSummary,
    groups: dict[tuple[str, ...], list[GroupSummary]],
) -> GateResult:
    """Check every configured threshold against an aggregated run."""
    if spec is None:
        return GateResult(configured=False, passed=True)

    unmeasurable: Status = {
        "warn": "skipped",
        "pass": "passed",
        "fail": "failed",
    }[spec.on_unmeasurable]  # type: ignore[assignment]

    out: list[Assertion] = []
    o = spec.overall
    if o.min_samples is not None:
        out.append(
            _check(
                "overall.min_samples",
                "overall",
                "",
                "n_samples",
                ">=",
                float(o.min_samples),
                float(overall.n_samples),
            )
        )
    if o.min_weighted_mean is not None:
        out.append(
            _check(
                "overall.min_weighted_mean",
                "overall",
                "",
                "weighted_mean",
                ">=",
                o.min_weighted_mean,
                overall.weighted_mean,
                unmeasurable_status=unmeasurable,
                skip_reason="no sample produced a weighted score",
            )
        )
    if o.max_failure_rate is not None:
        rate = overall.n_failed / overall.n_samples if overall.n_samples else None
        out.append(
            _check(
                "overall.max_failure_rate",
                "overall",
                "",
                "failure_rate",
                "<=",
                o.max_failure_rate,
                rate,
            )
        )
    if o.max_cost_usd is not None:
        out.append(
            _check(
                "overall.max_cost_usd",
                "overall",
                "",
                "cost_usd",
                "<=",
                o.max_cost_usd,
                overall.total_cost,
            )
        )
    if o.min_coverage is not None:
        worst = min((c.coverage for c in overall.criteria), default=None)
        out.append(
            _check(
                "overall.min_coverage",
                "overall",
                "",
                "coverage",
                ">=",
                o.min_coverage,
                worst,
            )
        )

    by_id = {c.criterion_id: c for c in overall.criteria}
    for gate in spec.criteria:
        found = by_id.get(gate.id)
        if found is None:
            raise ConfigError(
                f"gate names criterion {gate.id!r}, which this run has no scores for",
                hint=f"scored criteria: {', '.join(sorted(by_id)) or '(none)'}",
            )
        if gate.min_mean is not None:
            out.append(
                _check(
                    f"criterion.{gate.id}.min_mean",
                    "criterion",
                    gate.id,
                    "mean",
                    ">=",
                    gate.min_mean,
                    found.mean,
                    unmeasurable_status=unmeasurable,
                    skip_reason=(f"not gradeable: 0 of {found.n_total} samples scored"),
                )
            )
        if gate.max_mean is not None:
            out.append(
                _check(
                    f"criterion.{gate.id}.max_mean",
                    "criterion",
                    gate.id,
                    "mean",
                    "<=",
                    gate.max_mean,
                    found.mean,
                    unmeasurable_status=unmeasurable,
                    skip_reason="not gradeable",
                )
            )
        if gate.min_coverage is not None:
            out.append(
                _check(
                    f"criterion.{gate.id}.min_coverage",
                    "criterion",
                    gate.id,
                    "coverage",
                    ">=",
                    gate.min_coverage,
                    found.coverage,
                )
            )
        if gate.min_scored is not None:
            out.append(
                _check(
                    f"criterion.{gate.id}.min_scored",
                    "criterion",
                    gate.id,
                    "n_scored",
                    ">=",
                    float(gate.min_scored),
                    float(found.n_scored),
                )
            )

    for group_gate in spec.groups:
        summaries = groups.get(group_gate.by, [])
        if not summaries and group_gate.require_all_groups:
            out.append(
                Assertion(
                    id=f"group.{'+'.join(group_gate.by)}.present",
                    scope="group",
                    subject="+".join(group_gate.by),
                    metric="n_groups",
                    comparator=">=",
                    threshold=1.0,
                    observed=0.0,
                    status="failed",
                    reason="no groups were produced for this grouping",
                )
            )
        for summary in summaries:
            key = "/".join(summary.key)
            if group_gate.min_weighted_mean is not None:
                out.append(
                    _check(
                        f"group.{key}.min_weighted_mean",
                        "group",
                        key,
                        "weighted_mean",
                        ">=",
                        group_gate.min_weighted_mean,
                        summary.weighted_mean,
                        unmeasurable_status=unmeasurable,
                        skip_reason="no weighted score for this group",
                    )
                )
            if group_gate.max_failure_rate is not None:
                rate = (
                    summary.n_failed / summary.n_samples if summary.n_samples else None
                )
                out.append(
                    _check(
                        f"group.{key}.max_failure_rate",
                        "group",
                        key,
                        "failure_rate",
                        "<=",
                        group_gate.max_failure_rate,
                        rate,
                    )
                )

    return GateResult(
        configured=True,
        passed=not any(a.status == "failed" for a in out),
        assertions=tuple(out),
    )
