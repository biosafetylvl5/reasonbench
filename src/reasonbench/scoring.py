"""Rubric scoring: deterministic checks, the judge, and aggregation.

Deterministic criteria are pure functions. Judge criteria go out in one
structured-output call per sample, so every judged criterion for a sample is
graded together.

A criterion targeting the reasoning trace scores ``None`` when the sample has no
readable trace. ``None`` aggregates as missing coverage, not as a zero.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from jinja2 import UndefinedError

from reasonbench.config import (
    Check,
    ContainsCheck,
    Criterion,
    DeterministicCriterion,
    ExactCheck,
    Frozen,
    JudgeCriterion,
    JudgeSettings,
    NumericCheck,
    PromptSpec,
    RegexCheck,
    Rubric,
    Target,
    TextScope,
    looks_templated,
    render_template,
)
from reasonbench.errors import ConfigError
from reasonbench.openrouter import FatalAPIError, OpenRouterClient, RetryableError
from reasonbench.storage import SampleRow, ScoreRow

if TYPE_CHECKING:
    from collections.abc import Callable


CriterionT = TypeVar("CriterionT", DeterministicCriterion, JudgeCriterion)

# Servers that reject structured output say so in the body. Matching narrowly
# keeps a bad model slug or a real outage from being mistaken for one.
UNSUPPORTED_MARKERS = (
    "response_format",
    "json_schema",
    "guided",
    "structured",
    "not supported",
    "unsupported",
    "unrecognized",
)

# Strongest first. Each rung constrains the judge less than the one above.
Rung = Literal["strict", "json_object", "none"]
RUNGS: tuple[Rung, ...] = ("strict", "json_object", "none")

JSON_ONLY = "Reply with a single JSON object and nothing else. No prose, no code fence."

TARGET_WORDING = {
    Target.OUTPUT: "the final answer only; ignore the reasoning trace",
    Target.REASONING: "the reasoning trace only; ignore the final answer",
    Target.BOTH: "the reasoning trace and the final answer together",
}


def _check_regex(check: RegexCheck, text: str) -> bool:
    flags = re.IGNORECASE if check.ignore_case else 0
    return re.search(check.pattern, text, flags) is not None


def _check_exact(check: ExactCheck, text: str) -> bool:
    actual = text.strip() if check.strip else text
    expected = check.expected.strip() if check.strip else check.expected
    if check.ignore_case:
        return actual.casefold() == expected.casefold()
    return actual == expected


def _check_contains(check: ContainsCheck, text: str) -> bool:
    if check.ignore_case:
        return check.expected.casefold() in text.casefold()
    return check.expected in text


def _check_numeric(check: NumericCheck, text: str) -> bool:
    pattern = check.extract or r"-?\d+(?:\.\d+)?"
    match = re.search(pattern, text)
    if match is None:
        return False
    raw = match.group(1) if match.groups() else match.group(0)
    try:
        value = float(raw)
    except ValueError:
        return False
    return abs(value - check.expected) <= check.tol


# The discriminator on ``Check`` guarantees the lookup and the argument type
# agree, which the signature of a plain dict cannot express.
CHECKS: dict[str, Callable[[Any, str], bool]] = {
    "regex": _check_regex,
    "exact": _check_exact,
    "contains": _check_contains,
    "numeric": _check_numeric,
}


def apply_check(check: Check, text: str) -> bool:
    """Run one deterministic check against ``text``."""
    return CHECKS[check.type](check, text)


def apply_scope(text: str, scope: TextScope) -> str:
    """Narrow ``text`` to the slice a check should read.

    Blank lines are ignored, so trailing newlines do not defeat
    :attr:`TextScope.LAST_LINE`.
    """
    if scope is TextScope.FULL:
        return text
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1] if scope is TextScope.LAST_LINE else lines[0]


def select_text(row: SampleRow, target: Target) -> str | None:
    """Return the text a criterion should be applied to, or ``None`` for N/A.

    With no trace, ``BOTH`` falls back to the answer alone; the answer is
    still gradeable.
    """
    reasoning = row.readable_reasoning
    if target is Target.OUTPUT:
        return row.output or None
    if target is Target.REASONING:
        return reasoning
    parts = [p for p in (reasoning, row.output) if p]
    return "\n\n".join(parts) if parts else None


def _render_field(text: str, case_vars: dict[str, Any], where: str) -> str:
    try:
        return render_template(text, case_vars)
    except (UndefinedError, TypeError, ValueError) as exc:
        # A filter such as re_escape fails on StrictUndefined with TypeError,
        # before jinja reports the undefined name. Both mean the same thing:
        # the column this criterion needs is not in the row.
        raise ConfigError(
            f"{where}: cannot render against this case: {exc}",
            hint=f"available columns: {', '.join(sorted(case_vars)) or '(none)'}",
        ) from exc


def resolve_criterion(
    criterion: CriterionT, case_vars: dict[str, Any] | None
) -> CriterionT:
    """Render a criterion's templated fields against one case's variables.

    Only strings containing Jinja syntax are touched, so an untemplated rubric
    comes back byte-identical. Patterns are compiled here: a regex broken by a
    data value should name the criterion, not surface as a traceback later.
    """
    if not case_vars:
        return criterion
    where = f"criterion {criterion.id!r}"
    if isinstance(criterion, DeterministicCriterion):
        check = criterion.check
        updates: dict[str, Any] = {}
        for field in ("pattern", "expected", "extract"):
            value = getattr(check, field, None)
            if isinstance(value, str) and looks_templated(value):
                updates[field] = _render_field(value, case_vars, where)
        if not updates:
            return criterion
        check = check.model_copy(update=updates)
        if isinstance(check, RegexCheck):
            try:
                re.compile(check.pattern)
            except re.error as exc:
                raise ConfigError(
                    f"{where}: rendered pattern is not a valid regex: {exc}",
                    hint=f"pattern was {check.pattern!r}",
                ) from exc
        return criterion.model_copy(update={"check": check})

    guidance = criterion.guidance
    updates = {}
    if looks_templated(guidance.summary):
        updates["summary"] = _render_field(guidance.summary, case_vars, where)
    levels = {
        k: (_render_field(v, case_vars, where) if looks_templated(v) else v)
        for k, v in guidance.levels.items()
    }
    if levels != guidance.levels:
        updates["levels"] = levels
    if guidance.notes and looks_templated(guidance.notes):
        updates["notes"] = _render_field(guidance.notes, case_vars, where)
    if not updates:
        return criterion
    return criterion.model_copy(
        update={"guidance": guidance.model_copy(update=updates)}
    )


def score_deterministic(
    criterion: DeterministicCriterion,
    row: SampleRow,
) -> ScoreRow:
    """Score one deterministic criterion for one sample."""
    text = select_text(row, criterion.target)
    if text is None:
        return ScoreRow(
            sample_id=row.sample_id,
            criterion_id=criterion.id,
            judge_repeat=0,
            kind="deterministic",
            target=str(criterion.target),
            weight=criterion.weight,
            score=None,
            normalized=None,
            applicable=False,
            reason="no text available for this target",
        )
    scoped = apply_scope(text, criterion.scope)
    passed = apply_check(criterion.check, scoped)
    return ScoreRow(
        sample_id=row.sample_id,
        criterion_id=criterion.id,
        judge_repeat=0,
        kind="deterministic",
        target=str(criterion.target),
        weight=criterion.weight,
        score=float(passed),
        normalized=float(passed),
        applicable=True,
        reason=(
            f"{criterion.check.type} check on {criterion.scope} "
            f"{'passed' if passed else 'failed'}"
        ),
    )


def render_guidance(criterion: JudgeCriterion) -> str:
    """Render one criterion's guidance block for the judge prompt."""
    low, high = criterion.scale
    guidance = criterion.guidance
    lines = [
        f"### {criterion.id}   (integer score from {low} to {high})",
        f"Evaluate: {TARGET_WORDING[criterion.target]}.",
        "",
        guidance.summary.strip(),
    ]
    if guidance.levels:
        lines += ["", "Score anchors:"]
        lines += [
            f"  {score} = {guidance.levels[score].strip()}"
            for score in sorted(guidance.levels, reverse=True)
        ]
    if guidance.positive_indicators:
        lines += ["", "Evidence that raises the score:"]
        lines += [f"  + {item}" for item in guidance.positive_indicators]
    if guidance.negative_indicators:
        lines += ["", "Evidence that lowers the score:"]
        lines += [f"  - {item}" for item in guidance.negative_indicators]
    if guidance.notes:
        lines += ["", f"Notes: {guidance.notes.strip()}"]
    return "\n".join(lines)


def applicable_judge_criteria(
    rubric: Rubric,
    row: SampleRow,
) -> tuple[tuple[JudgeCriterion, ...], tuple[JudgeCriterion, ...]]:
    """Split judge criteria into ``(gradeable, not_applicable)`` for a sample."""
    gradeable, not_applicable = [], []
    for criterion in rubric.judge_criteria:
        if select_text(row, criterion.target) is None:
            not_applicable.append(criterion)
        else:
            gradeable.append(criterion)
    return tuple(gradeable), tuple(not_applicable)


def build_judge_prompt(  # noqa: PLR0913
    prompt: PromptSpec,
    row: SampleRow,
    criteria: tuple[JudgeCriterion, ...],
    settings: JudgeSettings,
    case_vars: dict[str, Any] | None = None,
    *,
    json_only: bool = False,
) -> str:
    """Build the judge's user message for one sample."""
    variant = next(v for v in prompt.variants if v.id == row.variant_id)
    _, question = prompt.render(variant, case_vars)
    trace = row.readable_reasoning

    sections = [
        "## Question that was posed",
        question.strip(),
        "",
        "## Response under evaluation",
    ]
    if not settings.blind:
        sections += [f"Model: {row.model}", ""]
    if trace:
        label = (
            "Reasoning trace (summary only)"
            if row.reasoning_availability == "summary_only"
            else "Reasoning trace"
        )
        sections += [f"### {label}", trace[: settings.max_chars].strip(), ""]
    else:
        sections += [
            "### Reasoning trace",
            "(this model returned no readable reasoning trace)",
            "",
        ]
    sections += [
        "### Final answer",
        row.output[: settings.max_chars].strip() or "(empty)",
        "",
        "## Criteria",
        *[render_guidance(resolve_criterion(c, case_vars)) + "\n" for c in criteria],
        "Score every criterion independently against its anchors. Judge only "
        "what the criterion names. Justify each score in one sentence, quoting "
        "the response where useful.",
    ]
    if json_only:
        # json_object mode promises valid JSON, not this shape; the lowest rung
        # promises nothing at all.
        sections += ["", _schema_sketch(criteria), "", JSON_ONLY]
    return "\n".join(sections)


def _schema_sketch(criteria: tuple[JudgeCriterion, ...]) -> str:
    """Describe the expected object for rungs with no schema enforcement."""
    fields = ", ".join(
        f'"{c.id}": {{"score": <{c.scale[0]}-{c.scale[1]}>, "justification": "..."}}'
        for c in criteria
    )
    return "Reply with this object: {" + fields + "}"


def build_judge_schema(criteria: tuple[JudgeCriterion, ...]) -> dict[str, Any]:
    """Build a strict JSON schema so judge scores come back already validated."""
    properties = {
        criterion.id: {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": criterion.scale[0],
                    "maximum": criterion.scale[1],
                },
                "justification": {"type": "string"},
            },
            "required": ["score", "justification"],
            "additionalProperties": False,
        }
        for criterion in criteria
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rubric_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": [c.id for c in criteria],
                "additionalProperties": False,
            },
        },
    }


def build_judge_request(  # noqa: PLR0913, PLR0917
    prompt: PromptSpec,
    row: SampleRow,
    criteria: tuple[JudgeCriterion, ...],
    settings: JudgeSettings,
    case_vars: dict[str, Any] | None = None,
    rung: Rung = "strict",
) -> dict[str, Any]:
    """Build the full chat-completions body for one judge call."""
    system = settings.persona
    if prompt.rubric.judge_instructions:
        system = f"{system}\n\n{prompt.rubric.judge_instructions.strip()}"
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": build_judge_prompt(
                    prompt,
                    row,
                    criteria,
                    settings,
                    case_vars,
                    json_only=rung != "strict",
                ),
            },
        ],
        "temperature": settings.temperature,
    }
    if rung == "strict":
        body["response_format"] = build_judge_schema(criteria)
    elif rung == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def parse_judge_response(
    raw: dict[str, Any],
    row: SampleRow,
    criteria: tuple[JudgeCriterion, ...],
    judge_repeat: int,
) -> list[ScoreRow]:
    """Turn a judge response into score rows, one per criterion."""
    content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
    try:
        parsed = json.loads(content or "{}")
    except json.JSONDecodeError:
        parsed = {}

    rows = []
    for criterion in criteria:
        entry = parsed.get(criterion.id) or {}
        raw_score = entry.get("score")
        low, high = criterion.scale
        clamped = False
        if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
            # bool is an int subclass, so {"score": true} would otherwise be 1.0.
            score, normalized = None, None
            applicable, reason = False, "judge returned no score for this criterion"
        elif not math.isfinite(float(raw_score)):
            score, normalized = None, None
            applicable = False
            reason = f"judge returned a non-finite score ({raw_score!r})"
        else:
            score = float(raw_score)
            bounded = min(max(score, float(low)), float(high))
            clamped = bounded != score
            normalized = (bounded - low) / (high - low)
            applicable, reason = True, str(entry.get("justification") or "")
            if clamped:
                # Unclamped this lands outside 0-1 and silently moves every
                # weighted mean the sample contributes to.
                reason = (
                    f"judge returned {score:g}, outside scale {low}-{high}; "
                    f"clamped to {bounded:g}. {reason}"
                )
        rows.append(
            ScoreRow(
                sample_id=row.sample_id,
                criterion_id=criterion.id,
                judge_repeat=judge_repeat,
                kind="judge",
                target=str(criterion.target),
                weight=criterion.weight,
                score=score,
                normalized=normalized,
                applicable=applicable,
                reason=reason,
                clamped=clamped,
            ),
        )
    return rows


def na_rows(
    row: SampleRow,
    criteria: tuple[JudgeCriterion, ...],
    judge_repeat: int = 0,
) -> list[ScoreRow]:
    """Build N/A score rows for criteria that cannot be graded on this sample."""
    return [
        ScoreRow(
            sample_id=row.sample_id,
            criterion_id=criterion.id,
            judge_repeat=judge_repeat,
            kind="judge",
            target=str(criterion.target),
            weight=criterion.weight,
            score=None,
            normalized=None,
            applicable=False,
            reason=f"no readable reasoning trace ({row.reasoning_availability})",
        )
        for criterion in criteria
    ]


def failure_rows(
    row: SampleRow,
    criteria: tuple[JudgeCriterion, ...],
    judge_repeat: int,
    reason: str,
) -> list[ScoreRow]:
    """Build N/A rows for criteria the judge could not score."""
    return [
        ScoreRow(
            sample_id=row.sample_id,
            criterion_id=criterion.id,
            judge_repeat=judge_repeat,
            kind="judge",
            target=str(criterion.target),
            weight=criterion.weight,
            score=None,
            normalized=None,
            applicable=False,
            reason=reason,
        )
        for criterion in criteria
    ]


class _UnsupportedFormatError(Exception):
    """The endpoint refused this response format, so try a weaker one."""


def _rejects_format(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in UNSUPPORTED_MARKERS)


MAX_DOWNGRADES = 2


class JudgeScorer:
    """Grades samples with the judge model.

    A reply that does not parse is retried before being recorded N/A.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        prompt: PromptSpec,
        settings: JudgeSettings,
        on_raw: Callable[[str, int, int, dict[str, Any]], None] | None = None,
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._settings = settings
        self._on_raw = on_raw
        self._rung = settings.structured_output
        self._downgrades = 0
        self.cost = 0.0
        self.parse_retries = 0

    async def _attempt(
        self,
        body: dict[str, Any],
        row: SampleRow,
        criteria: tuple[JudgeCriterion, ...],
        repeat: int,
        attempt: int,
    ) -> list[ScoreRow] | None:
        """Make one judge call. Return rows, or ``None`` if it must be retried."""
        try:
            raw = await self._client.complete(body)
        except FatalAPIError as exc:
            if self._rung != RUNGS[-1] and _rejects_format(str(exc)):
                raise _UnsupportedFormatError(str(exc)) from None
            return failure_rows(row, criteria, repeat, f"judge call failed: {exc}")
        except RetryableError as exc:
            return failure_rows(row, criteria, repeat, f"judge call failed: {exc}")

        self.cost += float((raw.get("usage") or {}).get("cost") or 0.0)
        if self._on_raw is not None:
            self._on_raw(row.sample_id, repeat, attempt, raw)

        rows = parse_judge_response(raw, row, criteria, repeat)
        return rows if all(r.applicable for r in rows) else None

    def _lower_rung(self) -> bool:
        """Drop to a less demanding response format. Returns False at the floor."""
        index = RUNGS.index(self._rung)
        if index + 1 >= len(RUNGS) or self._downgrades >= MAX_DOWNGRADES:
            return False
        self._rung = RUNGS[index + 1]
        self._downgrades += 1
        return True

    async def score(
        self, row: SampleRow, case_vars: dict[str, Any] | None = None
    ) -> list[ScoreRow]:
        """Return every judge score for one sample, including N/A rows."""
        gradeable, skipped = applicable_judge_criteria(self._prompt.rubric, row)
        rows = na_rows(row, skipped)
        if not gradeable:
            return rows

        for repeat in range(self._settings.repeats):
            attempt = 0
            while attempt <= self._settings.parse_retries:
                body = build_judge_request(
                    self._prompt,
                    row,
                    gradeable,
                    self._settings,
                    case_vars,
                    rung=self._rung,
                )
                try:
                    scored = await self._attempt(body, row, gradeable, repeat, attempt)
                except _UnsupportedFormatError as exc:
                    if self._lower_rung():
                        # A downgrade is not a parse failure. Counting it as an
                        # attempt means parse_retries of 0 never reaches the
                        # rung that works.
                        continue
                    rows += failure_rows(
                        row,
                        gradeable,
                        repeat,
                        f"judge rejected every response format: {exc}",
                    )
                    break
                if scored is not None:
                    rows += scored
                    break
                self.parse_retries += 1
                attempt += 1
            else:
                rows += failure_rows(
                    row,
                    gradeable,
                    repeat,
                    f"judge output did not parse after "
                    f"{self._settings.parse_retries + 1} attempts",
                )
        return rows


def score_sample_deterministic(
    rubric: Rubric,
    row: SampleRow,
    case_vars: dict[str, Any] | None = None,
) -> list[ScoreRow]:
    """Score every deterministic criterion for one sample."""
    out = []
    for criterion in rubric.criteria:
        if isinstance(criterion, DeterministicCriterion):
            resolved = resolve_criterion(criterion, case_vars)
            out.append(score_deterministic(resolved, row))
    return out


class CriterionSummary(Frozen):
    """Aggregated results for one criterion within one group."""

    criterion_id: str
    kind: str
    target: str
    weight: float
    mean: float | None
    stdev: float | None
    n_scored: int
    n_total: int

    @property
    def coverage(self) -> float:
        """Return the fraction of samples this criterion could be applied to."""
        return self.n_scored / self.n_total if self.n_total else 0.0


class GroupSummary(Frozen):
    """Aggregated results for one group of samples."""

    key: tuple[str, ...]
    n_samples: int
    n_failed: int
    n_scored: int
    weighted_mean: float | None
    weighted_stdev: float | None
    criteria: tuple[CriterionSummary, ...]
    mean_reasoning_tokens: float
    mean_completion_tokens: float
    mean_latency_s: float
    total_cost: float
    availability: dict[str, int]


def collapse_judge_repeats(
    scores: list[ScoreRow],
) -> dict[tuple[str, str], float | None]:
    """Average judge repeats so each ``(sample, criterion)`` has one value."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in scores:
        key = (row.sample_id, row.criterion_id)
        seen.add(key)
        if row.applicable and row.normalized is not None:
            buckets[key].append(row.normalized)
    return {
        key: (statistics.fmean(buckets[key]) if buckets[key] else None) for key in seen
    }


def weighted_sample_score(
    criteria: tuple[Criterion, ...],
    collapsed: dict[tuple[str, str], float | None],
    sample_id: str,
) -> float | None:
    """Return the weighted mean of applicable criteria for one sample."""
    pairs = [
        (c.weight, value)
        for c in criteria
        if (value := collapsed.get((sample_id, c.id))) is not None
    ]
    if not pairs:
        return None
    total_weight = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / total_weight


def aggregate(
    samples: list[SampleRow],
    scores: list[ScoreRow],
    rubric: Rubric,
    group_by: tuple[str, ...] = ("model",),
) -> list[GroupSummary]:
    """Aggregate samples and scores into per-group summaries."""
    collapsed = collapse_judge_repeats(scores)
    groups: dict[tuple[str, ...], list[SampleRow]] = defaultdict(list)
    for row in samples:
        groups[tuple(str(getattr(row, field)) for field in group_by)].append(row)
    if not groups and not group_by:
        # The whole-run group exists even when the run is empty, so a gate can
        # fail on min_samples rather than finding nothing to assert against.
        groups[()] = []

    summaries = []
    for key, rows in sorted(groups.items()):
        ok_rows = [r for r in rows if r.ok]
        per_sample = [
            value
            for r in ok_rows
            if (value := weighted_sample_score(rubric.criteria, collapsed, r.sample_id))
            is not None
        ]
        criteria_summaries = []
        for criterion in rubric.criteria:
            values = [
                value
                for r in ok_rows
                if (value := collapsed.get((r.sample_id, criterion.id))) is not None
            ]
            criteria_summaries.append(
                CriterionSummary(
                    criterion_id=criterion.id,
                    kind=criterion.kind,
                    target=str(criterion.target),
                    weight=criterion.weight,
                    mean=statistics.fmean(values) if values else None,
                    stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                    n_scored=len(values),
                    n_total=len(ok_rows),
                ),
            )
        availability: dict[str, int] = defaultdict(int)
        for r in ok_rows:
            availability[str(r.reasoning_availability)] += 1

        summaries.append(
            GroupSummary(
                key=key,
                n_samples=len(rows),
                n_failed=len(rows) - len(ok_rows),
                n_scored=len(per_sample),
                weighted_mean=statistics.fmean(per_sample) if per_sample else None,
                weighted_stdev=(
                    statistics.stdev(per_sample) if len(per_sample) > 1 else 0.0
                ),
                criteria=tuple(criteria_summaries),
                mean_reasoning_tokens=(
                    statistics.fmean([r.reasoning_tokens for r in ok_rows])
                    if ok_rows
                    else 0.0
                ),
                mean_completion_tokens=(
                    statistics.fmean([r.completion_tokens for r in ok_rows])
                    if ok_rows
                    else 0.0
                ),
                mean_latency_s=(
                    statistics.fmean([r.latency_s for r in ok_rows]) if ok_rows else 0.0
                ),
                total_cost=sum(r.cost for r in rows),
                availability=dict(availability),
            ),
        )
    return summaries
