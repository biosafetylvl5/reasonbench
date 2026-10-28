"""Responses made elsewhere, stored as a run so they can be graded.

A prompt whose dataset sets ``recorded`` carries one response per row. ``run``
stores those instead of calling a model, so ``score``, ``report`` and ``gate``
treat them like any other run. There is no sweep: each row is one sample, and
its case id is its sample id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasonbench.config import PromptSpec, Variant
from reasonbench.errors import ConfigError
from reasonbench.openrouter import ReasoningAvailability, ReasoningTrace, SampleResult
from reasonbench.sweep import Sample, render_fingerprint

if TYPE_CHECKING:
    from reasonbench.dataset import Case, CaseSet


def is_recorded(prompt: PromptSpec) -> bool:
    """Return whether this prompt's responses come from its dataset."""
    return prompt.dataset is not None and prompt.dataset.recorded is not None


def recorded_prompt(prompt: PromptSpec, cases: CaseSet) -> PromptSpec:
    """Return the prompt with a variant for every label in the variant column.

    A label that names a variant uses that variant's template; any other label
    uses the first variant's. The labels then group results, so ``report
    --group-by variant_id`` splits by whatever the column records.
    """
    known = {v.id for v in prompt.variants}
    template = prompt.variants[0].user
    labels = dict.fromkeys(
        c.recorded.variant
        for c in cases.cases
        if c.recorded and c.recorded.variant and c.recorded.variant not in known
    )
    if not labels:
        return prompt
    added = tuple(Variant(id=label, user=template) for label in labels)
    return prompt.model_copy(update={"variants": prompt.variants + added})


def recorded_samples(prompt: PromptSpec, cases: CaseSet) -> tuple[Sample, ...]:
    """Return one sample per case. ``prompt`` must come from recorded_prompt."""
    variants = {v.id: v for v in prompt.variants}
    out = []
    for case in cases.cases:
        if case.recorded is None:  # pragma: no cover - load_cases guarantees it
            raise ConfigError(f"case {case.case_id!r} has no recorded response")
        variant = variants[case.recorded.variant or prompt.variants[0].id]
        out.append(
            Sample(
                sample_id=case.case_id,
                prompt_id=prompt.id,
                variant_id=variant.id,
                model=case.recorded.model,
                temperature=0.0,
                reasoning_effort=None,
                repeat=0,
                case_id=case.case_id,
                prompt_fingerprint=render_fingerprint(prompt, variant, case),
            )
        )
    return tuple(out)


def recorded_result(
    sample: Sample, case: Case, source: str
) -> tuple[SampleResult, dict[str, Any]]:
    """Return the stored result for one recorded case, and its raw artifact."""
    if case.recorded is None:  # pragma: no cover - load_cases guarantees it
        raise ConfigError(f"case {case.case_id!r} has no recorded response")
    text = case.recorded.reasoning or ""
    trace = ReasoningTrace(
        availability=(
            ReasoningAvailability.FULL_TEXT if text else ReasoningAvailability.ABSENT
        ),
        text=text,
        source="recorded" if text else "none",
    )
    result = SampleResult(
        sample=sample,
        ok=True,
        output=case.recorded.output,
        reasoning=trace,
        served_model=case.recorded.model,
        provider="recorded",
    )
    raw = {
        "recorded_from": source,
        "line": case.source_line,
        "output": case.recorded.output,
        "reasoning": case.recorded.reasoning,
    }
    return result, raw
