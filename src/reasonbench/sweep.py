"""Expansion of a run config into the flat list of samples. No I/O."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import TYPE_CHECKING

from reasonbench.config import Frozen, PromptSpec, RunConfig

if TYPE_CHECKING:
    from reasonbench.config import Variant
    from reasonbench.dataset import Case, CaseSet


class Sample(Frozen):
    """One planned API call: a single cell of the sweep, at one repeat index."""

    sample_id: str
    prompt_id: str
    variant_id: str
    model: str
    temperature: float
    reasoning_effort: str | None
    repeat: int
    case_id: str = ""
    prompt_fingerprint: str = ""

    @property
    def cell(self) -> tuple[str, str, float, str]:
        """Return the grouping key shared by every repeat of this cell."""
        return (
            self.model,
            self.variant_id,
            self.temperature,
            self.reasoning_effort or "none",
        )


def render_fingerprint(prompt: PromptSpec, variant: Variant, case: Case | None) -> str:
    """Digest the material that shapes one request.

    Per variant, so editing ``cot`` does not invalidate ``plain``. The rubric is
    excluded: re-scoring a stored run against a rewritten rubric must stay free.
    """
    payload = json.dumps(
        {
            "system": prompt.system,
            "user": variant.user,
            "variables": prompt.variables,
            "case_vars": case.variables if case else {},
            "attachments": [
                {"id": a.id, "media_type": a.media_type}
                for a in prompt.attachments
                if a.applies_to(variant.id)
            ],
            "case_images": [
                {"column": i.column, "sha256": i.sha256}
                for i in (case.images if case else ())
                if i.applies_to(variant.id)
            ],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def make_sample_id(  # noqa: PLR0913
    *,
    prompt_id: str,
    variant_id: str,
    model: str,
    temperature: float,
    reasoning_effort: str | None,
    repeat: int,
    case_id: str = "",
    prompt_fingerprint: str = "",
) -> str:
    """Return a stable content-addressed id for one sample.

    The id depends only on what defines the call, so re-running an interrupted
    sweep can skip work already committed simply by comparing ids.
    """
    payload = json.dumps(
        {
            "prompt_id": prompt_id,
            "variant_id": variant_id,
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "repeat": repeat,
            "case_id": case_id,
            "prompt_fingerprint": prompt_fingerprint,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def expand(
    config: RunConfig, prompt: PromptSpec, cases: CaseSet | None = None
) -> tuple[Sample, ...]:
    """Return every sample in the cartesian product of the sweep axes.

    The product is ``models x cases x variants x temperature x reasoning_effort``,
    each repeated ``sweep.repeats`` times. With no dataset there is one implicit
    case, so a prompt that predates datasets expands exactly as before.
    """
    case_list: tuple[Case | None, ...] = cases.cases if cases else (None,)
    fingerprints = {
        (variant.id, case.case_id if case else ""): render_fingerprint(
            prompt, variant, case
        )
        for variant in prompt.variants
        for case in case_list
    }
    axes = itertools.product(
        config.models,
        case_list,
        prompt.variants,
        config.sweep.temperature,
        config.sweep.reasoning_effort,
        range(config.sweep.repeats),
    )
    out = []
    for model, case, variant, temperature, effort, repeat in axes:
        case_id = case.case_id if case else ""
        fingerprint = fingerprints[variant.id, case_id]
        out.append(
            Sample(
                sample_id=make_sample_id(
                    prompt_id=prompt.id,
                    variant_id=variant.id,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=effort,
                    repeat=repeat,
                    case_id=case_id,
                    prompt_fingerprint=fingerprint,
                ),
                prompt_id=prompt.id,
                variant_id=variant.id,
                model=model,
                temperature=temperature,
                reasoning_effort=effort,
                repeat=repeat,
                case_id=case_id,
                prompt_fingerprint=fingerprint,
            )
        )
    return tuple(out)


def estimate_cost_usd(
    samples: tuple[Sample, ...],
    *,
    prompt_tokens: int = 400,
    completion_tokens: int = 1200,
    price_per_mtok: float = 0.5,
) -> float:
    """Return a rough pre-flight cost estimate for a planned sweep.

    Crude: it catches an accidental 10,000-call sweep. It will not
    predict the invoice. Real cost comes from the API response.
    """
    tokens = len(samples) * (prompt_tokens + completion_tokens)
    return tokens / 1_000_000 * price_per_mtok
