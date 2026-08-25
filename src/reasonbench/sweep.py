"""Expansion of a run config into the flat list of samples. No I/O."""

from __future__ import annotations

import hashlib
import itertools
import json

from reasonbench.config import Frozen, PromptSpec, RunConfig


class Sample(Frozen):
    """One planned API call: a single cell of the sweep, at one repeat index."""

    sample_id: str
    prompt_id: str
    variant_id: str
    model: str
    temperature: float
    reasoning_effort: str | None
    repeat: int

    @property
    def cell(self) -> tuple[str, str, float, str]:
        """Return the grouping key shared by every repeat of this cell."""
        return (
            self.model,
            self.variant_id,
            self.temperature,
            self.reasoning_effort or "none",
        )


def make_sample_id(  # noqa: PLR0913
    *,
    prompt_id: str,
    variant_id: str,
    model: str,
    temperature: float,
    reasoning_effort: str | None,
    repeat: int,
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
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def expand(config: RunConfig, prompt: PromptSpec) -> tuple[Sample, ...]:
    """Return every sample in the cartesian product of the sweep axes.

    The product is ``models x variants x temperature x reasoning_effort``,
    each repeated ``sweep.repeats`` times.
    """
    axes = itertools.product(
        config.models,
        prompt.variants,
        config.sweep.temperature,
        config.sweep.reasoning_effort,
        range(config.sweep.repeats),
    )
    return tuple(
        Sample(
            sample_id=make_sample_id(
                prompt_id=prompt.id,
                variant_id=variant.id,
                model=model,
                temperature=temperature,
                reasoning_effort=effort,
                repeat=repeat,
            ),
            prompt_id=prompt.id,
            variant_id=variant.id,
            model=model,
            temperature=temperature,
            reasoning_effort=effort,
            repeat=repeat,
        )
        for model, variant, temperature, effort, repeat in axes
    )


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
