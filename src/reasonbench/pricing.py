"""Working out what a sample cost when the server will not say.

Only OpenRouter reports ``usage.cost``. Everywhere else the field is absent and
defaults to zero, which leaves ``budget_usd`` looking enforced while nothing
stops the spend. A local price table closes that, and ``require_pricing`` makes
an unpriced model an error in CI rather than a surprise on the invoice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from reasonbench.errors import ConfigError

if TYPE_CHECKING:
    from reasonbench.config import PriceEntry, RunConfig
    from reasonbench.openrouter import Usage

CostSource = Literal["reported", "priced", "unpriced"]

PER_MILLION = 1_000_000
# Above this many consecutive unpriced samples, a run configured to require
# pricing has proven the endpoint will not report one.
UNPRICED_PATIENCE = 5


def price_from(entry: PriceEntry, usage: Usage) -> float:
    """Return the USD cost of one call from a local price entry."""
    billable_prompt = max(usage.prompt_tokens - usage.cached_tokens, 0)
    cached_rate = (
        entry.cached_prompt if entry.cached_prompt is not None else entry.prompt
    )
    total = billable_prompt * entry.prompt + usage.cached_tokens * cached_rate
    total += usage.completion_tokens * entry.completion
    if entry.reasoning is not None:
        total += usage.reasoning_tokens * entry.reasoning
    return total / PER_MILLION


def price_usage(
    usage: Usage, model: str, config: RunConfig
) -> tuple[float, CostSource]:
    """Return ``(cost, source)`` for one sample.

    A cost the provider reported always wins: it already accounts for discounts,
    cache rebates and routing that no local table models.
    """
    if usage.cost > 0:
        return usage.cost, "reported"
    entry = config.pricing.entry_for(model)
    if entry is not None:
        return price_from(entry, usage), "priced"
    return 0.0, "unpriced"


def reports_cost(config: RunConfig) -> bool:
    """Return whether this endpoint is expected to report a cost itself."""
    if config.reports_cost is not None:
        return config.reports_cost
    return config.base_url is None or "openrouter.ai" in config.base_url


def requires_pricing(config: RunConfig, *, in_ci: bool) -> bool:
    """Return whether an unpriced model should stop the run before it starts."""
    if config.require_pricing is not None:
        return config.require_pricing
    return in_ci


def check_pricing(config: RunConfig, models: tuple[str, ...], *, in_ci: bool) -> None:
    """Fail before spending anything if the budget could not be enforced."""
    if not requires_pricing(config, in_ci=in_ci) or reports_cost(config):
        return
    unpriced = [m for m in models if config.pricing.entry_for(m) is None]
    if not unpriced:
        return
    raise ConfigError(
        f"no price for {', '.join(unpriced)}, and this endpoint does not "
        f"report one, so budget_usd would not be enforced",
        hint=(
            "add pricing.per_mtok entries, set max_total_tokens instead, or "
            "set require_pricing: false to accept an unenforced budget."
        ),
    )
