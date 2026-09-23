"""Cost accounting on endpoints that do not report one."""

from __future__ import annotations

import pytest
from conftest import CONFIGS

from reasonbench.config import RunConfig, load_run_config
from reasonbench.errors import ConfigError
from reasonbench.openrouter import parse_usage
from reasonbench.pricing import check_pricing, price_usage, reports_cost

USAGE = parse_usage(
    {
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 800,
            "total_tokens": 2000,
            "completion_tokens_details": {"reasoning_tokens": 600},
            "prompt_tokens_details": {"cached_tokens": 200},
        }
    }
)


@pytest.fixture
def base():
    return load_run_config(CONFIGS / "models.yaml").model_dump()


def config(base, **overrides) -> RunConfig:
    return RunConfig.model_validate({**base, **overrides})


def test_a_reported_cost_always_wins(base):
    usage = parse_usage({"usage": {"completion_tokens": 10, "cost": 0.5}})
    priced = config(base, pricing={"per_mtok": {"m": {"prompt": 99, "completion": 99}}})
    assert price_usage(usage, "m", priced) == (0.5, "reported")


def test_an_endpoint_without_a_cost_leaves_the_budget_unenforced(base):
    local = config(base, base_url="http://localhost:8000/v1")
    assert price_usage(USAGE, "m", local) == (0.0, "unpriced")


def test_a_local_price_table_restores_the_cost(base):
    local = config(
        base,
        base_url="http://localhost:8000/v1",
        pricing={"per_mtok": {"m": {"prompt": 1.25, "completion": 10.0}}},
    )
    cost, source = price_usage(USAGE, "m", local)
    # cached tokens are a subset of prompt tokens, so they are not added twice
    expected = ((1200 - 200) * 1.25 + 200 * 1.25 + 800 * 10.0) / 1_000_000
    assert source == "priced"
    assert cost == pytest.approx(expected)


def test_reasoning_tokens_are_not_billed_twice_by_default(base):
    """They are counted inside completion_tokens on nearly every server."""
    local = config(
        base,
        base_url="http://localhost:8000/v1",
        pricing={"per_mtok": {"m": {"prompt": 0.0, "completion": 1.0}}},
    )
    cost, _ = price_usage(USAGE, "m", local)
    assert cost == pytest.approx(800 / 1_000_000)


def test_reasoning_can_be_billed_separately_when_a_provider_does(base):
    local = config(
        base,
        base_url="http://localhost:8000/v1",
        pricing={
            "per_mtok": {"m": {"prompt": 0.0, "completion": 1.0, "reasoning": 2.0}}
        },
    )
    cost, _ = price_usage(USAGE, "m", local)
    assert cost == pytest.approx((800 * 1.0 + 600 * 2.0) / 1_000_000)


def test_a_cheaper_cached_rate_is_honoured(base):
    local = config(
        base,
        base_url="http://localhost:8000/v1",
        pricing={
            "per_mtok": {"m": {"prompt": 10.0, "completion": 0.0, "cached_prompt": 1.0}}
        },
    )
    cost, _ = price_usage(USAGE, "m", local)
    assert cost == pytest.approx((1000 * 10.0 + 200 * 1.0) / 1_000_000)


def test_a_wildcard_price_covers_every_model(base):
    local = config(
        base,
        base_url="http://localhost:8000/v1",
        pricing={"per_mtok": {"*": {"prompt": 0.0, "completion": 0.0}}},
    )
    assert price_usage(USAGE, "anything", local) == (0.0, "priced")


def test_openrouter_is_assumed_to_report_a_cost(base):
    assert reports_cost(config(base)) is True
    assert reports_cost(config(base, base_url="http://localhost:8000/v1")) is False
    assert reports_cost(config(base, base_url="http://x/v1", reports_cost=True)) is True


def test_ci_refuses_to_start_with_an_unenforceable_budget(base):
    local = config(base, base_url="http://localhost:8000/v1")
    with pytest.raises(ConfigError, match="budget_usd would not be enforced"):
        check_pricing(local, ("m",), in_ci=True)


def test_outside_ci_an_unpriced_model_is_allowed(base):
    local = config(base, base_url="http://localhost:8000/v1")
    check_pricing(local, ("m",), in_ci=False)


def test_a_priced_model_passes_the_ci_guard(base):
    local = config(
        base,
        base_url="http://localhost:8000/v1",
        pricing={"per_mtok": {"m": {"prompt": 1.0, "completion": 1.0}}},
    )
    check_pricing(local, ("m",), in_ci=True)


def test_require_pricing_can_be_waived_explicitly(base):
    local = config(base, base_url="http://localhost:8000/v1", require_pricing=False)
    check_pricing(local, ("m",), in_ci=True)
