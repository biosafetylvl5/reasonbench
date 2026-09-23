"""Which parameter carries reasoning_effort, and when the axis is a lie."""

from __future__ import annotations

import pytest
from conftest import CONFIGS, make_sample

from reasonbench.config import RunConfig, load_prompt, load_run_config
from reasonbench.dialects import Dialect, check_efforts, effort_body, infer
from reasonbench.errors import ConfigError
from reasonbench.openrouter import build_request


@pytest.fixture
def base():
    return load_run_config(CONFIGS / "models.yaml").model_dump()


def config(base, **overrides) -> RunConfig:
    return RunConfig.model_validate({**base, **overrides})


def sweep(*efforts):
    return {"repeats": 1, "temperature": [0.0], "reasoning_effort": list(efforts)}


def test_the_dialect_is_inferred_from_the_endpoint(base):
    assert infer(config(base)) is Dialect.OPENROUTER
    assert infer(config(base, base_url="https://api.openai.com/v1")) is Dialect.OPENAI
    assert infer(config(base, base_url="http://localhost:8000/v1")) is Dialect.GENERIC
    assert infer(config(base, dialect="openai")) is Dialect.OPENAI


def test_openrouter_requests_are_unchanged(base):
    """The shape every existing config and fixture depends on."""
    prompt = load_prompt(CONFIGS / "prompts" / "periodic-table.yaml")
    body = build_request(make_sample(reasoning_effort="high"), prompt, config(base))
    assert body["reasoning"] == {"effort": "high", "exclude": False}
    assert "reasoning_effort" not in body


def test_openai_uses_its_own_parameter(base):
    prompt = load_prompt(CONFIGS / "prompts" / "periodic-table.yaml")
    cfg = config(base, base_url="https://api.openai.com/v1")
    body = build_request(make_sample(reasoning_effort="high"), prompt, cfg)
    assert body["reasoning_effort"] == "high"
    assert "reasoning" not in body


def test_a_generic_server_toggles_thinking(base):
    prompt = load_prompt(CONFIGS / "prompts" / "periodic-table.yaml")
    cfg = config(base, base_url="http://localhost:8000/v1")
    body = build_request(make_sample(reasoning_effort="high"), prompt, cfg)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    off = build_request(make_sample(reasoning_effort="none"), prompt, cfg)
    assert off["chat_template_kwargs"] == {"enable_thinking": False}


def test_no_effort_sends_no_parameter(base):
    prompt = load_prompt(CONFIGS / "prompts" / "periodic-table.yaml")
    for url in (None, "https://api.openai.com/v1", "http://localhost:8000/v1"):
        cfg = config(base, base_url=url)
        body = build_request(make_sample(reasoning_effort=None), prompt, cfg)
        assert "reasoning" not in body
        assert "reasoning_effort" not in body
        assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("effort", ["xhigh", "max", "none"])
def test_an_effort_openai_cannot_express_is_an_error(effort):
    """Dropping it silently would leave a sweep axis in the report that never varied."""
    with pytest.raises(ConfigError, match="no equivalent"):
        effort_body(Dialect.OPENAI, effort)


def test_downgrade_is_available_but_not_the_default():
    body, applied = effort_body(Dialect.OPENAI, "xhigh", on_unsupported="downgrade")
    assert body == {"reasoning_effort": "high"}
    assert applied == "high"


def test_omit_drops_the_parameter():
    assert effort_body(Dialect.OPENAI, "max", on_unsupported="omit") == ({}, None)


def test_a_meaningful_axis_produces_no_warning(base):
    assert check_efforts(config(base, sweep=sweep("low", "high"))) == []


def test_an_axis_that_collapses_says_so(base):
    """On a generic server every level but none sends the same body."""
    notes = check_efforts(
        config(base, base_url="http://localhost:8000/v1", sweep=sweep("low", "high"))
    )
    assert notes == ["low, high send the same request"]


def test_a_downgraded_axis_says_so(base):
    notes = check_efforts(
        config(
            base,
            base_url="https://api.openai.com/v1",
            on_unsupported_effort="downgrade",
            sweep=sweep("high", "xhigh"),
        )
    )
    assert notes == ["high, xhigh send the same request"]


def test_the_effort_axis_is_checked_before_any_call(base):
    with pytest.raises(ConfigError, match="no equivalent"):
        check_efforts(
            config(base, base_url="https://api.openai.com/v1", sweep=sweep("max"))
        )
