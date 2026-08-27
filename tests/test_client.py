"""HTTP behaviour: retries, fatal errors, and failure isolation. All mocked."""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import load_fixture, make_sample

from reasonbench.openrouter import (
    API_URL,
    FatalAPIError,
    OpenRouterClient,
    RetryableError,
)


@pytest.fixture
def client():
    """Return a client with no retry backoff worth waiting for."""
    return OpenRouterClient("test-key", max_concurrency=2, max_retries=2, timeout_s=5)


@respx.mock
async def test_successful_call_returns_the_payload(client):
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    async with client as c:
        assert await c.complete({"model": "m"}) == {"ok": True}


@respx.mock
async def test_rate_limit_is_retried_then_succeeds(client):
    route = respx.post(API_URL).mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    async with client as c:
        assert await c.complete({"model": "m"}) == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_retries_are_bounded(client):
    route = respx.post(API_URL).mock(return_value=httpx.Response(503))
    async with client as c:
        with pytest.raises(RetryableError):
            await c.complete({"model": "m"})
    assert route.call_count == 3  # initial attempt plus max_retries


@respx.mock
async def test_client_error_is_not_retried(client):
    route = respx.post(API_URL).mock(return_value=httpx.Response(400, text="bad model"))
    async with client as c:
        with pytest.raises(FatalAPIError, match="bad model"):
            await c.complete({"model": "m"})
    assert route.call_count == 1


@respx.mock
async def test_timeout_is_retried(client):
    route = respx.post(API_URL).mock(
        side_effect=[httpx.ReadTimeout("timed out"), httpx.Response(200, json={})],
    )
    async with client as c:
        await c.complete({"model": "m"})
    assert route.call_count == 2


@respx.mock
async def test_error_body_on_a_200_is_still_an_error(client):
    respx.post(API_URL).mock(
        return_value=httpx.Response(200, json={"error": {"message": "no such model"}}),
    )
    async with client as c:
        with pytest.raises(FatalAPIError):
            await c.complete({"model": "m"})


@respx.mock
async def test_failed_sample_is_returned_not_raised(client, prompt_spec, run_config):
    """One bad model must not abort a sweep."""
    respx.post(API_URL).mock(return_value=httpx.Response(400, text="nope"))
    async with client as c:
        result, raw = await c.run_sample(make_sample(), prompt_spec, run_config)
    assert result.ok is False
    assert "nope" in (result.error or "")
    assert "error" in raw


@respx.mock
async def test_run_sample_parses_a_real_response(client, prompt_spec, run_config):
    respx.post(API_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("gemma_full_text")),
    )
    async with client as c:
        result, _ = await c.run_sample(make_sample(), prompt_spec, run_config)
    assert result.ok is True
    assert result.reasoning.readable
    assert result.usage.cost > 0
    assert result.latency_s >= 0


async def test_use_outside_context_manager_is_an_error(client):
    with pytest.raises(RuntimeError, match="context manager"):
        await client.complete({"model": "m"})
