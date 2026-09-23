"""A rejected key must stop the run, not fail every sample in it."""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import CONFIGS
from typer.testing import CliRunner

from reasonbench import ui
from reasonbench.cli import app
from reasonbench.config import RunConfig, load_run_config
from reasonbench.errors import ExitCode
from reasonbench.openrouter import DEFAULT_BASE_URL, AuthError, OpenRouterClient

MODELS_URL = f"{DEFAULT_BASE_URL}/models"
CHAT_URL = f"{DEFAULT_BASE_URL}/chat/completions"


@pytest.fixture(autouse=True)
def _reset_ui():
    ui.reset()
    yield
    ui.reset()


@pytest.fixture
def config() -> RunConfig:
    return load_run_config(CONFIGS / "models.yaml")


@pytest.mark.parametrize("status", [401, 402, 403])
@respx.mock
async def test_a_rejected_key_raises_auth_error(status, config):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(status, text="nope"))
    async with OpenRouterClient("bad", max_retries=0) as client:
        with pytest.raises(AuthError):
            await client.preflight(config.models[0])


@respx.mock
async def test_a_working_models_listing_is_enough(config):
    route = respx.get(MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    async with OpenRouterClient("good", max_retries=0) as client:
        await client.preflight(config.models[0])
    assert route.called


@respx.mock
async def test_a_server_without_a_models_route_falls_back_to_a_ping(config):
    """llama.cpp and several shims implement only chat-completions."""
    respx.get(MODELS_URL).mock(return_value=httpx.Response(404))
    chat = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {}}]})
    )
    async with OpenRouterClient("good", max_retries=0) as client:
        await client.preflight(config.models[0])
    assert chat.called


@respx.mock
async def test_a_bad_key_is_caught_by_the_fallback_ping(config):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(404))
    respx.post(CHAT_URL).mock(return_value=httpx.Response(401, text="nope"))
    async with OpenRouterClient("bad", max_retries=0) as client:
        with pytest.raises(AuthError):
            await client.preflight(config.models[0])


@respx.mock
async def test_an_unreachable_endpoint_does_not_stop_the_run(config):
    """Inconclusive is not the same as rejected."""
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("down"))
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="oops"))
    async with OpenRouterClient("unknown", max_retries=0) as client:
        with pytest.raises(Exception, match="HTTP 500"):
            await client.preflight(config.models[0])


@respx.mock
def test_run_exits_10_on_a_rejected_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad")
    respx.get(MODELS_URL).mock(return_value=httpx.Response(401, text="nope"))
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(CONFIGS / "models.yaml"),
            str(CONFIGS / "prompts" / "periodic-table.yaml"),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == ExitCode.AUTH
    # nothing was generated, so no run directory was left behind
    assert not list(tmp_path.iterdir())
