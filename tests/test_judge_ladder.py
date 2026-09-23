"""Falling back when a server will not do strict structured output."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from conftest import make_row

from reasonbench.openrouter import API_URL, OpenRouterClient
from reasonbench.scoring import JudgeScorer, build_judge_request


def good(criteria) -> httpx.Response:
    payload = {c.id: {"score": c.scale[1], "justification": "ok"} for c in criteria}
    return httpx.Response(
        200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
    )


def rejects_format() -> httpx.Response:
    return httpx.Response(
        400, text='{"error":"response_format json_schema is not supported"}'
    )


@pytest.fixture
def judged(prompt_spec):
    return prompt_spec.rubric.judge_criteria


def test_the_top_rung_sends_a_json_schema(prompt_spec, run_config, judged):
    body = build_judge_request(prompt_spec, make_row(), judged, run_config.judge)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_json_object_mode_still_describes_the_shape(prompt_spec, run_config, judged):
    """That mode guarantees valid JSON, not this JSON."""
    body = build_judge_request(
        prompt_spec, make_row(), judged, run_config.judge, rung="json_object"
    )
    assert body["response_format"] == {"type": "json_object"}
    content = body["messages"][-1]["content"]
    assert "single JSON object" in content
    assert judged[0].id in content


def test_the_bottom_rung_sends_no_response_format(prompt_spec, run_config, judged):
    body = build_judge_request(
        prompt_spec, make_row(), judged, run_config.judge, rung="none"
    )
    assert "response_format" not in body
    assert "single JSON object" in body["messages"][-1]["content"]


@respx.mock
async def test_a_rejected_format_falls_to_the_next_rung(
    prompt_spec, run_config, judged
):
    route = respx.post(API_URL).mock(side_effect=[rejects_format(), good(judged)])
    async with OpenRouterClient("k", max_retries=0) as client:
        scorer = JudgeScorer(client, prompt_spec, run_config.judge)
        rows = await scorer.score(make_row())
    assert route.call_count == 2
    assert all(r.applicable for r in rows if r.kind == "judge")


@respx.mock
async def test_a_downgrade_does_not_consume_a_parse_retry(
    prompt_spec, run_config, judged
):
    """With parse_retries at 0 a two-rung fall would otherwise never land."""
    settings = run_config.judge.model_copy(update={"parse_retries": 0})
    respx.post(API_URL).mock(
        side_effect=[rejects_format(), rejects_format(), good(judged)]
    )
    async with OpenRouterClient("k", max_retries=0) as client:
        scorer = JudgeScorer(client, prompt_spec, settings)
        rows = await scorer.score(make_row())
    assert scorer.parse_retries == 0
    assert all(r.applicable for r in rows if r.kind == "judge")


@respx.mock
async def test_an_unrelated_error_does_not_trigger_a_downgrade(prompt_spec, run_config):
    """A bad model slug must not be mistaken for an unsupported format."""
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(400, text='{"error":"no such model"}')
    )
    async with OpenRouterClient("k", max_retries=0) as client:
        scorer = JudgeScorer(client, prompt_spec, run_config.judge)
        rows = await scorer.score(make_row())
    assert route.call_count == 1
    assert any("judge call failed" in (r.reason or "") for r in rows)


@respx.mock
async def test_a_server_that_refuses_every_rung_is_reported(prompt_spec, run_config):
    respx.post(API_URL).mock(return_value=rejects_format())
    async with OpenRouterClient("k", max_retries=0) as client:
        scorer = JudgeScorer(client, prompt_spec, run_config.judge)
        rows = await scorer.score(make_row())
    judged_rows = [r for r in rows if r.kind == "judge"]
    assert judged_rows
    assert all(not r.applicable for r in judged_rows)
