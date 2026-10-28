"""Judging with a decisions model, such as Jev, on the Decisions API."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
import yaml
from conftest import load_fixture, make_row, make_sample
from typer.testing import CliRunner

from reasonbench import ui
from reasonbench.cli import app
from reasonbench.config import JudgeCriterion, JudgeSettings, Rubric
from reasonbench.errors import ConfigError, ExitCode
from reasonbench.openrouter import (
    API_URL,
    DECISIONS_URL,
    OpenRouterClient,
    decisions_url,
    parse_response,
)
from reasonbench.scoring import (
    JudgeScorer,
    build_decisions_request,
    check_decisions_rubric,
    parse_decisions_response,
)
from reasonbench.storage import RunStore, new_run_dir

JEV = JudgeSettings(model="typesafe/jev-1.14", api="decisions")


def criterion(**overrides) -> JudgeCriterion:
    defaults = {
        "id": "sound",
        "kind": "judge",
        "target": "reasoning",
        "scale": [1, 3],
        "guidance": {
            "summary": "Is the reasoning valid?",
            "levels": {3: "Flawless.", 2: "One slip.", 1: "Nonsense."},
            "negative_indicators": ["guesses"],
            "notes": "Be strict.",
        },
    }
    return JudgeCriterion.model_validate(defaults | overrides)


def answer(position: float, **extra) -> dict:
    return {"type": "score", "score": position, **extra}


@pytest.fixture
def jev_prompt(prompt_spec):
    """Return the shared prompt with an anchor for every score on its judged scale."""
    deterministic, judged = prompt_spec.rubric.criteria
    levels = {s: f"Level {s}." for s in range(5)}
    anchored = judged.model_copy(
        update={"guidance": judged.guidance.model_copy(update={"levels": levels})}
    )
    rubric = prompt_spec.rubric.model_copy(
        update={"criteria": (deterministic, anchored)}
    )
    return prompt_spec.model_copy(update={"rubric": rubric})


@pytest.fixture(autouse=True)
def _reset_ui():
    ui.reset()
    yield
    ui.reset()


def test_the_decisions_url_sits_beside_the_chat_base():
    assert decisions_url("https://openrouter.ai/api/v1") == DECISIONS_URL
    assert decisions_url("https://openrouter.ai/api/v1/") == DECISIONS_URL


def test_levels_go_lowest_first_so_position_maps_to_score(prompt_spec):
    body = build_decisions_request(prompt_spec, make_row(), (criterion(),), JEV)
    question = body["questions"]["sound"]
    assert question["type"] == "score"
    assert question["criteria"] == ["Nonsense.", "One slip.", "Flawless."]
    assert "the reasoning trace only" in question["instructions"]
    assert "guesses" in question["instructions"]
    assert "Be strict." in question["instructions"]
    assert body["model"] == "typesafe/jev-1.14"


def test_the_response_is_the_state(prompt_spec):
    body = build_decisions_request(prompt_spec, make_row(), (criterion(),), JEV)
    state = body["state"]
    assert state["grading_context"] == "Grade carefully."
    assert state["question"] == "What is 7? Answer with a number."
    assert state["reasoning_trace"] == "Seven is seven."
    assert state["final_answer"] == "7"
    assert "model" not in state, "a blind judge is not told the model"


def test_the_position_is_offset_by_the_bottom_of_the_scale():
    raw = {
        "answers": {
            "sound": answer(
                1.5, confidence=0.4, probabilities={"0": 0.0, "1": 0.5, "2": 0.5}
            )
        }
    }
    (row,) = parse_decisions_response(raw, make_row(), (criterion(),), 0)
    assert row.score == pytest.approx(2.5)
    assert row.normalized == pytest.approx(0.75)
    assert row.reason == "confidence 0.40; P(3)=0.50; P(2)=0.50"


def test_a_position_past_the_top_is_clamped():
    raw = {"answers": {"sound": answer(5)}}
    (row,) = parse_decisions_response(raw, make_row(), (criterion(),), 0)
    assert row.clamped
    assert row.normalized == 1.0


def test_a_missing_answer_is_not_applicable():
    (row,) = parse_decisions_response({"answers": {}}, make_row(), (criterion(),), 0)
    assert not row.applicable
    assert row.score is None


def test_a_criterion_without_every_level_cannot_be_asked():
    sparse = criterion(
        guidance={"summary": "Valid?", "levels": {3: "Flawless.", 1: "Nonsense."}}
    )
    with pytest.raises(ConfigError, match=r"\[2\] have none"):
        check_decisions_rubric(Rubric(criteria=(sparse,)))


@respx.mock
async def test_the_scorer_asks_the_decisions_api(jev_prompt):
    route = respx.post(DECISIONS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "answers": {"sound_reasoning": answer(3.0)},
                "usage": {"cost": 0.0001, "input_tokens": 900, "output_tokens": 0},
            },
        )
    )
    settings = JEV.model_copy(update={"repeats": 2})
    async with OpenRouterClient("k", max_retries=0) as client:
        scorer = JudgeScorer(client, jev_prompt, settings)
        rows = await scorer.score(make_row())
    assert route.call_count == 2
    assert [r.score for r in rows] == [3.0, 3.0]
    assert scorer.cost == pytest.approx(0.0002)
    body = json.loads(route.calls.last.request.content)
    assert "temperature" not in body
    assert "response_format" not in body


@respx.mock
async def test_a_decisions_model_on_chat_says_how_to_fix_it(prompt_spec):
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            400,
            text='{"error":{"message":"typesafe/jev-1.14 is a decisions model and '
            'cannot be used with the chat/completions endpoint.","code":400}}',
        )
    )
    settings = JudgeSettings(model="typesafe/jev-1.14")
    async with OpenRouterClient("k", max_retries=0) as client:
        rows = await JudgeScorer(client, prompt_spec, settings).score(make_row())
    (row,) = rows
    assert not row.applicable
    assert "judge.api: decisions" in row.reason


def make_run(tmp_path, run_config, prompt):
    run_dir = new_run_dir(tmp_path / "runs")
    with RunStore(run_dir, create=True) as store:
        raw = load_fixture("gemma_full_text")
        store.add_sample(parse_response(raw, make_sample(sample_id="s1"), 1.0), raw)
        store.write_manifest(
            {
                "run_config": run_config.model_dump(mode="json"),
                "prompt": prompt.model_dump(mode="json"),
            }
        )
    return run_dir


@pytest.fixture
def jev_run(tmp_path, run_config, jev_prompt):
    return make_run(tmp_path, run_config, jev_prompt)


@pytest.fixture
def sparse_run(tmp_path, run_config, prompt_spec):
    return make_run(tmp_path, run_config, prompt_spec)


def test_score_with_a_decisions_judge_records_it(jev_run, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    with respx.mock:
        route = respx.post(DECISIONS_URL).mock(
            return_value=httpx.Response(
                200, json={"answers": {"sound_reasoning": answer(4.0)}, "usage": {}}
            )
        )
        result = CliRunner().invoke(
            app,
            [
                "score",
                str(jev_run),
                "--judge-model",
                "typesafe/jev-1.14",
                "--judge-api",
                "decisions",
            ],
        )
    assert result.exit_code == ExitCode.OK, result.output
    assert route.called
    manifest = yaml.safe_load((jev_run / "manifest.yaml").read_text())
    assert manifest["run_config"]["judge"]["api"] == "decisions"


def test_a_rubric_a_decisions_judge_cannot_ask_is_refused_first(
    sparse_run, monkeypatch
):
    """The shared prompt anchors only 0 and 4 of its 0-4 scale."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    result = CliRunner().invoke(
        app, ["score", str(sparse_run), "--judge-api", "decisions"]
    )
    assert result.exit_code == ExitCode.USAGE
    assert not (sparse_run / "judge").exists()


def test_an_unknown_judge_api_is_a_usage_error(jev_run):
    result = CliRunner().invoke(app, ["score", str(jev_run), "--judge-api", "x"])
    assert result.exit_code == ExitCode.USAGE
