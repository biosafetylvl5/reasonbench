"""Deterministic checks, N/A propagation, judge prompt building, aggregation."""

from __future__ import annotations

import json

import pytest
from conftest import make_row

from reasonbench.config import DeterministicCriterion, Target, TextScope
from reasonbench.scoring import (
    JudgeScorer,
    aggregate,
    applicable_judge_criteria,
    apply_scope,
    build_judge_prompt,
    build_judge_request,
    build_judge_schema,
    na_rows,
    parse_judge_response,
    render_guidance,
    score_deterministic,
    score_sample_deterministic,
    select_text,
    weighted_sample_score,
)


def criterion(**overrides):
    payload = {
        "id": "c",
        "kind": "deterministic",
        "target": "output",
        "check": {"type": "regex", "pattern": r"\bB\b"},
    } | overrides
    return DeterministicCriterion.model_validate(payload)


@pytest.mark.parametrize(
    ("check", "text", "expected"),
    [
        ({"type": "regex", "pattern": r"\bB\b"}, "answer B", True),
        ({"type": "regex", "pattern": r"\bB\b"}, "But no", False),
        ({"type": "exact", "expected": "B"}, "  b  ", True),
        ({"type": "exact", "expected": "B", "ignore_case": False}, "b", False),
        ({"type": "contains", "expected": "group"}, "same GROUP here", True),
        ({"type": "numeric", "expected": 127.0}, "there are 127 moves", True),
        ({"type": "numeric", "expected": 127.0, "tol": 1.0}, "126 moves", True),
        ({"type": "numeric", "expected": 127.0}, "126 moves", False),
        ({"type": "numeric", "expected": 5.0}, "no digits here", False),
        (
            {"type": "numeric", "expected": 127.0, "extract": r"(\d+) moves"},
            "took 3 hours and 127 moves",
            True,
        ),
    ],
)
def test_check_types(check, text, expected):
    row = make_row(output=text)
    score = score_deterministic(criterion(check=check), row)
    assert score.applicable is True
    assert score.score == float(expected)


def test_scope_narrows_to_the_last_nonblank_line():
    assert apply_scope("first\nB\n\n", TextScope.LAST_LINE) == "B"
    assert apply_scope("first\nB", TextScope.FIRST_LINE) == "first"
    assert apply_scope("first\nB", TextScope.FULL) == "first\nB"
    assert apply_scope("   \n\n", TextScope.LAST_LINE) == ""


def test_last_line_scope_avoids_matching_an_enumerated_option():
    """A chain-of-thought answer that discusses B but concludes C."""
    text = "B. same group, plausible\nC. same period\n\nC"
    assert (
        score_deterministic(criterion(scope="full"), make_row(output=text)).score == 1
    )
    assert (
        score_deterministic(criterion(scope="last_line"), make_row(output=text)).score
        == 0
    )


def test_reasoning_target_is_na_without_a_readable_trace():
    row = make_row(reasoning_availability="encrypted_only", reasoning_text="")
    assert select_text(row, Target.REASONING) is None
    score = score_deterministic(criterion(target="reasoning"), row)
    assert score.applicable is False
    assert score.score is None
    assert score.normalized is None


def test_both_target_degrades_to_output_when_trace_is_missing():
    row = make_row(reasoning_availability="absent", reasoning_text="", output="B")
    assert select_text(row, Target.BOTH) == "B"
    assert score_deterministic(criterion(target="both"), row).applicable is True


def test_summary_only_trace_is_gradeable():
    row = make_row(
        reasoning_availability="summary_only",
        reasoning_text="",
        reasoning_summary="brief summary",
    )
    assert select_text(row, Target.REASONING) == "brief summary"


def test_na_rows_explain_themselves(prompt_spec):
    row = make_row(reasoning_availability="absent", reasoning_text="")
    _, skipped = applicable_judge_criteria(prompt_spec.rubric, row)
    rows = na_rows(row, skipped)
    assert len(rows) == 1
    assert rows[0].applicable is False
    assert "absent" in (rows[0].reason or "")


def test_judge_criteria_are_split_by_gradeability(prompt_spec):
    readable = make_row()
    gradeable, skipped = applicable_judge_criteria(prompt_spec.rubric, readable)
    assert len(gradeable) == 1
    assert not skipped

    blind_row = make_row(reasoning_availability="absent", reasoning_text="")
    gradeable, skipped = applicable_judge_criteria(prompt_spec.rubric, blind_row)
    assert not gradeable
    assert len(skipped) == 1


def test_guidance_renders_every_configured_section(prompt_spec):
    rendered = render_guidance(prompt_spec.rubric.judge_criteria[0])
    assert "integer score from 0 to 4" in rendered
    assert "reasoning trace only" in rendered
    assert "4 = Flawless." in rendered
    assert "+ shows the arithmetic" in rendered
    assert "- guesses" in rendered
    assert "Notes: Be strict." in rendered


def test_anchors_render_high_to_low(prompt_spec):
    rendered = render_guidance(prompt_spec.rubric.judge_criteria[0])
    assert rendered.index("4 = Flawless.") < rendered.index("0 = Nonsense.")


def test_blind_judging_hides_the_model_name(prompt_spec, run_config):
    row = make_row()
    blind = build_judge_prompt(
        prompt_spec,
        row,
        prompt_spec.rubric.judge_criteria,
        run_config.judge,
    )
    assert row.model not in blind

    named = build_judge_prompt(
        prompt_spec,
        row,
        prompt_spec.rubric.judge_criteria,
        run_config.judge.model_copy(update={"blind": False}),
    )
    assert row.model in named


def test_judge_prompt_includes_trace_and_answer(prompt_spec, run_config):
    prompt_text = build_judge_prompt(
        prompt_spec,
        make_row(),
        prompt_spec.rubric.judge_criteria,
        run_config.judge,
    )
    assert "Seven is seven." in prompt_text
    assert "What is 7?" in prompt_text


def test_judge_instructions_are_prepended_to_the_system_message(
    prompt_spec,
    run_config,
):
    body = build_judge_request(
        prompt_spec,
        make_row(),
        prompt_spec.rubric.judge_criteria,
        run_config.judge,
    )
    assert "Grade carefully." in body["messages"][0]["content"]
    assert body["temperature"] == 0.0


def test_judge_schema_constrains_scores_to_the_scale(prompt_spec):
    schema = build_judge_schema(prompt_spec.rubric.judge_criteria)
    prop = schema["json_schema"]["schema"]["properties"]["sound_reasoning"]
    assert prop["properties"]["score"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 4,
    }
    assert schema["json_schema"]["strict"] is True


def test_judge_response_is_normalized_onto_zero_one(prompt_spec):
    raw = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"sound_reasoning": {"score": 3, "justification": "ok"}},
                    ),
                },
            },
        ],
    }
    rows = parse_judge_response(
        raw,
        make_row(),
        prompt_spec.rubric.judge_criteria,
        judge_repeat=0,
    )
    assert rows[0].score == 3.0
    assert rows[0].normalized == pytest.approx(0.75)  # 3 of 0-4


def test_unparseable_judge_output_becomes_na(prompt_spec):
    raw = {"choices": [{"message": {"content": "not json"}}]}
    rows = parse_judge_response(
        raw,
        make_row(),
        prompt_spec.rubric.judge_criteria,
        judge_repeat=0,
    )
    assert rows[0].applicable is False
    assert rows[0].normalized is None


def test_weighted_score_ignores_na_criteria(prompt_spec):
    collapsed = {("abc123", "correct"): 1.0, ("abc123", "sound_reasoning"): None}
    assert (
        weighted_sample_score(prompt_spec.rubric.criteria, collapsed, "abc123") == 1.0
    )


def test_weighted_score_respects_weights(prompt_spec):
    collapsed = {("abc123", "correct"): 1.0, ("abc123", "sound_reasoning"): 0.0}
    # weights 2.0 and 1.0 -> (2*1 + 1*0) / 3
    assert weighted_sample_score(
        prompt_spec.rubric.criteria,
        collapsed,
        "abc123",
    ) == pytest.approx(2 / 3)


def test_weighted_score_is_none_when_nothing_applies(prompt_spec):
    collapsed = {("abc123", "correct"): None, ("abc123", "sound_reasoning"): None}
    assert (
        weighted_sample_score(prompt_spec.rubric.criteria, collapsed, "abc123") is None
    )


def test_aggregate_reports_coverage_not_zeros(prompt_spec):
    rows = [
        make_row(sample_id="s1", model="thinks"),
        make_row(
            sample_id="s2",
            model="silent",
            reasoning_availability="absent",
            reasoning_text="",
            reasoning_tokens=0,
        ),
    ]
    scores = [
        *score_sample_deterministic(prompt_spec.rubric, rows[0]),
        *score_sample_deterministic(prompt_spec.rubric, rows[1]),
        *na_rows(rows[1], prompt_spec.rubric.judge_criteria),
    ]
    summaries = aggregate(rows, scores, prompt_spec.rubric, group_by=("model",))
    by_model = {s.key[0]: s for s in summaries}

    silent = next(
        c for c in by_model["silent"].criteria if c.criterion_id == "sound_reasoning"
    )
    assert silent.mean is None
    assert silent.n_scored == 0
    assert silent.coverage == 0.0
    assert by_model["silent"].availability == {"absent": 1}


def test_aggregate_groups_by_multiple_fields(prompt_spec):
    rows = [
        make_row(sample_id="s1", model="m", variant_id="plain"),
        make_row(sample_id="s2", model="m", variant_id="cot"),
    ]
    summaries = aggregate(
        rows, [], prompt_spec.rubric, group_by=("model", "variant_id")
    )
    assert {s.key for s in summaries} == {("m", "plain"), ("m", "cot")}


def test_failed_samples_are_counted_but_not_scored(prompt_spec):
    rows = [make_row(sample_id="s1"), make_row(sample_id="s2", ok=False)]
    summaries = aggregate(rows, [], prompt_spec.rubric, group_by=("model",))
    assert summaries[0].n_samples == 2
    assert summaries[0].n_failed == 1


class FakeClient:
    """Returns a scripted sequence of judge responses."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    async def complete(self, body):  # noqa: ARG002
        """Return the next scripted response."""
        self.calls += 1
        return self._payloads.pop(0)


def judge_payload(scores):
    """Wrap a criterion->score mapping in a chat-completions response."""
    body = {
        cid: {"score": value, "justification": "because"}
        for cid, value in scores.items()
    }
    return {
        "choices": [{"message": {"content": json.dumps(body)}}],
        "usage": {"cost": 0.001},
    }


GARBAGE = {"choices": [{"message": {"content": "sorry, I cannot"}}], "usage": {}}


async def test_unparseable_judge_reply_is_retried(prompt_spec, run_config):
    client = FakeClient([GARBAGE, judge_payload({"sound_reasoning": 4})])
    scorer = JudgeScorer(client, prompt_spec, run_config.judge)

    rows = await scorer.score(make_row())

    assert client.calls == 2
    assert scorer.parse_retries == 1
    assert [r.applicable for r in rows] == [True]
    assert rows[0].score == 4.0


async def test_judge_gives_up_after_parse_retries(prompt_spec, run_config):
    settings = run_config.judge.model_copy(update={"parse_retries": 1})
    client = FakeClient([GARBAGE, GARBAGE])
    scorer = JudgeScorer(client, prompt_spec, settings)

    rows = await scorer.score(make_row())

    assert client.calls == 2
    assert rows[0].applicable is False
    assert "did not parse after 2 attempts" in (rows[0].reason or "")


async def test_missing_trace_is_na_without_calling_the_judge(prompt_spec, run_config):
    client = FakeClient([])
    scorer = JudgeScorer(client, prompt_spec, run_config.judge)

    rows = await scorer.score(
        make_row(reasoning_availability="absent", reasoning_text=""),
    )

    assert client.calls == 0
    assert scorer.cost == 0.0
    assert rows[0].applicable is False
    assert "absent" in (rows[0].reason or "")


async def test_judge_raw_responses_are_handed_to_the_callback(prompt_spec, run_config):
    saved = []
    client = FakeClient([judge_payload({"sound_reasoning": 2})])
    scorer = JudgeScorer(
        client,
        prompt_spec,
        run_config.judge,
        on_raw=lambda sid, repeat, attempt, _raw: saved.append((sid, repeat, attempt)),
    )

    await scorer.score(make_row())

    assert saved == [("abc123", 0, 0)]
    assert scorer.cost == 0.001
