"""The shipped prompt configs, checked against output shapes models produce."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from conftest import CONFIGS, make_row

from reasonbench.config import DeterministicCriterion, JudgeCriterion, load_prompt
from reasonbench.scoring import build_judge_schema, render_guidance, score_deterministic


@dataclass(frozen=True)
class Case:
    """The shape of one shipped prompt, and what counts as right and wrong."""

    filename: str
    correct: str
    wrong: str
    reasoning_criterion: str
    n_judge: int
    judge_instructions: bool
    # criteria whose guidance is the bare-string shorthand, so not anchored
    shorthand: frozenset[str] = frozenset()


CASES = [
    Case("periodic-table.yaml", "B", "C", "explains_via_valence_electrons", 3, True),
    Case("laplace-transform.yaml", "B", "C", "explains_domain_transformation", 2, True),
    Case(
        "grey-body.yaml",
        "1",
        "3",
        "distinguishes_grey_from_black",
        3,
        False,
        frozenset({"factually_sound"}),
    ),
]


@pytest.fixture(params=CASES, ids=lambda c: c.filename)
def case(request):
    """Return one shipped prompt together with its expected answers."""
    return request.param


@pytest.fixture
def shipped_prompt(case):
    """Return the loaded prompt for the current case."""
    return load_prompt(CONFIGS / "prompts" / case.filename)


def criterion_by_id(prompt, criterion_id):
    """Look up one criterion in a prompt's rubric."""
    return next(c for c in prompt.rubric.criteria if c.id == criterion_id)


def test_prompt_loads_and_has_both_variants(shipped_prompt):
    assert {v.id for v in shipped_prompt.variants} == {"plain", "cot"}
    assert shipped_prompt.system


def test_rubric_covers_both_kinds_and_all_targets(shipped_prompt):
    criteria = shipped_prompt.rubric.criteria
    assert {c.kind for c in criteria} == {"deterministic", "judge"}
    assert {str(c.target) for c in criteria} == {"output", "reasoning", "both"}


def test_rubric_has_the_shape_the_case_declares(shipped_prompt, case):
    assert len(shipped_prompt.rubric.judge_criteria) == case.n_judge
    assert bool(shipped_prompt.rubric.judge_instructions) is case.judge_instructions


def test_every_judge_criterion_is_fully_anchored(shipped_prompt, case):
    for criterion in shipped_prompt.rubric.judge_criteria:
        if criterion.id in case.shorthand:
            continue
        low, high = criterion.scale
        assert set(criterion.guidance.levels) == set(range(low, high + 1)), (
            f"{criterion.id} must describe every score on its scale"
        )
        assert criterion.guidance.negative_indicators
        assert criterion.guidance.notes


def test_guidance_renders_without_losing_configured_content(shipped_prompt):
    for criterion in shipped_prompt.rubric.judge_criteria:
        rendered = render_guidance(criterion)
        assert criterion.guidance.summary.split()[0] in rendered
        for level_text in criterion.guidance.levels.values():
            assert level_text.split()[0] in rendered
        for indicator in criterion.guidance.negative_indicators:
            assert indicator in rendered


def test_judge_schema_is_strict_and_complete(shipped_prompt):
    schema = build_judge_schema(shipped_prompt.rubric.judge_criteria)
    inner = schema["json_schema"]["schema"]
    assert schema["json_schema"]["strict"] is True
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {
        c.id for c in shipped_prompt.rubric.judge_criteria
    }


def test_deterministic_criteria_read_only_the_last_line(shipped_prompt):
    for criterion in shipped_prompt.rubric.criteria:
        if isinstance(criterion, DeterministicCriterion):
            assert str(criterion.scope) == "last_line"


def test_weights_prioritize_correctness_over_format(shipped_prompt):
    assert (
        criterion_by_id(shipped_prompt, "correct_answer").weight
        > criterion_by_id(shipped_prompt, "follows_answer_format").weight
    )


def test_reasoning_criterion_targets_the_trace(shipped_prompt, case):
    criterion = criterion_by_id(shipped_prompt, case.reasoning_criterion)
    assert isinstance(criterion, JudgeCriterion)
    assert str(criterion.target) == "reasoning"
    assert criterion.weight >= 2.0


def score_both(prompt, output):
    """Return ``(correct_answer, follows_answer_format)`` for one output."""
    row = make_row(output=output)
    return (
        score_deterministic(criterion_by_id(prompt, "correct_answer"), row),
        score_deterministic(criterion_by_id(prompt, "follows_answer_format"), row),
    )


@pytest.mark.parametrize(
    ("template", "correct", "well_formed"),
    [
        ("{correct}", True, True),
        ("**{correct}**", True, True),
        ("{correct}.", True, True),
        ("Answer: {correct}", True, False),
        ("Let me think.\n\nIt is clear.\n\n{correct}", True, True),
        ("1. no\n2. maybe\n\n{correct}", True, True),
        # Discusses the right option, then concludes the wrong one.
        ("{correct} looks plausible.\nBut on reflection, no.\n\n{wrong}", False, True),
        ("{wrong}", False, True),
        ("I am not sure", False, False),
        ("", False, False),
    ],
)
def test_answer_extraction(shipped_prompt, case, template, correct, well_formed):
    output = template.format(correct=case.correct, wrong=case.wrong)
    got_correct, got_format = score_both(shipped_prompt, output)
    assert bool(got_correct.score) is correct, f"correct_answer on {output!r}"
    assert bool(got_format.score) is well_formed, f"answer_format on {output!r}"


def test_empty_output_is_na_not_a_failed_check(shipped_prompt):
    """No output at all is unmeasurable, which differs from a wrong answer."""
    got_correct, _ = score_both(shipped_prompt, "")
    assert got_correct.applicable is False
    assert got_correct.score is None


def test_grey_body_accepts_the_word_as_well_as_the_number():
    """Numbered options invite word answers, so both are credited."""
    prompt = load_prompt(CONFIGS / "prompts" / "grey-body.yaml")
    for answer in ("1", "Grey", "grey", "Gray"):
        got_correct, _ = score_both(prompt, answer)
        assert got_correct.score == 1.0, answer
    for answer in ("3", "Black", "Opaque"):
        got_correct, _ = score_both(prompt, answer)
        assert got_correct.score == 0.0, answer


def test_lettered_prompts_do_not_credit_a_bare_number():
    """Guards against a copy-paste regex leaking between prompt files."""
    prompt = load_prompt(CONFIGS / "prompts" / "periodic-table.yaml")
    got_correct, _ = score_both(prompt, "1")
    assert got_correct.score == 0.0
