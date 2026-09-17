"""Cases, their images, and the escaping that keeps a data value out of the regex."""

from __future__ import annotations

import json

import pytest
from conftest import CONFIGS, make_row

from reasonbench.config import DeterministicCriterion, load_prompt
from reasonbench.dataset import load_cases
from reasonbench.errors import ConfigError
from reasonbench.scoring import resolve_criterion, score_deterministic

ANSWER_PATTERN = r"(?i)^(?:\W*answer\W*)?\W*{{ expected | re_escape }}(?!\w)"


def answer_criterion(pattern: str = ANSWER_PATTERN) -> DeterministicCriterion:
    return DeterministicCriterion.model_validate(
        {
            "id": "correct_answer",
            "kind": "deterministic",
            "target": "output",
            "scope": "last_line",
            "weight": 1.0,
            "check": {"type": "regex", "pattern": pattern},
        }
    )


def grades(expected: str, output: str, pattern: str = ANSWER_PATTERN) -> bool:
    criterion = resolve_criterion(answer_criterion(pattern), {"expected": expected})
    return bool(score_deterministic(criterion, make_row(output=output)).score)


@pytest.mark.parametrize(
    ("expected", "output", "want"),
    [
        ("B", "B", True),
        ("B", "Answer: B", True),
        ("B", "Answer: BQ", False),
        ("2.5", "2.5", True),
        ("2.5", "225", False),  # the decimal point must not match any digit
        (
            "a|b",
            "totally wrong b here",
            False,
        ),  # alternation must not split the pattern
        ("C++", "Answer: C++", True),  # (?!\w) not \b, or this cannot match itself
        ("C++", "C", False),
        ("50%", "50%", True),
        ("$5", "$5", True),
    ],
)
def test_answer_keys_from_data_grade_correctly(expected, output, want):
    assert grades(expected, output) is want


@pytest.mark.parametrize("expected", ["[Fe", "H2O)", "(unclosed", "a.b", "x*"])
def test_regex_hostile_answer_keys_match_themselves(expected):
    """Escaped, a metacharacter is just a character."""
    assert grades(expected, expected) is True


@pytest.mark.parametrize("expected", ["[Fe", "H2O)"])
def test_an_unescaped_template_reports_the_broken_pattern(expected):
    """Without re_escape the data can still break the regex; say which criterion."""
    with pytest.raises(ConfigError, match="correct_answer"):
        grades(expected, "anything", pattern=r"^{{ expected }}$")


def test_a_missing_column_raises_rather_than_matching_everything():
    """An empty render would produce a pattern that matches nearly any output."""
    with pytest.raises(ConfigError, match="correct_answer"):
        resolve_criterion(answer_criterion(), {"wrong_column": "B"})


def test_an_untemplated_criterion_is_returned_unchanged():
    plain = answer_criterion(r"(?i)^\W*[A-D]\W*$")
    assert resolve_criterion(plain, {"expected": "B"}) is plain


def test_dataset_rows_become_cases():
    path = CONFIGS / "prompts" / "mcq-dataset.yaml"
    cases = load_cases(load_prompt(path), path)
    assert cases is not None
    assert [c.case_id for c in cases.cases] == [
        "mcq-001",
        "mcq-002",
        "mcq-003",
        "mcq-004",
    ]
    assert cases.cases[0].source_line == 1
    assert cases.cases[0].variables["expected"] == "B"
    assert cases.sha256


def test_limit_selects_a_prefix():
    path = CONFIGS / "prompts" / "mcq-dataset.yaml"
    cases = load_cases(load_prompt(path), path, limit=2)
    assert cases is not None
    assert len(cases.cases) == 2


def test_image_columns_resolve_to_data_urls():
    path = CONFIGS / "prompts" / "chart-reading.yaml"
    cases = load_cases(load_prompt(path), path)
    assert cases is not None
    image = cases.cases[0].images[0]
    assert image.url.startswith("data:image/png;base64,")
    assert image.bytes_ > 0
    assert len(image.sha256) == 64
    # the image column is an attachment, not a template variable
    assert "chart" not in cases.cases[0].variables


def test_a_prompt_without_a_dataset_loads_no_cases():
    path = CONFIGS / "prompts" / "periodic-table.yaml"
    assert load_cases(load_prompt(path), path) is None


def test_duplicate_case_ids_are_rejected(tmp_path):
    data = tmp_path / "dupes.jsonl"
    data.write_text(
        json.dumps({"case_id": "a", "q": 1})
        + "\n"
        + json.dumps({"case_id": "a", "q": 2}),
        encoding="utf-8",
    )
    spec = load_prompt(CONFIGS / "prompts" / "mcq-dataset.yaml").model_copy(
        update={
            "dataset": load_prompt(
                CONFIGS / "prompts" / "mcq-dataset.yaml"
            ).dataset.model_copy(update={"path": str(data), "required_columns": ()})
        }
    )
    with pytest.raises(ConfigError, match="duplicate case ids"):
        load_cases(spec, tmp_path / "prompt.yaml")


def test_a_missing_required_column_is_reported(tmp_path):
    data = tmp_path / "thin.jsonl"
    data.write_text(json.dumps({"case_id": "a"}), encoding="utf-8")
    base = load_prompt(CONFIGS / "prompts" / "mcq-dataset.yaml")
    spec = base.model_copy(
        update={"dataset": base.dataset.model_copy(update={"path": str(data)})}
    )
    with pytest.raises(ConfigError, match="missing column"):
        load_cases(spec, tmp_path / "prompt.yaml")
