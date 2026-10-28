"""Grading responses that were made elsewhere and recorded in the dataset."""

from __future__ import annotations

import json
import sqlite3

import pytest
import respx
import yaml
from typer.testing import CliRunner

from reasonbench import ui
from reasonbench.cli import app
from reasonbench.config import load_prompt
from reasonbench.dataset import RECORDED_MODEL, load_cases
from reasonbench.errors import ConfigError, ExitCode
from reasonbench.recorded import recorded_prompt, recorded_samples

ROWS = [
    {
        "case_id": "c1--original",
        "question": "What is 7?",
        "expected": "7",
        "mutation": "original",
        "model": "agent/x",
        "reasoning": "Seven is seven.",
        "output": "7",
    },
    {
        "case_id": "c1--wrong",
        "question": "What is 7?",
        "expected": "7",
        "mutation": "wrong",
        "model": "agent/x",
        "reasoning": "",
        "output": "8",
    },
]

PROMPT = {
    "id": "rec",
    "title": "Recorded",
    "dataset": {
        "path": "cases.jsonl",
        "recorded": {"model": "model", "variant": "mutation"},
    },
    "variants": [{"id": "original", "user": "{{ question }}"}],
    "rubric": {
        "criteria": [
            {
                "id": "correct",
                "kind": "deterministic",
                "check": {"type": "exact", "expected": "{{ expected }}"},
            },
            {
                "id": "sound",
                "kind": "judge",
                "target": "both",
                "scale": [0, 2],
                "guidance": {
                    "summary": "Is it sound?",
                    "levels": {0: "Wrong.", 1: "Partly.", 2: "Right."},
                },
            },
        ]
    },
}


@pytest.fixture(autouse=True)
def _reset_ui():
    ui.reset()
    yield
    ui.reset()


def write_prompt(tmp_path, rows=ROWS, prompt=PROMPT):
    (tmp_path / "cases.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    path = tmp_path / "prompt.yaml"
    path.write_text(yaml.safe_dump(prompt), encoding="utf-8")
    return path


def write_config(tmp_path, judge=None):
    path = tmp_path / "judge.yaml"
    path.write_text(
        yaml.safe_dump({"judge": judge or {"model": "judge/x"}}), encoding="utf-8"
    )
    return path


def test_the_response_is_not_a_template_variable(tmp_path):
    path = write_prompt(tmp_path)
    cases = load_cases(load_prompt(path), path)
    first = cases.cases[0]
    assert "output" not in first.variables
    assert "reasoning" not in first.variables
    assert first.variables["mutation"] == "original"
    assert first.recorded.output == "7"
    assert first.recorded.reasoning == "Seven is seven."
    assert first.recorded.model == "agent/x"


def test_a_row_without_an_output_is_an_error(tmp_path):
    rows = [{k: v for k, v in ROWS[0].items() if k != "output"}]
    path = write_prompt(tmp_path, rows)
    with pytest.raises(ConfigError, match="output"):
        load_cases(load_prompt(path), path)


def test_a_case_id_that_cannot_name_a_file_is_an_error(tmp_path):
    path = write_prompt(tmp_path, [ROWS[0] | {"case_id": "c1/original"}])
    with pytest.raises(ConfigError, match="cannot name a recorded sample"):
        load_cases(load_prompt(path), path)


def test_without_a_model_column_the_model_is_recorded(tmp_path):
    prompt = PROMPT | {"dataset": {"path": "cases.jsonl", "recorded": {}}}
    path = write_prompt(tmp_path, prompt=prompt)
    cases = load_cases(load_prompt(path), path)
    assert cases.cases[0].recorded.model == RECORDED_MODEL
    assert cases.cases[0].recorded.variant is None


def test_each_label_gets_the_first_variants_template(tmp_path):
    path = write_prompt(tmp_path)
    prompt = load_prompt(path)
    cases = load_cases(prompt, path)
    labelled = recorded_prompt(prompt, cases)
    assert [v.id for v in labelled.variants] == ["original", "wrong"]
    assert labelled.variants[1].user == prompt.variants[0].user

    samples = recorded_samples(labelled, cases)
    assert [s.sample_id for s in samples] == ["c1--original", "c1--wrong"]
    assert [s.variant_id for s in samples] == ["original", "wrong"]
    assert {s.model for s in samples} == {"agent/x"}


@respx.mock(assert_all_called=False)
def test_run_stores_recorded_responses_without_calling_anything(tmp_path):
    """respx.mock with no routes fails any request that is made."""
    prompt = write_prompt(tmp_path)
    config = write_config(tmp_path)
    result = CliRunner().invoke(
        app, ["run", str(config), str(prompt), "--out", str(tmp_path / "runs")]
    )
    assert result.exit_code == ExitCode.OK, result.output
    (run_dir,) = (tmp_path / "runs").iterdir()
    with sqlite3.connect(run_dir / "results.sqlite") as conn:
        rows = dict(conn.execute("SELECT sample_id, output FROM samples"))
        trace = conn.execute(
            "SELECT reasoning_availability FROM samples WHERE sample_id = 'c1--wrong'"
        ).fetchone()[0]
    assert rows == {"c1--original": "7", "c1--wrong": "8"}
    assert trace == "absent"
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    assert manifest["run_config"]["models"] == ["agent/x"]
    assert [v["id"] for v in manifest["prompt"]["variants"]] == ["original", "wrong"]

    scored = CliRunner().invoke(app, ["score", str(run_dir), "--only", "deterministic"])
    assert scored.exit_code == ExitCode.OK, scored.output
    with sqlite3.connect(run_dir / "results.sqlite") as conn:
        passed = dict(conn.execute("SELECT sample_id, score FROM scores"))
    assert passed == {"c1--original": 1.0, "c1--wrong": 0.0}


def test_run_dry_run_stores_nothing(tmp_path):
    prompt = write_prompt(tmp_path)
    config = write_config(tmp_path)
    result = CliRunner().invoke(
        app,
        ["run", str(config), str(prompt), "--out", str(tmp_path / "runs"), "--dry-run"],
    )
    assert result.exit_code == ExitCode.OK, result.output
    assert not (tmp_path / "runs").exists()


def test_a_generated_run_still_needs_models(tmp_path):
    prompt = PROMPT | {"dataset": None}
    path = write_prompt(tmp_path, prompt=prompt)
    config = write_config(tmp_path)
    result = CliRunner().invoke(app, ["run", str(config), str(path), "--dry-run"])
    assert result.exit_code == ExitCode.USAGE
    assert "models list must not be empty" in result.stderr
