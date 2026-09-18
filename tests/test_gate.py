"""Thresholds, the n/a rule, and the exit codes CI depends on."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest
from conftest import load_fixture, make_row, make_sample
from typer.testing import CliRunner

from reasonbench import ui
from reasonbench.cli import app
from reasonbench.errors import ExitCode
from reasonbench.gate import CriterionGate, GateSpec, OverallGate, evaluate
from reasonbench.openrouter import parse_response
from reasonbench.scoring import aggregate
from reasonbench.storage import RunStore, ScoreRow, new_run_dir


@pytest.fixture(autouse=True)
def _reset_ui():
    ui.reset()
    yield
    ui.reset()


def _summaries(scores, prompt_spec, samples):
    overall = aggregate(samples, scores, prompt_spec.rubric, group_by=())[0]
    groups = {("model",): aggregate(samples, scores, prompt_spec.rubric, ("model",))}
    return overall, groups


def _score(sample_id, criterion, value, *, applicable=True, kind="judge"):
    return ScoreRow(
        sample_id=sample_id,
        criterion_id=criterion,
        judge_repeat=0,
        kind=kind,
        target="reasoning" if kind == "judge" else "output",
        weight=1.0,
        score=None if value is None else value * 4,
        normalized=value,
        applicable=applicable,
        reason="",
    )


@pytest.fixture
def run_with(tmp_path, run_config, prompt_spec):
    """Build a run directory whose stored scores the test chooses."""

    def build(scores, *, ok=True):
        run_dir = new_run_dir(tmp_path / "runs")
        raw = load_fixture("gemma_full_text")
        with RunStore(run_dir, create=True) as store:
            result = parse_response(raw, make_sample(sample_id="s1"), 1.0)
            if not ok:
                result = result.model_copy(update={"ok": False, "error": "boom"})
            store.add_sample(result, raw)
            store.add_scores(scores)
            store.write_manifest(
                {
                    "run_config": run_config.model_dump(mode="json"),
                    "prompt": prompt_spec.model_dump(mode="json"),
                }
            )
        return run_dir

    return build


def test_a_breached_threshold_fails(prompt_spec):
    samples = [make_row(sample_id="s1")]
    scores = [_score("s1", "sound_reasoning", 0.25)]
    overall, groups = _summaries(scores, prompt_spec, samples)
    spec = GateSpec(overall=OverallGate(min_samples=0, min_weighted_mean=0.9))
    result = evaluate(spec, overall, groups)
    assert result.passed is False
    assert result.n_failed == 1


def test_an_unmeasurable_criterion_is_skipped_not_failed(prompt_spec):
    """A model that exposes no trace must not be ranked down for it."""
    scores = [_score("s1", "sound_reasoning", None, applicable=False)]
    overall, groups = _summaries(scores, prompt_spec, [make_row(sample_id="s1")])
    spec = GateSpec(
        overall=OverallGate(min_samples=0),
        criteria=(CriterionGate(id="sound_reasoning", min_mean=0.9),),
    )
    result = evaluate(spec, overall, groups)
    assert result.n_failed == 0
    assert result.n_skipped == 1
    assert result.passed is True


def test_coverage_is_how_gradeability_is_asserted(prompt_spec):
    scores = [_score("s1", "sound_reasoning", None, applicable=False)]
    overall, groups = _summaries(scores, prompt_spec, [make_row(sample_id="s1")])
    spec = GateSpec(
        overall=OverallGate(min_samples=0),
        criteria=(CriterionGate(id="sound_reasoning", min_coverage=1.0),),
    )
    result = evaluate(spec, overall, groups)
    assert result.passed is False


def test_no_gate_means_no_verdict(prompt_spec):
    overall, groups = _summaries([], prompt_spec, [])
    result = evaluate(None, overall, groups)
    assert result.configured is False
    assert result.passed is True


def test_an_empty_run_cannot_pass(prompt_spec):
    overall, groups = _summaries([], prompt_spec, [])
    result = evaluate(GateSpec(), overall, groups)
    assert result.passed is False


def test_gate_command_exits_20_on_a_breach(run_with):
    run_dir = run_with([_score("s1", "sound_reasoning", 0.25)])
    result = CliRunner().invoke(app, ["gate", str(run_dir), "--fail-under", "0.99"])
    assert result.exit_code == ExitCode.GATE


def test_gate_command_exits_0_when_thresholds_hold(run_with):
    run_dir = run_with([_score("s1", "sound_reasoning", 1.0)])
    result = CliRunner().invoke(app, ["gate", str(run_dir), "--fail-under", "0.5"])
    assert result.exit_code == ExitCode.OK


def test_gate_and_fail_under_together_is_a_usage_error(run_with):
    run_dir = run_with([_score("s1", "sound_reasoning", 1.0)])
    result = CliRunner().invoke(
        app, ["gate", str(run_dir), "--fail-under", "0.5", "--gate", "g.yaml"]
    )
    assert result.exit_code == ExitCode.USAGE


def test_artifacts_are_written_even_when_the_gate_fails(run_with):
    run_dir = run_with([_score("s1", "sound_reasoning", 0.1)])
    result = CliRunner().invoke(app, ["gate", str(run_dir), "--fail-under", "0.99"])
    assert result.exit_code == ExitCode.GATE

    payload = json.loads((run_dir / "report.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["exit_code"] == int(ExitCode.GATE)
    assert payload["gate"]["passed"] is False

    tree = ET.parse(run_dir / "junit.xml")  # noqa: S314
    names = [s.get("name") for s in tree.getroot()]
    assert "reasonbench.gate" in names
