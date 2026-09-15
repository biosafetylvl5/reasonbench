"""Command behaviour: exit codes, and the bugs that used to destroy data."""

from __future__ import annotations

import sqlite3

import pytest
from conftest import load_fixture, make_sample
from typer.testing import CliRunner

from reasonbench import ui
from reasonbench.cli import app
from reasonbench.errors import ExitCode
from reasonbench.openrouter import parse_response
from reasonbench.storage import RunStore, ScoreRow, new_run_dir


@pytest.fixture(autouse=True)
def _reset_ui():
    ui.reset()
    yield
    ui.reset()


@pytest.fixture
def runner():
    return CliRunner()


def _score_count(run_dir):
    with sqlite3.connect(run_dir / "results.sqlite") as conn:
        return conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]


@pytest.fixture
def scored_run(tmp_path, run_config, prompt_spec):
    """Build a run directory with one stored sample and one judge score."""
    run_dir = new_run_dir(tmp_path / "runs")
    with RunStore(run_dir, create=True) as store:
        raw = load_fixture("gemma_full_text")
        store.add_sample(parse_response(raw, make_sample(sample_id="s1"), 1.0), raw)
        store.add_scores(
            [
                ScoreRow(
                    sample_id="s1",
                    criterion_id="sound_reasoning",
                    judge_repeat=0,
                    kind="judge",
                    target="reasoning",
                    weight=1.0,
                    score=3.0,
                    normalized=0.75,
                    applicable=True,
                    reason="fine",
                )
            ]
        )
        store.write_manifest(
            {
                "run_config": run_config.model_dump(mode="json"),
                "prompt": prompt_spec.model_dump(mode="json"),
            }
        )
    return run_dir


def test_version_exits_zero(runner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == ExitCode.OK
    assert "reasonbench" in result.stdout


def test_show_on_a_missing_run_dir_creates_nothing(runner, tmp_path):
    missing = tmp_path / "typo"
    result = runner.invoke(app, ["show", str(missing), "abc"])
    assert result.exit_code == ExitCode.NOT_FOUND
    assert not missing.exists()


def test_report_on_a_missing_run_dir_creates_nothing(runner, tmp_path):
    missing = tmp_path / "typo"
    result = runner.invoke(app, ["report", str(missing)])
    assert result.exit_code == ExitCode.NOT_FOUND
    assert not missing.exists()


def test_score_without_a_key_keeps_existing_scores(
    runner, scored_run, monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    before = _score_count(scored_run)
    assert before == 1
    result = runner.invoke(app, ["score", str(scored_run)])
    assert result.exit_code == ExitCode.AUTH
    assert _score_count(scored_run) == before


def test_unknown_group_by_field_is_a_usage_error(runner, scored_run):
    result = runner.invoke(app, ["report", str(scored_run), "--group-by", "modle"])
    assert result.exit_code == ExitCode.USAGE
    assert "model" in result.stderr


def test_empty_group_by_is_a_usage_error(runner, scored_run):
    result = runner.invoke(app, ["report", str(scored_run), "--group-by", ""])
    assert result.exit_code == ExitCode.USAGE


def test_ambiguous_sample_prefix_is_reported(runner, tmp_path):
    run_dir = new_run_dir(tmp_path / "runs")
    with RunStore(run_dir, create=True) as store:
        raw = load_fixture("gemma_full_text")
        for sid in ("ab1", "ab2"):
            store.add_sample(parse_response(raw, make_sample(sample_id=sid), 1.0), raw)
    result = runner.invoke(app, ["show", str(run_dir), "ab"])
    assert result.exit_code == ExitCode.USAGE
    assert "matches 2 samples" in result.stderr


def test_run_rejects_resume_combined_with_out(runner, tmp_path):
    result = runner.invoke(
        app,
        ["run", "m.yaml", "p.yaml", "--resume", str(tmp_path), "--out", str(tmp_path)],
    )
    assert result.exit_code != ExitCode.OK


def test_raw_writes_the_artifact_byte_for_byte(runner, scored_run):
    artifact = scored_run / "raw" / "s1.json"
    result = runner.invoke(app, ["raw", str(scored_run), "s1"])
    assert result.exit_code == ExitCode.OK
    assert result.stdout == artifact.read_text(encoding="utf-8")
