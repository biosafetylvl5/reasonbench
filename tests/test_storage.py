"""Persistence: raw artifacts, resume, budget accounting, and re-scoring."""

from __future__ import annotations

import yaml
from conftest import load_fixture, make_sample

from reasonbench.config import PromptSpec, RunConfig
from reasonbench.openrouter import SampleResult, parse_response
from reasonbench.storage import RunStore, ScoreRow, new_run_dir


def store_one(store, sample_id="s1", fixture="gemma_full_text"):
    """Persist one parsed fixture response under the given sample id."""
    raw = load_fixture(fixture)
    result = parse_response(raw, make_sample(sample_id=sample_id), 1.0)
    store.add_sample(result, raw)
    return result


def test_sample_round_trips_through_sqlite(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        original = store_one(store)
        (row,) = store.samples()
    assert row.sample_id == "s1"
    assert row.reasoning_availability == original.reasoning.availability
    assert row.readable_reasoning == original.reasoning.readable
    assert row.cost == original.usage.cost


def test_raw_response_is_written_verbatim(tmp_path):
    run_dir = new_run_dir(tmp_path)
    with RunStore(run_dir) as store:
        store_one(store)
    artifact = run_dir / "raw" / "s1.json"
    assert artifact.is_file()
    assert "reasoning_details" in artifact.read_text(encoding="utf-8")


def test_existing_ids_drive_resume(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        store_one(store, "s1")
        store_one(store, "s2")
        assert store.existing_sample_ids() == {"s1", "s2"}


def test_reopening_a_run_dir_keeps_its_samples(tmp_path):
    run_dir = new_run_dir(tmp_path)
    with RunStore(run_dir) as store:
        store_one(store)
    with RunStore(run_dir) as reopened:
        assert len(reopened.samples()) == 1
        assert reopened.total_cost() > 0


def test_total_cost_sums_samples(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        first = store_one(store, "s1")
        second = store_one(store, "s2")
        expected = first.usage.cost + second.usage.cost
        assert store.total_cost() == expected


def test_scores_can_be_cleared_without_touching_samples(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        store_one(store)
        store.add_scores(
            [
                ScoreRow(
                    sample_id="s1",
                    criterion_id="c",
                    judge_repeat=0,
                    kind="judge",
                    target="reasoning",
                    weight=1.0,
                    score=3.0,
                    normalized=0.75,
                    applicable=True,
                ),
            ],
        )
        assert len(store.scores()) == 1

        store.clear_scores()
        assert store.scores() == []
        assert len(store.samples()) == 1  # generations survive


def test_na_scores_round_trip_as_none(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        store_one(store)
        store.add_scores(
            [
                ScoreRow(
                    sample_id="s1",
                    criterion_id="c",
                    judge_repeat=0,
                    kind="judge",
                    target="reasoning",
                    weight=1.0,
                    score=None,
                    normalized=None,
                    applicable=False,
                    reason="no trace",
                ),
            ],
        )
        (row,) = store.scores()
    assert row.score is None
    assert row.applicable is False


def test_judge_repeats_are_stored_separately(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        store_one(store)
        store.add_scores(
            [
                ScoreRow(
                    sample_id="s1",
                    criterion_id="c",
                    judge_repeat=repeat,
                    kind="judge",
                    target="output",
                    weight=1.0,
                    score=float(repeat),
                    normalized=repeat / 4,
                    applicable=True,
                )
                for repeat in range(3)
            ],
        )
        assert len(store.scores()) == 3


def test_manifest_round_trips_the_rubric(tmp_path, run_config, prompt_spec):
    """`score` reconstructs the rubric from the manifest, so it must survive."""
    run_dir = new_run_dir(tmp_path)
    with RunStore(run_dir) as store:
        store.write_manifest(
            {
                "run_config": run_config.model_dump(mode="json"),
                "prompt": prompt_spec.model_dump(mode="json"),
            },
        )
    data = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert RunConfig.model_validate(data["run_config"]) == run_config
    assert PromptSpec.model_validate(data["prompt"]) == prompt_spec


def test_run_dir_name_includes_a_label(tmp_path):
    assert new_run_dir(tmp_path, "mylabel").name.endswith("_mylabel")


def test_resume_skips_successes_but_retries_failures(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        store_one(store, "good")
        store.add_sample(
            SampleResult(
                sample=make_sample(sample_id="bad"),
                ok=False,
                error="HTTP 500",
            ),
            {"error": "HTTP 500"},
        )
        assert store.existing_sample_ids() == {"good"}
        assert len(store.samples()) == 2  # the failure is still recorded


def test_retrying_a_failed_sample_overwrites_it(tmp_path):
    with RunStore(new_run_dir(tmp_path)) as store:
        store.add_sample(
            SampleResult(sample=make_sample(sample_id="s1"), ok=False, error="boom"),
            {"error": "boom"},
        )
        store_one(store, "s1")  # same id, now successful
        (row,) = store.samples()
        assert row.ok is True
        assert store.existing_sample_ids() == {"s1"}
