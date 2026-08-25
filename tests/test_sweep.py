"""Sweep expansion and the content-addressed sample ids that make resume work."""

from __future__ import annotations

from reasonbench.sweep import estimate_cost_usd, expand, make_sample_id


def test_expansion_is_the_full_cartesian_product(run_config, prompt_spec):
    samples = expand(run_config, prompt_spec)
    expected = 2 * 2 * 2 * 2 * 2  # models, variants, temps, efforts, repeats
    assert len(samples) == expected


def test_every_sample_id_is_unique(run_config, prompt_spec):
    samples = expand(run_config, prompt_spec)
    assert len({s.sample_id for s in samples}) == len(samples)


def test_expansion_is_deterministic(run_config, prompt_spec):
    first = expand(run_config, prompt_spec)
    second = expand(run_config, prompt_spec)
    assert [s.sample_id for s in first] == [s.sample_id for s in second]


def test_sample_id_depends_only_on_call_defining_fields():
    base = {
        "prompt_id": "p",
        "variant_id": "v",
        "model": "m",
        "temperature": 0.0,
        "reasoning_effort": "low",
        "repeat": 0,
    }
    assert make_sample_id(**base) == make_sample_id(**base)
    assert make_sample_id(**base) != make_sample_id(**{**base, "repeat": 1})
    assert make_sample_id(**base) != make_sample_id(**{**base, "temperature": 1.0})
    assert make_sample_id(**base) != make_sample_id(
        **{**base, "reasoning_effort": "high"},
    )


def test_repeats_of_one_cell_share_a_grouping_key(run_config, prompt_spec):
    samples = expand(run_config, prompt_spec)
    cell = samples[0].cell
    matching = [s for s in samples if s.cell == cell]
    assert len(matching) == run_config.sweep.repeats
    assert {s.repeat for s in matching} == set(range(run_config.sweep.repeats))


def test_cost_estimate_scales_with_sample_count(run_config, prompt_spec):
    samples = expand(run_config, prompt_spec)
    assert estimate_cost_usd(samples) > estimate_cost_usd(samples[:4])
    assert estimate_cost_usd(()) == 0.0
