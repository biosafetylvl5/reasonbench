"""Configuration schemas, YAML loading, and validation errors."""

from __future__ import annotations

import pytest
from conftest import CONFIGS

from reasonbench.config import (
    ConfigError,
    Guidance,
    JudgeCriterion,
    PromptSpec,
    RunConfig,
    load_prompt,
    load_run_config,
)


def test_shipped_configs_load():
    config = load_run_config(CONFIGS / "models.yaml")
    prompt = load_prompt(CONFIGS / "prompts" / "periodic-table.yaml")
    assert config.models
    assert config.judge.model
    assert len(prompt.variants) == 2
    assert len(prompt.rubric.criteria) == 5


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="No such config file"):
        load_run_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("models: [a\n  bad", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_run_config(path)


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="typoo"):
        RunConfig.model_validate(
            {"models": ["a/b"], "judge": {"model": "j"}, "typoo": 1},
        )


def test_empty_model_list_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        RunConfig.model_validate({"models": [], "judge": {"model": "j"}})


def test_duplicate_models_are_rejected():
    with pytest.raises(ValueError, match="duplicate model slugs"):
        RunConfig.model_validate(
            {"models": ["a/b", "a/b"], "judge": {"model": "j"}},
        )


def test_unknown_reasoning_effort_is_rejected():
    with pytest.raises(ValueError, match="unknown reasoning_effort"):
        RunConfig.model_validate(
            {
                "models": ["a/b"],
                "judge": {"model": "j"},
                "sweep": {"reasoning_effort": ["turbo"]},
            },
        )


def test_guidance_accepts_a_bare_string():
    guidance = Guidance.model_validate("just check it is right")
    assert guidance.summary == "just check it is right"
    assert guidance.levels == {}


def test_guidance_level_outside_scale_is_rejected():
    with pytest.raises(ValueError, match="fall outside scale"):
        JudgeCriterion.model_validate(
            {
                "id": "c",
                "kind": "judge",
                "scale": [0, 2],
                "guidance": {"summary": "s", "levels": {5: "impossible"}},
            },
        )


def test_inverted_scale_is_rejected():
    with pytest.raises(ValueError, match="must be increasing"):
        JudgeCriterion.model_validate(
            {"id": "c", "kind": "judge", "scale": [4, 0], "guidance": "s"},
        )


def test_duplicate_criterion_ids_are_rejected(prompt_spec):
    duplicated = prompt_spec.rubric.criteria[0]
    with pytest.raises(ValueError, match="duplicate criterion ids"):
        PromptSpec.model_validate(
            {
                "id": "p",
                "title": "t",
                "variants": [{"id": "v", "user": "u"}],
                "rubric": {
                    "criteria": [
                        duplicated.model_dump(),
                        duplicated.model_dump(),
                    ],
                },
            },
        )


def test_attachment_referencing_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variants"):
        PromptSpec.model_validate(
            {
                "id": "p",
                "title": "t",
                "variants": [{"id": "v", "user": "u"}],
                "attachments": [
                    {
                        "id": "a",
                        "url": "data:image/png;base64,AA",
                        "media_type": "image/png",
                        "attach_to": ["ghost"],
                    },
                ],
                "rubric": {
                    "criteria": [
                        {
                            "id": "c",
                            "kind": "deterministic",
                            "check": {"type": "contains", "expected": "x"},
                        },
                    ],
                },
            },
        )


def test_attachment_file_is_inlined_as_a_data_url(tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "p.yaml").write_text(
        """
id: p
title: t
variants: [{id: v, user: u}]
attachments:
  - id: pic
    path: pic.png
rubric:
  criteria:
    - {id: c, kind: deterministic, check: {type: contains, expected: x}}
""",
        encoding="utf-8",
    )
    prompt = load_prompt(tmp_path / "p.yaml")
    assert prompt.attachments[0].url.startswith("data:image/png;base64,")
    assert prompt.attachments[0].media_type == "image/png"


def test_missing_attachment_file_raises(tmp_path):
    (tmp_path / "p.yaml").write_text(
        """
id: p
title: t
variants: [{id: v, user: u}]
attachments: [{id: pic, path: absent.png}]
rubric:
  criteria:
    - {id: c, kind: deterministic, check: {type: contains, expected: x}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="attachment file not found"):
        load_prompt(tmp_path / "p.yaml")


def test_config_is_immutable(run_config):
    with pytest.raises(ValueError, match="frozen"):
        run_config.max_tokens = 1
