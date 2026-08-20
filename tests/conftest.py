"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from reasonbench.config import PromptSpec, RunConfig

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture
def run_config() -> RunConfig:
    """Return a small in-memory run configuration."""
    return RunConfig.model_validate(
        {
            "models": ["model/a", "model/b"],
            "sweep": {
                "repeats": 2,
                "temperature": [0.0, 1.0],
                "reasoning_effort": ["low", "high"],
            },
            "judge": {"model": "judge/x"},
        },
    )


@pytest.fixture
def prompt_spec() -> PromptSpec:
    """Return a two-variant prompt with one deterministic and one judge criterion."""
    return PromptSpec.model_validate(
        {
            "id": "p1",
            "title": "Test prompt",
            "system": "You are terse.",
            "variables": {"n": 7},
            "variants": [
                {"id": "plain", "user": "What is {{ n }}? Answer with a number."},
                {"id": "cot", "user": "What is {{ n }}? Think, then answer."},
            ],
            "rubric": {
                "judge_instructions": "Grade carefully.",
                "criteria": [
                    {
                        "id": "correct",
                        "kind": "deterministic",
                        "target": "output",
                        "scope": "last_line",
                        "weight": 2.0,
                        "check": {"type": "regex", "pattern": r"\b7\b"},
                    },
                    {
                        "id": "sound_reasoning",
                        "kind": "judge",
                        "target": "reasoning",
                        "weight": 1.0,
                        "scale": [0, 4],
                        "guidance": {
                            "summary": "Is the reasoning valid?",
                            "levels": {4: "Flawless.", 0: "Nonsense."},
                            "positive_indicators": ["shows the arithmetic"],
                            "negative_indicators": ["guesses"],
                            "notes": "Be strict.",
                        },
                    },
                ],
            },
        },
    )
