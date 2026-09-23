"""Checks the docs against the code."""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from reasonbench.config import DatasetSpec
from reasonbench.errors import ExitCode
from reasonbench.gate import GateSpec

DOCS = pathlib.Path(__file__).parent.parent / "docs"


def fenced(path: pathlib.Path, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", path.read_text(), re.S)


@pytest.mark.parametrize("code", list(ExitCode))
def test_every_exit_code_is_documented(code):
    text = (DOCS / "exit-codes.md").read_text()
    assert code.name in text, f"{code.name} is not in exit-codes.md"
    assert f"| {int(code)} |" in text, f"{int(code)} is not in the table"


def test_the_gate_example_validates():
    blocks = fenced(DOCS / "gate.md", "yaml")
    assert blocks, "gate.md has no yaml example"
    GateSpec.model_validate(yaml.safe_load(blocks[0]))


def test_the_dataset_example_validates():
    blocks = fenced(DOCS / "datasets.md", "yaml")
    assert blocks, "datasets.md has no yaml example"
    DatasetSpec.model_validate(yaml.safe_load(blocks[0])["dataset"])


def test_the_shipped_gate_template_validates():
    root = pathlib.Path(__file__).parent.parent / "configs"
    for path in [root / "gate.yaml", root / "ci" / "gate.yaml"]:
        GateSpec.model_validate(yaml.safe_load(path.read_text()))
