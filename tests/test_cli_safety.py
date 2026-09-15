"""Untrusted text must survive printing, and nothing may print around the ui layer."""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib

import pytest

from reasonbench import ui

HOSTILE = [
    "[/INST]",
    "[/]",
    "[/bold]",
    "[bold red]unclosed",
    "hello [a-z]+ world",
    "See [link](http://x) for details",
    "[type=missing, input_value={'a': 1}, input_type=dict]",
    "array[0] and dict[key]",
    "regex used: [a-z]+ and [0-9]{2}",
]

SRC = pathlib.Path(__file__).parent.parent / "src" / "reasonbench"
GUARDED = ["cli.py", "report.py"]


@pytest.fixture(autouse=True)
def _reset_ui():
    ui.reset()
    yield
    ui.reset()


@pytest.mark.parametrize("text", HOSTILE)
def test_blob_writes_untrusted_text_unchanged(text):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ui.configure()
        ui.blob(text)
    assert buf.getvalue().rstrip("\n") == text


def test_blob_strips_control_characters():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ui.configure()
        ui.blob("before\x1b[31mafter\x00end")
    assert "\x1b" not in buf.getvalue()
    assert "\x00" not in buf.getvalue()
    assert "before" in buf.getvalue()


@pytest.mark.parametrize("text", HOSTILE)
def test_status_escapes_its_arguments(text):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        ui.configure()
        ui.status("sample failed {}", text)
    assert text in buf.getvalue()


@pytest.mark.parametrize("text", HOSTILE)
def test_cell_does_not_parse_markup(text):
    assert ui.cell(text).plain == text


def _guarded_trees():
    for name in GUARDED:
        path = SRC / name
        if path.is_file():
            yield path, ast.parse(path.read_text(encoding="utf-8"))


def test_no_console_is_built_outside_ui():
    for path, tree in _guarded_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                assert name != "Console", f"{path.name} builds its own Console"


def test_guarded_modules_do_not_print_directly():
    banned = {"print", "print_json", "rule", "log"}
    for path, tree in _guarded_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                where = f"{path.name}:{node.lineno}"
                assert node.func.attr not in banned, (
                    f"{where} calls .{node.func.attr}(); print via reasonbench.ui"
                )


def test_ui_templates_are_literals():
    """Every ui.* call must pass a literal template, so arguments get escaped."""
    printers = {"status", "note", "warn", "error", "fail"}
    for path, tree in _guarded_trees():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in printers
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ui"
                and node.args
            ):
                first = node.args[0]
                where = f"{path.name}:{node.lineno} -> ui.{node.func.attr}()"
                assert isinstance(first, ast.Constant), (
                    f"{where} takes a computed template; pass a literal"
                )
                assert isinstance(first.value, str), f"{where} template is not a string"
