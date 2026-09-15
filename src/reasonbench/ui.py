"""The one place that builds a Console, and the only way to print.

Data goes to stdout, narration to stderr. Nothing here accepts an already
formatted string: every printer takes a literal template plus arguments, and
the arguments are escaped on the way in. Model output and judge text reach the
terminal through ``blob``, which does not parse markup at all.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NoReturn

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from reasonbench.errors import ExitCode, ReasonBenchError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.console import RenderableType

# C0/C1 control characters a model could emit to move the cursor or set colour.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

FALLBACK_WIDTH = 150


class Mode(StrEnum):
    """How much the tool says, and in what form."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


@dataclass
class _State:
    mode: Mode = Mode.NORMAL
    json_out: bool = False
    color: bool = True
    assume_yes: bool = False
    out: Console = field(default_factory=Console)
    err: Console = field(default_factory=lambda: Console(stderr=True))


_state = _State()


def _want_color(explicit_no_color: bool) -> bool:
    if explicit_no_color or os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return os.environ.get("TERM") != "dumb" and sys.stdout.isatty()


def configure(
    *,
    quiet: bool = False,
    verbose: int = 0,
    json_out: bool = False,
    no_color: bool = False,
    yes: bool = False,
) -> None:
    """Set the output mode for this invocation."""
    if quiet and verbose:
        raise typer.BadParameter("--quiet and --verbose are mutually exclusive")
    mode = (
        Mode.QUIET
        if quiet
        else (Mode.NORMAL, Mode.VERBOSE, Mode.DEBUG)[min(verbose, 2)]
    )
    color = _want_color(no_color) and not json_out
    width = (
        None if sys.stdout.isatty() else int(os.environ.get("COLUMNS", FALLBACK_WIDTH))
    )
    _state.mode = mode
    _state.json_out = json_out
    _state.color = color
    _state.assume_yes = yes
    _state.out = Console(
        width=width, no_color=not color, highlight=False, soft_wrap=False
    )
    _state.err = Console(stderr=True, width=width, no_color=not color, highlight=False)


def reset() -> None:
    """Restore defaults. Tests call this between invocations."""
    global _state  # noqa: PLW0603
    _state = _State()


def err_console() -> Console:
    """Return the stderr console, for widgets that render themselves."""
    return _state.err


def is_quiet() -> bool:
    """Return whether narration is suppressed."""
    return _state.mode is Mode.QUIET


def is_json() -> bool:
    """Return whether this invocation emits a JSON document."""
    return _state.json_out


def table(title: str, *, caption: str | None = None) -> Table:
    """Return a table styled for the current mode."""
    return Table(
        title=title,
        caption=caption,
        header_style="bold",
        box=box.HEAVY_HEAD if _state.color else box.SIMPLE,
        title_justify="left",
    )


def cell(value: object) -> Text:
    """Wrap a value for a table cell so its brackets are not read as markup."""
    return Text(_CONTROL.sub("", str(value)))


def data(renderable: RenderableType) -> None:
    """Print a trusted renderable to stdout."""
    if _state.json_out:
        return
    _state.out.print(renderable)


def blob(text: str, *, sanitize: bool = True) -> None:
    """Write untrusted text to stdout verbatim, without parsing markup."""
    if _state.json_out:
        return
    _state.out.out(_CONTROL.sub("", text) if sanitize else text, highlight=False)


def json_document(payload: Mapping[str, Any]) -> None:
    """Write exactly one JSON document to stdout."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


def _emit(template: str, args: tuple[object, ...]) -> None:
    _state.err.print(template.format(*(escape(str(a)) for a in args)), soft_wrap=True)


def status(template: str, /, *args: object) -> None:
    """Report progress on stderr."""
    if _state.mode is not Mode.QUIET:
        _emit(template, args)


def note(template: str, /, *args: object) -> None:
    """Report detail shown only with --verbose."""
    if _state.mode in (Mode.VERBOSE, Mode.DEBUG):
        _emit("[dim]" + template + "[/dim]", args)


def warn(template: str, /, *args: object) -> None:
    """Report something the user should see even when quiet."""
    _emit("[yellow]warning:[/yellow] " + template, args)


def error(template: str, /, *args: object) -> None:
    """Report a failure on stderr."""
    _emit("[bold red]error:[/bold red] " + template, args)


def render_error(exc: ReasonBenchError) -> None:
    """Print a typed error, its details, and any hint."""
    error("{}", exc.message)
    for line in exc.details:
        _state.err.print("  " + escape(str(line)), soft_wrap=True)
    if exc.hint:
        _state.err.print("\n[dim]hint:[/dim] " + escape(exc.hint), soft_wrap=True)


def fail(
    template: str,
    /,
    *args: object,
    code: ExitCode = ExitCode.USAGE,
    hint: str | None = None,
) -> NoReturn:
    """Print an error and exit with ``code``."""
    error(template, *args)
    if hint:
        _state.err.print("\n[dim]hint:[/dim] " + escape(hint), soft_wrap=True)
    raise typer.Exit(int(code))


def confirm(question: str, *, default: bool = False) -> bool:
    """Ask a yes/no question. Non-interactive runs take --yes or the default."""
    if _state.assume_yes:
        return True
    if not sys.stdin.isatty():
        return default
    return typer.confirm(question, default=default)
