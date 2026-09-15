"""Progress reporting that also works when stdout is not a terminal."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Protocol, Self

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from reasonbench import ui

if TYPE_CHECKING:
    from types import TracebackType

THROTTLE_S = 15.0


class Reporter(Protocol):
    """What the sweep and the judge call to report progress."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def advance(self, *, ok: bool = True, cost: float = 0.0) -> None:
        """Record one completed unit of work."""
        ...

    def retry(self, label: str, attempt: int, delay_s: float, reason: str) -> None:
        """Record a transient failure that is about to be retried."""
        ...


class _Null:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def advance(self, **_: object) -> None:
        return None

    def retry(self, *_args: object, **_kwargs: object) -> None:
        return None


class _Rich:
    def __init__(
        self, label: str, total: int, spent: float, *, show_cost: bool
    ) -> None:
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        if show_cost:
            columns.append(TextColumn("${task.fields[cost]:.4f}"))
        self._progress = Progress(*columns, console=ui.err_console())
        self._label, self._total, self._cost = label, total, spent
        self._show_cost = show_cost

    def __enter__(self) -> Self:
        self._progress.start()
        self._task = self._progress.add_task(
            self._label, total=self._total, cost=self._cost
        )
        return self

    def __exit__(self, *_: object) -> None:
        self._progress.stop()

    def advance(self, *, ok: bool = True, cost: float = 0.0) -> None:
        del ok
        self._cost += cost
        self._progress.update(self._task, advance=1, cost=self._cost)

    def retry(self, label: str, attempt: int, delay_s: float, reason: str) -> None:
        ui.note("retry {} attempt {} in {:.1f}s: {}", label, attempt, delay_s, reason)


class _Lines:
    """One throttled line at a time, for pipes and CI logs."""

    def __init__(
        self, label: str, total: int, spent: float, *, show_cost: bool
    ) -> None:
        self._label, self._total, self._cost = label, total, spent
        self._show_cost = show_cost
        self._done = self._ok = self._fail = 0
        self._every = max(1, total // 20)
        self._started = self._last = time.monotonic()

    def __enter__(self) -> Self:
        self._emit(force=True)
        return self

    def __exit__(self, *_: object) -> None:
        self._emit(force=True)

    def advance(self, *, ok: bool = True, cost: float = 0.0) -> None:
        self._done += 1
        self._cost += cost
        self._ok += int(ok)
        self._fail += int(not ok)
        due = self._done % self._every == 0
        if due or time.monotonic() - self._last >= THROTTLE_S or not ok:
            self._emit()

    def retry(self, label: str, attempt: int, delay_s: float, reason: str) -> None:
        ui.note("retry {} attempt {} in {:.1f}s: {}", label, attempt, delay_s, reason)

    def _emit(self, *, force: bool = False) -> None:
        del force
        self._last = time.monotonic()
        elapsed = time.monotonic() - self._started
        rate = self._done / elapsed * 60 if elapsed > 0 and self._done else 0.0
        pct = (self._done / self._total * 100) if self._total else 100.0
        cost = f" | ${self._cost:.4f}" if self._show_cost else ""
        ui.status(
            "[{:>4}/{:<4}] {:>3.0f}% | {:.0f}s | {:.0f}/min{} | ok {} fail {}",
            self._done,
            self._total,
            pct,
            elapsed,
            rate,
            cost,
            self._ok,
            self._fail,
        )


def reporter(
    label: str, total: int, *, spent: float = 0.0, show_cost: bool
) -> Reporter:
    """Return the progress reporter that suits the current output mode."""
    if ui.is_quiet() or ui.is_json():
        return _Null()
    if sys.stderr.isatty():
        return _Rich(label, total, spent, show_cost=show_cost)
    return _Lines(label, total, spent, show_cost=show_cost)
