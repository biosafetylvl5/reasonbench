"""Terminal and Markdown rendering of aggregated results."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from reasonbench import ui

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from rich.table import Table

    from reasonbench.scoring import GroupSummary
    from reasonbench.storage import SampleRow


# Below this many observations there is no spread to report.
MIN_FOR_SPREAD = 2


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_pm(mean: float | None, stdev: float | None, n: int) -> str:
    """Format ``mean +/- stdev``, keeping 'no spread' distinct from 'one sample'."""
    if mean is None:
        return "n/a"
    if n < MIN_FOR_SPREAD:
        return f"{mean:.2f} (n=1)"
    return f"{mean:.2f} ± {stdev or 0.0:.2f}"


def summary_table(summaries: list[GroupSummary], group_by: tuple[str, ...]) -> Table:
    """Build the headline per-group results table."""
    table = ui.table("Weighted rubric score")
    for field in group_by:
        table.add_column(field, style="cyan", no_wrap=True)
    table.add_column("n", justify="right")
    table.add_column("fail", justify="right")
    table.add_column("score", justify="right", style="bold")
    table.add_column("reasoning tok", justify="right")
    table.add_column("output tok", justify="right")
    table.add_column("latency s", justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("trace")

    for summary in summaries:
        traces = ", ".join(f"{k}={v}" for k, v in sorted(summary.availability.items()))
        table.add_row(
            *(ui.cell(k) for k in summary.key),
            str(summary.n_samples),
            str(summary.n_failed) if summary.n_failed else "-",
            fmt_pm(summary.weighted_mean, summary.weighted_stdev, summary.n_scored),
            f"{summary.mean_reasoning_tokens:.0f}",
            f"{summary.mean_completion_tokens:.0f}",
            f"{summary.mean_latency_s:.1f}",
            f"{summary.total_cost:.4f}",
            traces or "-",
        )
    return table


def criteria_table(summaries: list[GroupSummary], group_by: tuple[str, ...]) -> Table:
    """Build the per-criterion breakdown table."""
    table = ui.table("Per-criterion scores (normalized 0-1)")
    for field in group_by:
        table.add_column(field, style="cyan", no_wrap=True)
    table.add_column("criterion")
    table.add_column("kind", style="dim")
    table.add_column("target", style="dim")
    table.add_column("w", justify="right", style="dim")
    table.add_column("mean", justify="right", style="bold")
    table.add_column("coverage", justify="right")

    for summary in summaries:
        for index, criterion in enumerate(summary.criteria):
            coverage = f"{criterion.n_scored}/{criterion.n_total}"
            if criterion.n_scored == 0 and criterion.n_total:
                coverage = f"[yellow]{coverage}[/yellow]"
            table.add_row(
                *(
                    ui.cell(k)
                    for k in (summary.key if index == 0 else ("",) * len(group_by))
                ),
                ui.cell(criterion.criterion_id),
                ui.cell(criterion.kind),
                ui.cell(criterion.target),
                f"{criterion.weight:g}",
                fmt_pm(criterion.mean, criterion.stdev, criterion.n_scored),
                coverage,
            )
    return table


def _md_row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    """Render a Markdown table whose separator is derived from the headers."""
    out = [_md_row(headers), "|" + "---|" * len(headers)]
    out.extend(_md_row(r) for r in rows)
    return out


def render_markdown(
    summaries: list[GroupSummary],
    group_by: tuple[str, ...],
    *,
    title: str,
    prompt_title: str,
) -> str:
    """Render the aggregated results as a Markdown report."""
    group_headers = list(group_by) or ["all"]
    lines = [f"# {title}", "", f"Prompt: {prompt_title}", ""]

    lines += ["## Weighted rubric score", ""]
    lines += _md_table(
        [
            *group_headers,
            "n",
            "fail",
            "score",
            "reasoning tok",
            "output tok",
            "latency s",
            "cost $",
            "trace availability",
        ],
        (
            [
                *(summary.key or ("all",)),
                summary.n_samples,
                summary.n_failed,
                fmt_pm(summary.weighted_mean, summary.weighted_stdev, summary.n_scored),
                f"{summary.mean_reasoning_tokens:.0f}",
                f"{summary.mean_completion_tokens:.0f}",
                f"{summary.mean_latency_s:.1f}",
                f"{summary.total_cost:.4f}",
                ", ".join(f"{k}={v}" for k, v in sorted(summary.availability.items()))
                or "-",
            ]
            for summary in summaries
        ),
    )

    lines += ["", "## Per-criterion scores", ""]
    lines += _md_table(
        [*group_headers, "criterion", "kind", "target", "weight", "mean", "coverage"],
        (
            [
                *(summary.key or ("all",)),
                criterion.criterion_id,
                criterion.kind,
                criterion.target,
                f"{criterion.weight:g}",
                fmt_pm(criterion.mean, criterion.stdev, criterion.n_scored),
                f"{criterion.n_scored}/{criterion.n_total}",
            ]
            for summary in summaries
            for criterion in summary.criteria
        ),
    )

    lines += [
        "",
        "> `n/a` means the criterion could not be applied to that sample. "
        "Coverage counts the samples it could.",
        "",
    ]
    return "\n".join(lines)


def export_csv(samples: list[SampleRow], path: Path) -> None:
    """Write every sample to CSV for downstream analysis."""
    fields = [
        "sample_id",
        "model",
        "variant_id",
        "temperature",
        "reasoning_effort",
        "repeat",
        "ok",
        "reasoning_availability",
        "reasoning_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "latency_s",
        "output",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in samples:
            writer.writerow(row.model_dump(mode="json"))


def print_report(summaries: list[GroupSummary], group_by: tuple[str, ...]) -> None:
    """Print both tables to stdout."""
    ui.data("")
    ui.data(summary_table(summaries, group_by))
    ui.data("")
    ui.data(criteria_table(summaries, group_by))
    ui.data("")
