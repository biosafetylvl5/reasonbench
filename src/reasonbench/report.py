"""Terminal and Markdown rendering of aggregated results."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from reasonbench.scoring import GroupSummary
    from reasonbench.storage import SampleRow


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pm(mean: float | None, stdev: float | None) -> str:
    if mean is None:
        return "[dim]n/a[/dim]"
    return f"{mean:.2f} ± {stdev:.2f}" if stdev else f"{mean:.2f}"


def summary_table(summaries: list[GroupSummary], group_by: tuple[str, ...]) -> Table:
    """Build the headline per-group results table."""
    table = Table(title="Weighted rubric score", header_style="bold")
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
            *summary.key,
            str(summary.n_samples),
            str(summary.n_failed) if summary.n_failed else "-",
            _fmt_pm(summary.weighted_mean, summary.weighted_stdev),
            f"{summary.mean_reasoning_tokens:.0f}",
            f"{summary.mean_completion_tokens:.0f}",
            f"{summary.mean_latency_s:.1f}",
            f"{summary.total_cost:.4f}",
            traces or "-",
        )
    return table


def criteria_table(summaries: list[GroupSummary], group_by: tuple[str, ...]) -> Table:
    """Build the per-criterion breakdown table."""
    table = Table(title="Per-criterion scores (normalized 0-1)", header_style="bold")
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
                *(summary.key if index == 0 else ("",) * len(group_by)),
                criterion.criterion_id,
                criterion.kind,
                criterion.target,
                f"{criterion.weight:g}",
                _fmt_pm(criterion.mean, criterion.stdev),
                coverage,
            )
    return table


def render_markdown(
    summaries: list[GroupSummary],
    group_by: tuple[str, ...],
    *,
    title: str,
    prompt_title: str,
) -> str:
    """Render the aggregated results as a Markdown report."""
    header = " | ".join(group_by)
    lines = [
        f"# {title}",
        "",
        f"Prompt: {prompt_title}",
        "",
        "## Weighted rubric score",
        "",
        f"| {header} | n | fail | score | reasoning tok | output tok "
        "| latency s | cost $ | trace availability |",
        f"|{'---|' * (len(group_by) + 8)}",
    ]
    for summary in summaries:
        traces = ", ".join(f"{k}={v}" for k, v in sorted(summary.availability.items()))
        score = (
            "n/a"
            if summary.weighted_mean is None
            else f"{summary.weighted_mean:.2f} ± {summary.weighted_stdev or 0:.2f}"
        )
        lines.append(
            f"| {' | '.join(summary.key)} | {summary.n_samples} | "
            f"{summary.n_failed} | {score} | "
            f"{summary.mean_reasoning_tokens:.0f} | "
            f"{summary.mean_completion_tokens:.0f} | "
            f"{summary.mean_latency_s:.1f} | {summary.total_cost:.4f} | "
            f"{traces or '-'} |",
        )

    lines += [
        "",
        "## Per-criterion scores",
        "",
        f"| {header} | criterion | kind | target | weight | mean | coverage |",
        f"|{'---|' * (len(group_by) + 6)}",
    ]
    for summary in summaries:
        for criterion in summary.criteria:
            lines.append(
                f"| {' | '.join(summary.key)} | {criterion.criterion_id} | "
                f"{criterion.kind} | {criterion.target} | {criterion.weight:g} | "
                f"{_fmt(criterion.mean)} | "
                f"{criterion.n_scored}/{criterion.n_total} |",
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


def print_report(
    console: Console,
    summaries: list[GroupSummary],
    group_by: tuple[str, ...],
) -> None:
    """Print both tables to the terminal."""
    console.print()
    console.print(summary_table(summaries, group_by))
    console.print()
    console.print(criteria_table(summaries, group_by))
    console.print()
