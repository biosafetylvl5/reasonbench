"""Command line entry points."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from reasonbench import report as reporting
from reasonbench.config import (
    ConfigError,
    PromptSpec,
    RunConfig,
    Settings,
    load_prompt,
    load_run_config,
)
from reasonbench.openrouter import OpenRouterClient
from reasonbench.scoring import JudgeScorer, aggregate, score_sample_deterministic
from reasonbench.storage import RunStore, SampleRow, ScoreRow, new_run_dir
from reasonbench.sweep import Sample, estimate_cost_usd, expand

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Evaluate LLM reasoning traces and outputs against a rubric.",
)

# Rich falls back to 80 cols off-tty, which wraps the tables. Slugs are long.
console = Console(width=None if sys.stdout.isatty() else 150)

DATA_URL_PLACEHOLDER = "<inlined at run time>"


def _fail(message: str) -> None:
    console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code=1)


def _manifest(config: RunConfig, prompt: PromptSpec) -> dict[str, Any]:
    """Freeze both configs, eliding inlined attachment payloads."""
    prompt_dump = prompt.model_dump(mode="json")
    for attachment in prompt_dump.get("attachments", []):
        if str(attachment.get("url", "")).startswith("data:"):
            attachment["url"] = DATA_URL_PLACEHOLDER
    return {
        "run_config": config.model_dump(mode="json"),
        "prompt": prompt_dump,
    }


def _load_manifest_prompt(run_dir: Path) -> PromptSpec:
    path = run_dir / "manifest.yaml"
    if not path.is_file():
        _fail(f"no manifest.yaml in {run_dir}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptSpec.model_validate(data["prompt"])


def _load_manifest_config(run_dir: Path) -> RunConfig:
    data = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    return RunConfig.model_validate(data["run_config"])


def _plan_table(samples: tuple[Sample, ...], config: RunConfig) -> Table:
    table = Table(title="Planned sweep", header_style="bold")
    table.add_column("axis", style="cyan")
    table.add_column("values")
    table.add_column("n", justify="right")
    table.add_row("models", "\n".join(config.models), str(len(config.models)))
    variants = sorted({s.variant_id for s in samples})
    table.add_row("variants", ", ".join(variants), str(len(variants)))
    table.add_row(
        "temperature",
        ", ".join(str(t) for t in config.sweep.temperature),
        str(len(config.sweep.temperature)),
    )
    table.add_row(
        "reasoning_effort",
        ", ".join(str(e) for e in config.sweep.reasoning_effort),
        str(len(config.sweep.reasoning_effort)),
    )
    table.add_row("repeats", str(config.sweep.repeats), str(config.sweep.repeats))
    table.add_section()
    table.add_row("[bold]total samples[/bold]", "", f"[bold]{len(samples)}[/bold]")
    return table


async def _execute(
    config: RunConfig,
    prompt: PromptSpec,
    samples: tuple[Sample, ...],
    store: RunStore,
    api_key: str,
) -> None:
    """Run every sample concurrently, committing each as it completes."""
    spent = store.total_cost()
    async with OpenRouterClient(
        api_key,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_s=config.timeout_s,
    ) as client:
        tasks = [
            asyncio.create_task(client.run_sample(sample, prompt, config))
            for sample in samples
        ]
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("· ${task.fields[cost]:.4f}"),
            console=console,
        ) as progress:
            bar = progress.add_task("generating", total=len(tasks), cost=spent)
            try:
                for future in asyncio.as_completed(tasks):
                    result, raw = await future
                    store.add_sample(result, raw)
                    spent += result.usage.cost
                    progress.update(bar, advance=1, cost=spent)
                    if not result.ok:
                        console.print(
                            f"[yellow]sample failed[/yellow] "
                            f"{result.sample.model}: {result.error}",
                        )
                    if spent > config.budget_usd:
                        console.print(
                            f"[bold red]budget of ${config.budget_usd:.2f} "
                            f"exceeded (${spent:.4f}); stopping[/bold red]",
                        )
                        break
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


@app.command()
def run(
    models_yaml: Annotated[Path, typer.Argument(help="Path to models.yaml")],
    prompt_yaml: Annotated[Path, typer.Argument(help="Path to a prompt YAML")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate configs and print the matrix only."),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Continue an interrupted run directory."),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Root for run dirs.")] = Path(
        "runs",
    ),
    label: Annotated[
        str | None,
        typer.Option("--label", help="Suffix appended to the run directory name."),
    ] = None,
) -> None:
    """Execute the sweep and store every response."""
    try:
        config = load_run_config(models_yaml)
        prompt = load_prompt(prompt_yaml)
    except ConfigError as exc:
        _fail(str(exc))
        return

    samples = expand(config, prompt)
    console.print(_plan_table(samples, config))
    console.print(
        f"rough cost estimate: [bold]~${estimate_cost_usd(samples):.3f}[/bold] "
        f"(budget ${config.budget_usd:.2f})",
    )

    if dry_run:
        console.print("[dim]--dry-run: no API calls made[/dim]")
        return

    try:
        api_key = Settings.resolve()
    except ConfigError as exc:
        _fail(str(exc))
        return

    run_dir = resume or new_run_dir(out, label or prompt.id)
    with RunStore(run_dir) as store:
        store.write_manifest(_manifest(config, prompt))
        done = store.existing_sample_ids()
        pending = tuple(s for s in samples if s.sample_id not in done)
        if done:
            console.print(f"[dim]resuming: {len(done)} samples already stored[/dim]")
        if not pending:
            console.print("nothing to do; every sample is already stored")
        else:
            asyncio.run(_execute(config, prompt, pending, store, api_key))
        console.print(
            f"\nwrote [bold]{run_dir}[/bold] "
            f"(spent ${store.total_cost():.4f} on generation)",
        )


async def _judge_all(
    prompt: PromptSpec,
    config: RunConfig,
    store: RunStore,
    api_key: str,
) -> tuple[list[ScoreRow], float, int]:
    rows = store.samples()
    async with OpenRouterClient(
        api_key,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_s=config.timeout_s,
    ) as client:
        scorer = JudgeScorer(client, prompt, config.judge, on_raw=store.save_judge_raw)
        tasks = [asyncio.create_task(scorer.score(row)) for row in rows if row.ok]
        collected: list[ScoreRow] = []
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("judging", total=len(tasks))
            for future in asyncio.as_completed(tasks):
                collected += await future
                progress.update(bar, advance=1)
        return collected, scorer.cost, scorer.parse_retries


@app.command()
def score(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to score")],
    prompt_yaml: Annotated[
        Path | None,
        typer.Option("--prompt", help="Apply a different rubric than the run used."),
    ] = None,
    skip_judge: Annotated[
        bool,
        typer.Option("--skip-judge", help="Deterministic criteria only; no API calls."),
    ] = False,
) -> None:
    """Apply the rubric to a stored run. Makes no generation calls."""
    if not run_dir.is_dir():
        _fail(f"no such run directory: {run_dir}")

    try:
        prompt = (
            load_prompt(prompt_yaml)
            if prompt_yaml
            else _load_manifest_prompt(
                run_dir,
            )
        )
        config = _load_manifest_config(run_dir)
    except ConfigError as exc:
        _fail(str(exc))
        return

    with RunStore(run_dir) as store:
        store.clear_scores()
        samples = store.samples()
        deterministic = [
            row
            for sample in samples
            if sample.ok
            for row in score_sample_deterministic(prompt.rubric, sample)
        ]
        store.add_scores(deterministic)
        console.print(f"scored {len(deterministic)} deterministic criteria")

        if skip_judge or not prompt.rubric.judge_criteria:
            console.print("[dim]judge skipped[/dim]")
            return

        try:
            api_key = Settings.resolve()
        except ConfigError as exc:
            _fail(str(exc))
            return

        judged, cost, retries = asyncio.run(_judge_all(prompt, config, store, api_key))
        store.add_scores(judged)
        console.print(
            f"scored {len(judged)} judge criteria with "
            f"[bold]{config.judge.model}[/bold] (${cost:.4f})",
        )
        if retries:
            console.print(
                f"[dim]{retries} judge repl{'y' if retries == 1 else 'ies'} did not "
                f"parse and were retried[/dim]",
            )


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to report on")],
    group_by: Annotated[
        str,
        typer.Option("--group-by", help="Comma-separated sample fields to group by."),
    ] = "model",
) -> None:
    """Print aggregated results and write report.md plus samples.csv."""
    if not run_dir.is_dir():
        _fail(f"no such run directory: {run_dir}")

    prompt = _load_manifest_prompt(run_dir)
    fields = tuple(f.strip() for f in group_by.split(",") if f.strip())
    valid = set(SampleRow.model_fields)
    if unknown := set(fields) - valid:
        _fail(f"unknown group-by fields: {sorted(unknown)}")

    with RunStore(run_dir) as store:
        samples = store.samples()
        scores = store.scores()
        if not samples:
            _fail("run contains no samples")
        summaries = aggregate(samples, scores, prompt.rubric, group_by=fields)
        reporting.print_report(console, summaries, fields)

        markdown = reporting.render_markdown(
            summaries,
            fields,
            title=f"ReasonBench: {run_dir.name}",
            prompt_title=prompt.title,
        )
        (run_dir / "report.md").write_text(markdown, encoding="utf-8")
        reporting.export_csv(samples, run_dir / "samples.csv")
        console.print(
            f"wrote [bold]{run_dir / 'report.md'}[/bold] and "
            f"[bold]{run_dir / 'samples.csv'}[/bold]",
        )


@app.command()
def show(
    run_dir: Annotated[Path, typer.Argument(help="Run directory")],
    sample_id: Annotated[str, typer.Argument(help="Sample id (or a unique prefix)")],
) -> None:
    """Print one sample's reasoning trace, answer, and scores."""
    with RunStore(run_dir) as store:
        matches = [s for s in store.samples() if s.sample_id.startswith(sample_id)]
        if not matches:
            _fail(f"no sample starting with {sample_id!r}")
        row = matches[0]
        console.rule(f"{row.model} · {row.variant_id} · T={row.temperature}")
        console.print(f"[dim]trace availability:[/dim] {row.reasoning_availability}")
        if trace := row.readable_reasoning:
            console.rule("reasoning", style="dim")
            console.print(trace)
        console.rule("answer", style="dim")
        console.print(row.output)
        console.rule("scores", style="dim")
        for score_row in store.scores():
            if score_row.sample_id != row.sample_id:
                continue
            value = "n/a" if score_row.normalized is None else f"{score_row.score:g}"
            console.print(
                f"  {score_row.criterion_id:24s} {value:>6s}  {score_row.reason}"
            )


@app.command()
def raw(
    run_dir: Annotated[Path, typer.Argument(help="Run directory")],
    sample_id: Annotated[str, typer.Argument(help="Sample id (or a unique prefix)")],
) -> None:
    """Print the verbatim OpenRouter response stored for one sample."""
    matches = sorted((run_dir / "raw").glob(f"{sample_id}*.json"))
    if not matches:
        _fail(f"no raw artifact starting with {sample_id!r}")
    console.print_json(json.dumps(json.loads(matches[0].read_text(encoding="utf-8"))))


if __name__ == "__main__":  # pragma: no cover
    app()
