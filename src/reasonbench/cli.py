"""Command line entry points."""

from __future__ import annotations

import asyncio
import difflib
import functools
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml

from reasonbench import __version__, progress, ui
from reasonbench import report as reporting
from reasonbench.config import (
    PromptSpec,
    RunConfig,
    Settings,
    load_prompt,
    load_run_config,
)
from reasonbench.errors import (
    AmbiguousSampleError,
    ArtifactNotFoundError,
    ExitCode,
    ManifestError,
    ReasonBenchError,
    RunDirError,
    RunFailedError,
    SampleNotFoundError,
    UsageError,
)
from reasonbench.openrouter import OpenRouterClient
from reasonbench.scoring import JudgeScorer, aggregate, score_sample_deterministic
from reasonbench.storage import RunStore, SampleRow, ScoreRow, new_run_dir
from reasonbench.sweep import Sample, estimate_cost_usd, expand

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.table import Table

app = typer.Typer(
    add_completion=True,
    rich_markup_mode=None,
    help="Evaluate LLM reasoning traces and outputs against a rubric.",
)

DATA_URL_PLACEHOLDER = "<inlined at run time>"

QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Errors only.")]
VerboseOpt = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="More detail.")
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Emit one JSON document on stdout.")
]
NoColorOpt = Annotated[bool, typer.Option("--no-color", help="Disable colour.")]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Assume yes at prompts.")]


def handle_errors(fn: Callable[..., None]) -> Callable[..., None]:
    """Turn typed errors into a message and an exit code."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except ReasonBenchError as exc:
            ui.render_error(exc)
            raise typer.Exit(int(exc.exit_code)) from None
        except KeyboardInterrupt:
            ui.warn("interrupted; results committed so far are intact")
            raise typer.Exit(int(ExitCode.INTERRUPTED)) from None

    return wrapper


def _version(value: bool) -> None:
    if value:
        sys.stdout.write(f"reasonbench {__version__}\n")
        raise typer.Exit(int(ExitCode.OK))


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Evaluate LLM reasoning traces and outputs against a rubric."""
    del version


def _manifest(config: RunConfig, prompt: PromptSpec) -> dict[str, Any]:
    """Freeze both configs, eliding inlined attachment payloads."""
    prompt_dump = prompt.model_dump(mode="json")
    for attachment in prompt_dump.get("attachments", []):
        if str(attachment.get("url", "")).startswith("data:"):
            attachment["url"] = DATA_URL_PLACEHOLDER
    return {"run_config": config.model_dump(mode="json"), "prompt": prompt_dump}


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.yaml"
    if not path.is_file():
        raise ManifestError(f"no manifest.yaml in {run_dir}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict) or "prompt" not in data or "run_config" not in data:
        raise ManifestError(f"{path}: missing 'prompt' or 'run_config'")
    return data


def _manifest_prompt(run_dir: Path) -> PromptSpec:
    return PromptSpec.model_validate(_load_manifest(run_dir)["prompt"])


def _manifest_config(run_dir: Path) -> RunConfig:
    return RunConfig.model_validate(_load_manifest(run_dir)["run_config"])


def _plan_table(samples: tuple[Sample, ...], config: RunConfig) -> Table:
    table = ui.table("Planned sweep")
    table.add_column("axis", style="cyan")
    table.add_column("values")
    table.add_column("n", justify="right")
    table.add_row(
        ui.cell("models"),
        ui.cell("\n".join(config.models)),
        ui.cell(len(config.models)),
    )
    variants = sorted({s.variant_id for s in samples})
    table.add_row(
        ui.cell("variants"), ui.cell(", ".join(variants)), ui.cell(len(variants))
    )
    table.add_row(
        ui.cell("temperature"),
        ui.cell(", ".join(str(t) for t in config.sweep.temperature)),
        ui.cell(len(config.sweep.temperature)),
    )
    table.add_row(
        ui.cell("reasoning_effort"),
        ui.cell(", ".join(str(e) for e in config.sweep.reasoning_effort)),
        ui.cell(len(config.sweep.reasoning_effort)),
    )
    table.add_row(
        ui.cell("repeats"), ui.cell(config.sweep.repeats), ui.cell(config.sweep.repeats)
    )
    table.add_section()
    table.add_row(ui.cell("total samples"), ui.cell(""), ui.cell(len(samples)))
    return table


class RunOutcome:
    """What a sweep did, so the caller can pick an exit code."""

    def __init__(self) -> None:
        self.spent = 0.0
        self.n_ok = 0
        self.n_failed = 0
        self.stopped_reason: str | None = None
        self.failures: list[tuple[str, str, str]] = []


async def _execute(
    config: RunConfig,
    prompt: PromptSpec,
    samples: tuple[Sample, ...],
    store: RunStore,
    api_key: str,
) -> RunOutcome:
    """Run every sample concurrently, committing each as it completes."""
    outcome = RunOutcome()
    outcome.spent = store.total_cost()
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
        with progress.reporter(
            "generating", len(tasks), spent=outcome.spent, show_cost=True
        ) as bar:
            try:
                for future in asyncio.as_completed(tasks):
                    result, raw = await future
                    store.add_sample(result, raw)
                    outcome.spent += result.usage.cost
                    bar.advance(ok=result.ok, cost=result.usage.cost)
                    if result.ok:
                        outcome.n_ok += 1
                    else:
                        outcome.n_failed += 1
                        cell = (
                            f"{result.sample.variant_id} T={result.sample.temperature} "
                            f"effort={result.sample.reasoning_effort} "
                            f"rep={result.sample.repeat}"
                        )
                        outcome.failures.append(
                            (
                                result.sample.sample_id,
                                f"{result.sample.model} {cell}",
                                result.error or "",
                            )
                        )
                        ui.warn(
                            "{} {}: {}",
                            result.sample.sample_id[:12],
                            result.sample.model,
                            result.error or "",
                        )
                    if outcome.spent > config.budget_usd:
                        outcome.stopped_reason = "budget"
                        break
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    return outcome


@app.command()
@handle_errors
def run(
    models_yaml: Annotated[Path, typer.Argument(help="Path to models.yaml")],
    prompt_yaml: Annotated[Path, typer.Argument(help="Path to a prompt YAML")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the matrix only."),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Continue a run directory."),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Root for run dirs.")] = Path(
        "runs"
    ),
    label: Annotated[
        str | None, typer.Option("--label", help="Suffix for the run directory name.")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Deterministic run-directory suffix.")
    ] = None,
    allow_failures: Annotated[
        int,
        typer.Option("--allow-failures", help="Tolerate N failed samples."),
    ] = 0,
    quiet: QuietOpt = False,
    verbose: VerboseOpt = 0,
    no_color: NoColorOpt = False,
    yes: YesOpt = False,
) -> None:
    """Execute the sweep and store every response."""
    ui.configure(quiet=quiet, verbose=verbose, no_color=no_color, yes=yes)
    if resume is not None and (label is not None or out != Path("runs")):
        raise UsageError(
            "--resume cannot be combined with --out or --label",
            hint="the resumed directory already fixes both.",
        )

    config = load_run_config(models_yaml)
    prompt = load_prompt(prompt_yaml)

    samples = expand(config, prompt)
    ui.data(_plan_table(samples, config))
    ui.status(
        "estimate ~${:.3f} of ${:.2f} budget ({} samples)",
        estimate_cost_usd(samples),
        config.budget_usd,
        len(samples),
    )
    if dry_run:
        ui.status("--dry-run: no API calls made")
        return

    api_key = Settings.resolve()
    run_dir = resume or new_run_dir(out, label or prompt.id, run_id=run_id)
    with RunStore(run_dir, create=resume is None) as store:
        store.write_manifest(_manifest(config, prompt))
        done = store.existing_sample_ids()
        pending = tuple(s for s in samples if s.sample_id not in done)
        if done:
            ui.status("resuming: {} samples already stored", len(done))
        outcome = RunOutcome()
        if not pending:
            ui.status("nothing to do; every sample is already stored")
        else:
            outcome = asyncio.run(_execute(config, prompt, pending, store, api_key))

        ui.status("")
        ui.status(
            "wrote {}\n  {} samples, {} ok, {} failed, ${:.4f} of ${:.2f}",
            run_dir,
            len(samples),
            outcome.n_ok,
            outcome.n_failed,
            store.total_cost(),
            config.budget_usd,
        )
        if outcome.failures:
            ui.status("")
            ui.status("{} sample(s) failed:", len(outcome.failures))
            for sid, cell, err in outcome.failures[:10]:
                ui.status("  {}  {}\n    {}", sid[:12], cell, err)
            ui.status(
                "  retry with: reasonbench run {} {} --resume {}",
                models_yaml,
                prompt_yaml,
                run_dir,
            )
        else:
            ui.status("next: reasonbench score {}", run_dir)

        if outcome.stopped_reason == "budget":
            raise ReasonBenchError(
                f"budget of ${config.budget_usd:.2f} exceeded "
                f"(${store.total_cost():.4f}); run stopped",
            ) from None
        if pending and outcome.n_ok == 0:
            raise RunFailedError("every sample failed")
        if outcome.n_failed > allow_failures:
            raise RunFailedError(
                f"{outcome.n_failed} samples failed (allowed {allow_failures})",
                hint="raise --allow-failures to tolerate this.",
            )


async def _judge_all(
    prompt: PromptSpec, config: RunConfig, store: RunStore, api_key: str
) -> tuple[list[ScoreRow], float, int]:
    rows = store.samples()
    async with OpenRouterClient(
        api_key,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        timeout_s=config.timeout_s,
    ) as client:
        scorer = JudgeScorer(client, prompt, config.judge, on_raw=store.save_judge_raw)
        tasks = [asyncio.create_task(scorer.score(r)) for r in rows if r.ok]
        collected: list[ScoreRow] = []
        with progress.reporter("judging", len(tasks), show_cost=False) as bar:
            for future in asyncio.as_completed(tasks):
                collected += await future
                bar.advance()
        return collected, scorer.cost, scorer.parse_retries


@app.command()
@handle_errors
def score(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to score")],
    prompt_yaml: Annotated[
        Path | None, typer.Option("--prompt", help="Apply a different rubric.")
    ] = None,
    only: Annotated[
        str, typer.Option("--only", help="all, deterministic, or judge.")
    ] = "all",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = 0,
    no_color: NoColorOpt = False,
    yes: YesOpt = False,
) -> None:
    """Apply the rubric to a stored run. Makes no generation calls."""
    ui.configure(quiet=quiet, verbose=verbose, no_color=no_color, yes=yes)
    if only not in {"all", "deterministic", "judge"}:
        raise UsageError(f"--only must be all, deterministic or judge, not {only!r}")
    if not run_dir.is_dir():
        raise RunDirError(f"no such run directory: {run_dir}")

    prompt = load_prompt(prompt_yaml) if prompt_yaml else _manifest_prompt(run_dir)
    config = _manifest_config(run_dir)

    # Resolve the key before anything is deleted: scoring without one used to
    # wipe the stored judge scores and only then report the missing key.
    wants_judge = only in {"all", "judge"} and bool(prompt.rubric.judge_criteria)
    api_key = Settings.resolve() if wants_judge else None

    with RunStore(run_dir) as store:
        samples = store.samples()
        if only in {"all", "deterministic"}:
            store.clear_scores("deterministic")
            deterministic = [
                row
                for sample in samples
                if sample.ok
                for row in score_sample_deterministic(prompt.rubric, sample)
            ]
            store.add_scores(deterministic)
            ui.status("scored {} deterministic criteria", len(deterministic))

        if not wants_judge:
            ui.status("judge skipped")
            return

        store.clear_scores("judge")
        judged, cost, retries = asyncio.run(
            _judge_all(prompt, config, store, api_key or "")
        )
        store.add_scores(judged)
        ui.status(
            "scored {} judge criteria with {} (${:.4f})",
            len(judged),
            config.judge.model,
            cost,
        )
        if retries:
            ui.note("{} judge replies did not parse and were retried", retries)


@app.command()
@handle_errors
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to report on")],
    group_by: Annotated[
        str, typer.Option("--group-by", help="Comma-separated fields to group by.")
    ] = "model",
    quiet: QuietOpt = False,
    verbose: VerboseOpt = 0,
    json_out: JsonOpt = False,
    no_color: NoColorOpt = False,
) -> None:
    """Print aggregated results and write report.md plus samples.csv."""
    ui.configure(quiet=quiet, verbose=verbose, json_out=json_out, no_color=no_color)
    if not run_dir.is_dir():
        raise RunDirError(f"no such run directory: {run_dir}")

    prompt = _manifest_prompt(run_dir)
    fields = tuple(f.strip() for f in group_by.split(",") if f.strip())
    if not fields:
        raise UsageError(
            "--group-by needs at least one field",
            hint="try --group-by model",
        )
    valid = set(SampleRow.model_fields)
    if unknown := set(fields) - valid:
        near = [
            m
            for f in sorted(unknown)
            for m in difflib.get_close_matches(f, sorted(valid), n=1)
        ]
        raise UsageError(
            f"unknown --group-by field(s): {', '.join(sorted(unknown))}",
            hint=(f"did you mean {near[0]}?" if near else None),
        )

    with RunStore(run_dir) as store:
        samples = store.samples()
        scores = store.scores()
        if not samples:
            raise UsageError("run contains no samples")
        summaries = aggregate(samples, scores, prompt.rubric, group_by=fields)

        if json_out:
            ui.json_document(
                {
                    "schema_version": 1,
                    "run_dir": str(run_dir),
                    "group_by": list(fields),
                    "groups": [s.model_dump(mode="json") for s in summaries],
                }
            )
            return

        reporting.print_report(summaries, fields)
        markdown = reporting.render_markdown(
            summaries,
            fields,
            title=f"ReasonBench: {run_dir.name}",
            prompt_title=prompt.title,
        )
        (run_dir / "report.md").write_text(markdown, encoding="utf-8")
        reporting.export_csv(samples, run_dir / "samples.csv")
        ui.status("wrote {} and {}", run_dir / "report.md", run_dir / "samples.csv")


def _find_sample(store: RunStore, prefix: str) -> SampleRow:
    rows = store.samples()
    exact = [r for r in rows if r.sample_id == prefix]
    if exact:
        return exact[0]
    matches = [r for r in rows if r.sample_id.startswith(prefix)]
    if not matches:
        raise SampleNotFoundError(f"no sample starting with {prefix!r}")
    if len(matches) > 1:
        raise AmbiguousSampleError(
            f"sample prefix {prefix!r} matches {len(matches)} samples",
            details=tuple(
                f"{r.sample_id}  {r.model}  {r.variant_id}  T={r.temperature}  "
                f"effort={r.reasoning_effort}  rep={r.repeat}"
                for r in matches[:8]
            ),
            hint="use more characters of the id.",
        )
    return matches[0]


@app.command()
@handle_errors
def show(
    run_dir: Annotated[Path, typer.Argument(help="Run directory")],
    sample_id: Annotated[str, typer.Argument(help="Sample id, or a unique prefix")],
    chars: Annotated[
        int, typer.Option("--chars", help="Truncate trace and answer.")
    ] = 4000,
    quiet: QuietOpt = False,
    no_color: NoColorOpt = False,
) -> None:
    """Print one sample: metadata, error, trace, answer, and scores."""
    ui.configure(quiet=quiet, no_color=no_color)
    if not run_dir.is_dir():
        raise RunDirError(f"no such run directory: {run_dir}")

    with RunStore(run_dir) as store:
        row = _find_sample(store, sample_id)

        head = ui.table(row.sample_id)
        head.add_column("field", style="cyan")
        head.add_column("value")
        head.add_row(ui.cell("model"), ui.cell(row.model))
        if row.served_model and row.served_model != row.model:
            head.add_row(ui.cell("served"), ui.cell(row.served_model))
        if row.provider:
            head.add_row(ui.cell("provider"), ui.cell(row.provider))
        head.add_row(
            ui.cell("cell"),
            ui.cell(
                f"{row.variant_id}  T={row.temperature}  "
                f"effort={row.reasoning_effort}  repeat={row.repeat}"
            ),
        )
        head.add_row(ui.cell("status"), ui.cell("ok" if row.ok else "FAILED"))
        if row.finish_reason:
            head.add_row(ui.cell("finish"), ui.cell(row.finish_reason))
        head.add_row(ui.cell("trace"), ui.cell(row.reasoning_availability))
        head.add_row(
            ui.cell("tokens"),
            ui.cell(
                f"{row.prompt_tokens} in, {row.completion_tokens} out, "
                f"{row.reasoning_tokens} reasoning"
            ),
        )
        head.add_row(
            ui.cell("cost"), ui.cell(f"${row.cost:.6f}   {row.latency_s:.1f}s")
        )
        ui.data(head)

        if not row.ok:
            ui.data("")
            ui.data("error")
            ui.blob(row.error or "(no error recorded)")
            return

        if trace := row.readable_reasoning:
            ui.data("")
            ui.data(f"reasoning ({row.reasoning_availability}, {len(trace)} chars)")
            ui.blob(trace[:chars])
        ui.data("")
        ui.data("answer")
        ui.blob(row.output or "(empty)")

        rows = [s for s in store.scores() if s.sample_id == row.sample_id]
        if not rows:
            ui.data("")
            ui.status("no scores stored; run: reasonbench score {}", run_dir)
            return
        table = ui.table("scores")
        for col in ("criterion", "kind", "raw", "norm", "rep", "reason"):
            table.add_column(col)
        for s in rows:
            raw_value = "n/a" if s.score is None else f"{s.score:g}"
            norm = "n/a" if s.normalized is None else f"{s.normalized:.2f}"
            table.add_row(
                ui.cell(s.criterion_id),
                ui.cell(s.kind),
                ui.cell(raw_value),
                ui.cell(norm),
                ui.cell(f"r{s.judge_repeat}"),
                ui.cell(s.reason or ""),
            )
        ui.data("")
        ui.data(table)


@app.command()
@handle_errors
def raw(
    run_dir: Annotated[Path, typer.Argument(help="Run directory")],
    sample_id: Annotated[str, typer.Argument(help="Sample id, or a unique prefix")],
    judge: Annotated[
        bool, typer.Option("--judge", help="Show the judge artifact instead.")
    ] = False,
    list_only: Annotated[
        bool, typer.Option("--list", help="List artifacts for this sample.")
    ] = False,
    pretty: Annotated[
        bool, typer.Option("--pretty", help="Re-indent instead of byte-exact.")
    ] = False,
    quiet: QuietOpt = False,
    no_color: NoColorOpt = False,
) -> None:
    """Print a stored artifact exactly as it was written."""
    ui.configure(quiet=quiet, no_color=no_color)
    if not run_dir.is_dir():
        raise RunDirError(f"no such run directory: {run_dir}")

    with RunStore(run_dir) as store:
        row = _find_sample(store, sample_id)

    gen = run_dir / "raw" / f"{row.sample_id}.json"
    judged = sorted((run_dir / "judge").glob(f"{row.sample_id}_r*_a*.json"))
    if list_only:
        table = ui.table(f"artifacts for {row.sample_id}")
        table.add_column("kind")
        table.add_column("path")
        table.add_column("bytes", justify="right")
        if gen.is_file():
            table.add_row(
                ui.cell("generation"), ui.cell(gen), ui.cell(gen.stat().st_size)
            )
        for j in judged:
            table.add_row(ui.cell("judge"), ui.cell(j), ui.cell(j.stat().st_size))
        ui.data(table)
        return

    path = judged[0] if judge and judged else gen
    if judge and not judged:
        raise ArtifactNotFoundError(f"no judge artifact for {row.sample_id}")
    if not path.is_file():
        raise ArtifactNotFoundError(f"no artifact at {path}")
    if pretty:
        ui.blob(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    else:
        sys.stdout.buffer.write(path.read_bytes())


@app.command()
@handle_errors
def ls(
    root: Annotated[Path, typer.Argument(help="Run root")] = Path("runs"),
    limit: Annotated[int, typer.Option("--limit", help="How many to list.")] = 25,
    quiet: QuietOpt = False,
    no_color: NoColorOpt = False,
) -> None:
    """List run directories, newest first."""
    ui.configure(quiet=quiet, no_color=no_color)
    if not root.is_dir():
        raise RunDirError(f"no such run root: {root}")
    dirs = sorted(
        (d for d in root.iterdir() if (d / "results.sqlite").is_file()),
        key=lambda d: d.name,
        reverse=True,
    )[:limit]
    if not dirs:
        ui.status("no runs under {}", root)
        return
    table = ui.table(f"runs under {root}")
    for col in ("run", "samples", "failed", "scored", "cost $"):
        table.add_column(col)
    for d in dirs:
        with RunStore(d) as store:
            rows = store.samples()
            table.add_row(
                ui.cell(d.name),
                ui.cell(len(rows)),
                ui.cell(sum(1 for r in rows if not r.ok)),
                ui.cell("yes" if store.scores() else "no"),
                ui.cell(f"{store.total_cost():.4f}"),
            )
    ui.data(table)
