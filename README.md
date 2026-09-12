# ReasonBench

Runs a prompt across OpenRouter models and parameter settings, stores the
reasoning trace along with the final answer, and grades both against a rubric.

Traces come back in four shapes. Some models return full `reasoning.text`, some
only a `reasoning.summary`, some an encrypted blob, and some nothing at all.
ReasonBench records which of the four it got. A criterion aimed at the trace
scores n/a when there is no readable trace, not 0. Scoring it 0 ranks models on
whether they expose their thinking.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

The key goes in `OPENROUTER_API_KEY`, in `.env`, or in a file called
`openrouter.key`.

## Use

```bash
reasonbench run configs/models.yaml configs/prompts/periodic-table.yaml --dry-run
reasonbench run configs/models.yaml configs/prompts/periodic-table.yaml
reasonbench score runs/<dir>
reasonbench report runs/<dir> --group-by model,variant_id
```

`score` is a separate command from `run`. Rubrics change more often than
generations do, and re-scoring a stored run costs judge tokens only.

```bash
reasonbench show runs/<dir> <sample-id-prefix>   # trace, answer, scores
reasonbench raw  runs/<dir> <sample-id-prefix>   # the stored API response
```

## Configuration

`configs/models.yaml` holds the run: model slugs, sweep axes, concurrency, a
hard `budget_usd`, and the judge model.

`configs/prompts/<name>.yaml` holds the prompt, its variants, optional image or
PDF attachments, and the rubric.

Criteria are `deterministic` (a pure check) or `judge` (an integer scale scored
by the judge model), and each targets `output`, `reasoning`, or `both`. Judge
criteria take a `guidance` block: a `summary`, per-score `levels`,
`positive_indicators`, `negative_indicators`, and `notes`. A bare string is
shorthand for `summary`.

## Bundled prompts

Three multiple-choice items whose distractors map onto distinct reasoning
failures, so a model can reach the right answer for the wrong reason.

| File | Question | Answer |
|---|---|---|
| `periodic-table.yaml` | Why lithium and sodium behave alike | B |
| `laplace-transform.yaml` | Purpose of the Laplace transform | B |
| `grey-body.yaml` | Wavelength-independent emissivity | 1 (Grey) |

`grey-body.yaml` uses numbered options, which also covers answer extraction on
digits.

## Output

```
runs/2026-08-07T14-22-01_periodic-group-similarity/
  manifest.yaml            frozen copy of both configs
  results.sqlite           samples and scores tables
  raw/<id>.json            generation responses
  judge/<id>_r0_a0.json    judge responses, for auditing grades
  report.md
  samples.csv
```

Scores live in their own table, so `score` can be re-run against a rewritten
rubric without touching the generations.

A failed API call is stored as a failed sample instead of aborting the sweep,
and `--resume` retries it rather than skipping it.

A judge reply that will not parse is retried (`judge.parse_retries`) before
being recorded as n/a.

## Development

```bash
pytest -m "not integration"
ruff check src tests
mypy src
```

## License

Apache-2.0. See `LICENSE`.
