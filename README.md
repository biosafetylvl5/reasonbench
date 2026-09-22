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

To use it from another repository, install a tag:

```bash
pip install "reasonbench @ git+https://github.com/biosafetylvl5/reasonbench.git@v1"
```

The key goes in `OPENROUTER_API_KEY`, in `.env`, or in a file called
`openrouter.key`. Any OpenAI-style endpoint works: set `base_url` in
`models.yaml`, or `REASONBENCH_BASE_URL` in the environment.

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
reasonbench ls                                   # run directories, newest first
reasonbench show runs/<dir> <sample-id-prefix>   # trace, answer, scores
reasonbench raw  runs/<dir> <sample-id-prefix>   # the stored API response
reasonbench raw  runs/<dir> <prefix> --judge     # the judge response behind a grade
```

## Datasets

A prompt can point at a JSONL or CSV of cases instead of hard-coding one
question. Each row's columns become template variables, so the answer key
lives in the data and one rubric covers every row.

```yaml
dataset:
  path: ../cases/mcq.jsonl
  required_columns: [question, options, expected]

variants:
  - id: plain
    user: |
      {{ question }}

      {{ options }}

      Respond with a single letter.

rubric:
  criteria:
    - id: correct_answer
      kind: deterministic
      target: output
      scope: last_line
      weight: 3.0
      check:
        type: regex
        pattern: '(?i)^(?:\W*answer\W*)?\W*{{ expected | re_escape }}(?!\w)'
```

Use `re_escape` on any value interpolated into a pattern. Without it an answer
key of `2.5` matches the output `225`, and one of `a|b` splits the pattern at
the top level and matches almost anything.

A row can carry images:

```yaml
dataset:
  path: ../cases/charts.jsonl
  image_root: ../assets
  images:
    - column: chart
      media_type: image/png
```

Cells may be a path, an `http(s)` URL, a data URL, or a JSON array of those.

## Gating a run

`gate.yaml` holds thresholds. They live apart from `models.yaml` so a stored
run can be re-checked at a different bar without generating anything again.

```yaml
overall:
  min_weighted_mean: 0.70
  max_failure_rate: 0.10
criteria:
  - id: correct_answer
    min_mean: 0.80
```

```bash
reasonbench eval configs/models.yaml configs/prompts/mcq-dataset.yaml \
  --gate configs/gate.yaml
reasonbench gate runs/<dir> --gate configs/gate.yaml   # re-check, no API calls
```

A criterion with nothing to measure is skipped, not failed. Assert
gradeability with `min_coverage`, which is always measurable.

| Exit | Meaning |
|---|---|
| 0 | Everything passed |
| 2 | Bad invocation or configuration |
| 3 | Run directory, sample, or artifact not found |
| 10 | No API key, or the key was rejected |
| 11 | No sample succeeded |
| 12 | Budget exceeded |
| 20 | The run completed but a gate assertion failed |

`eval` writes `report.json` and `junit.xml` whichever way it exits, so CI shows
what happened rather than an empty artifact.

## In CI

GitHub Actions:

```yaml
- uses: biosafetylvl5/reasonbench@v1
  with:
    prompt: .reasonbench/prompts/support-triage.yaml
    gate: .reasonbench/gate.yaml
    api-key: ${{ secrets.OPENROUTER_API_KEY }}
    max-cases: '20'
```

GitLab:

```yaml
include:
  - remote: 'https://raw.githubusercontent.com/biosafetylvl5/reasonbench/v1/ci/gitlab/reasonbench.yml'

prompt-eval:
  extends: .reasonbench
  variables:
    REASONBENCH_PROMPT: .reasonbench/prompts/support-triage.yaml
    REASONBENCH_GATE: .reasonbench/gate.yaml
```

A fork pull request receives no secrets, so the Action declines to run live
there rather than failing with an auth error.

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

## Reference

- [docs/datasets.md](docs/datasets.md) covers case files, answer keys and images
- [docs/gate.md](docs/gate.md) covers thresholds and the n/a rule
- [docs/ci.md](docs/ci.md) covers GitHub Actions, GitLab, and other endpoints
- [docs/exit-codes.md](docs/exit-codes.md) covers what each exit code means

## Development

```bash
pytest -m "not integration"
ruff check src tests
mypy src
```

## License

Apache-2.0. See `LICENSE`.
