# ReasonBench

Runs a prompt across OpenRouter models and parameter settings, stores the
reasoning trace along with the final answer, and grades both against a rubric.

Some models return full `reasoning.text`, some only a `reasoning.summary`,
some an encrypted blob, and some nothing at all. ReasonBench records which. A
criterion aimed at the trace scores n/a when there is no readable trace, not 0.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

From another repository, install a tag:

```bash
pip install "reasonbench @ git+https://github.com/biosafetylvl5/reasonbench.git@v1"
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

Re-scoring a stored run costs judge tokens only.

```bash
reasonbench ls                                   # run directories, newest first
reasonbench show runs/<dir> <sample-id-prefix>   # trace, answer, scores
reasonbench raw  runs/<dir> <sample-id-prefix>   # the stored API response
reasonbench raw  runs/<dir> <prefix> --judge     # the judge response behind a grade
```

## Datasets

A prompt can point at a JSONL or CSV of cases. Each row's columns become
template variables.

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

Any value interpolated into a pattern needs `re_escape`, or the data rewrites
the pattern. Image columns, CSV handling and case ids are in
[docs/datasets.md](docs/datasets.md).

## Gating a run

`gate.yaml` holds thresholds.

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

Groups, coverage and the n/a rule are in [docs/gate.md](docs/gate.md). `eval`
exits 20 when a gate assertion fails; the rest are in
[docs/exit-codes.md](docs/exit-codes.md).

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

GitLab, fork pull requests, matrix jobs and non-OpenRouter endpoints are in
[docs/ci.md](docs/ci.md).

## Configuration

- `configs/models.yaml`: model slugs, sweep axes, concurrency, a hard
  `budget_usd`, and the judge model.
- `configs/prompts/<name>.yaml`: the prompt, its variants, optional image or
  PDF attachments, and the rubric.

Criteria are `deterministic` (a pure check) or `judge` (an integer scale scored
by the judge model), and each targets `output`, `reasoning`, or `both`. Judge
criteria take a `guidance` block: a `summary`, per-score `levels`,
`positive_indicators`, `negative_indicators`, and `notes`. A bare string is
shorthand for `summary`.

## Bundled prompts

| File | Question | Answer |
|---|---|---|
| `periodic-table.yaml` | Why lithium and sodium behave alike | B |
| `laplace-transform.yaml` | Purpose of the Laplace transform | B |
| `grey-body.yaml` | Wavelength-independent emissivity | 1 (Grey) |

Each distractor corresponds to a different reasoning failure.
`grey-body.yaml` uses numbered options.

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

A failed API call is stored as a failed sample instead of aborting the sweep.
`--resume` retries it.

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
