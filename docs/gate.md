# Gating a run

`gate.yaml`:

```yaml
version: 1
on_unmeasurable: warn          # warn, fail, or pass

overall:
  min_samples: 1               # an empty run must never pass
  min_weighted_mean: 0.70
  max_failure_rate: 0.10
  max_cost_usd: 1.00
  min_coverage: null           # worst coverage across all criteria

criteria:
  - id: correct_answer
    min_mean: 0.80
    max_mean: null             # for criteria where lower is better
    min_coverage: 1.0
    min_scored: 10

groups:
  - by: [model]
    min_weighted_mean: 0.50
    max_failure_rate: 0.20
    require_all_groups: true
```

```bash
reasonbench eval models.yaml prompt.yaml --gate gate.yaml
reasonbench gate runs/<dir> --gate gate.yaml     # re-check, no API calls
reasonbench gate runs/<dir> --fail-under 0.8     # a one-line gate
```

`--gate` and `--fail-under` are mutually exclusive.

## Unmeasurable samples

A criterion targeting the trace scores `n/a` when the model returned none.
`on_unmeasurable` sets the status of an assertion with nothing to measure:
`warn` marks it `skipped` (the default), `fail` marks it `failed`, `pass`
marks it `passed`.

Assert gradeability separately, with `min_coverage` or `min_scored`. Coverage
is always measurable, so those assertions fail normally.

## Judge outages

Put `min_coverage: 1.0` on a criterion with `target: both`. It falls back to
the output when there is no trace, so it fires on a judge outage rather than a
missing trace.

Deterministic criteria are scored without the judge. `min_coverage` on one
never detects an outage.

## Output

`report.json` lists every assertion with its observed value, threshold and
status. `junit.xml` carries two suites: one test per case and one test per
assertion. The gate suite is written only when a gate is configured.
