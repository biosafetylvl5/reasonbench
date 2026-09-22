# Gating a run

Thresholds live in their own file. They are policy, they differ between a
merge request and the default branch, and a stored run should be re-checkable
at a different bar without generating anything again.

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

## Unmeasurable is not zero

A criterion aimed at the reasoning trace cannot be scored when the model
returned no trace. Those samples score `n/a`, and an assertion with nothing to
measure is **skipped**, not failed.

Failing it would rank a model down for not exposing its reasoning rather than
for reasoning badly, which is the confusion this tool exists to prevent.

Assert gradeability separately, with `min_coverage` or `min_scored`. Coverage
is always measurable, so those assertions fail normally.

`on_unmeasurable: fail` is available, but it red-builds every honest run
against a model that does not disclose its thinking.

## Which criteria catch a judge outage

A criterion targeting `both` has full coverage whenever the output is
non-empty, so `min_coverage: 1.0` on one of those fires when the judge is
unreachable and never fires merely because a trace was missing.

`min_coverage` on a deterministic criterion can essentially never fail, so it
asserts nothing.

## Output

`report.json` lists every assertion with its observed value, threshold and
status. `junit.xml` carries two suites: one test per case, so a platform can
show which cases regressed, and one test per assertion, so it can show why
the job failed.
