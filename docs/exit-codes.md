# Exit codes

| Code | Name | When |
|---|---|---|
| 0 | OK | Ran to completion; every gate assertion held, or no gate was configured |
| 1 | INTERNAL | An unexpected exception. Re-run with `-vv` for the traceback |
| 2 | USAGE | Bad invocation or configuration, an unknown `--group-by` field, or a sweep over `max_samples` |
| 3 | NOT_FOUND | The run directory, sample, or artifact does not exist |
| 10 | AUTH | No API key could be resolved, or the endpoint rejected it |
| 11 | API | No sample succeeded, or more failed than `--allow-failures` permits |
| 12 | BUDGET | `budget_usd` was exceeded and the run stopped |
| 13 | TIMEOUT | The wall-clock deadline expired |
| 20 | GATE | The run completed, but at least one gate assertion was breached |
| 130 | INTERRUPTED | SIGINT |

Code 2 matches what Click already returns for a bad invocation, so scripts do
not have to distinguish the two.

## Infrastructure outranks policy

Codes 10 to 13 take precedence over 20. A gate verdict computed on a truncated
run is not meaningful, so a budget-stopped run exits 12 even when its partial
scores would have passed.

`report.json` and `junit.xml` are written whichever way the process exits, and
`report.json` carries the exit code at the top level, so a downstream job can
read the verdict without parsing prose.

## Budget is a stop threshold

Spending is checked after each sample commits, so up to `max_concurrency`
requests may already be in flight when the cap trips. Set `budget_usd` to
roughly 0.7 of the true ceiling if the difference matters.

A server that does not report `usage.cost` leaves the cap unenforced. That is
the case for most OpenAI-compatible endpoints other than OpenRouter; bound
those runs by sample count with `max_samples` instead.
