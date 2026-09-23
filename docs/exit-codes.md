# Exit codes

| Code | Name | When |
|---|---|---|
| 0 | OK | Ran to completion; gates passed, or none configured |
| 1 | INTERNAL | An unexpected exception. Re-run with `-vv` for the traceback |
| 2 | USAGE | Bad invocation or configuration, an unknown `--group-by` field, or a sweep over `max_samples` |
| 3 | NOT_FOUND | The run directory, sample, or artifact does not exist |
| 10 | AUTH | No API key could be resolved, or the endpoint rejected it |
| 11 | API | No sample succeeded, or more failed than `--allow-failures` permits |
| 12 | BUDGET | `budget_usd` was exceeded and the run stopped |
| 13 | TIMEOUT | The wall-clock deadline expired |
| 20 | GATE | Run completed; at least one gate assertion failed |
| 130 | INTERRUPTED | SIGINT |

## Precedence

Codes 10 to 13 take precedence over 20. A budget-stopped run exits 12 even
when its partial scores would have passed.

## Artifacts

`report.json` and `junit.xml` are written whichever way the process exits.
`report.json` carries the exit code at the top level.

## Budget

Spending is checked after each sample commits, so up to `max_concurrency`
requests may already be in flight when the cap trips. Set `budget_usd` to
roughly 0.7 of the true ceiling.

A server that does not report `usage.cost` leaves the cap unenforced. Most
OpenAI-compatible endpoints other than OpenRouter do not report it; bound
those runs with `max_samples`.
