# Datasets

A prompt can name a file of cases. Each row is one case; its columns are
template variables.

```yaml
dataset:
  path: ../cases/mcq.jsonl     # relative to this YAML
  format: auto                 # auto, jsonl, or csv
  id_column: case_id           # absent rows get row-0001, row-0002, ...
  required_columns: [question, expected]
  limit: null                  # cap rows, after select and sample
  select: []                   # explicit case ids, in this order
  sample: {n: 25, seed: 7}     # deterministic subsample
```

`--max-cases N` overrides `limit` for one invocation.

## File formats

- JSONL: one JSON object per line.
- CSV: the header row gives the column names. A cell that starts with `[` is
  parsed as a JSON array; nothing else is reinterpreted.

## Answer keys

Interpolate the expected value and escape it:

```yaml
check:
  type: regex
  pattern: '(?i)^(?:\W*answer\W*)?\W*{{ expected | re_escape }}(?!\w)'
```

Without `re_escape` the data rewrites the pattern:

| `expected` | Without escaping |
|---|---|
| `2.5` | matches the output `225`, because `.` is any character |
| `a\|b` | splits the pattern at the top level, discarding the anchor |
| `C++` | `+` is a quantifier; matches a bare `C` |
| `[Fe` | raises at check time, after generation |

Use `(?!\w)` rather than a trailing `\b`: `C++` and `50%` have no word
boundary after them.

A column named in the template but missing from a row is an error naming the
criterion and the case.

## Images

```yaml
dataset:
  path: ../cases/charts.jsonl
  image_root: ../assets        # defaults to the dataset file's directory
  max_image_bytes: 8000000
  images:
    - column: chart
      media_type: image/png    # inferred from the suffix when omitted
      required: false
      attach_to: all           # or a list of variant ids
```

A cell may be a path, an `http(s)` URL, a data URL, or a JSON array of those.
Local files are inlined as data URLs at load time. An image column is an
attachment and does not appear in the rendered prompt.

Only the reference, size and digest are stored in the run database.

## What ends up in the run

`case_id` becomes a sample field, so `--group-by case_id` works, and JUnit
names each test case after it. Sample ids include the case and a fingerprint of
the rendered prompt, so editing prompt text makes `--resume` regenerate.
