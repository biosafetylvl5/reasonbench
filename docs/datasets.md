# Datasets

A prompt can name a file of cases instead of hard-coding one question. Each
row becomes one case: its columns are template variables, so the answer key
lives in the data and one rubric covers every row.

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

`--max-cases N` overrides `limit` for one invocation, which is how a pull
request job runs a cheap subset of a large dataset.

## File formats

JSONL is one JSON object per line. CSV uses the header row for column names,
and a cell that starts with `[` is parsed as a JSON array; nothing else is
reinterpreted.

## Answer keys

Interpolate the expected value into a check, and escape it:

```yaml
check:
  type: regex
  pattern: '(?i)^(?:\W*answer\W*)?\W*{{ expected | re_escape }}(?!\w)'
```

`re_escape` is not optional. Without it the data rewrites the pattern:

| `expected` | Without escaping |
|---|---|
| `2.5` | matches the output `225`, because `.` is any character |
| `a\|b` | splits the pattern at the top level, discarding the anchor |
| `C++` | `+` is a quantifier; matches a bare `C` |
| `[Fe` | raises, after the generations are paid for |

Use `(?!\w)` rather than a trailing `\b`. After escaping, an answer ending in
punctuation such as `C++` or `50%` has no word boundary after it, so `\b`
would stop it matching itself.

A column named in the template but missing from a row is an error naming the
criterion and the case, not a silently permissive pattern.

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
Local files are read and inlined as data URLs when the dataset loads, so
request building stays free of file I/O. An image column is an attachment, not
a template variable, so it does not appear in the rendered prompt.

Only the reference, size and digest are stored in the run database; payloads
are not.

## What ends up in the run

`case_id` becomes a sample field, so `--group-by case_id` works, and JUnit
names each test case after it. Sample ids include the case and a fingerprint
of the rendered prompt, so editing prompt text means `--resume` regenerates
rather than reusing a stale answer.
