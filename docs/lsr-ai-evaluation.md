# Grading LSR-AI verification logs

LSR-AI checks severe-weather storm reports against radar. A supervisor agent
passes each report to a convective specialist agent, which calls MRMS radar
tools and writes an analysis. The supervisor then answers `Event Verification:
Yes/No` with notes. Both agents ran Claude 3.7 Sonnet on Bedrock.

We have two LSR-AI logs and no way to run more. This document describes how
those two logs were graded with ReasonBench, how to repeat it, and what came
out. With two logs nothing can be said about LSR-AI's accuracy. What can be
tested is the grading itself: whether a judge model, given this rubric, finds
flaws we know are there. That is done by planting flaws in copies of the two
logs and checking that the judge's scores drop.

## Files

| Path | What it is |
|---|---|
| `Examples of LSR-AI Output logs.docx` | The two DynamoDB items, pasted into Word |
| `scripts/lsr_ai_to_json.py` | DynamoDB item (docx, JSON, or a scan export) to plain JSON |
| `data/lsr-ai/<id>.json` | The two logs as plain JSON (`dynamodb/` keeps the typed originals) |
| `data/lsr-ai/mutations.yaml` | The planted flaws, and which criteria each should lower |
| `scripts/lsr_convert.py` | Writes the logs and their mutated copies as a ReasonBench dataset |
| `data/lsr-ai/cases.jsonl` | That dataset: 18 rows, each with its recorded answer and trace |
| `configs/prompts/lsr-verification.yaml` | The rubric, and the dataset's recorded columns |
| `configs/lsr-jev.yaml` | Judge settings: Jev 1.14 on the Decisions API, three repeats |
| `scripts/lsr_judge_check.py` | Compares a judge's scores with the planted flaws |
| `scripts/lsr_golden_to_jsonl.py` | Sean's Good/Bad codes from the golden set, as JSONL (not used here; see the end) |

## How it works

ReasonBench normally generates outputs itself through OpenRouter. It cannot
run LSR-AI, which is a multi-agent system with tools. Instead,
`lsr_convert.py` records LSR-AI's responses in the dataset (see
[recorded responses](datasets.md#recorded-responses)), and `reasonbench run`
stores them as a run without calling any model:

- **One sample per log and mutation.** The case id, which is also the sample
  id, is `<log>--<mutation>`, for example `0227test3--report_time`. The 2 logs and 16 mutations give 18
  samples. `variant_id` is the mutation (`original` for the unmodified log), so
  `report --group-by variant_id` splits results by planted flaw.
- **Question:** the report exactly as LSR-AI received it, for example
  `HAIL at None, latitude: 38.927, longitude: -97.157. Date is 20220429.  UTC
  time is 2348`.
- **Output:** the supervisor's final `Event Verification` message.
- **Reasoning trace:** every agent message in order, with tool calls written
  as `→ get_mesh(date=…, latitude=…, …)` and results as `← get_mesh: …`.
- **Radar grids are summarised.** Each tool returns up to 546 grid points; the
  full Example 2 transcript is about 55,000 characters. Each grid becomes one
  line: the value at the nearest point, every value within 1.5 km (the point
  and its eight neighbours on the 0.01° grid), the range within 5 and 10 km,
  the grid maximum with its distance and compass direction, and how many
  points are non-zero. The longest trace the judge sees is 3,349 characters.
  The rubric tells the judge these summaries are what the agent saw.
- **Case variables** hold the report, the correct verdict (`expected`), and
  the mutation's description and target criteria. Only `report_message` and
  `expected` reach the judge's prompt or the checks; the mutation details are
  for `lsr_judge_check.py`. The recorded `model` column names the system
  that produced the response (LSR-AI's Bedrock model).

### The rubric

Three deterministic checks, scored without a model:

| Criterion | Weight | Passes when |
|---|---|---|
| `verdict_format` | 0.5 | The first line is `Event Verification: Yes` or `No` |
| `verdict_correct` | 3.0 | That verdict equals `expected` |
| `hail_core_checked` | 0.5 | The trace contains a `get_mesh` call |

Seven judge criteria, each on a 0–4 scale with anchored levels:

| Criterion | Weight | Asks |
|---|---|---|
| `grounded_numbers` | 2 | Does every cited value and product match a tool result? |
| `internally_consistent` | 1 | Is the analysis free of contradictions? |
| `evidence_supports_verdict` | 2 | Does the verdict follow from the evidence, with the link stated? |
| `missing_data_handling` | 1 | Are empty tool results treated as missing, not as evidence? |
| `calibrated_confidence` | 1 | Does stated confidence match the evidence? |
| `faithful_summary` | 1 | Do the supervisor's verdict and notes match the specialist? |
| `addresses_report` | 2 | Were the report's time, place and (if given) magnitude checked? |

The correct verdicts come from Storm Events, not from LSR-AI:

- **Example 1** (`18cf24b6…`, Liberty MO, 13 Feb 2026 1700 UTC): there is no
  Storm Events report near that place and date, and the radar is all zero. It
  looks like a constructed negative case. Expected **No**.
- **Example 2** (`0227test3`, central Kansas, 29 Apr 2022 2348 UTC): trained
  spotters reported 4.0, 1.75 and 2.75 inch hail 3–5 km away between 2345 and
  0000. Expected **Yes**. Whoever built this test should confirm which report
  it came from, since no single report matches its time and coordinates.

### The mutations

Each mutation in `data/lsr-ai/mutations.yaml` is a list of literal or regex
edits to the report, the specialist's messages, or the supervisor's answer. An
edit that matches nothing stops the conversion, so a typo cannot silently produce
an unmodified copy.

| Mutation | What changes | Should lower |
|---|---|---|
| `fixed` | Removes the original's flaw (Ex 1: empty tables called ambiguous, in both specialist and supervisor; Ex 2: the dBZ contradiction) | Should *raise* `missing_data_handling` / `internally_consistent` |
| `wrong_number` | A quoted value no longer matches the grid (Ex 1: flag 1.0 not 0.0; Ex 2: MESH 27.4–31.1 mm not 7.4–11.1) | `grounded_numbers` |
| `phantom_tool` | Cites a product never fetched (Ex 1: VIL; Ex 2: composite reflectivity > 65 dBZ) | `grounded_numbers` |
| `flipped_verdict` | Supervisor verdict flipped, notes unchanged | `faithful_summary`, `evidence_supports_verdict`, `internally_consistent` |
| `bare_conclusion` | All reasoning replaced by one sentence; High confidence kept | `evidence_supports_verdict`, `calibrated_confidence` |
| `contradiction` (Ex 1) | Conclusion says hail likely fell nearby, against "complete absence" of echoes | `internally_consistent` |
| `report_magnitude` (Ex 2) | Report now says 4.00 inch hail; grid MESH peaks at 39.4 mm | `addresses_report` |
| `report_time` | Report time moved (Ex 1: 1700→1830; Ex 2: 90 minutes later); the agent still analysed the original time | `addresses_report` |
| `report_location` | Report moved 30 km north; the agent still analysed the original point | `addresses_report` |

The three `report_*` mutations follow the reasons Sean most often gave for
marking a report Bad in the golden set: timing (71 of 140), location (42) and
magnitude (28).

## Running it

### 1. Set up

```bash
uv venv && uv pip install -e ".[dev]"
```

### 2. Convert the logs (already done; repeat for new logs)

```bash
.venv/bin/python scripts/lsr_ai_to_json.py "Examples of LSR-AI Output logs.docx" data/lsr-ai
.venv/bin/python scripts/lsr_convert.py data/lsr-ai
# data/lsr-ai/cases.jsonl  (18 cases, longest trace 3349 chars)
```

`lsr_ai_to_json.py` also accepts a DynamoDB JSON export, one item, a list, or
`{"Items": [...]}`. A new log needs an entry in `mutations.yaml` with at least
`expected:` and an empty `mutations: []`; the converter only reads logs listed
there.

### 3. Smoke test with a free local judge

Done first to check the pipeline end to end before spending anything.

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &      # default 4096 is too tight
ollama pull qwen2.5:0.5b-instruct              # ~400 MB

.venv/bin/reasonbench run configs/lsr-jev.yaml \
    configs/prompts/lsr-verification.yaml --label lsr-ai-qwen0.5b
# wrote runs/<dir>  (stores the 18 recorded responses; calls nothing)

OPENROUTER_API_KEY=ollama .venv/bin/reasonbench score runs/<dir> \
    --judge-model qwen2.5:0.5b-instruct \
    --judge-base-url http://localhost:11434/v1 --judge-api chat
.venv/bin/python scripts/lsr_judge_check.py runs/<dir>
```

`OPENROUTER_API_KEY=ollama` satisfies the key check; Ollama ignores it. Judge
prompts were about 2,300 tokens, so nothing was truncated at 8,192.

### 4. Score with an OpenRouter judge

```bash
export OPENROUTER_API_KEY=...                  # or .env, or openrouter.key

.venv/bin/reasonbench run configs/lsr-jev.yaml \
    configs/prompts/lsr-verification.yaml --label lsr-ai-jev
.venv/bin/reasonbench score runs/<dir>
```

`configs/lsr-jev.yaml` sets the judge to `typesafe/jev-1.14` with
`api: decisions` and `repeats: 3`. Grading every sample three times is what
makes the repeat spread measurable. Jev is a decisions model, so it is asked
one Score question per criterion on OpenRouter's Decisions API, with the
rubric's five levels as the ordered choices. Its score is a
probability-weighted level, so it can fall between two integers.
`reasonbench raw runs/<dir> <sample> --judge` shows its confidence and level
probabilities.

A judge is fixed per run directory. For a chat judge, run again (or copy the
directory) and score with, for example, `--judge-model
deepseek/deepseek-v4.1-flash-1017 --judge-api chat`. `reasonbench eval
configs/lsr-jev.yaml configs/prompts/lsr-verification.yaml` does `run`,
`score` and `report` in one step.

### 5. Read the results

```bash
.venv/bin/python scripts/lsr_judge_check.py runs/<dir> [runs/<other-judge> ...]
.venv/bin/reasonbench report runs/<dir> --group-by variant_id
.venv/bin/reasonbench show runs/<dir> 0227test3--original     # trace, answer, scores
.venv/bin/reasonbench raw runs/<dir> 0227test3--original --judge   # judge reply
```

`lsr_judge_check.py` prints, for each mutation and target criterion, the mean
judge score on the original and on the copy (0–4), and whether the flaw was
caught:

- **ok**: the score dropped (for `fixed`, did not drop).
- **MISS**: it did not.
- **flr**: the original already scored under half a point, so no drop was
  possible. Excluded from the caught count.

It then gives one line per judge: flaws caught, *collateral drops* (a drop of
a full point or more on a criterion the mutation did not target), the mean
standard deviation across judge repeats, and missing scores. With several run
directories the judges appear side by side.

## Results

All runs are under `runs/` (not tracked by git).

| Run | Judge | Caught | Collateral | Repeat spread | Cost |
|---|---|---|---|---|---|
| `2026-10-19T18-46-25Z-2c82_lsr-ai-qwen0.5b` | qwen2.5:0.5b-instruct, 1 repeat | 5 / 13 (7 at floor) | 8 | n/a | free |
| `2026-10-19T18-50-48Z-d76d_…` | deepseek-v4.1-flash-1017, 3 repeats | 18 / 20 | 9 | 0.10 | $0.027 |
| `2026-10-19T19-05-47Z-6d41_…` | same | 21 / 22 | 7 | 0.12 | $0.025 |
| **`2026-10-19T19-20-59Z-7d24_…`** | **same** | **20 / 22** | **5** | **0.10** | **$0.044** |
| **`2026-10-28T17-52-56Z-fe66_lsr-ai-jev`** | **typesafe/jev-1.14, 3 repeats** | **21 / 22** | **10** | **0.03** | **$0.0063** |

The last two runs are the ones to use. They grade identical samples with an
identical rubric, so they differ only in the judge. The changes between the
DeepSeek runs:

1. **qwen2.5 0.5B** proved the pipeline (18/18 samples judged, every reply
   parsed) and nothing else. It gave near-uniform scores with generic
   justifications.
2. **First DeepSeek run.** Most collateral drops were the judge being right
   about my mutations. Flipping the verdict does create a contradiction with
   the notes, so `internally_consistent` was added to its targets. Example 1's
   `fixed` copy changed the specialist's wording but not the supervisor's, so
   the notes now misstated the specialist; the supervisor edit was added.
3. **Second DeepSeek run.** The judge marked Example 2 down because it could
   not find VIL 13.5 in the summary. The value is real, but the summary then
   gave only ranges. Summaries now list every value within 1.5 km.
4. **Third DeepSeek run.** The judge now confirms the quoted values exist and
   objects only to the narrowed ranges, which is a fair criticism.

The two DeepSeek runs before the last are superseded: they used the earlier
mutations or summaries. The Jev run is compared with DeepSeek below.

### The two real logs (final run, three judge repeats)

| Criterion | Example 1 (No) | Example 2 (Yes) |
|---|---|---|
| `verdict_correct` / `verdict_format` / `hail_core_checked` | pass / pass / pass | pass / pass / pass |
| `grounded_numbers` | 4, 4, 4 | 2, 2, 3 |
| `internally_consistent` | 4, 4, 4 | 2, 2, 2 |
| `evidence_supports_verdict` | 4, 2, 3 | 3, 3, 3 |
| `missing_data_handling` | 0, 0, 0 | 4, 4, 4 |
| `calibrated_confidence` | 2, 2, 2 | 3, 3, 3 |
| `faithful_summary` | 4, 4, 4 | 4, 4, 4 |
| `addresses_report` | 4, 4, 4 | 4, 4, 4 |
| Weighted rubric score | 0.85 | 0.85 |

These agree with a manual reading of the logs:

- **Example 1** reads three empty tables (MESH, −20 °C reflectivity,
  composite) as "confirming that there were no storm echoes." An empty table
  can also be a failed fetch. The all-zero precipitation flag does support
  "No", which is why the verdict criteria stay high while
  `missing_data_handling` is 0 and confidence is marked down.
- **Example 2** says reflectivity of "47–50 dBZ" is "well above the 50 dBZ
  threshold." Its values are real grid points (flag 7, MESH 7.4 mm, VIL 15.2,
  47.5 dBZ at the nearest point), but the quoted ranges are narrower than the
  surrounding values (VIL "13.5 to 15.2" where the 1.5 km values run 9.5 to
  20.9).

Jev, on the same two logs, gives these scores (mean of three repeats, Ex 1 /
Ex 2): `grounded_numbers` 3.9 / 2.75, `internally_consistent` 3.8 / 1.9,
`evidence_supports_verdict` 3.95 / 3.75, `missing_data_handling` 0.09 / 3.99,
`calibrated_confidence` 3.3 / 3.4, `faithful_summary` 3.9 / 3.5,
`addresses_report` 3.6 / 3.8. Its weighted rubric scores are 0.89 / 0.88. It
finds the same weaknesses as DeepSeek (Ex 1's empty-table handling, Ex 2's
dBZ contradiction and narrowed ranges). It is more generous on
`calibrated_confidence` (3.3 against DeepSeek's 2 for Ex 1) and on
`evidence_supports_verdict`.

### Planted flaws (final DeepSeek run)

| Mutation | Criterion | Ex 1 orig → copy | Ex 2 orig → copy |
|---|---|---|---|
| fixed | missing_data_handling / internally_consistent (should rise) | 0.00 → 3.33 ok | 2.00 → 4.00 ok |
| wrong_number | grounded_numbers | 4.00 → 1.67 ok | 2.33 → 1.00 ok |
| phantom_tool | grounded_numbers | 4.00 → 1.00 ok | 2.33 → 1.00 ok |
| flipped_verdict | evidence_supports_verdict | 3.00 → 0.00 ok | 3.00 → 0.00 ok |
| flipped_verdict | faithful_summary | 4.00 → 0.00 ok | 4.00 → 0.00 ok |
| flipped_verdict | internally_consistent | 4.00 → 0.00 ok | 2.00 → 0.00 ok |
| bare_conclusion | evidence_supports_verdict | 3.00 → 2.00 ok | 3.00 → 1.67 ok |
| bare_conclusion | calibrated_confidence | 2.00 → 2.00 **MISS** | 3.00 → 4.00 **MISS** |
| contradiction | internally_consistent | 4.00 → 1.67 ok | — |
| report_magnitude | addresses_report | — | 4.00 → 2.00 ok |
| report_time | addresses_report | 4.00 → 1.00 ok | 4.00 → 1.00 ok |
| report_location | addresses_report | 4.00 → 1.00 ok | 4.00 → 1.00 ok |

`verdict_correct` failed on both flipped copies and passed everywhere else.

**The two misses.** In both `bare_conclusion` copies the judge rated
confidence against the radar data still visible in the trace rather than
against the reasoning the agent wrote, so it did not lower the score. That is
a reasonable reading of the criterion as worded ("the evidence presented"), so
the fix is in the rubric: say the evidence the agent *states*. The same
mutation was caught on `evidence_supports_verdict` in both examples.

**The five collateral drops** are mostly `calibrated_confidence` falling when
another flaw goes unaddressed (for example, High confidence after analysing
the wrong time). That is defensible behaviour, not noise.

### Planted flaws: Jev 1.14

Jev caught 21 of 22. It caught everything DeepSeek did, and also Ex 1's
`bare_conclusion` on `calibrated_confidence` (3.30 → 2.98: a third of a point,
a small drop). It missed Ex 2's `bare_conclusion` on `calibrated_confidence`
(3.38 → 3.66), for the reason DeepSeek gave. Its drops are as clean as
DeepSeek's: every `flipped_verdict` target falls to under 0.1, every
`report_location`/`report_time` to about 1, and `report_magnitude` to 2.23.

Its repeats barely move (0.03 points of spread against DeepSeek's 0.10). That
is because each score is an expected level rather than a sampled one.

Jev has ten collateral drops to DeepSeek's five. Most follow from the
mutation. `wrong_number` makes the answer disagree with the trace, which also
lowers `internally_consistent` and `faithful_summary`. `phantom_tool` lowers
`faithful_summary`, and Ex 1's `fixed` copy loses `calibrated_confidence`.
Two shared with DeepSeek, on Ex 1's `fixed` copy (`internally_consistent`,
`evidence_supports_verdict`), suggest that mutation's edit reads awkwardly.

The cost was $0.0063 for 54 judgements, a seventh of DeepSeek's. A
decisions model writes no justification, though, so a score can only be
explained through its level probabilities.

## What this does and does not show

- It shows this judge, with this rubric, reliably notices the planted kinds of
  flaw, and scores consistently across repeats (0.1 points of spread on a 0–4
  scale).
- It does not show the judge catches natural errors, which are usually
  subtler than planted ones. The flaws were written by the same person who
  wrote the rubric.
- It says nothing about LSR-AI's accuracy: two logs, both with correct
  verdicts.
- The reference for trusting the judge is human grading. Sean (or another
  forecaster) should grade the two original traces on this rubric, and ideally
  the copies, so the judge's scores can be compared with theirs.

## Findings about LSR-AI itself

- **The agent is never given the reported magnitude.** Both opening messages
  carry event, place, date and time only, although the DynamoDB item stores
  the magnitude (Example 2: 4.0 inches). LSR-AI cannot detect a report whose
  size is wrong, which was Sean's reason for 28 of the 140 Bad reports.
- **Every radar product is fetched once, at the report time.** A storm that
  passed 20 minutes later is not visible to it. Timing was the most common
  reason Sean marked a report Bad (71 of 140). This comes from reading two
  traces and should be confirmed with the LSR-AI team.
- Both behaviours are what `addresses_report` measures, and the `report_*`
  mutations show the judge detects them.

## Next steps

1. Reword `calibrated_confidence` to judge against the evidence the agent
   states, and rerun.
2. ~~Add a second judge from another provider.~~ Done with Jev 1.14 (above).
   A third judge would show whether Jev's extra collateral drops or
   DeepSeek's misses are the outlier.
3. Have Sean grade the two original traces on the rubric.
4. If more LSR-AI logs become available, convert them, add each to
   `mutations.yaml` with its Storm Events verdict, and rerun `lsr_convert.py`. The golden set
   (`scripts/lsr_golden_to_jsonl.py`, 1,044 coded reports) supplies the
   correct verdicts and the reviewer's reason for each Bad one.
