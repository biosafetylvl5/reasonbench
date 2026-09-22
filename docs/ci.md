# Running in CI

## GitHub Actions

```yaml
jobs:
  prompts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: biosafetylvl5/reasonbench@v1
        with:
          prompt: .reasonbench/prompts/support-triage.yaml
          models: .reasonbench/models.yaml
          gate: .reasonbench/gate.yaml
          api-key: ${{ secrets.OPENROUTER_API_KEY }}
          max-cases: '20'
```

The action installs its own checkout, so the job image needs no `git` and the
consumer's dependencies are untouched. Pin the version by pinning the `uses:`
ref.

Outputs: `passed`, `exit-code`, `weighted-mean`, `total-cost-usd`,
`report-json`, `junit-xml`.

### Forks

A pull request from a fork receives no secrets. The action declines to run
live there rather than failing with an auth error. Run the live gate on pushes
to the default branch and on `workflow_dispatch`; `pull_request_target` is not
a workaround, because it combines fork-authored configuration with full secret
access.

### Matrix jobs

`github.run_id` and `github.sha` are properties of the run, so every matrix leg
sees the same values. The action derives a per-leg suffix from
`strategy.job-index` plus a hash of its inputs, and uses it for both the run
directory and the artifact name. `upload-artifact@v4` rejects duplicate names,
and `overwrite: true` is not the fix: it makes duplicates succeed by clobbering
the other leg's report.

## GitLab CI

```yaml
include:
  - remote: 'https://raw.githubusercontent.com/biosafetylvl5/reasonbench/v1/ci/gitlab/reasonbench.yml'

prompt-eval:
  extends: .reasonbench
  variables:
    REASONBENCH_PROMPT: .reasonbench/prompts/support-triage.yaml
    REASONBENCH_GATE: .reasonbench/gate.yaml
```

`OPENROUTER_API_KEY` must be a masked variable. If it is also **protected** it
is unavailable on merge requests from unprotected branches, which is the most
common way this appears to work on the default branch and silently skip on
merge requests. The shipped rules make that skip explicit rather than letting
the job fail with an auth error.

The image needs `git` for a VCS install. Pin a full 40-character commit SHA:
pip only reuses its wheel cache for immutable URLs, so a branch or tag rebuilds
on every pipeline.

`.reasonbench-validate` needs no key and no network, so it is safe on every
merge request including forks.

## Other endpoints

Any OpenAI-style server works. Set `base_url` in `models.yaml`, or
`REASONBENCH_BASE_URL` in the environment:

```yaml
base_url: http://localhost:11434/v1   # Ollama
```

Traces are read from `reasoning_details`, `reasoning_content`, `reasoning`,
`thinking`, and Anthropic-style thinking content parts, so a self-hosted vLLM
or Ollama reports a readable trace rather than reporting none.

A server that does not return `usage.cost` leaves `budget_usd` unenforced.
Bound those runs by `max_samples` instead.
