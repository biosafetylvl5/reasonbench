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

The action installs its own checkout, so the job image needs no `git`. Pin the
`uses:` ref.

Outputs: `passed`, `exit-code`, `weighted-mean`, `total-cost-usd`,
`report-json`, `junit-xml`.

### Forks

A fork pull request gets no secrets, and the action skips the live run there.
Run the live gate on pushes to the default branch and on `workflow_dispatch`.
Do not use `pull_request_target`: it runs fork-authored config with secret
access.

### Matrix jobs

`github.run_id` and `github.sha` are identical across legs, so the action
suffixes the run directory and artifact name with `strategy.job-index` and a
hash of the leg's inputs.

Do not set `overwrite: true` on `upload-artifact@v4`. It clobbers the other
leg's report.

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

`OPENROUTER_API_KEY` must be masked. A protected variable is unavailable on
merge requests from unprotected branches; the shipped rules skip the job there.

The image needs `git` for a VCS install. Pin a full 40-character commit SHA;
pip rebuilds the wheel every pipeline for a branch or tag.

`.reasonbench-validate` needs no key and no network. Safe on fork merge
requests.

## Other endpoints

Any OpenAI-style server works. Set `base_url` in `models.yaml`, or
`REASONBENCH_BASE_URL` in the environment:

```yaml
base_url: http://localhost:11434/v1   # Ollama
```

Traces are read from `reasoning_details`, `reasoning_content`, `reasoning`,
`thinking`, and Anthropic-style thinking content parts. Every shape is tried,
whatever the endpoint is configured as.

`reasoning_effort` is sent as `reasoning: {effort}` on OpenRouter,
`reasoning_effort` on OpenAI, and `chat_template_kwargs.enable_thinking`
elsewhere. Set `dialect` to override the guess. A level the endpoint cannot
express is an error; `on_unsupported_effort` can `downgrade` or `omit` it
instead. `--dry-run` reports levels that end up sending the same request.

The judge asks for a strict JSON schema, then `json_object`, then plain JSON,
dropping a rung when the server rejects the format. Set
`judge.structured_output` to start lower.

`budget_usd` needs a price when the server omits `usage.cost`; see
[exit-codes.md](exit-codes.md).

The key is checked once before the sweep. `--no-preflight` skips it.
