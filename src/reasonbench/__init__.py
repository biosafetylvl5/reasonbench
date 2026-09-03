"""ReasonBench: evaluate LLM reasoning traces and outputs against a rubric."""

from reasonbench.config import (
    ConfigError,
    Guidance,
    JudgeSettings,
    PromptSpec,
    Rubric,
    RunConfig,
    Settings,
    Target,
    load_prompt,
    load_run_config,
)
from reasonbench.openrouter import (
    OpenRouterClient,
    ReasoningAvailability,
    ReasoningTrace,
    SampleResult,
    Usage,
    build_request,
    parse_reasoning,
    parse_response,
)
from reasonbench.scoring import JudgeScorer, aggregate, score_sample_deterministic
from reasonbench.storage import RunStore, SampleRow, ScoreRow, new_run_dir
from reasonbench.sweep import Sample, expand, make_sample_id

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "Guidance",
    "JudgeScorer",
    "JudgeSettings",
    "OpenRouterClient",
    "PromptSpec",
    "ReasoningAvailability",
    "ReasoningTrace",
    "Rubric",
    "RunConfig",
    "RunStore",
    "Sample",
    "SampleResult",
    "SampleRow",
    "ScoreRow",
    "Settings",
    "Target",
    "Usage",
    "aggregate",
    "build_request",
    "expand",
    "load_prompt",
    "load_run_config",
    "make_sample_id",
    "new_run_dir",
    "parse_reasoning",
    "parse_response",
    "score_sample_deterministic",
]
