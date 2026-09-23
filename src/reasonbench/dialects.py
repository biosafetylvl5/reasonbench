"""How each family of server wants reasoning asked for.

Only the request varies. Responses are parsed by trying every known shape,
because a gateway can proxy anything and a configured dialect is a claim about
what we send, not a promise about what comes back.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from reasonbench.errors import ConfigError

if TYPE_CHECKING:
    from reasonbench.config import RunConfig


class Dialect(StrEnum):
    """Which parameter carries the reasoning request."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GENERIC = "generic"


# OpenRouter normalises these per provider. Plain OpenAI accepts a narrower
# set, and nothing maps xhigh or max onto it.
OPENAI_EFFORTS = frozenset({"minimal", "low", "medium", "high"})

# Where a downgrade is allowed, this is the nearest supported level.
DOWNGRADE = {"xhigh": "high", "max": "high"}


def infer(config: RunConfig) -> Dialect:
    """Guess the dialect from the endpoint when it is not stated."""
    if config.dialect is not None:
        return Dialect(config.dialect)
    base = config.base_url
    if base is None or "openrouter.ai" in base:
        return Dialect.OPENROUTER
    if "api.openai.com" in base:
        return Dialect.OPENAI
    return Dialect.GENERIC


def effort_body(  # noqa: PLR0911
    dialect: Dialect,
    effort: str | None,
    *,
    on_unsupported: Literal["error", "downgrade", "omit"] = "error",
    thinking_param: str = "auto",
) -> tuple[dict[str, Any], str | None]:
    """Return ``(body fragment, effort actually applied)``.

    An effort the target cannot express is an error by default. Dropping it
    silently would leave the sweep axis in the report while every cell on it
    sent the same request.
    """
    if effort is None:
        return {}, None

    if dialect is Dialect.OPENROUTER:
        return {"reasoning": {"effort": effort, "exclude": False}}, effort

    if dialect is Dialect.OPENAI:
        if effort in OPENAI_EFFORTS:
            return {"reasoning_effort": effort}, effort
        return _unsupported(dialect, effort, on_unsupported, "reasoning_effort")

    if thinking_param == "none":
        return _unsupported(dialect, effort, on_unsupported, "nothing")
    if thinking_param == "reasoning_effort":
        return {"reasoning_effort": effort}, effort
    enabled = effort != "none"
    return {"chat_template_kwargs": {"enable_thinking": enabled}}, effort


def _unsupported(
    dialect: Dialect,
    effort: str,
    policy: str,
    parameter: str,
) -> tuple[dict[str, Any], str | None]:
    if policy == "omit":
        return {}, None
    if policy == "downgrade" and effort in DOWNGRADE:
        applied = DOWNGRADE[effort]
        return {parameter: applied}, applied
    raise ConfigError(
        f"the {dialect} dialect has no equivalent for reasoning_effort {effort!r}",
        hint=(
            "drop it from sweep.reasoning_effort, or set "
            "on_unsupported_effort to downgrade or omit."
        ),
    )


def check_efforts(config: RunConfig) -> list[str]:
    """Validate the sweep's effort axis against the endpoint, before any call.

    Raises when a level cannot be expressed. Returns a description of every
    pair of levels that end up sending the same request, because comparing
    those cells afterwards would be comparing a level against itself.
    """
    dialect = infer(config)
    sent: dict[str, list[str]] = {}
    for effort in config.sweep.reasoning_effort:
        body, _ = effort_body(
            dialect,
            effort,
            on_unsupported=config.on_unsupported_effort,
            thinking_param=config.thinking_param,
        )
        key = json.dumps(body, sort_keys=True)
        sent.setdefault(key, []).append("none" if effort is None else effort)
    return [
        f"{', '.join(levels)} send the same request"
        for levels in sent.values()
        if len(levels) > 1
    ]
