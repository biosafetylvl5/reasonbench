"""Request building, response parsing, and the async OpenRouter client.

``build_request``, ``parse_reasoning`` and ``parse_response`` are pure and
tested against recorded responses. ``OpenRouterClient`` holds the session, the
semaphore, and the retry policy.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from reasonbench.config import Attachment, Frozen, PromptSpec, RunConfig, Variant
from reasonbench.sweep import Sample

if TYPE_CHECKING:
    from types import TracebackType

    from reasonbench.dataset import Case

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Kept as a module constant: the test suite routes respx on it.
API_URL = f"{DEFAULT_BASE_URL}/chat/completions"
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
CLIENT_ERROR_STATUS = 400


class RetryableError(Exception):
    """A transient failure worth retrying (rate limit, 5xx, timeout)."""


class FatalAPIError(Exception):
    """A non-retryable API failure; recorded as a failed sample."""


class ReasoningAvailability(StrEnum):
    """What kind of reasoning trace, if any, came back with a response."""

    FULL_TEXT = "full_text"
    SUMMARY_ONLY = "summary_only"
    ENCRYPTED_ONLY = "encrypted_only"
    ABSENT = "absent"


class ReasoningTrace(Frozen):
    """A normalized reasoning trace extracted from one response."""

    availability: ReasoningAvailability
    text: str = ""
    summary: str = ""
    token_count: int | None = None
    block_types: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()

    @property
    def readable(self) -> str | None:
        """Return trace text a judge can grade, or ``None``."""
        if self.availability is ReasoningAvailability.FULL_TEXT:
            return self.text or None
        if self.availability is ReasoningAvailability.SUMMARY_ONLY:
            return self.summary or None
        return None


class Usage(Frozen):
    """Token counts and cost, as reported by OpenRouter."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0


class SampleResult(Frozen):
    """The outcome of one API call, successful or not."""

    sample: Sample
    ok: bool
    output: str = ""
    reasoning: ReasoningTrace = ReasoningTrace(
        availability=ReasoningAvailability.ABSENT,
    )
    usage: Usage = Usage()
    latency_s: float = 0.0
    served_model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    error: str | None = None


def _media_part(name: str, url: str, *, is_pdf: bool) -> dict[str, Any]:
    if is_pdf:
        return {"type": "file", "file": {"filename": f"{name}.pdf", "file_data": url}}
    return {"type": "image_url", "image_url": {"url": url}}


def _attachment_part(attachment: Attachment) -> dict[str, Any]:
    return _media_part(attachment.id, attachment.url, is_pdf=attachment.is_pdf)


def build_messages(
    prompt: PromptSpec, variant: Variant, case: Case | None = None
) -> list[dict[str, Any]]:
    """Return the ``messages`` array for one prompt variant and case.

    Text-only prompts use a plain string content field; anything with an
    attachment or a case image uses the content-parts form.
    """
    system_text, user_text = prompt.render(variant, case.variables if case else None)
    parts = [
        _attachment_part(a) for a in prompt.attachments if a.applies_to(variant.id)
    ]
    parts += [
        _media_part(i.column, i.url, is_pdf=i.is_pdf)
        for i in (case.images if case else ())
        if i.applies_to(variant.id)
    ]

    content: str | list[dict[str, Any]] = (
        [{"type": "text", "text": user_text}, *parts] if parts else user_text
    )

    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": content})
    return messages


def build_request(
    sample: Sample,
    prompt: PromptSpec,
    config: RunConfig,
    case: Case | None = None,
) -> dict[str, Any]:
    """Return the JSON body for one sample's chat-completions call.

    ``reasoning_effort`` of ``None`` omits the parameter entirely; the literal
    string ``"none"`` sends an explicit request to disable reasoning.
    OpenRouter normalizes ``effort`` per provider, so no per-model handling is
    needed here.
    """
    variant = next(v for v in prompt.variants if v.id == sample.variant_id)
    body: dict[str, Any] = {
        "model": sample.model,
        "messages": build_messages(prompt, variant, case),
        "temperature": sample.temperature,
        "max_tokens": config.max_tokens,
    }
    if sample.reasoning_effort is not None:
        body["reasoning"] = {"effort": sample.reasoning_effort, "exclude": False}
    return body


def parse_reasoning(message: dict[str, Any], usage: Usage) -> ReasoningTrace:
    """Extract a normalized trace from a response ``message``.

    Reads both the flat ``reasoning`` string and the structured
    ``reasoning_details`` blocks, preferring the blocks when present because
    they identify *which kind* of trace was returned.
    """
    plaintext = (message.get("reasoning") or "").strip()
    blocks = message.get("reasoning_details") or []

    texts, summaries, encrypted = [], [], 0
    block_types, formats = [], []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type", ""))
        block_types.append(kind)
        if fmt := block.get("format"):
            formats.append(str(fmt))
        if kind == "reasoning.text" and (value := block.get("text")):
            texts.append(str(value))
        elif kind == "reasoning.summary" and (value := block.get("summary")):
            summaries.append(str(value))
        elif kind == "reasoning.encrypted" and block.get("data"):
            encrypted += 1

    text = "\n".join(texts).strip() or plaintext
    summary = "\n".join(summaries).strip()

    if text:
        availability = ReasoningAvailability.FULL_TEXT
    elif summary:
        availability = ReasoningAvailability.SUMMARY_ONLY
    elif encrypted:
        availability = ReasoningAvailability.ENCRYPTED_ONLY
    else:
        availability = ReasoningAvailability.ABSENT

    return ReasoningTrace(
        availability=availability,
        text=text,
        summary=summary,
        token_count=usage.reasoning_tokens or None,
        block_types=tuple(dict.fromkeys(block_types)),
        formats=tuple(dict.fromkeys(formats)),
    )


def parse_usage(raw: dict[str, Any]) -> Usage:
    """Extract token counts and cost. Usage accounting is always on."""
    usage = raw.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=usage.get("prompt_tokens") or 0,
        completion_tokens=usage.get("completion_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
        cached_tokens=prompt_details.get("cached_tokens") or 0,
        cost=float(usage.get("cost") or 0.0),
    )


def parse_response(
    raw: dict[str, Any],
    sample: Sample,
    latency_s: float,
) -> SampleResult:
    """Turn a raw OpenRouter response into a :class:`SampleResult`."""
    choices = raw.get("choices") or []
    if not choices:
        return SampleResult(
            sample=sample,
            ok=False,
            latency_s=latency_s,
            error=str(raw.get("error") or "response contained no choices"),
        )

    choice = choices[0]
    message = choice.get("message") or {}
    usage = parse_usage(raw)
    return SampleResult(
        sample=sample,
        ok=True,
        output=(message.get("content") or "").strip(),
        reasoning=parse_reasoning(message, usage),
        usage=usage,
        latency_s=latency_s,
        served_model=raw.get("model"),
        provider=raw.get("provider"),
        finish_reason=choice.get("finish_reason"),
    )


class OpenRouterClient:
    """Async OpenRouter client with bounded concurrency and retries."""

    def __init__(
        self,
        api_key: str,
        *,
        max_concurrency: int = 8,
        max_retries: int = 4,
        timeout_s: float = 300.0,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "reasonbench",
        }
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post_once(self, body: dict[str, Any]) -> dict[str, Any]:
        """Issue a single POST, mapping transient failures to retryable errors."""
        if self._client is None:
            raise RuntimeError("OpenRouterClient must be used as a context manager")
        try:
            response = await self._client.post(self._url, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"transport error: {exc}") from exc

        if response.status_code in RETRYABLE_STATUS:
            raise RetryableError(f"HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code >= CLIENT_ERROR_STATUS:
            raise FatalAPIError(f"HTTP {response.status_code}: {response.text[:400]}")

        payload: dict[str, Any] = response.json()
        # OpenRouter can return a 200 whose body carries an error object.
        if payload.get("error") and not payload.get("choices"):
            message = str(payload["error"])
            # No machine-readable retryability flag on this path, so match the text.
            if any(s in message for s in ("rate", "429", "timeout", "overload")):
                raise RetryableError(message[:200])
            raise FatalAPIError(message[:400])
        return payload

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST a chat-completions request, retrying transient failures."""
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential_jitter(initial=1, max=30),
                retry=retry_if_exception_type(RetryableError),
                reraise=True,
            ):
                with attempt:
                    return await self._post_once(body)
        raise FatalAPIError("retry loop exited without a result")  # pragma: no cover

    async def run_sample(
        self,
        sample: Sample,
        prompt: PromptSpec,
        config: RunConfig,
        case: Case | None = None,
    ) -> tuple[SampleResult, dict[str, Any]]:
        """Execute one sample, returning its result and the raw response body.

        A failed call comes back as ``ok=False``; it does not raise.
        """
        body = build_request(sample, prompt, config, case)
        started = time.perf_counter()
        try:
            raw = await self.complete(body)
        except (RetryableError, FatalAPIError) as exc:
            elapsed = time.perf_counter() - started
            return (
                SampleResult(
                    sample=sample,
                    ok=False,
                    latency_s=elapsed,
                    error=str(exc),
                ),
                {"error": str(exc)},
            )
        elapsed = time.perf_counter() - started
        return parse_response(raw, sample, elapsed), raw
