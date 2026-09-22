"""Trace extraction across the shapes OpenAI-style servers actually return.

A configured dialect is a claim about the request, not a promise about the
reply, so parsing tries every known shape rather than switching on config.
"""

from __future__ import annotations

import pytest
from conftest import load_fixture, make_sample

from reasonbench.openrouter import (
    ReasoningAvailability,
    extract_output_text,
    parse_reasoning,
    parse_response,
    parse_usage,
)

BILLED = parse_usage(
    {
        "usage": {
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 300},
        }
    }
)
UNBILLED = parse_usage({})


def test_reasoning_content_is_a_readable_trace():
    """vLLM, SGLang, Ollama, DeepSeek and friends use this field."""
    trace = parse_reasoning(
        {"content": "B", "reasoning_content": "Li and Na are both group 1."}, BILLED
    )
    assert trace.availability is ReasoningAvailability.FULL_TEXT
    assert trace.source == "reasoning_content"
    assert "group 1" in (trace.readable or "")


def test_thinking_content_parts_are_a_readable_trace():
    trace = parse_reasoning(
        {
            "content": [
                {"type": "thinking", "thinking": "step one"},
                {"type": "text", "text": "B"},
            ]
        },
        UNBILLED,
    )
    assert trace.availability is ReasoningAvailability.FULL_TEXT
    assert "content.thinking" in trace.block_types


def test_redacted_thinking_is_encrypted_not_absent():
    trace = parse_reasoning(
        {"content": [{"type": "redacted_thinking"}, {"type": "text", "text": "B"}]},
        UNBILLED,
    )
    assert trace.availability is ReasoningAvailability.ENCRYPTED_ONLY
    assert trace.readable is None


def test_a_content_list_still_yields_the_answer():
    """Without this the answer is empty and every deterministic check fails."""
    assert (
        extract_output_text(
            {
                "content": [
                    {"type": "thinking", "thinking": "x"},
                    {"type": "text", "text": "B"},
                ]
            }
        )
        == "B"
    )


def test_a_content_list_round_trips_through_parse_response():
    raw = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "B"},
                    ],
                },
            }
        ]
    }
    result = parse_response(raw, make_sample(), 1.0)
    assert result.ok is True
    assert result.output == "B"
    assert result.reasoning.readable == "hmm"


def test_an_echoed_trace_is_not_doubled():
    """A gateway may copy one trace into two fields; concatenating inflates it."""
    trace = parse_reasoning(
        {"content": "B", "reasoning_content": "the trace", "reasoning": "the trace"},
        UNBILLED,
    )
    assert trace.text == "the trace"


def test_reasoning_content_wins_over_a_gateway_copy():
    trace = parse_reasoning(
        {"reasoning_content": "native", "reasoning": "truncated copy"}, UNBILLED
    )
    assert trace.text == "native"
    assert trace.source == "reasoning_content"


def test_billed_but_absent_separates_the_two_silent_cases():
    billed = parse_reasoning({"content": "Yes"}, BILLED)
    never = parse_reasoning({"content": "B"}, UNBILLED)
    assert billed.availability is never.availability is ReasoningAvailability.ABSENT
    assert billed.billed_but_absent is True
    assert never.billed_but_absent is False


@pytest.mark.parametrize(
    ("fixture", "availability", "source"),
    [
        ("gemma_full_text", ReasoningAvailability.FULL_TEXT, "reasoning_details"),
        ("summary_only", ReasoningAvailability.SUMMARY_ONLY, "reasoning_details"),
        ("encrypted_only", ReasoningAvailability.ENCRYPTED_ONLY, "reasoning_details"),
        ("llama_no_reasoning", ReasoningAvailability.ABSENT, "none"),
        ("openai_reasoning", ReasoningAvailability.ABSENT, "none"),
    ],
)
def test_openrouter_parsing_is_unchanged(fixture, availability, source):
    """The only guard against widening the parser altering OpenRouter results."""
    trace = parse_response(load_fixture(fixture), make_sample(), 1.0).reasoning
    assert trace.availability is availability
    assert trace.source == source
