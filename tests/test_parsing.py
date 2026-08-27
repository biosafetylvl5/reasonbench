"""Reasoning-trace extraction."""

from __future__ import annotations

import pytest
from conftest import load_fixture, make_sample

from reasonbench.config import Attachment
from reasonbench.openrouter import (
    ReasoningAvailability,
    build_messages,
    build_request,
    parse_reasoning,
    parse_response,
    parse_usage,
)


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("gemma_full_text", ReasoningAvailability.FULL_TEXT),
        ("summary_only", ReasoningAvailability.SUMMARY_ONLY),
        ("encrypted_only", ReasoningAvailability.ENCRYPTED_ONLY),
        ("llama_no_reasoning", ReasoningAvailability.ABSENT),
        ("openai_reasoning", ReasoningAvailability.ABSENT),
    ],
)
def test_availability_is_classified(fixture, expected):
    raw = load_fixture(fixture)
    result = parse_response(raw, make_sample(), latency_s=1.0)
    assert result.reasoning.availability is expected


def test_full_text_trace_is_readable():
    raw = load_fixture("gemma_full_text")
    trace = parse_response(raw, make_sample(), 1.0).reasoning
    assert trace.readable
    assert "reasoning.text" in trace.block_types
    assert trace.token_count is not None
    assert trace.token_count > 0


def test_summary_only_falls_back_to_summary_text():
    raw = load_fixture("summary_only")
    trace = parse_response(raw, make_sample(), 1.0).reasoning
    assert trace.readable == trace.summary
    assert "settled on shared group membership" in trace.readable


def test_encrypted_trace_is_not_readable_despite_costing_tokens():
    raw = load_fixture("encrypted_only")
    trace = parse_response(raw, make_sample(), 1.0).reasoning
    assert trace.readable is None
    assert trace.token_count == 256  # billed for thinking we cannot see


def test_absent_trace_has_no_tokens():
    raw = load_fixture("llama_no_reasoning")
    trace = parse_response(raw, make_sample(), 1.0).reasoning
    assert trace.readable is None
    assert trace.token_count is None


def test_absent_traces_differ_in_whether_thinking_was_billed():
    """Two fixtures, both absent: one was billed for thinking, one never thought."""
    billed = parse_response(
        load_fixture("openai_reasoning"), make_sample(), 1.0
    ).reasoning
    never = parse_response(
        load_fixture("llama_no_reasoning"), make_sample(), 1.0
    ).reasoning
    assert billed.availability is never.availability is ReasoningAvailability.ABSENT
    assert billed.token_count == 320
    assert never.token_count is None


def test_usage_and_cost_are_extracted():
    usage = parse_usage(load_fixture("gemma_full_text"))
    assert usage.completion_tokens > 0
    assert usage.reasoning_tokens > 0
    assert usage.cost > 0


def test_plain_reasoning_string_without_details_is_full_text():
    trace = parse_reasoning({"reasoning": "step one"}, parse_usage({}))
    assert trace.availability is ReasoningAvailability.FULL_TEXT
    assert trace.text == "step one"


def test_unknown_block_types_are_ignored_gracefully():
    message = {"reasoning_details": [{"type": "reasoning.future", "mystery": 1}]}
    trace = parse_reasoning(message, parse_usage({}))
    assert trace.availability is ReasoningAvailability.ABSENT
    assert trace.block_types == ("reasoning.future",)


def test_response_without_choices_is_a_failure():
    result = parse_response({"error": "boom"}, make_sample(), 1.0)
    assert result.ok is False
    assert "boom" in (result.error or "")


def test_effort_none_omits_the_reasoning_parameter(run_config, prompt_spec):
    sample = make_sample(reasoning_effort=None)
    assert "reasoning" not in build_request(sample, prompt_spec, run_config)


def test_effort_set_sends_it_verbatim(run_config, prompt_spec):
    body = build_request(make_sample(reasoning_effort="high"), prompt_spec, run_config)
    assert body["reasoning"] == {"effort": "high", "exclude": False}
    assert body["max_tokens"] == run_config.max_tokens


def test_variables_are_rendered_into_the_prompt(prompt_spec):
    messages = build_messages(prompt_spec, prompt_spec.variants[0])
    assert messages[0]["role"] == "system"
    assert "What is 7?" in messages[1]["content"]


def test_image_attachment_becomes_an_image_url_part(prompt_spec):
    image = Attachment(
        id="img",
        url="data:image/png;base64,AAAA",
        media_type="image/png",
    )
    spec = prompt_spec.model_copy(update={"attachments": (image,)})
    parts = build_messages(spec, spec.variants[0])[1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/png")


def test_pdf_attachment_becomes_a_file_part(prompt_spec):
    pdf = Attachment(
        id="doc",
        url="data:application/pdf;base64,AAAA",
        media_type="application/pdf",
    )
    spec = prompt_spec.model_copy(update={"attachments": (pdf,)})
    parts = build_messages(spec, spec.variants[0])[1]["content"]
    assert parts[1]["type"] == "file"
    assert parts[1]["file"]["filename"] == "doc.pdf"


def test_attachment_targeting_excludes_other_variants(prompt_spec):
    image = Attachment(
        id="img",
        url="data:image/png;base64,AAAA",
        media_type="image/png",
        attach_to=("cot",),
    )
    spec = prompt_spec.model_copy(update={"attachments": (image,)})
    plain = build_messages(spec, spec.variants[0])[1]["content"]
    cot = build_messages(spec, spec.variants[1])[1]["content"]
    assert isinstance(plain, str)  # no parts, so plain text
    assert isinstance(cot, list)
