"""Unit tests for src.lib.evaluation.llm_judge."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.lib.evaluation.llm_judge import JudgeVerdict, judge_answer
from src.lib.evaluation.llm_judge.evaluator import _JudgeSchema


def _client_returning(
    correct: bool = True, reason: str = "ok", confidence: str = "high"
) -> MagicMock:
    """Mock an OpenAI client whose `responses.parse` returns a structured verdict.

    Structured outputs hand back a parsed Pydantic object on `output_parsed`,
    not a JSON string to decode -- so the mock mirrors that shape.
    """
    client = MagicMock()
    client.responses = MagicMock()
    resp = MagicMock()
    resp.output_parsed = _JudgeSchema(correct=correct, reason=reason, confidence=confidence)
    client.responses.parse = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_judge_parses_correct_verdict() -> None:
    client = _client_returning(correct=True, reason="exact match", confidence="high")
    v = await judge_answer("what was net income?", 100.0, "100.0", client)
    assert v.correct is True
    assert v.reason == "exact match"
    assert v.confidence == "high"


@pytest.mark.asyncio
async def test_judge_parses_incorrect_verdict() -> None:
    client = _client_returning(correct=False, reason="off by 10x", confidence="high")
    v = await judge_answer("what was net income?", 100.0, "1000.0", client)
    assert v.correct is False
    assert v.reason == "off by 10x"


@pytest.mark.asyncio
async def test_judge_no_parsed_verdict_fails_closed() -> None:
    """If the model refuses (output_parsed is None), fail closed to correct=False."""
    client = MagicMock()
    client.responses = MagicMock()
    resp = MagicMock()
    resp.output_parsed = None
    client.responses.parse = AsyncMock(return_value=resp)
    v = await judge_answer("q", 1.0, "1.0", client)
    assert v.correct is False
    assert "judge_error" in v.reason
    assert v.confidence == "low"


@pytest.mark.asyncio
async def test_judge_api_error_returns_false_with_reason() -> None:
    """API errors must never silently pass an answer -- they return correct=False."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.parse = AsyncMock(side_effect=RuntimeError("rate limit"))
    v = await judge_answer("q", 1.0, "1.0", client)
    assert v.correct is False
    assert "judge_error" in v.reason
    assert v.confidence == "low"


@pytest.mark.asyncio
async def test_judge_uses_structured_schema_and_default_model() -> None:
    """The judge call uses the JSON structured-output schema and defaults to gpt-5-mini.

    A judge should be at least as capable as the model under evaluation; the old
    gpt-4o-mini default judged gpt-5-mini outputs with a weaker model.
    """
    client = _client_returning()
    await judge_answer("q", 1.0, "1.0", client)
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["text_format"] is _JudgeSchema


@pytest.mark.asyncio
async def test_judge_includes_question_gold_predicted_in_input() -> None:
    """The input must contain the question, gold, and predicted for context."""
    client = _client_returning()
    await judge_answer("what was net income in 2007?", 60.94, "60.94", client)
    input_text = client.responses.parse.call_args.kwargs["input"]
    assert "what was net income in 2007?" in input_text
    assert "60.94" in input_text


@pytest.mark.asyncio
async def test_verdict_dataclass_defaults() -> None:
    """JudgeVerdict has sensible defaults for confidence."""
    v = JudgeVerdict(correct=True, reason="x")
    assert v.confidence == "high"


@pytest.mark.asyncio
async def test_llm_judge_evaluator_delegates_to_judge_answer() -> None:
    """LLMJudgeEvaluator adapts JudgeVerdict into a common EvalVerdict."""
    from src.lib.evaluation.llm_judge import LLMJudgeEvaluator
    client = _client_returning(correct=True, reason="exact match", confidence="high")
    v = await LLMJudgeEvaluator(client=client, model="gpt-5-mini").evaluate("q", 100.0, "100.0")
    assert v.correct is True
    assert v.reason == "exact match"


@pytest.mark.asyncio
async def test_llm_judge_evaluator_propagates_error() -> None:
    """API errors propagate as correct=False with low confidence."""
    from src.lib.evaluation.llm_judge import LLMJudgeEvaluator
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.parse = AsyncMock(side_effect=RuntimeError("rate limit"))
    v = await LLMJudgeEvaluator(client=client, model="gpt-5-mini").evaluate("q", 1.0, "1.0")
    assert v.correct is False
    assert v.confidence == "low"
