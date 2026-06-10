"""Unit tests for AnswerStrategy implementations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.convfinqa.agent.agent import ToolAnswerOutcome, ToolTrace
from src.convfinqa.dataset.models import ConvFinQARecord
from src.lib.answers import (
    AnswerResult,
    AnswerStrategy,
    RawDocStrategy,
    SinglePassGraphStrategy,
    ToolCallingStrategy,
)


@pytest.fixture
def record() -> ConvFinQARecord:
    return ConvFinQARecord.model_validate({
        "id": "Single_TEST/2009/page_1.pdf-1",
        "doc": {
            "pre_text": "Some pre text.",
            "post_text": "Some post text.",
            "table": {"2009": {"net income": 100.0}},
        },
        "dialogue": {
            "conv_questions": ["what was net income?"],
            "conv_answers": ["100"],
            "turn_program": ["100"],
            "executed_answers": [100.0],
            "qa_split": [False],
        },
        "features": {
            "num_dialogue_turns": 1,
            "has_type2_question": False,
            "has_duplicate_columns": False,
            "has_non_numeric_values": False,
        },
    })


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.answer = AsyncMock(return_value="100")
    agent.answer_with_tools = AsyncMock(return_value=ToolAnswerOutcome(answer="100"))
    return agent


# ---- Protocol conformance ---------------------------------------------------

def test_single_pass_satisfies_answer_strategy_protocol() -> None:
    """SinglePassGraphStrategy must satisfy the AnswerStrategy Protocol at runtime."""
    assert isinstance(SinglePassGraphStrategy(), AnswerStrategy)


def test_tool_calling_satisfies_answer_strategy_protocol() -> None:
    """ToolCallingStrategy must satisfy the AnswerStrategy Protocol at runtime."""
    assert isinstance(ToolCallingStrategy(), AnswerStrategy)


# ---- SinglePassGraphStrategy ----------------------------------------------------

@pytest.mark.asyncio
async def test_single_pass_delegates_to_agent_answer(mock_agent, record) -> None:
    """SinglePassGraphStrategy must call agent.answer() with the correct args."""
    strategy = SinglePassGraphStrategy()
    result = await strategy.answer(mock_agent, "conv-1", record, "what was net income?")

    mock_agent.answer.assert_awaited_once_with("conv-1", record, "what was net income?")
    assert result.answer == "100"


@pytest.mark.asyncio
async def test_single_pass_returns_empty_trace(mock_agent, record) -> None:
    """SinglePassGraphStrategy always returns an empty trace (no tool calls)."""
    strategy = SinglePassGraphStrategy()
    result = await strategy.answer(mock_agent, "conv-1", record, "q")

    assert result.trace == []
    assert result.loop_cap_exhausted is False
    assert result.fell_back_to_single_pass is False


@pytest.mark.asyncio
async def test_single_pass_returns_answer_result_type(mock_agent, record) -> None:
    """Return type must be AnswerResult."""
    result = await SinglePassGraphStrategy().answer(mock_agent, "conv-1", record, "q")
    assert isinstance(result, AnswerResult)


# ---- ToolCallingStrategy ---------------------------------------------------

@pytest.mark.asyncio
async def test_tool_calling_delegates_to_agent_answer_with_tools(mock_agent, record) -> None:
    """ToolCallingStrategy must call agent.answer_with_tools() with the correct args."""
    strategy = ToolCallingStrategy()
    await strategy.answer(mock_agent, "conv-1", record, "what was net income?")

    mock_agent.answer_with_tools.assert_awaited_once_with("conv-1", record, "what was net income?")


@pytest.mark.asyncio
async def test_tool_calling_maps_trace_from_outcome(mock_agent, record) -> None:
    """ToolCallingStrategy must copy trace, loop_cap_exhausted, and fell_back fields."""
    trace = [ToolTrace("compute", {"expression": "100-50"}, 50.0)]
    mock_agent.answer_with_tools = AsyncMock(return_value=ToolAnswerOutcome(
        answer="50",
        trace=trace,
        loop_cap_exhausted=True,
        fell_back_to_single_pass=True,
    ))

    result = await ToolCallingStrategy().answer(mock_agent, "conv-1", record, "q")

    assert result.answer == "50"
    assert result.trace == trace
    assert result.loop_cap_exhausted is True
    assert result.fell_back_to_single_pass is True


@pytest.mark.asyncio
async def test_tool_calling_returns_answer_result_type(mock_agent, record) -> None:
    """Return type must be AnswerResult."""
    result = await ToolCallingStrategy().answer(mock_agent, "conv-1", record, "q")
    assert isinstance(result, AnswerResult)


# ---- AnswerResult defaults -------------------------------------------------

def test_answer_result_defaults_are_falsy() -> None:
    """AnswerResult with only answer set must have safe defaults for all other fields."""
    r = AnswerResult(answer="42")
    assert r.trace == []
    assert r.loop_cap_exhausted is False
    assert r.fell_back_to_single_pass is False


# ---- Strategy names --------------------------------------------------------

# ---- RawDocStrategy --------------------------------------------------------

def test_raw_doc_satisfies_answer_strategy_protocol() -> None:
    """RawDocStrategy must satisfy the AnswerStrategy Protocol at runtime."""
    assert isinstance(RawDocStrategy(), AnswerStrategy)


@pytest.mark.asyncio
async def test_raw_doc_delegates_to_agent_answer_raw_doc(mock_agent, record) -> None:
    """RawDocStrategy must call agent.answer_raw_doc() -- not agent.answer()."""
    mock_agent.answer_raw_doc = AsyncMock(return_value="42")
    result = await RawDocStrategy().answer(mock_agent, "conv-1", record, "q?")

    mock_agent.answer_raw_doc.assert_awaited_once_with("conv-1", record, "q?")
    mock_agent.answer.assert_not_called()
    assert result.answer == "42"


@pytest.mark.asyncio
async def test_raw_doc_returns_empty_trace(mock_agent, record) -> None:
    """RawDocStrategy produces no tool trace."""
    mock_agent.answer_raw_doc = AsyncMock(return_value="42")
    result = await RawDocStrategy().answer(mock_agent, "conv-1", record, "q")

    assert result.trace == []
    assert result.loop_cap_exhausted is False


@pytest.mark.asyncio
async def test_raw_doc_returns_answer_result_type(mock_agent, record) -> None:
    """Return type must be AnswerResult."""
    mock_agent.answer_raw_doc = AsyncMock(return_value="42")
    result = await RawDocStrategy().answer(mock_agent, "conv-1", record, "q")
    assert isinstance(result, AnswerResult)


# ---- Strategy names --------------------------------------------------------

def test_strategy_names_are_distinct() -> None:
    """Each strategy must have a unique name (used for logging / output paths)."""
    names = {SinglePassGraphStrategy.name, RawDocStrategy.name, ToolCallingStrategy.name}
    assert len(names) == 3
