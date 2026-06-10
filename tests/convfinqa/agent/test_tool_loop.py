"""Unit tests for FinancialAgent.answer_with_tools: tool-loop orchestration.

The tests mock the Responses API to emit a controlled sequence of function_call
items followed by a final structured answer, and they mock the ToolRegistry to
return controlled values for each dispatch. The agent's job is the loop
discipline: cap respect, fallback on exhaustion, trace recording.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.convfinqa.agent.agent import MAX_TOOL_CALLS, FinancialAgent

from tests.conftest import make_record


def _record():
    return make_record(rid="Single_TEST/2007/page_1.pdf-1", table={"2007": {"net income": 100.0}})


def _fc(call_id: str, name: str, args_json: str) -> MagicMock:
    """Build a mock Responses `function_call` output item."""
    item = MagicMock()
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = args_json
    return item


def _resp(function_calls: list | None = None, answer: str | None = None) -> MagicMock:
    """Build a mock `responses.parse` return.

    Pass ``function_calls`` for a tool-calling turn (no final answer), or
    ``answer`` for the closing turn (structured ``output_parsed``).
    """
    from src.convfinqa.agent.agent import FinancialAnswer

    resp = MagicMock()
    resp.output = function_calls or []
    if answer is not None:
        resp.output_parsed = FinancialAnswer(reasoning="r", determinable=True, answer=answer)
        resp.output_text = f'{{"answer": "{answer}"}}'
    else:
        resp.output_parsed = None
        resp.output_text = ""
    return resp


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.new_conversation = MagicMock(return_value="conv-1")
    store.format_history = MagicMock(return_value="")
    store.save_turn = MagicMock()
    store.save_messages = MagicMock()
    store.load_messages = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_graph() -> MagicMock:
    graph = MagicMock()
    graph.search = AsyncMock(return_value=[])
    return graph


@pytest.fixture
def mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.tools_for_openai = MagicMock(return_value=[])
    registry.dispatch = AsyncMock(return_value=100.0)
    return registry


@pytest.fixture
def mock_openai() -> MagicMock:
    client = MagicMock()
    # Mode 3 tool loop + single-pass fallback both go through responses.parse.
    client.responses = MagicMock()
    client.responses.parse = AsyncMock()
    client.responses.create = AsyncMock()
    return client


def _make_agent(mock_graph, mock_store, mock_openai) -> FinancialAgent:
    return FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)


# Happy paths ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_tool_calls_returns_immediate_answer(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """If the first response has no function_calls, the agent returns the answer immediately."""
    mock_openai.responses.parse.side_effect = [_resp(answer="100")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "what was net income?", registry=mock_registry,
    )

    assert outcome.answer == "100"
    assert outcome.trace == []
    assert outcome.loop_cap_exhausted is False
    assert outcome.fell_back_to_single_pass is False
    mock_store.save_turn.assert_called_once()


@pytest.mark.asyncio
async def test_single_tool_call_then_answer(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """One function call, then a final answer."""
    mock_openai.responses.parse.side_effect = [
        _resp(function_calls=[_fc("c1", "compute", '{"expression": "100 - 0"}')]),
        _resp(answer="100"),
    ]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "what was net income in 2007?", registry=mock_registry,
    )

    assert outcome.answer == "100"
    assert len(outcome.trace) == 1
    assert outcome.trace[0].tool_name == "compute"
    assert outcome.trace[0].args == {"expression": "100 - 0"}
    mock_registry.dispatch.assert_awaited_once_with("compute", {"expression": "100 - 0"})


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_response_all_dispatched(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """The model can return multiple function_calls in one response. Dispatch all."""
    mock_openai.responses.parse.side_effect = [
        _resp(function_calls=[
            _fc("c1", "search_document", '{"query": "net income 2007"}'),
            _fc("c2", "compute", '{"expression": "100 - 0"}'),
        ]),
        _resp(answer="100"),
    ]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "what was net income?", registry=mock_registry,
    )

    assert len(outcome.trace) == 2
    assert {t.tool_name for t in outcome.trace} == {"search_document", "compute"}


# Loop cap and fallback ------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_cap_triggers_warning_message_and_still_completes(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """MAX_TOOL_CALLS tool calls trigger the cap-warning; the model then answers."""
    tool_responses = [
        _resp(function_calls=[_fc(f"c{i}", "compute", '{"expression": "1+1"}')])
        for i in range(MAX_TOOL_CALLS)
    ]
    mock_openai.responses.parse.side_effect = tool_responses + [_resp(answer="100")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "q", registry=mock_registry,
    )

    assert outcome.answer == "100"
    assert outcome.loop_cap_exhausted is True
    assert outcome.fell_back_to_single_pass is False
    assert len(outcome.trace) == MAX_TOOL_CALLS

    # Cap-warning system message must NOT be persisted: it's a transient
    # control-flow nudge tracked by index and excluded before save_messages.
    saved_msgs = mock_store.save_messages.call_args.args[1]
    for msg in saved_msgs:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        assert "tool budget" not in (content or "")


@pytest.mark.asyncio
async def test_loop_cap_exhausted_falls_back_to_single_pass(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """If the model keeps calling tools after the cap-warning, fall back to answer()."""
    tool_responses = [
        _resp(function_calls=[_fc(f"c{i}", "compute", '{"expression": "1+1"}')])
        for i in range(MAX_TOOL_CALLS + 1)
    ]
    # The single-pass fallback makes its own responses.parse call.
    mock_openai.responses.parse.side_effect = tool_responses + [_resp(answer="42")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "q", registry=mock_registry,
    )

    assert outcome.loop_cap_exhausted is True
    assert outcome.fell_back_to_single_pass is True
    assert outcome.answer == "42"


# Edge cases -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_tool_args_dispatched_as_empty_dict(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """Bad JSON in tool args should not crash the loop; dispatched as {}."""
    mock_openai.responses.parse.side_effect = [
        _resp(function_calls=[_fc("c1", "compute", "not-valid-json")]),
        _resp(answer="100"),
    ]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "q", registry=mock_registry,
    )

    mock_registry.dispatch.assert_awaited_once_with("compute", {})
    assert outcome.answer == "100"


@pytest.mark.asyncio
async def test_default_registry_built_when_none_passed(
    mock_graph, mock_store, mock_openai
) -> None:
    """If registry is None, the agent builds a default ConvFinQAToolRegistry."""
    mock_openai.responses.parse.side_effect = [_resp(answer="100")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools("conv-1", _record(), "q")  # no registry passed

    assert outcome.answer == "100"


@pytest.mark.asyncio
async def test_first_turn_persists_system_user_and_assistant(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """On the FIRST turn (no prior_messages), save_messages persists
    [system, user, assistant] -- including the system item with the
    pre-loaded table -- so the next turn can replay it.
    """
    mock_store.load_messages = MagicMock(return_value=[])
    mock_openai.responses.parse.side_effect = [_resp(answer="42")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    await agent.answer_with_tools(
        "conv-1", _record(), "what was X?", registry=mock_registry,
    )

    mock_store.save_messages.assert_called_once()
    saved_conv_id, saved_msgs = mock_store.save_messages.call_args.args
    assert saved_conv_id == "conv-1"
    roles = [m["role"] for m in saved_msgs]
    assert roles == ["system", "user", "assistant"]
    assert "Table (pre-loaded, full):" in saved_msgs[0]["content"]
    assert saved_msgs[1] == {"role": "user", "content": "what was X?"}
    assert saved_msgs[2]["role"] == "assistant"
    assert "42" in saved_msgs[2]["content"]


@pytest.mark.asyncio
async def test_subsequent_turn_persists_only_new_messages(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """On a subsequent turn, save_messages persists ONLY the new [user, assistant]
    -- the system and prior dialogue are already in the DB and not re-saved."""
    mock_store.load_messages = MagicMock(return_value=[
        {"role": "system", "content": "<persisted system from turn 1>"},
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": '{"answer": "1"}'},
    ])
    mock_openai.responses.parse.side_effect = [_resp(answer="2")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    await agent.answer_with_tools(
        "conv-1", _record(), "follow-up", registry=mock_registry,
    )

    mock_store.save_messages.assert_called_once()
    _, saved_msgs = mock_store.save_messages.call_args.args
    roles = [m["role"] for m in saved_msgs]
    assert roles == ["user", "assistant"]
    assert saved_msgs[0] == {"role": "user", "content": "follow-up"}
    assert saved_msgs[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_prior_messages_replayed_in_input(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """Prior turns' items (loaded from ``load_messages``) must be replayed in `input`.

    The whole thread stays in the Responses `input` array on the next turn.
    """
    prior_messages = [
        {"role": "system", "content": "<persisted system with table>"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": '{"answer": "100"}'},
    ]
    mock_store.load_messages = MagicMock(return_value=prior_messages)
    mock_openai.responses.parse.side_effect = [_resp(answer="100")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    await agent.answer_with_tools(
        "conv-1", _record(), "follow-up question", registry=mock_registry,
    )

    sent = mock_openai.responses.parse.call_args.kwargs["input"]
    # [persisted_system, prior_user, prior_assistant, current_user]
    assert sent[0] == {"role": "system", "content": "<persisted system with table>"}
    assert sent[1] == {"role": "user", "content": "first question"}
    assert sent[2] == {"role": "assistant", "content": '{"answer": "100"}'}
    assert sent[3] == {"role": "user", "content": "follow-up question"}


@pytest.mark.asyncio
async def test_tool_trace_preserved_across_fallback(
    mock_graph, mock_store, mock_openai, mock_registry
) -> None:
    """Even when falling back, the trace from the tool calls is kept."""
    tool_responses = [
        _resp(function_calls=[_fc(f"c{i}", "compute", '{"expression": "1+1"}')])
        for i in range(MAX_TOOL_CALLS + 1)
    ]
    mock_openai.responses.parse.side_effect = tool_responses + [_resp(answer="0")]
    agent = _make_agent(mock_graph, mock_store, mock_openai)

    outcome = await agent.answer_with_tools(
        "conv-1", _record(), "q", registry=mock_registry,
    )

    # All MAX_TOOL_CALLS+1 tool dispatches are in the trace (even though fallback was used)
    assert len(outcome.trace) == MAX_TOOL_CALLS + 1
