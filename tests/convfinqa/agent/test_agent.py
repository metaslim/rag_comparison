"""Unit tests for FinancialAgent with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.convfinqa.dataset.models import ConvFinQARecord

SAMPLE_RECORD = ConvFinQARecord.model_validate({
    "id": "Single_TEST/2009/page_1.pdf",
    "doc": {
        "pre_text": "Context text.",
        "post_text": "Post context.",
        "table": {"2009": {"net income": 100.0}, "2008": {"net income": 90.0}},
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


def _make_parsed_response(answer: str, reasoning: str = "step-by-step reasoning") -> MagicMock:
    """Mock a `client.responses.parse` return: structured `output_parsed` + `output_text`."""
    from src.convfinqa.agent.agent import FinancialAnswer

    response = MagicMock()
    response.output_parsed = FinancialAnswer(
        reasoning=reasoning, determinable=True, answer=answer
    )
    response.output_text = f'{{"answer": "{answer}"}}'
    return response


def _make_text_response(text: str) -> MagicMock:
    """Mock a `client.responses.create` return: plain `output_text` (query rewriter)."""
    response = MagicMock()
    response.output_text = text
    return response


@pytest.fixture
def mock_graph() -> MagicMock:
    graph = MagicMock()
    graph.search = AsyncMock(return_value=["net income in 2009 was 100.0"])
    return graph


@pytest.fixture
def mock_store(tmp_path) -> MagicMock:
    from src.lib.conversation import ConversationStore
    return ConversationStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def mock_openai() -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    # Single-pass answer path -> responses.parse (structured JSON output).
    client.responses.parse = AsyncMock(return_value=_make_parsed_response("100.0"))
    # Query-rewriter path -> responses.create (plain text).
    client.responses.create = AsyncMock(return_value=_make_text_response("rewritten query"))
    return client


@pytest.fixture
def agent(mock_graph, mock_store, mock_openai):
    from src.convfinqa.agent import FinancialAgent
    return FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)


@pytest.mark.asyncio
async def test_answer_uses_graph_results(agent, mock_graph, mock_openai) -> None:
    conv_id = agent._store.new_conversation()
    answer = await agent.answer(conv_id, SAMPLE_RECORD, "what was net income in 2009?")
    assert answer == "100.0"
    mock_graph.search.assert_called_once_with(SAMPLE_RECORD.id, "what was net income in 2009?")


@pytest.mark.asyncio
async def test_answer_saves_turn_to_history(agent, mock_store) -> None:
    conv_id = agent._store.new_conversation()
    await agent.answer(conv_id, SAMPLE_RECORD, "what was net income?")
    history = mock_store.load_history(conv_id)
    assert len(history) == 1
    assert history[0]["question"] == "what was net income?"
    assert history[0]["answer"] == "100.0"


@pytest.mark.asyncio
async def test_answer_uses_fallback_when_graph_empty(agent, mock_graph, mock_openai) -> None:
    mock_graph.search = AsyncMock(return_value=[])
    conv_id = agent._store.new_conversation()
    await agent.answer(conv_id, SAMPLE_RECORD, "what was revenue?")
    call_args = mock_openai.responses.parse.call_args
    prompt = call_args.kwargs["input"][1]["content"]
    assert "[Table]" in prompt


@pytest.mark.asyncio
async def test_answer_replays_prior_messages_in_message_stack(agent, mock_openai) -> None:
    """Prior turns' user + assistant messages are replayed in the messages
    array (canonical OpenAI multi-turn pattern), not concatenated as a
    Q->A string in the current user prompt.
    """
    conv_id = agent._store.new_conversation()
    # Save a prior turn via both code paths the agent uses for persistence.
    # Realistic prior_messages: includes the system message persisted on
    # turn 1, just like the real load_messages would return.
    agent._store.save_turn(conv_id, SAMPLE_RECORD.id, "Q1", "A1")
    agent._store.save_messages(conv_id, [
        {"role": "system", "content": "<persisted system from turn 1>"},
        {"role": "user", "content": "Q1 user content"},
        {"role": "assistant", "content": "A1 assistant content"},
    ])
    await agent.answer(conv_id, SAMPLE_RECORD, "what is the difference?")
    call_args = mock_openai.responses.parse.call_args
    sent = call_args.kwargs["input"]
    # [persisted_system, prior_user, prior_assistant, current_user]
    # No fresh system prepended -- prior already has it from turn 1.
    assert sent[0] == {"role": "system", "content": "<persisted system from turn 1>"}
    assert sent[1] == {"role": "user", "content": "Q1 user content"}
    assert sent[2] == {"role": "assistant", "content": "A1 assistant content"}
    assert sent[3]["role"] == "user"
    # Current user message should NOT contain a "Conversation so far:" string
    # -- history is now native messages, not an inlined summary.
    assert "Conversation so far" not in sent[3]["content"]
    assert "what is the difference?" in sent[3]["content"]


def test_finalize_answer_evaluates_expression() -> None:
    """The structured `answer` field is evaluated with simpleeval -- no regex."""
    from src.convfinqa.agent.agent import FinancialAnswer, _finalize_answer
    assert _finalize_answer(
        FinancialAnswer(reasoning="r", determinable=True, answer="11 - 8")
    ) == "3"
    assert _finalize_answer(
        FinancialAnswer(reasoning="r", determinable=True, answer="25587.0")
    ) == "25587.0"


def test_finalize_answer_unknown_when_not_determinable() -> None:
    """determinable=False maps to the 'unknown' sentinel the evaluators expect."""
    from src.convfinqa.agent.agent import FinancialAnswer, _finalize_answer
    assert _finalize_answer(
        FinancialAnswer(reasoning="r", determinable=False, answer="")
    ) == "unknown"


@pytest.mark.asyncio
async def test_rewrite_query_returns_original_question_on_exception(
    mock_graph, mock_store
) -> None:
    """If the OpenAI rewriter call raises, _rewrite_query falls back to the raw question."""
    from src.convfinqa.agent import FinancialAgent

    flaky_client = MagicMock()
    flaky_client.responses = MagicMock()
    flaky_client.responses.create = AsyncMock(side_effect=RuntimeError("openai down"))
    agent = FinancialAgent(graph=mock_graph, store=mock_store, openai_client=flaky_client)

    # Non-empty history is required, else _rewrite_query short-circuits at the top.
    out = await agent._rewrite_query(
        "what about 2010?", history="Q1: what was net income in 2009? → 100"
    )
    assert out == "what about 2010?"


# ---- ContextBuilder structure tests -----------------------------------------

def test_mode2_fallback_includes_all_sections(mock_graph, mock_store, mock_openai) -> None:
    """Mode 2 fallback passes the full document verbatim: pre_text, table, post_text."""
    from src.convfinqa.agent import FinancialAgent
    mock_graph.search = AsyncMock(return_value=[])
    agent = FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)

    import asyncio
    conv_id = mock_store.new_conversation()
    asyncio.run(agent.answer(conv_id, SAMPLE_RECORD, "test?"))
    user_msg = mock_openai.responses.parse.call_args.kwargs["input"][1]["content"]

    assert "Context text." in user_msg
    assert "Post context." in user_msg
    assert "net income" in user_msg


def test_facts_path_includes_all_facts(mock_graph, mock_store, mock_openai) -> None:
    """Mode 1 facts path passes all retrieved facts to the model."""
    from src.convfinqa.agent import FinancialAgent
    facts = [f"FACT_{i}" for i in range(10)]
    mock_graph.search = AsyncMock(return_value=facts)
    agent = FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)

    import asyncio
    conv_id = mock_store.new_conversation()
    asyncio.run(agent.answer(conv_id, SAMPLE_RECORD, "test?"))
    user_msg = mock_openai.responses.parse.call_args.kwargs["input"][1]["content"]

    for fact in facts:
        assert fact in user_msg
    assert "net income" in user_msg


def test_answer_raw_doc_skips_graph_search_and_includes_full_document(
    mock_graph, mock_store, mock_openai
) -> None:
    """answer_raw_doc() must not call graph.search and must include the full document."""
    from src.convfinqa.agent import FinancialAgent
    agent = FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)

    import asyncio
    conv_id = mock_store.new_conversation()
    asyncio.run(agent.answer_raw_doc(conv_id, SAMPLE_RECORD, "test?"))

    mock_graph.search.assert_not_called()
    user_msg = mock_openai.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "Context text." in user_msg
    assert "Post context." in user_msg
    assert "net income" in user_msg


def test_format_table_grid_handles_empty_and_non_numeric_cells() -> None:
    """The helper handles empty tables, integer-valued floats, fractional floats,
    integers, strings, and None values. Covers all branches of _fmt."""
    from src.convfinqa.agent.agent import _format_table_grid
    assert _format_table_grid({}) == "(empty table)"
    # Integer-valued float -> int form; fractional float at full precision;
    # plain int, str, None all pass through.
    grid = _format_table_grid({
        "2008": {
            "int_valued_float": 12000000.0,
            "fractional": 2.61,
            "plain_int": 5,
            "string": "n/a",
            "none_value": None,
        }
    })
    assert "12000000" in grid  # no scientific notation
    assert "2.61" in grid
    assert "n/a" in grid


def test_graph_and_store_properties_expose_injected_deps(
    mock_graph, mock_store, mock_openai
) -> None:
    """The public agent.graph / agent.store properties return the same
    instances passed at construction. Used by the eval runner."""
    from src.convfinqa.agent import FinancialAgent
    agent = FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)
    assert agent.graph is mock_graph
    assert agent.store is mock_store


def test_client_property_exposes_openai_client(
    mock_graph, mock_store, mock_openai
) -> None:
    """agent.client returns the injected OpenAI client (used by LLMJudgeEvaluator)."""
    from src.convfinqa.agent import FinancialAgent
    agent = FinancialAgent(graph=mock_graph, store=mock_store, openai_client=mock_openai)
    assert agent.client is mock_openai


def test_temp_kwargs_omits_temperature_for_no_temperature_models() -> None:
    """Reasoning-model prefixes and gpt-5-* must not receive a temperature kwarg."""
    from src.convfinqa.agent.agent import _temp_kwargs
    assert _temp_kwargs("o1-mini") == {}
    assert _temp_kwargs("o3") == {}
    assert _temp_kwargs("o4-mini") == {}
    assert _temp_kwargs("gpt-5-mini") == {}
    assert _temp_kwargs("gpt-4o-mini") == {"temperature": 0}
    assert _temp_kwargs("gpt-4-turbo") == {"temperature": 0}


def test_build_user_prompt_combines_context_and_question() -> None:
    """build_user_prompt places context before the question in the user message."""
    from src.convfinqa.agent.prompts import build_user_prompt
    out = build_user_prompt(context="table data here", question="follow-up?")
    assert "table data here" in out
    assert "Current question: follow-up?" in out


def test_to_responses_input_filters_tool_role_and_null_content() -> None:
    """_to_responses_input drops tool-role messages and assistant messages with no content."""
    from src.convfinqa.agent.agent import _to_responses_input
    thread = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "100"},
        {"role": "assistant", "content": '{"answer": "100"}'},
    ]
    result = _to_responses_input(thread)
    roles = [m["role"] for m in result]
    assert roles == ["system", "user", "assistant"]
    assert result[2]["content"] == '{"answer": "100"}'


def test_to_responses_tools_flattens_chat_completions_schema() -> None:
    """_to_responses_tools unnests the 'function' wrapper and adds strict=False."""
    from src.convfinqa.agent.agent import _to_responses_tools
    chat_schema = [{
        "type": "function",
        "function": {
            "name": "compute",
            "description": "eval math",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]
    result = _to_responses_tools(chat_schema)
    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["name"] == "compute"
    assert tool["description"] == "eval math"
    assert "parameters" in tool
    assert tool["strict"] is False
