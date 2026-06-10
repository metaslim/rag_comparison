"""Unit tests for lib.conversation.ConversationStore."""

import pytest

from src.lib.conversation import ConversationStore


@pytest.fixture
def store(tmp_path) -> ConversationStore:
    return ConversationStore(db_path=str(tmp_path / "test_conversations.db"))


def test_new_conversation_returns_unique_ids(store: ConversationStore) -> None:
    id1 = store.new_conversation()
    id2 = store.new_conversation()
    assert id1 != id2


def test_save_and_load_turn(store: ConversationStore) -> None:
    conv_id = store.new_conversation()
    store.save_turn(conv_id, "record-1", "what was net income?", "100")
    history = store.load_history(conv_id)
    assert len(history) == 1
    assert history[0]["question"] == "what was net income?"
    assert history[0]["answer"] == "100"


def test_load_history_ordered_by_turn(store: ConversationStore) -> None:
    conv_id = store.new_conversation()
    store.save_turn(conv_id, "record-1", "Q1", "A1")
    store.save_turn(conv_id, "record-1", "Q2", "A2")
    store.save_turn(conv_id, "record-1", "Q3", "A3")
    history = store.load_history(conv_id)
    assert [h["question"] for h in history] == ["Q1", "Q2", "Q3"]


def test_load_history_empty(store: ConversationStore) -> None:
    assert store.load_history("unknown-id") == []


def test_format_history_with_turns(store: ConversationStore) -> None:
    conv_id = store.new_conversation()
    store.save_turn(conv_id, "record-1", "what was net cash in 2009?", "206588")
    store.save_turn(conv_id, "record-1", "what about in 2008?", "181001")
    formatted = store.format_history(conv_id)
    assert "Q1: what was net cash in 2009? → 206588" in formatted
    assert "Q2: what about in 2008? → 181001" in formatted


def test_format_history_empty(store: ConversationStore) -> None:
    assert store.format_history("unknown-id") == ""


def test_multiple_conversations_isolated(store: ConversationStore) -> None:
    id1 = store.new_conversation()
    id2 = store.new_conversation()
    store.save_turn(id1, "record-1", "Q1", "A1")
    store.save_turn(id2, "record-2", "Q2", "A2")
    assert len(store.load_history(id1)) == 1
    assert len(store.load_history(id2)) == 1
    assert store.load_history(id1)[0]["question"] == "Q1"


def test_format_history_limited_to_max_turns(store: ConversationStore) -> None:
    conv_id = store.new_conversation()
    for i in range(15):
        store.save_turn(conv_id, "record-1", f"Q{i}", f"A{i}")
    formatted = store.format_history(conv_id, max_turns=10)
    lines = formatted.strip().split("\n")
    assert len(lines) == 10


# ---- Full-message thread (Mode 3 tool-calling) -----------------------------

def test_save_and_load_messages_roundtrip(store: ConversationStore) -> None:
    """save_messages + load_messages returns the thread in chronological order."""
    conv_id = store.new_conversation()
    turn_1 = [
        {"role": "user", "content": "what was net income?"},
        {"role": "assistant", "content": "ANSWER: 100"},
    ]
    store.save_messages(conv_id, turn_1)
    assert store.load_messages(conv_id) == turn_1


def test_load_messages_concatenates_across_turns(store: ConversationStore) -> None:
    """Sequential save_messages calls accumulate into one ordered thread."""
    conv_id = store.new_conversation()
    store.save_messages(conv_id, [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ])
    store.save_messages(conv_id, [
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ])
    loaded = store.load_messages(conv_id)
    assert [m["content"] for m in loaded] == ["Q1", "A1", "Q2", "A2"]


def test_save_messages_preserves_tool_calls(store: ConversationStore) -> None:
    """Assistant messages with tool_calls round-trip cleanly through JSON."""
    conv_id = store.new_conversation()
    tool_call = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "search_document", "arguments": '{"query": "net income"}'},
    }
    store.save_messages(conv_id, [
        {"role": "user", "content": "what was net income?"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_abc", "content": '[{"row": "net income", "value": 100}]'},
        {"role": "assistant", "content": "ANSWER: 100"},
    ])
    loaded = store.load_messages(conv_id)
    assert loaded[1]["tool_calls"] == [tool_call]
    assert loaded[2]["tool_call_id"] == "call_abc"
    # None content is omitted from the dict (OpenAI accepts missing `content`
    # for assistant messages that only carry tool_calls).
    assert "content" not in loaded[1]


def test_save_messages_roundtrips_responses_tool_items(store: ConversationStore) -> None:
    """Responses API function_call / function_call_output items (no `role`) are
    stored verbatim in item_json and replayed unchanged, interleaved with
    ordinary messages."""
    conv_id = store.new_conversation()
    thread = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what was net income?"},
        {"type": "function_call", "call_id": "fc_1", "name": "search_document",
         "arguments": '{"query": "net income"}'},
        {"type": "function_call_output", "call_id": "fc_1",
         "output": '[{"row": "net income", "value": 100}]'},
        {"role": "assistant", "content": '{"answer": "100"}'},
    ]
    store.save_messages(conv_id, thread)
    assert store.load_messages(conv_id) == thread


def test_load_messages_empty(store: ConversationStore) -> None:
    assert store.load_messages("unknown-conv") == []


def test_save_messages_isolated_per_conversation(store: ConversationStore) -> None:
    """Two conversations never see each other's messages."""
    id1, id2 = store.new_conversation(), store.new_conversation()
    store.save_messages(id1, [{"role": "user", "content": "from conv 1"}])
    store.save_messages(id2, [{"role": "user", "content": "from conv 2"}])
    assert store.load_messages(id1) == [{"role": "user", "content": "from conv 1"}]
    assert store.load_messages(id2) == [{"role": "user", "content": "from conv 2"}]
