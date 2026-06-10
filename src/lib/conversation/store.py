"""SQLite-backed conversation history store. Implements the HistoryStore protocol.

Two parallel histories live here:

1. ``conversations`` table: legacy `(question, answer)` per turn. Mode 1
   (single-pass) and Mode 2 (raw doc) continue to use this via ``save_turn`` /
   ``format_history``; the formatted string goes into the user prompt.

2. ``messages`` table: full OpenAI / Vercel-style message thread. Mode 3
   (tool-calling) uses this via ``save_messages`` / ``load_messages``: every
   prior turn's user message, assistant tool-call response, tool result, and
   final assistant answer is replayed verbatim to the LLM on the next turn.
   This matches OpenAI's documented multi-turn function-calling pattern.
"""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

# Default number of recent turns surfaced by ``format_history``. Kept here
# (in the lib layer) so prompt code can import it from store, not the
# other way around (lib must not depend on convfinqa).
DEFAULT_MAX_HISTORY_TURNS = 10

# Sentinel stored in the (NOT NULL) `role` column for non-message thread items
# (Responses `function_call` / `function_call_output`). The real item lives in
# `item_json`; load keys off that, so this value is never read back -- it only
# satisfies the legacy NOT NULL constraint on databases created before the
# `item_json` column existed.
_RESPONSES_ITEM_ROLE = "_responses_item"


class ConversationStore:
    """Persists and retrieves per-session conversation history.

    Implements the HistoryStore protocol. Backed by SQLite: zero dependencies,
    persists across sessions, queryable by conversation_id.
    """

    def __init__(self, db_path: str) -> None:
        path = db_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT NOT NULL,
                record_id       TEXT NOT NULL,
                turn            INTEGER NOT NULL,
                question        TEXT NOT NULL,
                answer          TEXT NOT NULL,
                timestamp       TEXT NOT NULL
            )
        """)
        # Full-message thread for Mode 3 (tool-calling). Each row is one
        # OpenAI-style message: role + content (+ tool_calls JSON for
        # assistant messages that fired tools, + tool_call_id for tool-role
        # messages). `seq` orders messages within a turn; `turn` groups
        # them so we can replay in chronological order.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                conversation_id TEXT NOT NULL,
                turn            INTEGER NOT NULL,
                seq             INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT,
                tool_calls      TEXT,
                tool_call_id    TEXT,
                item_json       TEXT,
                timestamp       TEXT NOT NULL,
                PRIMARY KEY (conversation_id, turn, seq)
            )
        """)
        # `item_json` was added after the original schema shipped: it stores
        # non-message thread items verbatim (Responses API `function_call` /
        # `function_call_output` items, which have no `role`). Add it
        # defensively so existing databases pick it up without a migration.
        try:
            self._conn.execute("ALTER TABLE messages ADD COLUMN item_json TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        self._conn.commit()

    def new_conversation(self) -> str:
        """Generate a new unique conversation ID (UUID4)."""
        return str(uuid.uuid4())

    def save_turn(self, conversation_id: str, record_id: str, question: str, answer: str) -> None:
        """Persist a Q&A turn to the database."""
        turn = self._next_turn(conversation_id)
        self._conn.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, record_id, turn, question, answer, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def _next_turn(self, conversation_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(turn) FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return ((row[0] or 0) + 1) if row else 1

    def load_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return all turns for a conversation ordered by turn number."""
        rows = self._conn.execute(
            "SELECT question, answer FROM conversations WHERE conversation_id = ? ORDER BY turn",
            (conversation_id,),
        ).fetchall()
        return [{"question": q, "answer": a} for q, a in rows]

    def format_history(self, conversation_id: str, max_turns: int = DEFAULT_MAX_HISTORY_TURNS) -> str:
        """Format conversation history as a string for inclusion in an LLM prompt.

        Limits to the most recent max_turns to prevent unbounded context growth.
        """
        history = self.load_history(conversation_id)
        if not history:
            return ""
        recent = history[-max_turns:]
        lines = [f"Q{i + 1}: {turn['question']} → {turn['answer']}" for i, turn in enumerate(recent)]
        return "\n".join(lines)

    # ---- Full-message thread (Mode 3 tool-calling) -----------------------

    def save_messages(
        self, conversation_id: str, messages: list[dict]
    ) -> None:
        """Append a turn's thread items to the full-message thread.

        Two item shapes are persisted:

        - **Messages** (have a ``role``): Chat-Completions / Responses message
          dicts (``role`` + ``content``; assistant tool calls add ``tool_calls``;
          tool-role messages add ``tool_call_id``). Stored across the typed
          columns so existing readers keep working.
        - **Responses tool items** (no ``role``): ``function_call`` /
          ``function_call_output`` items emitted by the Mode 3 tool loop. These
          don't fit the message columns, so the whole dict is stored verbatim in
          ``item_json`` and replayed as-is.

        We assign a new turn number (max(turn)+1) and `seq` within the turn
        so replay order is deterministic. The first call for a conversation
        sees turn=1.
        """
        turn = self._next_message_turn(conversation_id)
        ts = datetime.now(UTC).isoformat()
        for seq, msg in enumerate(messages):
            role = msg.get("role")
            # Items without a `role` are Responses tool items -- persist them
            # verbatim in item_json (the message columns don't fit them). The
            # role column gets a sentinel so the NOT NULL constraint holds; load
            # keys off item_json, not the role, so the sentinel is never read.
            is_item = role is None
            item_json = json.dumps(msg) if is_item else None
            # Name columns explicitly: `item_json` was added via ALTER on
            # pre-existing databases, which appends it AFTER `timestamp`, while a
            # fresh CREATE lists it before. Positional VALUES would map columns
            # differently across the two; naming them keeps the insert correct
            # regardless of physical column order.
            self._conn.execute(
                "INSERT INTO messages "
                "(conversation_id, turn, seq, role, content, tool_calls, "
                "tool_call_id, item_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    turn,
                    seq,
                    _RESPONSES_ITEM_ROLE if is_item else role,
                    msg.get("content"),
                    json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None,
                    msg.get("tool_call_id"),
                    item_json,
                    ts,
                ),
            )
        self._conn.commit()

    def _next_message_turn(self, conversation_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(turn) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return ((row[0] or 0) + 1) if row else 1

    def load_messages(self, conversation_id: str) -> list[dict]:
        """Return the full thread for a conversation in chronological order.

        Returns the items as stored: role-bearing messages (``[{role, content,
        tool_calls?, tool_call_id?}, ...]``) plus any Responses API tool items
        (``function_call`` / ``function_call_output``) reconstructed verbatim
        from ``item_json``. Ready to replay as Responses ``input``. JSON
        `tool_calls` are decoded; SQL NULL columns are dropped from the dict.
        """
        rows = self._conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, item_json FROM messages "
            "WHERE conversation_id = ? ORDER BY turn, seq",
            (conversation_id,),
        ).fetchall()
        out: list[dict] = []
        for role, content, tool_calls, tool_call_id, item_json in rows:
            # Non-message items (Responses tool items) were stored verbatim.
            if item_json is not None:
                out.append(json.loads(item_json))
                continue
            msg: dict = {"role": role}
            if content is not None:
                msg["content"] = content
            if tool_calls is not None:
                msg["tool_calls"] = json.loads(tool_calls)
            if tool_call_id is not None:
                msg["tool_call_id"] = tool_call_id
            out.append(msg)
        return out
