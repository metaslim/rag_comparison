"""Conversation history protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class HistoryStore(Protocol):
    """Interface for conversation history persistence."""

    def new_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        ...

    def save_turn(self, conversation_id: str, record_id: str, question: str, answer: str) -> None:
        """Persist a Q&A turn."""
        ...

    def load_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return all turns ordered by turn number."""
        ...

    def format_history(self, conversation_id: str) -> str:
        """Return formatted history string for LLM prompt injection."""
        ...

    def save_messages(self, conversation_id: str, messages: list[dict]) -> None:
        """Append a turn's full OpenAI-style messages to the thread."""
        ...

    def load_messages(self, conversation_id: str) -> list[dict]:
        """Return the full message thread, ready for OpenAI's `messages` arg."""
        ...
