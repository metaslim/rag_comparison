"""Base protocols and types for the AnswerStrategy pattern."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexedRecord(Protocol):
    """Minimal record interface: just the document identifier.

    ``ConvFinQARecord`` satisfies this at runtime. Keeping it minimal means
    lib/ has no dependency on convfinqa/ domain types.
    """

    id: str


@runtime_checkable
class TraceEntry(Protocol):
    """One tool invocation in a tool-calling trace."""

    tool_name: str


@runtime_checkable
class ToolOutcome(Protocol):
    """Return shape of agent.answer_with_tools(): used by ToolCallingStrategy."""

    answer: str
    trace: list[TraceEntry]
    loop_cap_exhausted: bool
    fell_back_to_single_pass: bool


@runtime_checkable
class Agent(Protocol):
    """Minimal agent interface required by the answer strategies.

    ``FinancialAgent`` satisfies this at runtime. Keeping it minimal means
    lib/ has no dependency on the concrete agent class.
    """

    async def answer(
        self, conversation_id: str, record: IndexedRecord, question: str
    ) -> str:
        """Single-pass answer using graph-RAG (Mode 1) or raw doc fallback (Mode 2)."""
        ...

    async def answer_raw_doc(
        self, conversation_id: str, record: IndexedRecord, question: str
    ) -> str:
        """Answer using the full raw document, skipping graph search (Mode 2 explicit)."""
        ...

    async def answer_with_tools(
        self, conversation_id: str, record: IndexedRecord, question: str
    ) -> ToolOutcome:
        """Function-calling tool loop (Mode 3)."""
        ...


@dataclass
class AnswerResult:
    """Uniform return type across all strategies.

    ``trace`` is empty for single-pass modes; populated by Mode 3. The runner
    uses ``if result.trace:`` to decide whether to log tool details.
    """

    answer: str
    trace: list[TraceEntry] = field(default_factory=list)
    loop_cap_exhausted: bool = False
    fell_back_to_single_pass: bool = False


@runtime_checkable
class AnswerStrategy(Protocol):
    """One execution path for answering a conversational question.

    Implement this protocol to add a new mode. No edits to existing code (OCP).
    """

    name: str

    async def answer(
        self,
        agent: Agent,
        conversation_id: str,
        record: IndexedRecord,
        question: str,
    ) -> AnswerResult:
        """Answer ``question`` and return a uniform result."""
        ...
