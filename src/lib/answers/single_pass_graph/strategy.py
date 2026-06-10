"""SinglePassGraphStrategy: Mode 1 (graph-RAG) and Mode 2 (raw document fallback)."""

from src.lib.answers.base.strategy import Agent, AnswerResult, IndexedRecord


class SinglePassGraphStrategy:
    """Mode 1 / Mode 2: single-pass Chain-of-Thought.

    Mode 1 fires when the record is indexed in the graph (graph-RAG retrieval).
    Mode 2 fires automatically when graph search returns nothing (raw document
    fallback). The selection is internal to the agent; the strategy surface
    stays the same either way.
    """

    name = "single-pass-graph"

    async def answer(
        self,
        agent: Agent,
        conversation_id: str,
        record: IndexedRecord,
        question: str,
    ) -> AnswerResult:
        """Delegate to agent.answer() and wrap in a uniform AnswerResult."""
        ans = await agent.answer(conversation_id, record, question)
        return AnswerResult(answer=ans)
