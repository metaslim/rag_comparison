"""RawDocStrategy: Mode 2 -- raw document, no graph search."""

from src.lib.answers.base.strategy import Agent, AnswerResult, IndexedRecord


class RawDocStrategy:
    """Mode 2: raw document fallback -- no graph retrieval.

    Passes the full pre_text + table + post_text directly to the model.
    Use when the record is intentionally not indexed (``--no-index``) or
    when graph retrieval should be bypassed entirely.

    Delegates to ``agent.answer_raw_doc()`` which builds context from the
    document without querying the graph.
    """

    name = "raw-doc"

    async def answer(
        self,
        agent: Agent,
        conversation_id: str,
        record: IndexedRecord,
        question: str,
    ) -> AnswerResult:
        """Delegate to agent.answer_raw_doc() and wrap in a uniform AnswerResult."""
        ans = await agent.answer_raw_doc(conversation_id, record, question)
        return AnswerResult(answer=ans)
