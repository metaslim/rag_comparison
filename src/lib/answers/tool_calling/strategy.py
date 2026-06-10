"""ToolCallingStrategy: Mode 3 function-calling loop."""

from src.lib.answers.base.strategy import Agent, AnswerResult, IndexedRecord


class ToolCallingStrategy:
    """Mode 3: function-calling loop driven by the Responses API.

    The model has three tools: search_document (parallel table + narrative search),
    compute (arithmetic evaluator), and compare (yes/no comparator).
    """

    name = "tool-calling"

    async def answer(
        self,
        agent: Agent,
        conversation_id: str,
        record: IndexedRecord,
        question: str,
    ) -> AnswerResult:
        """Delegate to agent.answer_with_tools() and wrap in a uniform AnswerResult."""
        outcome = await agent.answer_with_tools(conversation_id, record, question)
        return AnswerResult(
            answer=outcome.answer,
            trace=outcome.trace,
            loop_cap_exhausted=outcome.loop_cap_exhausted,
            fell_back_to_single_pass=outcome.fell_back_to_single_pass,
        )
