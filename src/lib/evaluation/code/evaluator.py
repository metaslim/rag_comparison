"""CodeEvaluator: seven-path numeric-tolerance evaluator wrapped as an Evaluator."""

from src.lib.evaluation.base import EvalVerdict, is_correct


class CodeEvaluator:
    """Default evaluator. Deterministic, free, fast.

    Wraps the pure `is_correct(predicted, gold)` function so the runner can
    treat code and LLM evaluators identically through the `Evaluator` Protocol.
    """

    name = "code"

    async def evaluate(
        self, question: str, gold: str | float, predicted: str
    ) -> EvalVerdict:
        """Apply the seven tolerance paths and return a structured verdict."""
        return EvalVerdict(
            correct=is_correct(predicted, gold),
            reason="seven-path numeric tolerance",
            confidence="high",
        )
