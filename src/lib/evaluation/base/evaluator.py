"""Evaluator Protocol: common interface for code-based and LLM-based evaluators.

Both the seven-path code evaluator and the LLM judge are *strategies* for
answering the same question: given (question, gold, predicted), is the
prediction correct? Sharing a Protocol lets the runner call any number of
evaluators in the same loop without per-evaluator special cases.

Open/Closed: add a new evaluator (ensemble, regex, human-rated cache) by
writing one class that implements `Evaluator`. The runner doesn't change.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EvalVerdict:
    """One evaluator's decision on a single (question, gold, predicted) triple.

    `confidence` is "high" or "low": implementations set "low" when the
    judgment is close (e.g. sign ambiguity, unit interpretation, or an API
    error that fell through to a fail-closed default).
    """

    correct: bool
    reason: str = ""
    confidence: str = "high"


@runtime_checkable
class Evaluator(Protocol):
    """Single-question evaluator with a stable name for result keying.

    The `name` attribute identifies the evaluator in `EvaluationResult`
    counters and disagreement logs (e.g. "code", "judge", "ensemble").
    """

    name: str

    async def evaluate(
        self, question: str, gold: str | float, predicted: str
    ) -> EvalVerdict:
        """Return a verdict on whether predicted matches gold for question."""
        ...
