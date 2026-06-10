"""Tests for the Evaluator protocol and EvalVerdict base types."""

from unittest.mock import MagicMock

from src.lib.evaluation.base import Evaluator, EvalVerdict
from src.lib.evaluation.code import CodeEvaluator
from src.lib.evaluation.llm_judge import LLMJudgeEvaluator


def test_eval_verdict_defaults() -> None:
    """`EvalVerdict` has sensible defaults for reason and confidence."""
    v = EvalVerdict(correct=True)
    assert v.reason == ""
    assert v.confidence == "high"


def test_code_evaluator_implements_protocol() -> None:
    """CodeEvaluator must satisfy the Evaluator Protocol."""
    assert isinstance(CodeEvaluator(), Evaluator)


def test_llm_judge_evaluator_implements_protocol() -> None:
    """LLMJudgeEvaluator must satisfy the Evaluator Protocol."""
    assert isinstance(LLMJudgeEvaluator(client=MagicMock(), model="gpt-4o-mini"), Evaluator)


def test_evaluators_have_distinct_names() -> None:
    """CodeEvaluator and LLMJudgeEvaluator have distinct, stable names."""
    assert CodeEvaluator().name == "code"
    assert LLMJudgeEvaluator(client=MagicMock(), model="x").name == "judge"
