"""Reusable evaluation utilities.

Structure:
  base evaluator protocol + EvalVerdict -- evaluator.py
  metrics (is_correct, normalise_answer) -- metrics.py
  code/      -- CodeEvaluator
  llm_judge/ -- LLMJudgeEvaluator, judge_answer, JudgeVerdict
"""

from src.lib.evaluation.base import is_correct, normalise_answer

__all__ = ["is_correct", "normalise_answer"]
