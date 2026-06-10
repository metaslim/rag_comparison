"""Tests for CodeEvaluator."""

import pytest

from src.lib.evaluation.code import CodeEvaluator


@pytest.mark.asyncio
async def test_correct_match() -> None:
    v = await CodeEvaluator().evaluate("what was net income?", 100.0, "100")
    assert v.correct is True
    assert v.confidence == "high"


@pytest.mark.asyncio
async def test_wrong_value() -> None:
    v = await CodeEvaluator().evaluate("what was net income?", 100.0, "999")
    assert v.correct is False


@pytest.mark.asyncio
async def test_seven_path_tolerance_percent_vs_decimal() -> None:
    """Delegates to the seven-path is_correct (percent-vs-decimal path)."""
    v = await CodeEvaluator().evaluate("what percent change?", 0.141, "14.1")
    assert v.correct is True
