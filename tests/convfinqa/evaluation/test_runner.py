"""Unit tests for convfinqa.evaluation: EvaluationResult + the run_evaluation loop."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.convfinqa.evaluation.runner import EvaluationResult, run_evaluation

from tests.conftest import make_record


def _record(rid: str, gold_answers: list[float], questions: list[str] | None = None):
    qs = questions or [f"q{i}" for i in range(len(gold_answers))]
    return make_record(rid=rid, answers=gold_answers, questions=qs)


def _make_agent(answers_by_question: dict[str, str], is_indexed: bool = True):
    """Build a mock agent with the minimum surface run_evaluation touches.

    The runner reads ``agent.graph`` and ``agent.store`` (public properties
    on FinancialAgent). Tests mock those names directly so the runner gets
    the AsyncMocks below, not auto-generated MagicMock attributes.
    """
    agent = MagicMock()
    agent.graph = MagicMock()
    agent.graph.is_indexed = AsyncMock(return_value=is_indexed)
    agent.graph.index_record = AsyncMock()
    agent.store = MagicMock()
    agent.store.new_conversation = MagicMock(return_value="conv-id")

    async def answer(conv_id, record, question):
        return answers_by_question.get(question, "0")

    agent.answer = AsyncMock(side_effect=answer)
    return agent


# EvaluationResult basics ----------------------------------------------------

def test_accuracy_calculation() -> None:
    result = EvaluationResult(total=10, correct=7)
    assert result.accuracy == pytest.approx(0.7)


def test_accuracy_zero_total() -> None:
    result = EvaluationResult()
    assert result.accuracy == 0.0


def test_turn_accuracy() -> None:
    result = EvaluationResult(
        by_turn={"1": {"total": 10, "correct": 9}, "2": {"total": 10, "correct": 6}}
    )
    assert result.turn_accuracy("1") == pytest.approx(0.9)
    assert result.turn_accuracy("2") == pytest.approx(0.6)


def test_unknown_turn_returns_zero() -> None:
    result = EvaluationResult()
    assert result.turn_accuracy("99") == 0.0


def test_secondary_accuracy_zero_total_returns_zero() -> None:
    """`secondary_accuracy` must avoid div-by-zero when no questions ran."""
    result = EvaluationResult()
    assert result.secondary_accuracy("judge") == 0.0


# run_evaluation -------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_evaluation_scores_correct_answers() -> None:
    """Predicted answer matching gold counts as correct."""
    record = _record("Single_R1/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"})
    result = await run_evaluation(agent, [record], auto_index=False)
    assert result.total == 1
    assert result.correct == 1
    assert result.accuracy == 1.0


@pytest.mark.asyncio
async def test_run_evaluation_scores_wrong_answers() -> None:
    """Predicted answer different from gold counts as incorrect."""
    record = _record("Single_R2/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "999"})
    result = await run_evaluation(agent, [record], auto_index=False)
    assert result.total == 1
    assert result.correct == 0


@pytest.mark.asyncio
async def test_run_evaluation_records_per_turn_breakdown() -> None:
    """Turn-position breakdown buckets turns 1/2/3 separately and groups 4+."""
    record = _record(
        "Single_R3/2009/page_1.pdf-1",
        [10.0, 20.0, 30.0, 40.0, 50.0],
        questions=["q0", "q1", "q2", "q3", "q4"],
    )
    # Get t1 + t4 right, others wrong
    agent = _make_agent({"q0": "10", "q3": "40"})
    result = await run_evaluation(agent, [record], auto_index=False)
    assert result.total == 5
    assert result.by_turn["1"] == {"total": 1, "correct": 1}
    assert result.by_turn["2"]["correct"] == 0
    assert result.by_turn["4+"]["total"] == 2  # turns 4 and 5 grouped


@pytest.mark.asyncio
async def test_run_evaluation_auto_indexes_missing_records() -> None:
    """auto_index=True triggers index_record when is_indexed returns False."""
    record = _record("Single_R4/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"}, is_indexed=False)
    await run_evaluation(agent, [record], auto_index=True)
    agent.graph.index_record.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_evaluation_skips_indexing_when_already_indexed() -> None:
    """auto_index=True is a no-op when the record is already in the graph."""
    record = _record("Single_R5/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"}, is_indexed=True)
    await run_evaluation(agent, [record], auto_index=True)
    agent.graph.index_record.assert_not_called()


@pytest.mark.asyncio
async def test_run_evaluation_continues_when_indexing_fails() -> None:
    """An indexing exception should be logged and the eval should still run."""
    record = _record("Single_R6/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"}, is_indexed=False)
    agent.graph.index_record = AsyncMock(side_effect=RuntimeError("neo4j down"))
    result = await run_evaluation(agent, [record], auto_index=True)
    # Eval still happened, scored correct
    assert result.total == 1
    assert result.correct == 1


@pytest.mark.asyncio
async def test_run_evaluation_handles_answer_exception() -> None:
    """Agent errors per turn are caught, counted as wrong, and don't halt the run."""
    record = _record("Single_R7/2009/page_1.pdf-1", [100.0, 200.0], questions=["q0", "q1"])
    agent = _make_agent({"q0": "100"})

    async def flaky_answer(conv_id, record, question):
        if question == "q1":
            raise RuntimeError("agent boom")
        return "100"

    agent.answer = AsyncMock(side_effect=flaky_answer)
    result = await run_evaluation(agent, [record], auto_index=False)
    assert result.total == 2
    assert result.correct == 1  # only q0 succeeded


@pytest.mark.asyncio
async def test_run_evaluation_writes_output_file(tmp_path) -> None:
    """Final results are persisted to output_path as JSON."""
    record = _record("Single_R8/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"})
    out = tmp_path / "eval.json"
    await run_evaluation(agent, [record], output_path=str(out), auto_index=False)
    saved = json.loads(out.read_text())
    assert saved["total"] == 1
    assert saved["correct"] == 1


@pytest.mark.asyncio
async def test_run_evaluation_routes_through_tool_path_when_tool_strategy_set() -> None:
    """ToolCallingStrategy should call agent.answer_with_tools and record trace + cap."""
    from src.convfinqa.agent.agent import ToolAnswerOutcome, ToolTrace
    from src.lib.answers import ToolCallingStrategy
    record = _record("Single_T/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"}, is_indexed=True)
    agent.answer_with_tools = AsyncMock(return_value=ToolAnswerOutcome(
        answer="100",
        trace=[ToolTrace("search_document", {"query": "net income 2009"}, {"table_cells": [100.0], "narrative_snippets": []})],
        loop_cap_exhausted=True,
    ))

    result = await run_evaluation(agent, [record], auto_index=False, strategy=ToolCallingStrategy())

    assert result.total == 1
    assert result.correct == 1
    assert result.loop_cap_exhausted == 1
    assert len(result.tool_traces) == 1
    assert result.tool_traces[0]["record_id"] == record.id
    assert result.tool_traces[0]["trace"][0]["tool_name"] == "search_document"


@pytest.mark.asyncio
async def test_run_evaluation_with_secondary_evaluator_records_agreements() -> None:
    """A secondary evaluator runs alongside the primary and its count is recorded by name."""
    from src.lib.evaluation.base import EvalVerdict
    from src.lib.evaluation.code import CodeEvaluator

    class _AlwaysCorrectEvaluator:
        name = "judge"
        async def evaluate(self, question, gold, predicted):
            return EvalVerdict(correct=True, reason="match")

    record = _record("Single_T/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"}, is_indexed=True)
    result = await run_evaluation(
        agent, [record], auto_index=False,
        evaluators=[CodeEvaluator(), _AlwaysCorrectEvaluator()],
    )

    assert result.secondary_correct["judge"] == 1
    assert result.disagreements == []
    assert result.secondary_accuracy("judge") == 1.0


@pytest.mark.asyncio
async def test_run_evaluation_logs_secondary_evaluator_disagreements() -> None:
    """When a secondary evaluator disagrees with the primary, the case is logged."""
    from src.lib.evaluation.base import EvalVerdict
    from src.lib.evaluation.code import CodeEvaluator

    class _AlwaysCorrectEvaluator:
        name = "judge"
        async def evaluate(self, question, gold, predicted):
            return EvalVerdict(correct=True, reason="semantic equivalence")

    record = _record("Single_T/2009/page_1.pdf-1", [100.0])
    # Predicted is wrong per CodeEvaluator but the secondary says correct.
    agent = _make_agent({"q0": "999"}, is_indexed=True)
    result = await run_evaluation(
        agent, [record], auto_index=False,
        evaluators=[CodeEvaluator(), _AlwaysCorrectEvaluator()],
    )

    # Primary (CodeEvaluator) says wrong (0/1); secondary "judge" says right (1/1)
    assert result.correct == 0
    assert result.secondary_correct["judge"] == 1
    assert len(result.disagreements) == 1
    d = result.disagreements[0]
    assert d["evaluator"] == "judge"
    assert d["primary_correct"] is False
    assert d["secondary_correct"] is True
    assert d["secondary_reason"] == "semantic equivalence"


@pytest.mark.asyncio
async def test_run_evaluation_tracks_secondary_per_turn_accuracy() -> None:
    """secondary_by_turn is populated and accessible via secondary_turn_accuracy()."""
    from src.lib.evaluation.base import EvalVerdict
    from src.lib.evaluation.code import CodeEvaluator

    class _Always:
        name = "judge"
        def __init__(self, correct: bool) -> None:
            self._correct = correct
        async def evaluate(self, question, gold, predicted):
            return EvalVerdict(correct=self._correct)

    # Two-turn record: judge says wrong on turn 1, right on turn 2
    record = _record("Single_T/2009/page_1.pdf-1", [100.0, 200.0], questions=["q0", "q1"])
    agent = _make_agent({"q0": "100", "q1": "200"}, is_indexed=True)

    # Custom judge that flips between turns by tracking calls
    class _FlipJudge:
        name = "judge"
        def __init__(self) -> None:
            self.calls = 0
        async def evaluate(self, question, gold, predicted):
            self.calls += 1
            return EvalVerdict(correct=(self.calls != 1))  # wrong on first call only

    flip = _FlipJudge()
    result = await run_evaluation(
        agent, [record], auto_index=False,
        evaluators=[CodeEvaluator(), flip],
    )

    # Judge per-turn: turn 1 = 0/1, turn 2 = 1/1
    assert result.secondary_by_turn["judge"]["1"] == {"total": 1, "correct": 0}
    assert result.secondary_by_turn["judge"]["2"] == {"total": 1, "correct": 1}
    assert result.secondary_turn_accuracy("judge", "1") == 0.0
    assert result.secondary_turn_accuracy("judge", "2") == 1.0


def test_secondary_turn_accuracy_unknown_returns_zero() -> None:
    """`secondary_turn_accuracy` returns 0 for unknown evaluator/turn combos."""
    result = EvaluationResult()
    assert result.secondary_turn_accuracy("judge", "99") == 0.0


@pytest.mark.asyncio
async def test_run_evaluation_default_evaluators_is_code_only() -> None:
    """When `evaluators` is omitted, only CodeEvaluator runs; secondary counters stay empty."""
    record = _record("Single_T/2009/page_1.pdf-1", [100.0])
    agent = _make_agent({"q0": "100"}, is_indexed=True)
    result = await run_evaluation(agent, [record], auto_index=False)
    assert result.correct == 1
    assert result.secondary_correct == {}
    assert result.disagreements == []


@pytest.mark.asyncio
async def test_run_evaluation_checkpoints_every_10_records(tmp_path) -> None:
    """Partial results are flushed at the 10-record boundary to survive crashes."""
    # Build 10 records (we hit the (rec_idx + 1) % 10 == 0 branch on the last one)
    records = [_record(f"Single_R{i}/2009/page_1.pdf-1", [100.0]) for i in range(10)]
    agent = _make_agent({"q0": "100"})
    out = tmp_path / "eval.json"
    result = await run_evaluation(agent, records, output_path=str(out), auto_index=False)
    assert result.total == 10
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["total"] == 10
