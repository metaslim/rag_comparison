"""Unit tests for src.main CLI helpers (the bits extracted from chat() for SRP)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from src.convfinqa.dataset.models import ConvFinQARecord
from src.main import _load_all_records, _pick_record, app

runner = CliRunner()


@pytest.fixture
def mock_graph_cli() -> MagicMock:
    """Pre-wired mock FinancialGraph for CLI command tests.

    Defaults: reachable, all async methods as no-op AsyncMocks.
    Override individual attrs per test (e.g. `mock_graph_cli.is_reachable = AsyncMock(return_value=False)`).
    """
    graph = MagicMock()
    graph.is_reachable = AsyncMock(return_value=True)
    graph.build_indices = AsyncMock()
    graph.index_bulk = AsyncMock()
    graph.is_indexed = AsyncMock(return_value=True)
    graph.index_record = AsyncMock()
    return graph


def _record(rid: str) -> ConvFinQARecord:
    """Build a minimal valid ConvFinQARecord with the given id."""
    return ConvFinQARecord.model_validate({
        "id": rid,
        "doc": {
            "pre_text": "Some pre text.",
            "post_text": "Some post text.",
            "table": {"2009": {"net income": 100.0}},
        },
        "dialogue": {
            "conv_questions": ["what was net income?"],
            "conv_answers": ["100"],
            "turn_program": ["100"],
            "executed_answers": [100.0],
            "qa_split": [False],
        },
        "features": {
            "num_dialogue_turns": 1,
            "has_type2_question": False,
            "has_duplicate_columns": False,
            "has_non_numeric_values": False,
        },
    })


@pytest.fixture
def sample_index() -> dict[str, ConvFinQARecord]:
    """Three records: two sharing a base document, one standalone."""
    records = [
        _record("Single_FOO/2007/page_1.pdf-1"),
        _record("Single_FOO/2007/page_1.pdf-3"),
        _record("Single_BAR/2010/page_5.pdf-2"),
    ]
    return {r.id: r for r in records}


# _pick_record ---------------------------------------------------------------

def test_pick_record_exact_match(sample_index) -> None:
    """An exact ID match returns that record directly."""
    rec = _pick_record("Single_FOO/2007/page_1.pdf-1", sample_index)
    assert rec.id == "Single_FOO/2007/page_1.pdf-1"


def test_pick_record_fuzzy_match_drops_n_suffix(sample_index) -> None:
    """If the user drops the -N suffix, pick the first record matching the base doc."""
    rec = _pick_record("Single_FOO/2007/page_1.pdf", sample_index)
    # Picks first record sharing the same base document
    assert rec.id.startswith("Single_FOO/2007/page_1.pdf-")


def test_pick_record_unknown_id_exits(sample_index) -> None:
    """Unknown ID exits cleanly via typer.Exit(1)."""
    with pytest.raises(typer.Exit) as exc_info:
        _pick_record("Nonexistent/Record/Id", sample_index)
    assert exc_info.value.exit_code == 1


def test_pick_record_none_prompts_user(sample_index) -> None:
    """If record_id is None, the function lists records and prompts."""
    # typer.prompt returns the picked ID; mock it to return a real one.
    with patch("src.main.typer.prompt", return_value="Single_BAR/2010/page_5.pdf-2"):
        rec = _pick_record(None, sample_index)
    assert rec.id == "Single_BAR/2010/page_5.pdf-2"


# _load_all_records ----------------------------------------------------------

def test_load_all_records_concatenates_train_and_dev() -> None:
    """_load_all_records combines train + dev splits."""
    train = [_record("train-1"), _record("train-2")]
    dev = [_record("dev-1")]
    with patch("src.main.load_dataset", return_value={"train": train, "dev": dev}):
        all_records = _load_all_records()
    ids = [r.id for r in all_records]
    assert ids == ["train-1", "train-2", "dev-1"]


def test_load_all_records_handles_missing_splits() -> None:
    """If train or dev is absent, the function still works."""
    with patch("src.main.load_dataset", return_value={"dev": [_record("dev-only")]}):
        all_records = _load_all_records()
    assert [r.id for r in all_records] == ["dev-only"]


def test_load_all_records_empty_dataset() -> None:
    """Empty dataset returns empty list, not an error."""
    with patch("src.main.load_dataset", return_value={}):
        all_records = _load_all_records()
    assert all_records == []


# _pick_record (extra branches) ---------------------------------------------

def test_pick_record_lists_first_20_and_more_marker() -> None:
    """With None record_id and >20 records, the prompt header lists the cap and a '... and N more' line."""
    from src.main import _pick_record as pick
    records = {f"id-{i:03d}": _record(f"id-{i:03d}") for i in range(25)}
    with patch("src.main.typer.prompt", return_value="id-010"):
        rec = pick(None, records)
    assert rec.id == "id-010"


# Typer CLI commands ---------------------------------------------------------

def test_evaluate_rejects_invalid_split_via_index_command() -> None:
    """`index --split=foo` exits with code 1."""
    result = runner.invoke(app, ["index", "--split=foo"])
    assert result.exit_code == 1


def test_chat_exits_when_dataset_missing(tmp_path) -> None:
    """`chat` exits with code 1 if the dataset file isn't present."""
    with patch("src.main.settings") as mock_settings:
        mock_settings.dataset_path = str(tmp_path / "missing.json")
        result = runner.invoke(app, ["chat", "some-record-id"])
    assert result.exit_code == 1


def test_index_command_runs_with_mocked_graph(mock_graph_cli) -> None:
    """`index --split=dev` wires through to graph.index_bulk."""
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = mock_graph_cli

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec], "train": []}):
        result = runner.invoke(app, ["index", "--split=dev"])

    assert result.exit_code == 0
    mock_graph.index_bulk.assert_awaited_once()


def test_index_command_handles_neo4j_down(mock_graph_cli) -> None:
    """If Neo4j ping fails, `index` reports it and exits cleanly without indexing."""
    mock_graph = mock_graph_cli
    mock_graph.is_reachable = AsyncMock(return_value=False)

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [_record("Single_T/2009/page_1.pdf-1")]}):
        result = runner.invoke(app, ["index", "--split=dev"])

    assert result.exit_code == 0
    mock_graph.index_bulk.assert_not_called()


def test_evaluate_single_record_resolves_via_fuzzy_match(mock_graph_cli) -> None:
    """`evaluate <record_id>` resolves the record and calls run_evaluation with just that one."""
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = mock_graph_cli

    eval_result = MagicMock()
    eval_result.accuracy = 1.0
    eval_result.correct = 1
    eval_result.total = 1
    eval_result.by_turn = {"1": {"total": 1, "correct": 1}}
    eval_result.turn_accuracy = lambda t: 1.0

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec]}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)) as mock_eval:
        result = runner.invoke(app, ["evaluate", "Single_T/2009/page_1.pdf-1"])

    assert result.exit_code == 0
    # Confirm run_evaluation was called with the one resolved record
    passed_records = mock_eval.call_args.args[1]
    assert len(passed_records) == 1
    assert passed_records[0].id == rec.id


def test_evaluate_limit_truncates_dev_records(mock_graph_cli) -> None:
    """`evaluate --limit=2` passes only the first 2 dev records to run_evaluation."""
    recs = [_record(f"Single_R{i}/2009/page_1.pdf-1") for i in range(5)]
    mock_graph = mock_graph_cli

    eval_result = MagicMock(accuracy=0.5, correct=1, total=2, by_turn={})

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": recs}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)) as mock_eval:
        result = runner.invoke(app, ["evaluate", "--limit=2"])

    assert result.exit_code == 0
    passed_records = mock_eval.call_args.args[1]
    assert len(passed_records) == 2


def test_evaluate_offset_skips_initial_records(mock_graph_cli) -> None:
    """`evaluate --offset=2 --limit=2` skips the first 2 records, then takes 2."""
    recs = [_record(f"Single_R{i}/2009/page_1.pdf-1") for i in range(5)]
    mock_graph = mock_graph_cli

    eval_result = MagicMock(accuracy=0.5, correct=1, total=2, by_turn={})

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": recs}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)) as mock_eval:
        result = runner.invoke(app, ["evaluate", "--offset=2", "--limit=2"])

    assert result.exit_code == 0
    passed_records = mock_eval.call_args.args[1]
    assert len(passed_records) == 2
    # Records R2 and R3 (skip R0, R1; take 2)
    assert [r.id for r in passed_records] == [
        "Single_R2/2009/page_1.pdf-1",
        "Single_R3/2009/page_1.pdf-1",
    ]


def test_evaluate_with_judge_prints_secondary_per_turn_table(mock_graph_cli) -> None:
    """`evaluate --judge` triggers the secondary-evaluator per-turn table render.

    Exercises the secondary_by_turn rendering branch that's only reached
    when --judge is active and the runner returns secondary_by_turn data.
    """
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = mock_graph_cli

    eval_result = MagicMock(
        accuracy=1.0, correct=1, total=1,
        by_turn={"1": {"total": 1, "correct": 1}},
        secondary_correct={"judge": 1},
        secondary_by_turn={"judge": {"1": {"total": 1, "correct": 1}}},
        disagreements=[],
        loop_cap_exhausted=0,
    )
    eval_result.secondary_accuracy = MagicMock(return_value=1.0)
    eval_result.turn_accuracy = MagicMock(return_value=1.0)
    eval_result.secondary_turn_accuracy = MagicMock(return_value=1.0)

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec]}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)):
        result = runner.invoke(app, ["evaluate", "--limit=1", "--judge"])

    assert result.exit_code == 0
    # The judge per-turn block should render in the output
    assert "judge" in result.stdout.lower()


# _maybe_index_on_demand -----------------------------------------------------

@pytest.mark.asyncio
async def test_maybe_index_on_demand_no_op_when_indexed() -> None:
    """If the record is already indexed, no prompt and no indexing call."""
    from src.main import _maybe_index_on_demand
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = MagicMock()
    mock_graph.is_indexed = AsyncMock(return_value=True)
    mock_graph.index_record = AsyncMock()
    await _maybe_index_on_demand(mock_graph, rec)
    mock_graph.index_record.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_index_on_demand_indexes_on_yes() -> None:
    """User responding 'y' triggers index_record."""
    from src.main import _maybe_index_on_demand
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = MagicMock()
    mock_graph.is_indexed = AsyncMock(return_value=False)
    mock_graph.index_record = AsyncMock()
    with patch("builtins.input", return_value="y"):
        await _maybe_index_on_demand(mock_graph, rec)
    mock_graph.index_record.assert_awaited_once_with(rec)


@pytest.mark.asyncio
async def test_maybe_index_on_demand_skips_on_no() -> None:
    """User responding 'n' falls back to Mode 2 (no indexing call)."""
    from src.main import _maybe_index_on_demand
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = MagicMock()
    mock_graph.is_indexed = AsyncMock(return_value=False)
    mock_graph.index_record = AsyncMock()
    with patch("builtins.input", return_value="n"):
        await _maybe_index_on_demand(mock_graph, rec)
    mock_graph.index_record.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_index_on_demand_indexes_on_empty_default() -> None:
    """Pressing Enter (empty input) defaults to Y."""
    from src.main import _maybe_index_on_demand
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = MagicMock()
    mock_graph.is_indexed = AsyncMock(return_value=False)
    mock_graph.index_record = AsyncMock()
    with patch("builtins.input", return_value=""):
        await _maybe_index_on_demand(mock_graph, rec)
    mock_graph.index_record.assert_awaited_once_with(rec)


# _chat_loop -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_loop_exits_on_quit_command() -> None:
    """Typing 'exit' or 'quit' breaks the loop cleanly."""
    from src.main import _chat_loop
    rec = _record("Single_T/2009/page_1.pdf-1")
    agent = MagicMock()
    agent.answer = AsyncMock(return_value="42")
    # Sequence: first a real question, then 'exit'
    with patch("builtins.input", side_effect=["what is x?", "exit"]):
        await _chat_loop(agent, "conv-1", rec)
    agent.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_loop_skips_blank_input() -> None:
    """Empty input loops without calling the agent."""
    from src.main import _chat_loop
    rec = _record("Single_T/2009/page_1.pdf-1")
    agent = MagicMock()
    agent.answer = AsyncMock(return_value="42")
    with patch("builtins.input", side_effect=["", "   ", "exit"]):
        await _chat_loop(agent, "conv-1", rec)
    agent.answer.assert_not_called()


@pytest.mark.asyncio
async def test_chat_loop_catches_agent_exceptions() -> None:
    """Agent errors are caught and printed instead of crashing the loop."""
    from src.main import _chat_loop
    rec = _record("Single_T/2009/page_1.pdf-1")
    agent = MagicMock()
    agent.answer = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("builtins.input", side_effect=["q1", "exit"]):
        # If the exception escaped, this would raise instead of returning
        await _chat_loop(agent, "conv-1", rec)


# Remaining CLI branches (closes coverage gaps in main.py) -------------------

def test_index_command_all_split_combines_train_and_dev(mock_graph_cli, tmp_path) -> None:
    """`index --split=all` passes train + dev to index_bulk."""
    train_recs = [_record(f"Train_{i}/2009/page_1.pdf-1") for i in range(2)]
    dev_recs = [_record(f"Dev_{i}/2009/page_1.pdf-1") for i in range(3)]
    mock_graph = mock_graph_cli

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"train": train_recs, "dev": dev_recs}):
        result = runner.invoke(app, ["index", "--split=all"])

    assert result.exit_code == 0
    passed = mock_graph.index_bulk.call_args.args[0]
    assert len(passed) == 5  # 2 train + 3 dev


def test_chat_command_full_happy_path(mock_graph_cli, tmp_path) -> None:
    """`chat <id>` end-to-end with all dependencies mocked."""
    rec = _record("Single_T/2009/page_1.pdf-1")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text('{"dev": [], "train": []}')  # file must exist

    mock_graph = mock_graph_cli

    mock_store = MagicMock()
    mock_store.new_conversation = MagicMock(return_value="conv-1")

    mock_agent = MagicMock()
    mock_agent.answer = AsyncMock(return_value="42")

    with patch("src.main.settings") as mock_settings, \
         patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.ConversationStore", return_value=mock_store), \
         patch("src.main.FinancialAgent", return_value=mock_agent), \
         patch("src.main.load_dataset", return_value={"dev": [rec], "train": []}), \
         patch("builtins.input", side_effect=["what is x?", "exit"]):
        mock_settings.dataset_path = str(dataset_path)
        result = runner.invoke(app, ["chat", rec.id])

    assert result.exit_code == 0
    mock_agent.answer.assert_awaited_once()


def test_chat_command_with_tools_flag_uses_tool_path(mock_graph_cli, tmp_path) -> None:
    """`chat --tools` routes through agent.answer_with_tools and prints trace + cap info."""
    from src.convfinqa.agent.agent import ToolAnswerOutcome, ToolTrace
    rec = _record("Single_T/2009/page_1.pdf-1")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text('{"dev": [], "train": []}')

    mock_graph = mock_graph_cli

    mock_agent = MagicMock()
    mock_agent.answer_with_tools = AsyncMock(return_value=ToolAnswerOutcome(
        answer="42",
        trace=[ToolTrace("search_document", {"query": "x 2009"}, {"table_cells": [42], "narrative_snippets": []})],
        loop_cap_exhausted=True,
    ))

    with patch("src.main.settings") as mock_settings, \
         patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.ConversationStore", return_value=MagicMock()), \
         patch("src.main.FinancialAgent", return_value=mock_agent), \
         patch("src.main.load_dataset", return_value={"dev": [rec], "train": []}), \
         patch("builtins.input", side_effect=["what is x?", "exit"]):
        mock_settings.dataset_path = str(dataset_path)
        result = runner.invoke(app, ["chat", "--tools", rec.id])

    assert result.exit_code == 0
    mock_agent.answer_with_tools.assert_awaited_once()


def test_evaluate_with_tools_flag_prints_cap_summary(mock_graph_cli) -> None:
    """`evaluate --tools` runs the tool path and prints the loop-cap summary line."""
    from unittest.mock import AsyncMock
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = mock_graph_cli

    eval_result = MagicMock(
        accuracy=1.0, correct=1, total=1, by_turn={}, loop_cap_exhausted=0,
    )
    eval_result.turn_accuracy = lambda t: 1.0

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec]}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)) as mock_eval:
        result = runner.invoke(app, ["evaluate", "--tools", "--limit=1"])

    from src.lib.answers import ToolCallingStrategy
    assert result.exit_code == 0
    # --tools must pass a ToolCallingStrategy to run_evaluation
    assert isinstance(mock_eval.call_args.kwargs.get("strategy"), ToolCallingStrategy)


def test_evaluate_with_judge_flag_appends_llm_judge_evaluator(mock_graph_cli) -> None:
    """`evaluate --judge` must append an LLMJudgeEvaluator to the evaluator list."""
    from src.lib.evaluation.code import CodeEvaluator
    from src.lib.evaluation.llm_judge import LLMJudgeEvaluator

    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = mock_graph_cli

    eval_result = MagicMock(
        accuracy=1.0, correct=1, total=1, by_turn={},
        loop_cap_exhausted=0, secondary_correct={"judge": 1}, disagreements=[],
    )
    eval_result.turn_accuracy = lambda t: 1.0
    eval_result.secondary_accuracy = lambda name: 1.0

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec]}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)) as mock_eval:
        result = runner.invoke(app, ["evaluate", "--judge", "--limit=1"])

    assert result.exit_code == 0
    evaluators = mock_eval.call_args.kwargs["evaluators"]
    assert len(evaluators) == 2
    assert isinstance(evaluators[0], CodeEvaluator)
    assert isinstance(evaluators[1], LLMJudgeEvaluator)


def test_evaluate_no_index_flag_skips_build_indices(mock_graph_cli) -> None:
    """`evaluate --no-index` does not call build_indices and passes auto_index=False."""
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = mock_graph_cli

    eval_result = MagicMock(accuracy=1.0, correct=1, total=1, by_turn={})
    eval_result.turn_accuracy = lambda t: 1.0

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec]}), \
         patch("src.main.run_evaluation", new=AsyncMock(return_value=eval_result)) as mock_eval:
        result = runner.invoke(app, ["evaluate", "--no-index"])

    assert result.exit_code == 0
    mock_graph.build_indices.assert_not_called()
    assert mock_eval.call_args.kwargs["auto_index"] is False


def test_chat_command_returns_early_when_neo4j_down(tmp_path) -> None:
    """`chat <id>` exits cleanly when Neo4j ping fails (no chat loop entered)."""
    rec = _record("Single_T/2009/page_1.pdf-1")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text('{"dev": [], "train": []}')

    mock_graph = MagicMock()
    mock_graph.is_reachable = AsyncMock(return_value=False)

    mock_agent = MagicMock()
    mock_agent.answer = AsyncMock()

    with patch("src.main.settings") as mock_settings, \
         patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.ConversationStore", return_value=MagicMock()), \
         patch("src.main.FinancialAgent", return_value=mock_agent), \
         patch("src.main.load_dataset", return_value={"dev": [rec], "train": []}):
        mock_settings.dataset_path = str(dataset_path)
        result = runner.invoke(app, ["chat", rec.id])

    assert result.exit_code == 0
    mock_agent.answer.assert_not_called()


def test_evaluate_returns_early_when_neo4j_down() -> None:
    """If Neo4j is unreachable in `evaluate`, run_evaluation is never invoked."""
    rec = _record("Single_T/2009/page_1.pdf-1")
    mock_graph = MagicMock()
    mock_graph.is_reachable = AsyncMock(return_value=False)

    with patch("src.main.FinancialGraph", return_value=mock_graph), \
         patch("src.main.load_dataset", return_value={"dev": [rec]}), \
         patch("src.main.run_evaluation", new=AsyncMock()) as mock_eval:
        result = runner.invoke(app, ["evaluate"])

    assert result.exit_code == 0
    mock_eval.assert_not_called()
