"""ConvFinQA evaluation runner: measures agent accuracy on the dev set."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console

from src.convfinqa.dataset.models import ConvFinQARecord
from src.lib.answers.base.strategy import Agent
from src.lib.answers import AnswerStrategy, SinglePassGraphStrategy
from src.lib.evaluation.base import Evaluator
from src.lib.evaluation.code import CodeEvaluator
from src.lib.graph.interfaces import GraphIndexer
from src.lib.util import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class EvaluationResult:
    """Aggregate results across all evaluators run on the same predictions.

    `correct` is the *primary* evaluator's count (drives the headline
    accuracy figure). Additional evaluators get keyed counts in
    `secondary_correct` and per-turn breakdowns in `secondary_by_turn`.
    Disagreements with the primary verdict are recorded in `disagreements`
    for post-hoc analysis.
    """

    total: int = 0
    correct: int = 0
    by_turn: dict[str, dict[str, int]] = field(default_factory=dict)
    # Secondary evaluators: keyed by evaluator.name -> count of correct verdicts
    secondary_correct: dict[str, int] = field(default_factory=dict)
    # Per-turn breakdown for each secondary evaluator: name -> turn -> {total, correct}
    secondary_by_turn: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    # Cases where a secondary evaluator disagreed with the primary
    disagreements: list[dict] = field(default_factory=list)
    # Tool-loop telemetry (zero when --tools not used)
    loop_cap_exhausted: int = 0
    tool_traces: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Overall accuracy from the primary evaluator."""
        return self.correct / self.total if self.total > 0 else 0.0

    def secondary_accuracy(self, name: str) -> float:
        """Accuracy from a secondary evaluator by its `name`."""
        if self.total == 0:
            return 0.0
        return self.secondary_correct.get(name, 0) / self.total

    def turn_accuracy(self, turn: str) -> float:
        """Per-turn accuracy from the primary evaluator."""
        t = self.by_turn.get(turn, {})
        total = t.get("total", 0)
        return t.get("correct", 0) / total if total > 0 else 0.0

    def secondary_turn_accuracy(self, name: str, turn: str) -> float:
        """Per-turn accuracy from a secondary evaluator."""
        t = self.secondary_by_turn.get(name, {}).get(turn, {})
        total = t.get("total", 0)
        return t.get("correct", 0) / total if total > 0 else 0.0


async def run_evaluation(
    agent: Agent,
    records: list[ConvFinQARecord],
    output_path: str | None = None,
    auto_index: bool = True,
    evaluators: list[Evaluator] | None = None,
    indexer: GraphIndexer | None = None,
    strategy: AnswerStrategy | None = None,
) -> EvaluationResult:
    """Run agent over records and return accuracy metrics.

    For each record: check indexed -> index if missing -> evaluate all turns ->
    move to next record. Interleaved flow lets the caller see real-time progress.

    ``strategy`` selects the execution path (``SinglePassGraphStrategy`` for Modes
    1/2, ``ToolCallingStrategy`` for Mode 3). Defaults to ``SinglePassGraphStrategy``.
    Adding a new mode means passing a new strategy -- no edits here required.

    ``evaluators`` controls scoring. The first evaluator is *primary* and drives
    the headline accuracy; the rest run side by side and their counts plus any
    disagreements with the primary are recorded for analysis. Defaults to a
    single ``CodeEvaluator()`` when not provided.

    ``indexer`` is the ``GraphIndexer`` used for per-record auto-indexing. When
    omitted it falls back to ``agent.graph`` (which satisfies ``GraphIndexer`` at
    runtime when the concrete ``FinancialGraph`` is injected). Pass it explicitly
    to keep the agent's declared dependency on ``GraphSearcher`` only.
    """
    if evaluators is None:
        evaluators = [CodeEvaluator()]
    primary, secondaries = evaluators[0], evaluators[1:]
    _strategy: AnswerStrategy = strategy if strategy is not None else SinglePassGraphStrategy()
    # Use the explicit indexer when provided; fall back to agent.graph which
    # satisfies GraphIndexer at runtime when FinancialGraph is injected.
    _indexer: GraphIndexer = indexer if indexer is not None else agent.graph  # type: ignore[assignment]

    result = EvaluationResult()
    # Initialise secondary counters to 0 so every secondary appears in the
    # result even when it never marks anything correct (simpler downstream).
    for sec in secondaries:
        result.secondary_correct.setdefault(sec.name, 0)
    total_records = len(records)

    for rec_idx, record in enumerate(records):
        # Per-record auto-index: index this record if it isn't in the graph yet
        if auto_index:
            if not await agent.graph.is_indexed(record.id):
                msg = f"[{rec_idx + 1}/{total_records}] Indexing {record.id} (~20-30s)..."
                try:
                    with console.status(f"[dim]{msg}[/dim]", spinner="dots"):
                        await _indexer.index_record(record)
                except Exception as e:
                    logger.warning(
                        "Indexing failed for %s, falling back to Mode 2: %s", record.id, e
                    )
            else:
                logger.debug(
                    "[%d/%d] Already indexed: %s", rec_idx + 1, total_records, record.id
                )

        conv_id = agent.store.new_conversation()

        for turn_idx, (question, gold) in enumerate(
            zip(record.dialogue.conv_questions, record.dialogue.executed_answers, strict=True)
        ):
            turn_key = str(turn_idx + 1) if turn_idx < 3 else "4+"
            tool_trace_summary = ""  # populated only on the tool path
            try:
                spinner_msg = (
                    f"[dim][{rec_idx + 1}/{total_records}] T{turn_idx + 1}: "
                    f"{question[:60]}...[/dim]"
                )
                with console.status(spinner_msg, spinner="dots"):
                    outcome = await _strategy.answer(agent, conv_id, record, question)
                    predicted = outcome.answer
                    if outcome.loop_cap_exhausted:
                        result.loop_cap_exhausted += 1
                    if outcome.trace:
                        tool_trace_summary = " | tools=" + " -> ".join(
                            t.tool_name for t in outcome.trace
                        )
                        result.tool_traces.append({
                            "record_id": record.id,
                            "turn": turn_idx + 1,
                            "question": question,
                            "trace": [
                                {"tool_name": t.tool_name, "args": t.args, "result": t.result}
                                for t in outcome.trace
                            ],
                            "fell_back_to_single_pass": outcome.fell_back_to_single_pass,
                        })
            except Exception as e:
                logger.warning("Error on record %s turn %d: %s", record.id, turn_idx + 1, e)
                predicted = "error"

            # Run every evaluator: primary drives headline accuracy,
            # secondaries get their own counters + per-turn breakdowns.
            primary_verdict = await primary.evaluate(question, gold, predicted)
            sec_verdicts = {
                sec.name: await sec.evaluate(question, gold, predicted)
                for sec in secondaries
            }

            # Build a verdict tag that fits every evaluator into one log line.
            def _mark(name: str, ok: bool) -> str:
                tag = "✅" if ok else "❌"
                return f"{tag}{name}"

            verdict_str = _mark(primary.name, primary_verdict.correct)
            for sec in secondaries:
                verdict_str += " " + _mark(sec.name, sec_verdicts[sec.name].correct)

            logger.info(
                "%s [%s] T%d: Q=%s | gold=%s | got=%s%s",
                verdict_str, record.id, turn_idx + 1,
                question[:50], gold, predicted, tool_trace_summary,
            )

            # Aggregate secondary counts + per-turn breakdowns; log disagreements.
            for sec in secondaries:
                sec_verdict = sec_verdicts[sec.name]
                if sec_verdict.correct:
                    result.secondary_correct[sec.name] += 1
                result.secondary_by_turn.setdefault(sec.name, {})
                result.secondary_by_turn[sec.name].setdefault(turn_key, {"total": 0, "correct": 0})
                result.secondary_by_turn[sec.name][turn_key]["total"] += 1
                if sec_verdict.correct:
                    result.secondary_by_turn[sec.name][turn_key]["correct"] += 1
                if sec_verdict.correct != primary_verdict.correct:
                    result.disagreements.append({
                        "evaluator": sec.name,
                        "record_id": record.id,
                        "turn": turn_idx + 1,
                        "question": question,
                        "gold": gold,
                        "predicted": predicted,
                        "primary_correct": primary_verdict.correct,
                        "secondary_correct": sec_verdict.correct,
                        "secondary_reason": sec_verdict.reason,
                    })

            result.total += 1
            if primary_verdict.correct:
                result.correct += 1

            result.by_turn.setdefault(turn_key, {"total": 0, "correct": 0})
            result.by_turn[turn_key]["total"] += 1
            if primary_verdict.correct:
                result.by_turn[turn_key]["correct"] += 1

            logger.debug(
                "[%d/%d] %s T%d: predicted=%s gold=%s correct=%s",
                rec_idx + 1, total_records, record.id[:40], turn_idx + 1,
                predicted, gold, primary_verdict.correct,
            )

        if (rec_idx + 1) % 10 == 0:
            logger.info("Progress: %d/%d, accuracy: %.1f%%", rec_idx + 1, total_records, result.accuracy * 100)
            # Checkpoint: save partial results so crash doesn't lose all progress
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text(json.dumps(asdict(result), indent=2))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(asdict(result), indent=2))
        logger.info("Results saved to %s", output_path)

    return result
