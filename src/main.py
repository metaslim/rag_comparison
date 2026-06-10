"""Main Typer CLI app for ConvFinQA."""

import asyncio
import re
from pathlib import Path
from typing import Optional

import typer
from rich import print as rich_print
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.convfinqa.agent import FinancialAgent
from src.convfinqa.agent.agent import MAX_TOOL_CALLS
from src.convfinqa.dataset.loader import build_index, load_dataset
from src.convfinqa.dataset.models import ConvFinQARecord
from src.convfinqa.evaluation import run_evaluation
from src.lib.answers import RawDocStrategy, SinglePassGraphStrategy, ToolCallingStrategy
from src.lib.conversation import ConversationStore
from src.lib.evaluation.code import CodeEvaluator
from src.lib.evaluation.llm_judge import LLMJudgeEvaluator
from src.lib.graph import FinancialGraph

console = Console()

app = typer.Typer(
    name="main",
    help="ConvFinQA financial question answering agent",
    add_completion=True,
    no_args_is_help=True,
)


async def _require_neo4j(graph: FinancialGraph) -> bool:
    """Verify Neo4j is reachable. Returns False (and prints) if not."""
    if not await graph.is_reachable():
        rich_print("[red]Neo4j is not running. Start it with: docker-compose up -d[/red]")
        return False
    return True


def _load_all_records() -> list[ConvFinQARecord]:
    """Load and concatenate the train and dev splits."""
    dataset = load_dataset(settings.dataset_path)
    return dataset.get("train", []) + dataset.get("dev", [])


def _pick_record(record_id: str | None, index: dict[str, ConvFinQARecord]) -> ConvFinQARecord:
    """Resolve a record ID to a ConvFinQARecord.

    - If `record_id` is None, prints the first 20 IDs and prompts the user.
    - If the exact ID is not found, fuzzy-matches by stripping the trailing -N
      suffix and returns the first match (with a note printed).
    - Exits cleanly with code 1 if no match is found.
    """
    if record_id is None:
        rich_print("\n[bold]Available record IDs:[/bold]")
        all_ids = list(index.keys())
        for rid in all_ids[:20]:
            rich_print(f"  {rid}")
        if len(all_ids) > 20:
            rich_print(f"  ... and {len(all_ids) - 20} more")
        record_id = typer.prompt("\nEnter a record ID")

    record = index.get(record_id)
    if record is not None:
        return record

    # Fuzzy match: the user may have dropped the -N suffix.
    base = re.sub(r"-\d+$", "", record_id)
    matches = [r for rid, r in index.items() if re.sub(r"-\d+$", "", rid) == base]
    if matches:
        rich_print(f"[dim]Matched base document: using {matches[0].id}[/dim]")
        return matches[0]

    rich_print(f"[red]Record not found: {record_id}[/red]")
    raise typer.Exit(1)


async def _maybe_index_on_demand(
    graph: FinancialGraph, record: ConvFinQARecord
) -> None:
    """If the record isn't yet in the graph, prompt the user to index it."""
    if await graph.is_indexed(record.id):
        return

    rich_print("[yellow]Record not indexed in graph.[/yellow]")
    rich_print("[dim]  Y: index now (~20s), then chat using graph search[/dim]")
    rich_print("[dim]  N: chat immediately using raw document (pre_text + table + post_text), no graph[/dim]")
    choice = input("Index this record now for best results? [Y/n]: ").strip().lower()
    if choice in {"", "y", "yes"}:
        with console.status("[bold]Indexing record (~20s)...[/bold]", spinner="dots"):
            await graph.index_record(record)
        rich_print("[green]Indexed. Chatting with graph search.[/green]")
    else:
        rich_print("[dim]Chatting with raw document context (no graph search).[/dim]")


async def _chat_loop(
    agent: FinancialAgent,
    conv_id: str,
    record: ConvFinQARecord,
    strategy: "SinglePassGraphStrategy | ToolCallingStrategy | None" = None,
) -> None:
    """Interactive REPL: read user question, print answer, repeat until quit."""
    from src.lib.answers import AnswerStrategy
    _strategy: AnswerStrategy = strategy if strategy is not None else SinglePassGraphStrategy()
    mode_hint = f" ({_strategy.name})" if not isinstance(_strategy, SinglePassGraphStrategy) else ""
    rich_print(f"\n[bold]Chatting about{mode_hint}:[/bold] {record.id}")
    rich_print("Type [bold]exit[/bold] or [bold]quit[/bold] to stop\n")
    while True:
        message = input(">>> ")
        if not message.strip():
            continue
        if message.strip().lower() in {"exit", "quit"}:
            break
        try:
            with console.status("[dim]thinking...[/dim]", spinner="dots"):
                outcome = await _strategy.answer(agent, conv_id, record, message)
            if outcome.trace:
                steps = " -> ".join(t.tool_name for t in outcome.trace)
                rich_print(f"[dim]tools: {steps}[/dim]")
            if outcome.loop_cap_exhausted:
                rich_print("[yellow]tool budget exhausted; used fallback[/yellow]")
            rich_print(f"[blue][bold]assistant:[/bold] {outcome.answer}[/blue]")
        except Exception as e:
            rich_print(f"[red]Error: {e}[/red]")


@app.command()
def index(
    split: str = typer.Option(
        "dev",
        "--split",
        help="Dataset split to index: 'dev' (default, 421 records), 'train' (3037), or 'all' (3458).",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        help="Parallel indexing concurrency. Default 1 to stay safe under OpenAI rate limits.",
    ),
) -> None:
    """Build the Graphiti + Neo4j knowledge graph from the dataset.

    Defaults to the dev split (the only one used by `evaluate`). Use --split=all
    to also index train (significantly longer and costlier).
    """
    if split not in {"dev", "train", "all"}:
        rich_print(f"[red]Invalid --split: {split!r}. Choose from: dev, train, all.[/red]")
        raise typer.Exit(1)

    rich_print("[bold]Loading dataset...[/bold]")
    dataset = load_dataset(settings.dataset_path)
    if split == "all":
        records = dataset.get("train", []) + dataset.get("dev", [])
    else:
        records = dataset.get(split, [])
    rich_print(f"Indexing split=[bold]{split}[/bold]: [bold]{len(records)}[/bold] records")

    graph = FinancialGraph()

    async def _run() -> None:
        if not await _require_neo4j(graph):
            return
        await graph.build_indices()
        await graph.index_bulk(records, concurrency=concurrency)
        rich_print(f"[green]Indexed {len(records)} records (split={split})[/green]")

    asyncio.run(_run())


@app.command()
def chat(
    record_id: str = typer.Argument(default=None, help="ID of the record to chat about"),
    tools: bool = typer.Option(
        False,
        "--tools",
        help="Use the tool-calling agent path (the model can probe schema, lookup exact cells, compute mid-chain).",
    ),
) -> None:
    """Ask conversational questions about a specific financial document."""
    if not Path(settings.dataset_path).exists():
        rich_print(f"[red]Dataset not found: {settings.dataset_path}[/red]")
        raise typer.Exit(1)

    all_records = _load_all_records()
    record = _pick_record(record_id, build_index(all_records))

    graph = FinancialGraph()
    store = ConversationStore(db_path=settings.conversations_db_path)
    agent = FinancialAgent(
        graph=graph, store=store,
        model=settings.openai_model,
        query_rewriter_model=settings.query_rewriter_model,
    )
    conv_id = store.new_conversation()

    async def _run() -> None:
        if not await _require_neo4j(graph):
            return
        await _maybe_index_on_demand(graph, record)
        chat_strategy = ToolCallingStrategy() if tools else SinglePassGraphStrategy()
        await _chat_loop(agent, conv_id, record, strategy=chat_strategy)

    asyncio.run(_run())


@app.command()
def evaluate(
    record_id: str = typer.Argument(
        default=None,
        help="Optional: evaluate a single record by ID. Supports fuzzy -N suffix match. If omitted, evaluates the dev set (subject to --limit).",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Evaluate only the first N dev records (e.g. 100 to reproduce the REPORT.md figure). Ignored when record_id is given.",
    ),
    offset: Optional[int] = typer.Option(
        None,
        "--offset",
        help="Skip the first N dev records before applying --limit. e.g. --offset=20 --limit=80 gives records 21-100. Ignored when record_id is given.",
    ),
    no_index: bool = typer.Option(
        False,
        "--no-index",
        help="Skip per-record auto-indexing. Un-indexed records fall back to Mode 2 (raw document).",
    ),
    tools: bool = typer.Option(
        False,
        "--tools",
        help="Use the tool-calling agent path (slower, more API calls, better on column-ambiguity records). Opt-in until validated.",
    ),
    judge: bool = typer.Option(
        False,
        "--judge",
        help="Also run LLM-as-judge alongside the code evaluator. Logs disagreements so you can see where the two evaluators differ. Costs one extra LLM call per question.",
    ),
) -> None:
    """Run evaluation on dev records and print accuracy metrics.

    Two modes:
      uv run main evaluate                     # all 421 dev records
      uv run main evaluate --limit=100         # first 100 (reproduces REPORT.md figure)
      uv run main evaluate <record_id>         # single record only

    Runs per-record: for each record, check if indexed in the graph, index it
    if not, then evaluate all its turns, then move to the next record. Index
    and eval logs interleave so the user sees real-time progress.

    Pass --no-index to skip per-record indexing (un-indexed records will use
    the Mode 2 raw-document fallback).
    """
    rich_print("[bold]Loading dev set...[/bold]")
    dataset = load_dataset(settings.dataset_path)
    dev_records = dataset.get("dev", [])

    if record_id is not None:
        # Single-record mode: resolve via the same fuzzy matcher chat uses.
        index = build_index(dev_records)
        target = _pick_record(record_id, index)
        dev_records = [target]
    else:
        if offset is not None:
            dev_records = dev_records[offset:]
        if limit is not None:
            dev_records = dev_records[:limit]

    rich_print(f"Evaluating on [bold]{len(dev_records)}[/bold] record(s)")
    if no_index:
        rich_print("[dim]--no-index set: skipping per-record indexing (Mode 2 fallback for missing records)[/dim]")
    else:
        rich_print("[dim]Per-record flow: check indexed -> index if missing -> evaluate -> next record[/dim]")

    async def _run() -> None:
        graph = FinancialGraph()
        store = ConversationStore(db_path=settings.conversations_db_path)
        agent = FinancialAgent(
            graph=graph, store=store,
            model=settings.openai_model,
            query_rewriter_model=settings.query_rewriter_model,
        )

        if not await _require_neo4j(graph):
            return

        # Ensure schema is in place (cheap; idempotent) before any per-record indexing.
        if not no_index:
            await graph.build_indices()

        # Build the evaluator list from CLI flags. CodeEvaluator is always the
        # primary; --judge appends the LLM-as-judge secondary.
        evaluators = [CodeEvaluator()]
        if judge:
            evaluators.append(LLMJudgeEvaluator(client=agent.client, model=settings.judge_model))

        # Mode-aware output path so parallel/sequential Mode 1 vs Mode 3 runs
        # don't overwrite each other. Default `eval_results.json` is preserved
        # for unmodified callers (no --tools, no --no-index).
        base = Path(settings.eval_results_path)
        suffix = "_mode_3" if tools else "_mode_2" if no_index else "_mode_1"
        output_path = str(base.with_stem(base.stem + suffix))

        eval_strategy = ToolCallingStrategy() if tools else RawDocStrategy() if no_index else SinglePassGraphStrategy()
        result = await run_evaluation(
            agent,
            dev_records,
            output_path=output_path,
            auto_index=not no_index,
            evaluators=evaluators,
            indexer=graph,
            strategy=eval_strategy,
        )

        primary_name = evaluators[0].name

        # One accuracy table per evaluator. Primary first, then any secondaries.
        primary_table = Table(title=f"{primary_name} evaluator")
        primary_table.add_column("Metric", style="bold")
        primary_table.add_column("Accuracy", justify="right")
        primary_table.add_row("Overall", f"{result.accuracy:.1%}")
        primary_table.add_section()
        for turn in sorted(result.by_turn.keys()):
            primary_table.add_row(f"Turn {turn}", f"{result.turn_accuracy(turn):.1%}")
        rich_print(primary_table)

        for sec in evaluators[1:]:
            sec_table = Table(title=f"{sec.name} evaluator")
            sec_table.add_column("Metric", style="bold")
            sec_table.add_column("Accuracy", justify="right")
            sec_table.add_row("Overall", f"{result.secondary_accuracy(sec.name):.1%}")
            sec_table.add_section()
            for turn in sorted(result.secondary_by_turn.get(sec.name, {}).keys()):
                sec_table.add_row(
                    f"Turn {turn}", f"{result.secondary_turn_accuracy(sec.name, turn):.1%}"
                )
            rich_print(sec_table)

        rich_print(f"\n[bold]Total:[/bold] {result.correct}/{result.total} correct ({primary_name} evaluator)")
        for sec in evaluators[1:]:
            sec_correct = result.secondary_correct.get(sec.name, 0)
            sec_disagreements = sum(1 for d in result.disagreements if d.get("evaluator") == sec.name)
            rich_print(
                f"[bold]{sec.name}:[/bold] {sec_correct}/{result.total} correct "
                f"({result.secondary_accuracy(sec.name):.1%}), "
                f"{sec_disagreements} disagreement(s) with {primary_name} evaluator"
            )
        if tools:
            rich_print(
                f"[dim]Tool path: {result.loop_cap_exhausted} question(s) exhausted the "
                f"{MAX_TOOL_CALLS}-call cap.[/dim]"
            )

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    app()
