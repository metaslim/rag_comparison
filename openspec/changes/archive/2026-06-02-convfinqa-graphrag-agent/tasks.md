# Tasks (final status)

Headline: **231 unit tests, 100% coverage. Graph-RAG 84.2%, Long Context 82.5% (code evaluator).**

## Final architecture (post-refactoring)

Key path and structural changes from original spec:

- `src/convfinqa/models.py` → `src/convfinqa/dataset/models.py`
- `src/convfinqa/data.py` → `src/convfinqa/dataset/loader.py`
- `src/convfinqa/graph/` → `src/lib/graph/` (graph.py split into `search.py` + `index.py` for SRP)
- `src/lib/interfaces.py` → split: `lib/graph/interfaces.py` (GraphSearcher/GraphIndexer/GraphStore/IndexableRecord), `lib/conversation/interfaces.py` (HistoryStore)
- Modes implemented as `AnswerStrategy` protocol: `lib/answers/single_pass_graph/`, `lib/answers/raw_doc/`
- `MAX_CONTEXT_CHARS` removed; full document passed verbatim in Mode 2
- `_next_turn` uses `MAX(turn)` not `COUNT(*)` (bug fix)
- Sequential triplet inserts (no semaphore) to eliminate write-ordering race condition
- `_FinancialGraphiti(Graphiti)` subclass wraps private `_search` API as stable `search_with_config()`
- `graph.is_reachable()` public method; CLI no longer accesses `_graphiti.driver` directly

## 1. Dependencies & Config

- [x] 1.1 Add `graphiti-core==0.29.1`, `openai`, `chonkie`, `simpleeval`, `pytest`, `pytest-asyncio`, `ruff`, `typer`, `rich`.
- [x] 1.2 Update `openspec/config.yaml` with project context.

## 2. Configuration

- [x] 2.1 `src/config/settings.py` with `Settings` class and `_require()` for clear missing-key errors.
- [x] 2.2 Neo4j config: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- [x] 2.3 All modules import via `from src.config import settings`.

## 3. Data Models & Loading

- [x] 3.1 `src/convfinqa/dataset/models.py`: Pydantic v2 models (`ConvFinQARecord`, `Document`, `Dialogue`, `Features`).
- [x] 3.2 `src/convfinqa/dataset/loader.py`: `load_dataset`, `build_index` (O(1) lookup).
- [x] 3.3 Tests passing.

## 4. Graphiti + Neo4j Knowledge Graph

- [x] 4.1 `src/lib/graph/graph.py`: `FinancialGraph` facade; `src/lib/graph/search.py` (`_GraphSearcherImpl`); `src/lib/graph/index.py` (`_GraphIndexerImpl`).
- [x] 4.2 Sequential triplet inserts per cell (`valid_at` from `_parse_year()`). Removed semaphore to eliminate write-ordering race.
- [x] 4.3 `add_episode_bulk`: pre_text + post_text LLM extraction, sentence-chunked.
- [x] 4.4 `is_indexed(record_id)`: takes record_id, converts to group_id internally. Persistent idempotency via Neo4j query.
- [x] 4.5 `record_id_to_group_id`: strips `-N` suffix, deduplicates shared PDFs.
- [x] 4.6 `_parse_year` with edge cases tested.
- [x] 4.7 `index_bulk` with checkpoint resume and concurrency knob.
- [x] 4.8 Stable UUIDs (md5 of `group_id::name`) for entity dedup.
- [x] 4.9 `_FinancialGraphiti(Graphiti)` subclass: `search_with_config()` wraps private `_search` API.
- [x] 4.10 `is_reachable()` public method on `FinancialGraph`.
- [x] 4.11 `lib/graph/interfaces.py`: `GraphSearcher`, `GraphIndexer`, `GraphStore`, `IndexableRecord`, `IndexableDocument` protocols.
- [x] 4.12 Tests passing.

## 5. Conversation Store (SQLite)

- [x] 5.1 `src/lib/conversation/store.py`: `ConversationStore` implementing `HistoryStore` protocol.
- [x] 5.2 `lib/conversation/interfaces.py`: `HistoryStore` protocol.
- [x] 5.3 `_next_turn` uses `MAX(turn)` not `COUNT(*)`.
- [x] 5.4 `format_history(max_turns=10)` bounded to prevent unbounded context growth.
- [x] 5.5 Tests passing.

## 6. Financial Agent

- [x] 6.1 `src/convfinqa/agent/agent.py`: `FinancialAgent` depends on `GraphSearcher` + `HistoryStore` protocols.
- [x] 6.2 `ContextBuilder` extracted (SRP). `_resolve_answer`, `_rewrite_query` extracted as module-level functions.
- [x] 6.3 `agent.client` property exposed (DIP fix).
- [x] 6.4 `answer_raw_doc()` method for explicit Mode 2 path.
- [x] 6.5 `_run_single_pass()` shared by `answer()` and `answer_raw_doc()`.
- [x] 6.6 Fallback path passes `TOOL_SYSTEM_PROMPT` override to persist correct context.
- [x] 6.7 `_NO_TEMPERATURE_PREFIXES` tuple replaces fragile string-contains temperature gate.
- [x] 6.8 `src/convfinqa/agent/prompts.py`: `SYSTEM_PROMPT`. `MAX_CONTEXT_CHARS` removed; Mode 2 passes full document.
- [x] 6.9 Tests passing (100% coverage).

## 7. Answer Strategies (OCP)

- [x] 7.1 `src/lib/answers/base/`: `AnswerResult`, `AnswerStrategy`, `Agent`, `IndexedRecord`, `ToolOutcome`, `TraceEntry` protocols.
- [x] 7.2 `src/lib/answers/single_pass_graph/`: `SinglePassGraphStrategy` (Mode 1: graph-RAG).
- [x] 7.3 `src/lib/answers/raw_doc/`: `RawDocStrategy` (Mode 2: full raw document).
- [x] 7.4 Runner and CLI depend on `AnswerStrategy` protocol; no `if use_tools:` branching.
- [x] 7.5 Tests passing.

## 8. Evaluation

- [x] 8.1 `src/lib/evaluation/base/metrics.py`: `normalise_answer`, `is_correct`. `sign_ok` removed; sign conventions delegated to LLM judge.
- [x] 8.2 `scaled` flag prevents compound rescaling across tolerance paths.
- [x] 8.3 `src/lib/evaluation/code/`: `CodeEvaluator`.
- [x] 8.4 `src/convfinqa/evaluation/runner.py`: `run_evaluation` with `AnswerStrategy` + `GraphIndexer` params.
- [x] 8.5 Per-record auto-indexing flow. JSON checkpoint at `data/eval_results_mode_N.json` via `Path.with_stem()`.
- [x] 8.6 Tests passing.

## 9. CLI

- [x] 9.1 `index --split {dev,train,all}`: split-aware, resumable.
- [x] 9.2 `chat <record_id>`: fuzzy `-N` suffix matching, on-demand indexing prompt, strategy-based dispatch.
- [x] 9.3 `evaluate --limit N --no-index --tools --judge`.
- [x] 9.4 `graph.is_reachable()` used for Neo4j health check (no private attribute access).

## 10. Infrastructure

- [x] 10.1 `docker-compose.yml`: Neo4j 5.26. Auto-restores `dev.dump` on first start.
- [x] 10.2 `data/neo4j_dump/dev.dump`: 39 MB pre-built graph (100 dev records).
- [x] 10.3 `lib/util/logger.py`: Graphiti internal warnings suppressed via `logging.getLogger("graphiti_core").setLevel(ERROR)`.

## 11. Code Quality

- [x] 11.1 SOLID: ISP (GraphSearcher/GraphIndexer split), SRP (FinancialAgent concerns extracted, FinancialGraph split to search.py + index.py), OCP (AnswerStrategy), DIP (protocols throughout), DRY (_run_search extracted).
- [x] 11.2 No `Any` in lib/ interfaces: proper protocols everywhere.
- [x] 11.3 231 unit tests, 100% line coverage, `ruff` clean.
- [x] 11.4 Test folder mirrors src/ structure exactly.

## 12. Documentation

- [x] 12.1 `REPORT.md`: method, key findings, error analysis, future work. Engineer voice throughout.
- [x] 12.2 `README.md`: setup, three modes, architecture, troubleshooting.
- [x] 12.3 `docs/hybrid_graph_schema.md`: annotated Neo4j graph example.
