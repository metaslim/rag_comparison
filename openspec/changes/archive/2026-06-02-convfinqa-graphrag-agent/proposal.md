## Why

The ConvFinQA assignment requires an LLM-driven system that can answer multi-turn conversational questions over financial documents, where later questions depend on prior answers and involve numerical reasoning. The existing boilerplate returned only a hardcoded response. We needed a real agent backed by a temporal knowledge graph and persistent conversation history.

## What Changes

- **New**: Graphiti 0.29.1 + Neo4j 5.26 temporal knowledge graph. Hybrid indexing: table cells as direct triplets (no LLM, exact numbers), pre/post text as LLM-extracted episodes. Search scoped per document via `group_id`.
- **New**: SQLite-backed conversation store with bounded history formatting for LLM prompts.
- **New**: Reasoning-model agent (`gpt-5-mini`) combining hybrid graph search + conversation history + Chain-of-Thought prompting. Two retrieval modes: graph search (Mode 1) and raw-document fallback (Mode 2).
- **New**: Evaluation pipeline over the dev set with seven tolerance paths (5% relative + 1e-3 absolute, plus scale/percent/sign/inverse normalisation).
- **New**: Pydantic v2 models matching the ConvFinQA dataset schema.
- **New**: CLI per-record auto-indexing during eval, on-demand indexing in chat, resumable batch indexing.
- **Modified**: `src/main.py` with `chat`, `evaluate`, and `index` commands; `docker-compose.yml` auto-restores pre-built graph dump on first start.

## Capabilities

### New Capabilities

- `graph`: Graphiti 0.29.1 + Neo4j 5.26. Hybrid indexing (triplets + episodes). `_parse_year` for `valid_at` timestamps. Stable UUIDs for entity dedup. `record_id_to_group_id` for per-document scoping. Hybrid `COMBINED_HYBRID_SEARCH_RRF` recipe with full-chunk preservation. Checkpoint-resumable bulk indexing.
- `conversation-store`: SQLite. Save/load/format history per session. Decimal-format enforced to prevent scale drift across turns.
- `financial-agent`: `gpt-5-mini` reasoning model. Chain-of-Thought output (`Step 1: resolve refs → Step 2: locate → Step 3: compute → ANSWER:`). `simpleeval` for exact math on the final expression. Eight curated prompt rules added in response to concrete failures and regression-tested. Two retrieval modes.
- `evaluation`: Seven normalisation paths (unit scale 1B/1M/1K, multiply-by-scale, percent-vs-decimal, reverse percent, reverse-percent both-small, inverse ratio, sign flip). Per-turn (Q1 / Q2 / Q3 / Q4+) and simple-vs-hybrid breakdowns. Per-record auto-indexing flow.
- `data-models`: Pydantic v2 models for `ConvFinQARecord`, `Document`, `Dialogue`, `Features`.

### Modified Capabilities

- `cli`: `index --split {dev,train,all}` (resumable), `chat <record_id>` (with on-demand indexing prompt + fuzzy `-N` suffix matching), `evaluate --limit N --no-index` (per-record auto-index then evaluate).

## Impact

- Dependencies: `graphiti-core==0.29.1`, `neo4j` Python driver, `openai`, `chonkie` (sentence chunker), `simpleeval`, `pytest`, `pytest-asyncio`, `ruff`, `typer`, `rich`.
- Module structure:
  - `src/main.py`: CLI entry point (Typer)
  - `src/config/`: app settings loaded from `.env`
  - `src/convfinqa/`: domain layer (models, data, graph, agent, evaluation runner)
  - `src/lib/`: reusable infrastructure (conversation store, evaluation metrics, util, interfaces)
- Tests: 64 unit tests with mocked external dependencies.
- Data: `data/neo4j/data/` (Neo4j volume), `data/neo4j_dump/dev.dump` (39M, 100 pre-indexed records, committed and auto-loaded), `data/conversations.db` (SQLite), `data/index_checkpoint.json` (resumable indexing state).
- Infra: `docker-compose.yml` (Neo4j 5.26 with APOC plugin and auto-restore entrypoint).
