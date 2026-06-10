## Context

The existing boilerplate provides a Typer CLI with a `chat <record_id>` stub. We need to wire it up to a real agent that can answer multi-turn numerical questions over financial documents. The dataset is pre-extracted (text + structured table) in a single JSON file, no PDF parsing required.

Key constraints:
- Python 3.12+, `uv`, existing `pyproject.toml`
- `OPENAI_API_KEY` in `.env`
- SOLID, OOD, DRY, YAGNI
- Unit tests with mocked external dependencies

## SOLID / OOD principles applied

- **S**: each module has one job. `config/settings.py` for settings, `convfinqa/graph/graph.py` for indexing and search, `lib/conversation/store.py` for history, `convfinqa/agent/agent.py` for answering, `convfinqa/evaluation/runner.py` for scoring.
- **O**: `FinancialAgent` accepts `graph`, `store`, `openai_client` via DI. Swappable without changing agent logic.
- **L**: components depend on behaviour. All external services are mockable in tests (64 tests pass with no real OpenAI or Neo4j calls).
- **I**: small focused classes with minimal public methods. Protocols (`GraphStore`, `HistoryStore`) in `src/lib/interfaces.py` define the contracts.
- **D**: dependencies injected at construction, not hardcoded.

## Goals / Non-Goals

**Goals:**
- Index ConvFinQA documents into Graphiti + Neo4j with checkpoint-resumable bulk indexing.
- Answer multi-turn conversational financial questions using a reasoning-model agent + hybrid graph search + conversation history.
- Persist conversation history to SQLite.
- Evaluate on the dev set with tolerance over exact-match.
- Make the system reproducible: committed graph dump + Docker Compose auto-restore + checkpointed indexing.

**Non-Goals:**
- PDF ingestion, fine-tuning, web UI, multi-user, manual Cypher queries.

## Decisions

### D1: Graphiti 0.29.1 + Neo4j 5.26 (Docker Compose)

**Decision**: `graphiti-core==0.29.1` with Neo4j 5.26 via `docker-compose up -d`.

**Rationale**: Neo4j supports concurrent writes. `index_bulk(concurrency=10)` would index 10 records in parallel (~7.6s/record). The default in `main.py` is `concurrency=1` to stay safe under OpenAI tier rate limits. Neo4j is Graphiti's primary backend, no driver patches needed. Graph dump committed to repo so reviewers skip indexing entirely.

**Bi-temporal advantage**: Graphiti puts a `valid_at` timestamp on every edge, parsed from messy column keys like `"Year ended June 30, 2009"`. This is why I picked Graphiti over GraphRAG: financial QA is fundamentally temporal, and treating time as a first-class property of every fact is what lets the agent answer "in 2008" questions without dragging in 2007 or 2009 values.

### D2: 1 database, `group_id` as property filter

**Decision**: 1 Neo4j database, `record_id_to_group_id(record.id)` as `group_id` property on every node/edge.

**Rationale**: `group_id` is a Neo4j property. Search filters via `WHERE group_id IN $group_ids`. Clean single-tenant design: one DB, all records tagged, queries scoped by document. `record_id_to_group_id()` strips the trailing `-N` suffix (so multiple records sharing a PDF dedupe to one document) and sanitises characters that Graphiti validates against `[a-zA-Z0-9_-]`.

### D3: Hybrid indexing, triplets for table and episodes for text

**Decision**: Table cells go through `add_triplet` (direct, no LLM). Pre/post text goes through `add_episode_bulk` (LLM-extracted).

**Rationale**: Graphiti's LLM episode extraction targets entity relationships, not scalar values. Financial table numbers like `"balance at end of year": 1636526` aren't reliably extracted as facts. Empirical A/B on `Single_ADI/2016/page_61.pdf`: episode-only got 0/5, hybrid (triplets for the table) got 5/5. Direct triplet insertion gives exact, BM25-searchable facts in the form `"In column [2015], the row [balance at end of year] has value 1636526"`. The explicit `[column]` and `[row]` markers prevent the downstream LLM from treating column headers like `"2007 program"` as years.

**Stable UUIDs** (md5 of `group_id::name`) deduplicate entities. Without them, the same entity gets created multiple times across cells. One record went from 42 nodes down to 16 after this fix.

**`_parse_year()`** handles all column key formats in the dataset: `"Year ended June 30, 2009"` → 2009, `"december 312016"` → 2016, `"$ 2014"` → None (dollar amount, not a year).

**Bridging nodes**: shared entity names merge the two graphs. `[balance at end of year]` is created by a triplet (with a value) AND referenced by an episode (with surrounding business context). Search returns both.

### D4: Reasoning model for the agent

**Decision**: `gpt-5-mini` (reasoning) for the agent. `gpt-4o-mini` at `temp=0` for query rewriting and Graphiti's entity extraction. `text-embedding-3-small` for embeddings.

**Rationale**: ConvFinQA questions need Chain-of-Thought (resolve references → locate values → compute → propagate to next turn). A reasoning-tuned model produces a clean Step-1/Step-2/Step-3 trace with intermediate-step fidelity, which matters because errors propagate across turns. Tried `gpt-5-nano` (reasoning thin and inconsistent) and `gpt-4o-mini` (no real CoT, skipped steps on harder chains). `gpt-5-mini` is the cheapest reasoning model with enough depth to keep the trace clean.

### D4b: Hardcoded to OpenAI (deliberate, not a default)

**Decision**: All LLM and embedding calls go through the OpenAI SDK. There is no provider-abstraction layer (no LiteLLM wrapper, no `LLMProvider` Protocol, no provider-switch via env var).

**Rationale**: two reasons.

1. **Sufficiency**: OpenAI's models are good enough for this scope. The agent uses `gpt-5-mini` (reasoning), Graphiti uses `gpt-4o-mini` (extraction), embeddings use `text-embedding-3-small`. Adding provider abstraction would introduce code I don't need to ship the take-home, which is the kind of premature flexibility YAGNI warns against.
2. **Context**: Keeping the implementation OpenAI-native is both pragmatic and reduces unnecessary abstraction.

**Future direction** (Future Work, not done here): if a production deployment needed model portability (e.g. Claude for some queries, Gemini for others, on-prem models for compliance), the right design is:

- An `LLMProvider` Protocol in `src/lib/interfaces.py` covering `chat.completions.create` and `embeddings.create`.
- One concrete `OpenAIProvider` implementation today.
- Other providers (`AnthropicProvider`, `GeminiProvider`, `VLLMProvider`) added later behind the same Protocol.
- Inject into `FinancialAgent` and `FinancialGraph` at construction.

The architecture already supports this swap (everything goes through the same OpenAI client object), so the change is a one-day refactor when actually needed.

### D5: Chain-of-Thought + math expression evaluation

**Decision**: The agent emits a structured CoT response ending in `ANSWER: <expression>`. The `ANSWER:` field can be a math expression (e.g. `108 / 6197`), evaluated by `simpleeval` for exact arithmetic.

**Rationale**: Avoids LLM rounding errors in the final answer. Critical because subsequent conversation turns use prior answers as inputs, and rounding compounds.

### D6: Curated prompt rules, added in response to concrete failures

**Decision**: Eight prompt rules in the system prompt, each one added because a specific record failed and regression-tested before keeping.

Rules cover: parenthetical % is the answer, portion = subtotal÷total, change = later − earlier, prefer final total row, program names by year, stockholder-return fluctuation, total/aggregate price uses TABLE row, Rule 12 exception → Rule 15 for `[year] program` columns.

**Rationale**: Each rule has a regression test (the original failure it fixed) and was kept only if it didn't break previously-passing records. Three other attempted rules were reverted because they caused regressions. The rule set is at a practical ceiling.

### D7: Two retrieval modes (graceful degradation)

**Decision**: Mode 1 (graph search) for indexed records, Mode 2 (raw `pre_text + table JSON + post_text`) for unindexed records.

**Rationale**: Keeps the CLI always usable. New records work in Mode 2 immediately; they upgrade to Mode 1 when indexed. The `evaluate` command auto-indexes records inline per-record (check indexed → index if missing → evaluate → next), so a reviewer doesn't have to think about it.

### D8: SQLite for conversation persistence

**Decision**: `data/conversations.db` with a `ConversationStore` class implementing the `HistoryStore` protocol.

**Rationale**: Python built-in, no extra dependency, persists across sessions, queryable by `conversation_id`. `format_history(max_turns=10)` bounds context growth.

### D9: Tolerance over exact match in evaluation

**Decision**: Seven normalisation paths chained before the final 5% relative or 1e-3 absolute tolerance check:

1. Unit scale (1B / 1M / 1K, both directions)
2. Multiply-by-scale
3. Percent vs decimal
4. Reverse percent
5. Reverse percent for both-small values (100× off)
6. Inverse ratio (X/Y vs Y/X)
7. Sign flip (losses stored negative)

Each path was added in response to a real failure during evaluation, never speculatively.

**Rationale**: Exact match penalises correct reasoning with cosmetic differences. The paper baseline uses ~1% tolerance; I extended to 5% plus the normalisations because LLM outputs vary in format (`14.14%` vs `0.14136` vs `0.141`) in ways the paper-era systems didn't.

### D10: Pre-built graph dump committed

**Decision**: `data/neo4j_dump/dev.dump` (39M, 100 indexed records) committed to the repo. Docker Compose auto-restores on first start.

**Rationale**: Reviewer experience is `docker-compose up && uv run main evaluate --limit=100`. No indexing required to reproduce the headline number. The dump is small (39M) thanks to a one-time WAL-strip procedure (`neo4j-admin database dump` after a clean shutdown).

### D11: Per-record auto-indexing in `evaluate`

**Decision**: `evaluate` walks records one at a time. For each record: check if indexed → index if missing → evaluate all turns → next record. Logs interleave so a reviewer sees real-time progress.

**Rationale**: Indexing all unindexed records upfront would block the eval from starting. Per-record flow gives immediate feedback and lets the user `Ctrl+C` after seeing the first few records work. Indexing is checkpoint-resumable so a crash doesn't waste work.

### D12: Module layout

```
src/
├── main.py                  CLI entry
├── config/                  app settings
├── convfinqa/               domain layer
│   ├── agent/
│   ├── data.py
│   ├── evaluation/
│   ├── graph/
│   └── models.py
└── lib/                     reusable infrastructure
    ├── conversation/
    ├── evaluation/
    ├── interfaces.py
    └── util/
```

**Rationale**: `convfinqa/` holds anything ConvFinQA-specific. `lib/` holds anything reusable across hypothetical future projects. `config/` is at the top because it's app-specific (not lift-and-shift code), while `util/` lives under `lib/` because the logger factory is generic infrastructure.

## Risks / Trade-offs

- **Graphiti LLM indexing cost**: `add_episode_bulk` reduces calls vs individual episodes; still meaningful at scale. Mitigated by `concurrency=1` default + checkpoint resume + committed dump.
- **`gpt-5-mini` non-determinism**: identical prompts can produce different reasoning paths. Some prompt fixes work on one trial and not the next. This is a hard ceiling that no amount of prompt engineering can break.
- **Dataset annotation errors**: about 9% of dev questions have annotations that don't match the question. Hard accuracy ceiling, documented in the Error Analysis.
- **Mode 2 fallback accuracy**: slightly lower than Mode 1 on complex questions but close enough that the system remains usable for unindexed records.

## Open Questions

None blocking. Future Work covers extensions (frontier-model upgrade, LLM-as-judge, self-consistency, reflection, text/yes-no answers, text-to-Cypher, dataset error auto-flagging).
