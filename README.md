# ConvFinQA: Conversational Financial QA Agent

Multi-turn QA over financial 10-K documents, using a temporal knowledge graph (Graphiti + Neo4j) for retrieval.

**Results on 100 dev records (349 questions):**

| Mode | Code Acc | Judge Acc |
|---|---|---|
| Graph-RAG | 84.2% | 77.4% |
| Long Context | 82.5% | 77.4% |
| **Agentic RAG** | **85.7%** | **79.7%** |

Same `gpt-5-mini` in all three, so any difference between modes is retrieval strategy and tool use, not the model. `gpt-5-mini` is non-deterministic, so each figure carries ~+/-1 point of run-to-run variance. Graph-RAG and Long Context are statistically tied (1.7pp, within the noise band); Agentic RAG's 3.2pp lead over Long Context holds across runs and is the one real gap.

Agentic RAG effective accuracy is **93.1%** (325/349) after excluding the 26 dataset-annotation errors where the gold answer contradicts the source. Of the 50 failures: 26 dataset bugs, 3 evaluator scale/sign gaps, 2 ambiguous phrasing, 19 genuine model errors (5.4%).

**Stack:** Python 3.12 · Graphiti 0.29.1 · Neo4j 5.26 · GPT-5-mini · SQLite · Docker Compose

See **[REPORT.md](REPORT.md)** for the full method, error analysis, and design decisions. See **[docs/hybrid_graph_schema.md](docs/hybrid_graph_schema.md)** for an annotated example of a real record's Neo4j graph.

## Three execution modes

The agent has three modes. Graph-RAG is the default; `--no-index` forces Long Context; `--tools` enables Agentic RAG.

> **Why retrieval at all?** These dev records are short enough to fit in a prompt, so Long Context is competitive here. Real 10-Ks run 200 pages / ~150K tokens: you cannot load one whole, and even if you could the extra context dilutes attention. The graph exists to find the needle, not dump the haystack.
>
> **Even with 1M-token context windows**, the three core limitations of Long Context remain. Rereading cost: the model still processes every token on every query turn. Needle-in-a-haystack: attention dilutes as context grows; a fact buried deep in a 1M-token window is harder to surface reliably than one returned by a targeted search. Infinite dataset: a full financial filing history across years, subsidiaries, and geographies is measured in terabytes; no context window reaches that. Long Context becomes more viable as windows grow, but retrieval is still the only approach that scales to unbounded data.
>
> See [REPORT.md](REPORT.md) for the full reasoning.

**Graph-RAG:** Runs a hybrid BM25 + embedding search over the indexed graph and answers from the returned facts. Default for indexed records.

**Long Context:** Passes the full document directly into the prompt. No retrieval. Falls back automatically when a record is not indexed, or when `--no-index` is set.

**Agentic RAG:** The model calls `search_document`, `compute`, and `compare` explicitly. `search_document` searches table cells and narrative text in parallel in a single call, so the model always sees both sources and picks the value whose type matches the question. This is the highest-accuracy mode on this dataset (85.7% code).

Both Graph-RAG and Agentic RAG query the same graph. The difference is that Graph-RAG returns a ranked mixed list and the model picks from it in one pass. Agentic RAG forces the model to actively verify both sources. Graph-RAG remains the default because it is cheaper and faster to reproduce.

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Docker + Docker Compose
- OpenAI API key

### Steps

```bash
# 1. Configure your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 2. Start Neo4j. The first run auto-restores the pre-built graph (~10s)
docker-compose up -d

# 3. Install Python dependencies
uv sync

# 4. (Optional) Verify the graph loaded
docker exec metaslim-neo4j-1 cypher-shell -u neo4j -p password \
  "MATCH (n:Entity) RETURN count(DISTINCT n.group_id) AS docs"
# Expected output: docs: 91 on a fresh clone.
# (100 records share 91 unique documents, some records like
# `Single_X/YEAR/page_Z.pdf-1` and `-2` come from the same PDF page.)
# This count grows by 1 per additional record you index later, so anything
# 91+ indicates a healthy load.
```

The committed graph dump (`data/neo4j_dump/dev.dump`, 39M) is auto-loaded on first `docker-compose up`. **No indexing required to get started.**

> **Scope of the pre-built graph:** 100 of 421 dev records are pre-indexed (24%). The full dataset is 3,458 records total (3,037 train + 421 dev); I indexed only the evaluated subset because (a) full indexing takes ~7 hours and ~$50-80 in OpenAI tokens, and (b) only the dev set is needed for evaluation.
>
> Records 101 to 421 do **not** need to be pre-indexed. `uv run main evaluate` auto-indexes each one inline as it visits the record. You can also batch-index with `uv run main index` or index a single record on demand via `uv run main chat`.

Neo4j Browser UI is available at **http://localhost:7474** (user `neo4j`, password `password`) for ad-hoc Cypher queries.

---

## Usage

Two top-level commands: `chat` (interactive REPL) and `evaluate` (scored batch run). Both work on a record that is **already indexed in the graph**, a record that **is not indexed but you opt-in to index it inline**, or a record that **is not indexed and you stay un-indexed** (raw-document fallback). Both commands accept `--tools` to switch from single-pass CoT to the function-calling tool loop.

The examples below use:
- **Indexed example**: `Single_MRO/2007/page_134.pdf-1`, one of the 100 dev records in the committed graph dump. Questions are about a weighted-average-exercise-price table.
- **Un-indexed example**: `Single_CDNS/2006/page_30.pdf-3`, a dev record beyond the committed 100 with no nodes in the graph until you index it. Questions are about Cadence Design Systems' stock performance value.

Verify the un-indexed example actually starts un-indexed in your environment:

```bash
docker exec metaslim-neo4j-1 cypher-shell -u neo4j -p password \
  "MATCH (n:Entity {group_id: 'Single_CDNS-2006-page_30-pdf'}) RETURN count(n) AS cnt"
# Expected on a fresh dump: cnt 0
```

### Chat (interactive)

#### Case A: pre-indexed record

**Without `--tools`** (Graph-RAG (single-pass)):

```bash
uv run main chat "Single_MRO/2007/page_134.pdf-1"
```

```
Chatting about: Single_MRO/2007/page_134.pdf-1
Type exit or quit to stop

>>> what was the weighted average exercise price per share in 2007?
assistant: 60.94
>>> and what was it in 2005?
assistant: 25.14
>>> what was, then, the change over the years?
assistant: 35.8
```

Gold: 60.94 / 25.14 / 35.8 (3/3 match). Graph search returns the top 30-50 facts per question; conversation history resolves "and what was it in 2005?".

**With `--tools`** (Agentic RAG, function-calling loop):

```bash
uv run main chat --tools "Single_MRO/2007/page_134.pdf-1"
```

```
Chatting about (tools): Single_MRO/2007/page_134.pdf-1
Type exit or quit to stop

>>> what was the weighted average exercise price per share in 2007?
assistant: 60.94
>>> and what was it in 2005?
assistant: 25.14
>>> what was, then, the change over the years?
tools: compute
assistant: 35.8
```

Turn 3 routes through `compute`; T1, T2, T4 resolve from history or the pre-loaded table. When the model calls `search_document`, it gets both table cells and narrative snippets in one call and picks the value whose type matches the question (catches the table-ratio-vs-narrative-amount failure mode demonstrated on CB/2010 in REPORT.md).

#### Case B: un-indexed record, choose to index inline

The CLI detects the missing index and prompts. Same flow whether or not `--tools` is set; once indexed, the record stays in Neo4j permanently.

```
$ uv run main chat "Single_CDNS/2006/page_30.pdf-3"

Record not indexed in graph.
  Y: index now (~20s), then chat using graph search
  N: chat immediately using raw document (pre_text + table + post_text), no graph
Index this record now for best results? [Y/n]: y
  [triplets] Single_CDNS/2006/page_30.pdf-3: 24 cells
  [triplets] done (3.7s)
  [episodes] Single_CDNS/2006/page_30.pdf-3: 4 chunks (embedding + LLM extraction)
  [episodes] done (26.3s)
Indexed record: Single_CDNS/2006/page_30.pdf-3

>>> what was the performance value of the cadence design systems inc in 2005?
assistant: 81.52
>>> what was, then, the change in its performance value, considering 2004 as a base?
assistant: -18.48
>>> and how much does this change represent in relation to that original amount?
assistant: -0.1848
```

Gold: 81.52 / -18.48 / -0.1848 (3/3). Add `--tools` for the same flow with Agentic RAG.

#### Case C: un-indexed record, decline indexing (Long Context fallback)

```
$ uv run main chat "Single_CDNS/2006/page_30.pdf-3"

Record not indexed in graph.
  Y: index now (~20s), then chat using graph search
  N: chat immediately using raw document (pre_text + table + post_text), no graph
Index this record now for best results? [Y/n]: n
Chatting with raw document context (no graph search).

>>> what was the performance value of the cadence design systems inc in 2005?
assistant: 81.52
```

Long Context dumps the full `pre_text + table + post_text` into the user prompt instead of retrieved facts. Slightly higher token cost per question, but no graph round-trip and no indexing wait. `--tools` works on top of Long Context too: the pre-loaded table grid plus `compute` still function, while `search_document` returns empty results because nothing was indexed.

#### Agentic RAG (`--tools`)

The model has **three** tools and a **pre-loaded full table** in the initial user prompt. Agentic RAG's structural difference from Graph-RAG: the table is unconditionally visible (Graph-RAG only shows search results), so the model can't miss a cell because hybrid retrieval buried it. Narrative stays search-based because most of it is irrelevant per question.

**The three tools:**

| Tool | Signature | Purpose |
|---|---|---|
| `search_document` | `(query: str) -> {table_cells: [...], narrative_snippets: [...]}` | Parallel hybrid search over table cells AND narrative. Returns both result sets; model picks the value whose type matches the question (dollar amount vs ratio vs percentage-point). |
| `compute` | `(reasoning: str, expression: str) -> float` | Math evaluator. `reasoning` anchors which prior-turn values each number comes from before the expression is evaluated. |
| `compare` | `(reasoning: str, expr_a: str, expr_b: str) -> "yes"\|"no"` | Comparison for yes/no questions ("did X grow?", "is A greater than B?"). Evaluates both expressions and compares. |

The **full table** is converted from its raw JSON form (`{column: {row: value}}`) into a `tabulate` grid and pre-loaded into the system prompt, so the model sees a human-readable table rather than raw JSON, which is harder to scan and index on. The model can read any cell directly without spending a tool call.

**Loop discipline:** hard cap of 15 tool calls per question. After the cap, the agent forces an `ANSWER:`. If the model still refuses, it falls back to the single-pass path so the worst case is bounded by current accuracy.

**`search_document` design:** merging table and narrative search into one tool eliminates the source-routing failure at the architecture level. The model always receives both result sets in a single call and picks the value whose type matches the question. No application-level sequencing enforcement is needed. See the CB/2010 trap in REPORT.md for a concrete example of this failure mode.

```bash
uv run main chat --tools "Single_MRO/2007/page_134.pdf-1"
```

**Expected trace shape under the 3-tool design:**

- *Value lookup* ("what was X in YEAR?"): `search_document` → inspect table_cells and narrative_snippets → `compute` if arithmetic needed.
- *Subset / qualifier* ("for the senior notes"): `search_document` → answer lives in narrative_snippets → `compute` if needed.
- *Follow-up needing a new value*: `search_document` for the new year/column.
- *Pure derivation* (both values in history): `compute` only.

**Trade-offs:** the pre-loaded table removes the failure mode where hybrid retrieval ranks the right cell off the surfaced facts. Latency is comparable to Graph-RAG when only `compute` is invoked; ~2x when narrative search is needed. Agentic RAG is the highest-accuracy mode on this dataset (85.7% raw code / 93.1% effective vs Long Context's 82.5%) at ~3x the per-question token cost.

#### Browsing available records

```bash
uv run main chat            # No argument: lists the first 20 record IDs and prompts you to pick
```

If you type a record ID without the `-N` suffix (e.g. `Single_MRO/2007/page_134.pdf` instead of `...pdf-1`), the CLI fuzzy-matches the base document and tells you which exact record it picked.

### Evaluate (scored)

Scope flags (which records to evaluate):

```bash
uv run main evaluate                                  # all 421 dev records
uv run main evaluate --limit=100                      # first 100 (the records in the committed dump)
uv run main evaluate --offset=20 --limit=80           # records 21-100
uv run main evaluate "Single_MRO/2007/page_134.pdf-1" # single record (fuzzy on the -N suffix)
```

Mode / evaluator flags (combinable with any scope):

| Flag | Effect |
|---|---|
| (no flag) | Graph-RAG ( single-pass) with the code evaluator only |
| `--tools` | Switch to Agentic RAG (function-calling loop) |
| `--judge` | Add LLM-as-judge alongside the code evaluator (records disagreements) |
| `--no-index` | Skip auto-indexing of missing records; force Long Context fallback |

The runner walks records one at a time: check indexed -> index if missing (or skip with `--no-index`) -> evaluate all turns -> next record. Results persist to `data/eval_results.json` after every 10 records.

> **To reproduce the 85.7% Agentic RAG headline,** run `uv run main evaluate --limit=100 --tools --judge`. All 100 records are pre-indexed in the committed dump, so no per-record index work happens.

#### Case A: pre-indexed record (all flag combinations)

**Default (Graph-RAG, code evaluator only):**

```
$ uv run main evaluate "Single_MRO/2007/page_134.pdf-1"

Loading dev set...
Evaluating on 1 record(s)
Per-record flow: check indexed -> index if missing -> evaluate -> next record
✅code T1: Q=what was the weighted average exercise price per s | gold=60.94   | got=60.94
✅code T2: Q=and what was it in 2005?                           | gold=25.14   | got=25.14
✅code T3: Q=what was, then, the change over the years?         | gold=35.8    | got=35.8
✅code T4: Q=what was the weighted average exercise price per s | gold=25.14   | got=25.14
✅code T5: Q=and how much does that change represent in relatio | gold=1.42403 | got=1.424

    code evaluator
┏━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric  ┃ Accuracy ┃
┡━━━━━━━━━╇━━━━━━━━━━┩
│ Overall │   100.0% │
├─────────┼──────────┤
│ Turn 1  │   100.0% │
│ Turn 2  │   100.0% │
│ Turn 3  │   100.0% │
│ Turn 4+ │   100.0% │
└─────────┴──────────┘

Total: 5/5 correct (code evaluator)
```

**`--judge` (Graph-RAG + dual evaluator):**

```
$ uv run main evaluate --judge "Single_MRO/2007/page_134.pdf-1"

✅code ✅judge T1: ... | gold=60.94   | got=60.94
✅code ✅judge T2: ... | gold=25.14   | got=25.14
✅code ✅judge T3: ... | gold=35.8    | got=35.8
✅code ✅judge T4: ... | gold=25.14   | got=25.14
✅code ✅judge T5: ... | gold=1.42403 | got=1.424

(code evaluator and judge evaluator tables render separately, both 100%)

Total: 5/5 correct (code evaluator)
judge: 5/5 correct (100.0%), 0 disagreement(s) with code evaluator
```

**`--tools` (Agentic RAG, code evaluator only):**

```
$ uv run main evaluate --tools "Single_MRO/2007/page_134.pdf-1"

✅code T1: ... | gold=60.94   | got=60.94
✅code T2: ... | gold=25.14   | got=25.14
✅code T3: ... | gold=35.8    | got=35.8                 | tools=compute
✅code T4: ... | gold=25.14   | got=25.14
✅code T5: ... | gold=1.42403 | got=1.424025457438345    | tools=compute

Total: 5/5 correct (code evaluator)
Tool path: 0 question(s) exhausted the 15-call cap.
```

The `tools=` suffix shows the tool sequence per turn. T1, T2, T4 call `search_document` and pick from both result sets; T3 and T5 route through `compute` only (pure derivation from history). `search_document` returns both table_cells and narrative_snippets in one call, so the model always compares both sources (relevant for records like CB/2010 where the table holds a ratio but the narrative holds the dollar amount the question wants).

**`--tools --judge` (Agentic RAG + dual evaluator, the headline command):**

```
$ uv run main evaluate --tools --judge "Single_MRO/2007/page_134.pdf-1"

✅code ✅judge T1: ... | gold=60.94   | got=60.94
✅code ✅judge T2: ... | gold=25.14   | got=25.14
✅code ✅judge T3: ... | gold=35.8    | got=35.8                 | tools=compute
✅code ✅judge T4: ... | gold=25.14   | got=25.14
✅code ✅judge T5: ... | gold=1.42403 | got=1.424025457438345    | tools=compute

Total: 5/5 correct (code evaluator)
judge: 5/5 correct (100.0%), 0 disagreement(s) with code evaluator
Tool path: 0 question(s) exhausted the 15-call cap.
```

#### Case B: un-indexed record, auto-indexed inline then evaluated

Default behavior: the runner indexes the record (~20 s of `gpt-4o-mini` extraction) before evaluating. Works for both Graph-RAG and Agentic RAG.

```
$ uv run main evaluate "Single_CDNS/2006/page_30.pdf-3"

Per-record flow: check indexed -> index if missing -> evaluate -> next record
  [triplets] Single_CDNS/2006/page_30.pdf-3: 24 cells
  [triplets] done (3.7s)
  [episodes] Single_CDNS/2006/page_30.pdf-3: 4 chunks (embedding + LLM extraction)
  [episodes] done (26.3s)
Indexed record: Single_CDNS/2006/page_30.pdf-3
✅code T1: Q=what was the performance value of the cadence desi | gold=81.52   | got=81.52
✅code T2: Q=what was, then, the change in its performance valu | gold=-18.48  | got=-18.48
✅code T3: Q=and how much does this change represent in relatio | gold=-0.1848 | got=-0.1848

Total: 3/3 correct (code evaluator)
```

Indexing is checkpoint-resumable via `data/index_checkpoint.json`, so crashes (rate limit, network, quota) pick up where they left off. With `--tools`, the same flow runs Agentic RAG after indexing.

#### Case C: un-indexed record, force Long Context (no indexing)

`--no-index` skips the inline indexing and falls back to the full-document path (Long Context). Useful for ad-hoc checks where you don't want to pay the indexing cost.

```
$ uv run main evaluate --no-index "Single_CDNS/2006/page_30.pdf-3"

Evaluating on 1 record(s)
--no-index set: skipping per-record indexing (Long Context fallback for missing records)
✅code T1: Q=what was the performance value of the cadence desi | gold=81.52   | got=81.52
✅code T2: Q=what was, then, the change in its performance valu | gold=-18.48  | got=-18.48
✅code T3: Q=and how much does this change represent in relatio | gold=-0.1848 | got=-0.1848

Total: 3/3 correct (code evaluator)
```

`--no-index` combines with `--tools` and `--judge` the same way.

#### Disagreement logging (`--judge`)

When code and judge disagree, the case is appended to `data/eval_results.json` under `disagreements` with the question, gold, predicted, both verdicts, and the judge's one-sentence reason. Useful for spotting cases where the seven-path code evaluator is too lenient (e.g., inverse-ratio false positives the judge catches) or too strict (e.g., balance-sheet sign-convention edge cases the judge accepts).

---

## Tests

```bash
uv run pytest tests/                              # 240 unit tests, mocked external deps
uv run pytest tests/ --cov=src --cov-report=term  # with coverage (currently 100%)
uv run ruff check src/ tests/                     # lint
```

CI runs all three on every push to `main` and `submission` (`.github/workflows/ci.yml`).

---

## Batch indexing (optional)

You usually don't need this. `uv run main evaluate` auto-indexes records inline as it visits them, and `uv run main chat` prompts for indexing when you pick an un-indexed record. But if you want to pre-build a whole split ahead of time:

```bash
uv run main index                    # default: dev split (421 records)
uv run main index --split=train      # train split (3037 records)
uv run main index --split=all        # train + dev (3458 records)

bash scripts/export_dump.sh dev      # dump the rebuilt graph to data/neo4j_dump/dev.dump
```

**Time and cost** at the default `concurrency=1` (avoids OpenAI tier rate limits):

| Split | Records | Wall time | API cost |
|---|---|---|---|
| dev (after the 100 already in the dump) | 321 | ~2.5 to 3 hours | ~$3 to $5 |
| train | 3,037 | ~25 to 30 hours | ~$40 to $60 |
| all | 3,458 | ~28 to 35 hours | ~$45 to $65 |

If your OpenAI tier supports it, raise `concurrency` (e.g. `--concurrency 10` on `index`) to cut wall time by roughly 10x at the same cost. Indexing is **resumable** via `data/index_checkpoint.json` after rate limits, network glitches, or quota issues.

---

## Architecture

```
src/
├── main.py                            # Typer CLI: chat / evaluate / index
├── config/
│   └── settings.py                    # settings loaded from .env
├── convfinqa/                         # domain layer (ConvFinQA-specific)
│   ├── agent/
│   │   ├── agent.py                   # FinancialAgent: answer() / answer_raw_doc() / answer_with_tools()
│   │   ├── prompts.py                 # SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT
│   │   └── tool_calling/              # tool surface (Agentic RAG)
│   │       ├── interfaces.py          # ToolRegistry protocol
│   │       ├── tool_registry.py       # ConvFinQAToolRegistry dispatcher
│   │       └── tools.py               # search_document / compute / compare
│   ├── dataset/
│   │   ├── loader.py                  # load_dataset, build_index
│   │   └── models.py                  # ConvFinQARecord, Document, Dialogue (Pydantic v2)
│   └── evaluation/
│       └── runner.py                  # eval loop, EvaluationResult, per-turn breakdowns
└── lib/                               # infrastructure / reusable
    ├── answers/                       # answer strategies (OCP: add new mode = new file)
    │   ├── base/                      # AnswerResult, AnswerStrategy, Agent, IndexedRecord protocols
    │   ├── single_pass_graph/         # Graph-RAG: graph-RAG single-pass CoT
    │   ├── raw_doc/                   # Long Context: full raw document, no graph
    │   └── tool_calling/              # Agentic RAG: function-calling loop + tools + registry
    ├── conversation/
    │   ├── interfaces.py              # HistoryStore protocol
    │   └── store.py                   # SQLite-backed conversation history
    ├── evaluation/
    │   ├── base/                      # Evaluator protocol, EvalVerdict, metrics
    │   ├── code/                      # CodeEvaluator (numeric tolerance, deterministic)
    │   └── llm_judge/                 # LLMJudgeEvaluator (semantic, --judge opt-in)
    ├── graph/
    │   ├── interfaces.py              # GraphSearcher / GraphIndexer / GraphStore / IndexableRecord
    │   ├── graph.py                   # FinancialGraph facade → _GraphSearcherImpl + _GraphIndexerImpl
    │   └── utils.py                   # record_id_to_group_id, year parsing
    └── util/
        └── logger.py                  # logger factory

tests/                                 # 240 unit tests, mirrors src/ structure, 100% coverage
data/
├── convfinqa_dataset.json             # source dataset (committed)
├── neo4j_dump/dev.dump                # pre-built graph (39 MB, 100 records, auto-loaded)
├── index_checkpoint.json              # resumable indexing checkpoint
└── conversations.db                   # SQLite history (created on first run)
```

`src/convfinqa/` is domain-specific code (agent, dataset models, eval runner). `src/lib/` is infrastructure that could be reused elsewhere (graph, conversation store, evaluators, answer strategies). The paper's best result was GPT-3 + simple prompting at 45.15%. Long Context adds only a stronger model and scores 82.5%. Graph-RAG and Long Context are within the noise band (1.7pp); Agentic RAG's 3.2pp lead is the one real gap. See [REPORT.md](REPORT.md) for the full reasoning.

---

## Troubleshooting

**Neo4j won't start, or "database unavailable"**
- Check the container is healthy: `docker ps` (look for "healthy" in the STATUS column)
- View logs: `docker logs metaslim-neo4j-1 | tail -50`
- Restart: `docker-compose restart neo4j`

**Eval is slow on first run**
- That's expected for un-indexed records. `uv run main evaluate` indexes each missing record inline before evaluating it (~20 to 30 seconds per record at the default concurrency=1). To skip indexing and use Long Context fallback instead, pass `--no-index`.

**OpenAI rate limit or quota exhausted**
- Three model knobs are configurable via env (see `.env.example`):
  - `OPENAI_MODEL` (default `gpt-5-mini`): the agent's CoT reasoning model.
  - `GRAPHITI_MODEL` (default `gpt-4o-mini`): Graphiti's entity/edge extraction during indexing.
  - `JUDGE_MODEL` (default `gpt-4o-mini`): the LLM-as-judge model when `--judge` is set.
- All go through the same `OPENAI_API_KEY`. Both indexing and eval are resumable, so fix the quota issue and re-run the same command.

**Stale `data/index_checkpoint.json`**
- If you wipe Neo4j data and want to re-index from scratch, also delete `data/index_checkpoint.json`. Otherwise the checkpoint will skip records.

---

## Documentation

- **[REPORT.md](REPORT.md)**: full report. Method, architecture decisions, two-evaluator framing, error analysis, how `--tools` and `--judge` augment the analysis, future work.
- **[dataset.md](dataset.md)**: notes on the ConvFinQA dataset structure.
- **[openspec/changes/](openspec/changes/)**: three spec-driven proposals authored before implementation:
  - `convfinqa-graphrag-agent/`: base system (Modes 1 + 2, code evaluator)
  - `convfinqa-tool-calling-agent/`: Agentic RAG (`--tools` function-calling agent)
  - `convfinqa-llm-as-judge/`: `--judge` LLM-as-judge evaluator
