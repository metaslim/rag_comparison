# ConvFinQA Report

A multi-turn QA agent over financial 10-K documents. It uses a temporal knowledge graph (Graphiti / Neo4j) for retrieval, with three execution modes: Graph-RAG, Long Context, and Agentic RAG, each representing a different strategy for getting the right information into the model's context window.

---

## Key findings

- **Match the model to the task, not just "use the best one."** `gpt-5-mini` (a reasoning model) does the multi-step numerical CoT; `gpt-4o-mini` (non-reasoning) handles the narrow, structured jobs (query rewrite, entity extraction, the judge). A reasoning model on those would cost more for no gain.
- **Agentic RAG's edge comes from the tool machinery, not retrieval alone.** Same `gpt-5-mini` in all three modes. Graph-RAG (84.2%) only ties Long Context (82.5%), within noise; retrieval alone does not move accuracy on these short records. Agentic RAG's +3.2pp over Long Context comes from on-demand re-querying + `search_document` (which always returns both table and narrative results) + `compute`. Targeted retrieval's real payoff is cost and scale (200-page 10-Ks Long Context can't fit), not accuracy here.
- **Most "model failures" are bad ground truth.** Of 50 Agentic RAG failures: 26 are dataset-annotation errors (gold contradicts the document), 3 are evaluator gaps where the model was right, 2 are ambiguous phrasing. Only 19 are genuine. Raw 85.7% becomes an effective 93.1%. Per-failure classification is not optional on a noisy benchmark.
- **The code evaluator is a moving target; the LLM judge isn't.** Each answer format (inverse ratio, scale, percent-vs-decimal) needs a new tolerance path, and knowing those paths tempts you to game the format instead of being correct. The judge handles equivalence semantically; running both and logging disagreements caught the sign-convention cases.
- **Trust gaps, not points.** `gpt-5-mini` ignores `temperature=0`, so every figure carries ~+/-1pp run-to-run (a Long Context re-run moved +4 questions with no code change). Sub-1-point differences between modes are noise; only gaps that clear the band, like Agentic RAG's lead, are real.

## Results

Evaluated on 100 of 421 dev records (349 questions). Two evaluators ran on every question: a deterministic code evaluator and an LLM judge. The headline configuration is Agentic RAG (`uv run main evaluate --limit=100 --tools --judge`).

| Mode | Code Acc | Judge Acc |
|---|---|---|
| Graph-RAG | 84.2% (294/349) | 77.4% (270/349) |
| Long Context | 82.5% (288/349) | 77.4% (270/349) |
| **Agentic RAG** | **85.7% (299/349)** | **79.7% (278/349)** |

All three are single full 100-record runs. Same `gpt-5-mini` throughout, so any difference between modes is retrieval strategy and tool use, not the model.

On precision: Graph-RAG vs Long Context (1.7pp) sits inside the +/-1pp run-to-run noise and should be read as a tie. Agentic RAG's lead over Long Context (3.2pp) holds up across runs and is the one real gap.

I classified every one of the 50 Agentic RAG failures against the source document. 26 are dataset-annotation errors where the gold answer contradicts the source (these fail regardless of architecture). Excluding only those, effective accuracy is **93.1%** (325/349). The genuine model error rate is **5.4%** (19/349). Full breakdown in Error Analysis.

For reference, the paper (Chen et al., 2022) reports GPT-3 best prompting at 45.15%, FinQANet at 68.90%, and human expert at 89.44% on the full dev set. Those figures are on 421 records; ours are on 100, so treat the comparison as directional only.

---

## Method

### Context injection: why it's the core problem

Large language models are trained on static data and frozen at their training cutoff. They have no access to private documents, domain-specific corpora, or anything that postdates training. Any LLM-based application must solve the same fundamental problem: how to get the right information into the model's context window at query time.

Three broad strategies exist, and this application implements all three as switchable execution modes.

| Strategy | How it works | Strengths | Weaknesses |
|---|---|---|---|
| Graph-RAG | Index documents offline; retrieve top-ranked facts at query time | Scales to any corpus; low per-query token cost | Retrieval lottery: wrong chunks rank highest and the answer never reaches the model (silent failure); index must be kept in sync with source data when documents change (graph store + embedding model + extraction pipeline: multiple moving parts, multiple places for things to break) |
| Long Context | Inject the full document directly into the prompt | No retrieval lottery; full cross-document visibility; no index infrastructure to maintain | Rereading cost on every query turn; needle-in-a-haystack as context grows (attention dilutes on large documents); bounded by context window |
| Agentic RAG | Model holds retrieval tools and decides what to fetch and when | Avoids rereading cost; can re-query to reduce silent failure | Most complex to implement; non-deterministic across runs; tool cap can cause give-up failures; carries the same indexing infrastructure overhead as Graph-RAG |

### Why a temporal graph (Graphiti)

Plain chunked RAG was the obvious starting point for the retrieval path. I rejected it because in financial documents a table cell is meaningless without the narrative that explains it, and those two things are often pages apart. Fixed-size chunk boundaries cut that link. A graph preserves it: each table cell becomes a graph edge, and narrative entities are connected to table entities regardless of where they appear in the source document.

Financial data is also inherently time-anchored. `"Net income was 93 million"` without a year is useless. Graphiti gives each edge a `valid_at` timestamp, so the same entity accumulates a timeline across years and follow-up questions like "and in 2008?" just work.

### Hybrid indexing: triplets for tables, episodes for narrative

Tables go in as direct triplets. Each cell becomes a Graphiti edge with the exact value verbatim. No LLM in the path because LLM extraction paraphrases numbers and loses precision. Narrative text (pre_text, post_text) is chunked and processed by Graphiti's LLM to extract entities and relations; that's where the semantic search comes from.

See [docs/hybrid_graph_schema.md](docs/hybrid_graph_schema.md) for an annotated example.

### Three execution modes

Each mode is a separate `AnswerStrategy` class. Adding a new mode means a new file; nothing existing changes.

1. **Graph-RAG.** The default for indexed records. Runs a hybrid BM25 + embedding search, returns the top-ranked facts from both table and narrative, and answers in one pass.
2. **Long Context.** The fallback when a record is not indexed, or when `--no-index` is set. The full `pre_text`, table, and `post_text` go directly into the prompt, with no retrieval.
3. **Agentic RAG.** Requires `--tools`. The model gets the full table pre-loaded and three tools: `search_document` (parallel hybrid search over table cells AND narrative, returning both result sets in one call), `compute` (arithmetic evaluator), and `compare` (yes/no comparison for questions like "did X grow?"). The model inspects both result sets from each `search_document` call and picks the value whose type matches the question.

All three modes render the raw table JSON as a `tabulate` grid before injecting it into the prompt, and share 16 financial-domain prompt rules in a single `_DOMAIN_RULES` constant.

### Conversation history

Every prior turn's messages are persisted to SQLite and replayed verbatim. The model sees the actual thread, not a Q→A summary. A separate lightweight summary feeds the Graph-RAG query rewriter, which only needs coreference context.

### Structured outputs via Responses API

All answer paths use the OpenAI Responses API (`client.responses.parse`) with a Pydantic schema (`FinancialAnswer`) as `text_format`. The schema is enforced server-side, so the model physically cannot return malformed JSON. The three fields (`reasoning`, `determinable`, `answer`) arrive as a typed object; no string scraping, no regex, no `json.loads`. If `answer` contains a math expression like `"108 / 6197"`, `simpleeval` evaluates it precisely. If the model sets `determinable=False`, the answer resolves to `"unknown"` without any string matching.

This replaces an earlier `ANSWER:` prefix protocol that required regex extraction and broke on edge cases. The structured approach is more robust: the schema's `Field(description=...)` is the authoritative field-level instruction, so the prompt only carries domain logic (financial conventions, CoT steps), not output format guidance.

### LLM choices

The agent uses `gpt-5-mini` for reasoning. It produces clean chain-of-thought traces and costs less than full `gpt-5`, which matters when running 349 questions with a judge on each.

Everything else uses `gpt-4o-mini`. The query rewriter runs at `temp=0` because it is a narrow coreference task; determinism matters more than capability. Graphiti's extraction model is the library default; the pre-cleaned ConvFinQA narrative does not need a stronger model. The judge also runs `gpt-4o-mini` at `temp=0` for the same reason: short structured output, no creativity required.

Embeddings use `text-embedding-3-small`. The hybrid search pairs it with BM25, so any recall gap from the embedding model is partially covered by lexical matching.

### Evaluation: two evaluators, side by side

I ran two evaluators on every question. The code evaluator is the primary: it handles numeric tolerance (scale equivalence, percent vs decimal, inverse ratio) and is deterministic and free to run. It deliberately doesn't handle sign conventions; whether a loss is stored as negative or positive is a semantic judgment that depends on context, so that gets delegated to the judge.

The LLM judge (`--judge`, opt-in) uses `gpt-4o-mini` to decide whether predicted and gold are semantically equivalent. It runs alongside the code evaluator and logs every disagreement with a one-sentence rationale. I never silently overrule one with the other; disagreements are signals worth inspecting.

In practice, all disagreements were "code passed, judge rejected": cases where the numeric paths accept a value but the judge flags a semantic issue. That's the right failure mode. The judge is stricter by design.

---

## Error Analysis

### Results on records 1-100 (n=349 questions, `--judge`)

| Mode | Code Acc | Judge Acc | Failures |
|---|---|---|---|
| Graph-RAG | 84.2% (294/349) | 77.4% | 55 |
| Long Context | 82.5% (288/349) | 77.4% | 61 |
| **Agentic RAG** | **85.7% (299/349)** | **79.7%** | 50 |

All three are single full 100-record runs. Reproduce with `uv run main evaluate --limit=100 --judge` (Graph-RAG), `--no-index --judge` (Long Context), and `--tools --judge` (Agentic RAG).

### What the 50 Agentic RAG failures actually are

I went through every failing turn and checked it against the source document and the gold `turn_program`. They fall into four categories:

| Category | Turns | Records |
|---|---|---|
| Dataset annotation error | 26 | 10 |
| Genuine model error | 19 | 9 |
| Evaluator scale/sign gap | 3 | 1 |
| Borderline phrasing | 2 | 2 |

**Dataset annotation errors (26 turns, 10 records).** The gold answer contradicts the source; not recoverable. The model is often right and still scored wrong. Examples: STT/2008 has corrupted min-column values; ETR/2017/114 has the two companies' payment values swapped in the gold program; GS/2012's gold divides by 1000 while its own question text says "divided by 100"; AMAT/2013 and AES/2003 ask for figures that are not present in the provided document; UNP/2006's gold subtracts a billions figure from a millions figure; BLL/2007 counts a five-year span as four.

**Genuine model errors (19 turns, 9 records).** The source supports the gold and the model slipped. This is the real headroom. Two sub-patterns: wrong-cell selection in multi-dimensional tables (IPG/2008 reads the wrong period row, AMT/2010 the wrong quarter, C/2010 and PNC/2013 the wrong denominator), and giving up with "unknown" at the 15-call tool cap when the value was sitting in the narrative (CB/2010, DRE/2013).

**Evaluator scale/sign gaps (3 turns, 1 record).** ZBH/2003: the model returned -0.512 (a signed fraction) where the gold is 51.2 (percentage points). Semantically equivalent, but the code evaluator does not normalize a combined divide-by-100-and-sign difference. A judge accepts it; the code evaluator does not.

**Borderline phrasing (2 turns, 2 records).** BLK/2014 and VLO/2018 ask for a "total portion" / "total number" that reads as a raw amount, but the gold wants it as a fraction of a total. Defensible either way.

### How Agentic RAG compares to the other two modes

The three modes are closer than the headline numbers suggest. Graph-RAG (84.2%) and Long Context (82.5%) are within the ±1pp run-to-run noise and should be read as a tie. Agentic RAG (85.7%) is 3.2pp ahead of Long Context, a gap that holds across runs, but only 1.5pp ahead of Graph-RAG, which does not clear the noise band. The honest conclusion is that Agentic RAG clearly beats Long Context; whether it beats Graph-RAG on this sample cannot be determined.

Against Long Context, the gap is explained by attention dilution: Long Context hands the model the full document and relies on it to find the right number. It often doesn't. Long Context's 61 failures are dominated by misses where the value was present in the document but the model didn't focus on it. Agentic RAG's on-demand retrieval forces it to search explicitly rather than scan passively.

Against Graph-RAG, the mechanism Agentic RAG adds is `search_document`: Graph-RAG answers in one pass from a ranked mixed list and can anchor on a table value that matches the question's words but has the wrong value type. `search_document` always returns both table_cells and narrative_snippets in one call, so the model is structurally forced to compare both sources. The CB/2010 ratio-vs-dollar trap that previously required an application-level guard is now resolved at the tool architecture level: the model sees `-6.8` (table ratio) and `$814 million` (narrative) together, and its interleaved reasoning picks the right one.

#### A concrete example of the trap

Record `Single_CB/2010/page_88.pdf-1`, turn 2: *"and what was it in 2008?"* (it = net favorable prior period development). Gold answer: `814` ($814 million). Both sources contain something called "prior period development", but only one holds the number the question wants:

| Source | What it holds for 2008 | Right? |
|---|---|---|
| Table row `prior period development` | `-6.8` (a percentage-point contribution to the loss ratio) | No (wrong value type) |
| Narrative (post_text) | "...net favorable prior period development ... of $576 million and $814 million in 2009 and 2008" | Yes |

The table row name is a perfect match for the question's words. A model that sees only the table finds `-6.8`, labels it as "prior period development 2008", and is tempted to answer. But `-6.8` is a ratio; the dollar amount the question actually wants (`814`) lives only in the footnote. Plain Graph-RAG, handed both in one ranked list, tends to anchor on the table value and commit. With `search_document`, the model receives both `-6.8` (table_cells) and the narrative snippet containing `$814 million` (narrative_snippets) in the same tool result. Its interleaved reasoning then picks `814` because the question asks for a dollar amount. The trap is resolved at the tool level rather than through application-level enforcement.

### Limitations

#### Dataset quality

The biggest single bucket is not the model: 26 of the 50 Agentic RAG failures are dataset bugs, gold answers that contradict the source document. No amount of architectural work recovers these.

#### Model headroom

The genuine model headroom is 19 turns across 9 records, in two patterns. Wrong-cell selection in multi-dimensional tables (IPG/2008 picks the right program column but the wrong period row; C/2010 and PNC/2013 pick the wrong denominator) is the larger one. The other is giving up with "unknown" at the 15-call tool cap when the value was actually in the narrative (CB/2010, DRE/2013); a higher cap or better narrative ranking would recover those. A stronger base model would help the first pattern more than further prompt tuning.

Indexing takes 20-30 seconds per document, which is fine for batch pre-indexing but awkward for one-off questions on unseen documents.

#### Scale

Long Context works in this evaluation only because ConvFinQA records are pre-extracted short excerpts. In production, a single large company's 10-K runs 200-400 pages; a full filing history across years, subsidiaries, and geographies is effectively an infinite dataset, far beyond any context window. A context window measured in millions of tokens is still a drop in the bucket compared to enterprise data measured in terabytes. Long Context cannot scale to this; a retrieval layer is the only viable approach when the data is unbounded.

The graph is also scoped per document today. Cross-document queries ("how did Bank A's tier-2 compare to Bank B's?") do not work without a federation layer.

**A note on 1M-token context windows.** Current frontier models offer context windows of one million tokens or more (roughly 700,000 words, enough to fit an entire novel series in a single prompt). That capacity makes Long Context genuinely viable for bounded, single-document use cases that were impractical just a few years ago, and it shifts the RAG-vs-Long-Context tradeoff meaningfully. But it does not eliminate the fundamental limitations:

- **Rereading cost persists.** A 1M-token context still requires the model to process every token on every query turn. Prompt caching can offset this for static documents, but any document that changes between sessions still pays the full input cost.
- **Needle-in-a-haystack scales with context size.** Model attention dilutes as context grows. A fact buried in the middle of a 1M-token window is harder to retrieve reliably than a fact surfaced by a targeted retrieval step. The problem does not disappear; it grows with the window.
- **Enterprise data still exceeds any context window.** A full filing history across years, subsidiaries, and geographies is measured in terabytes, not megabytes. Even a 1M-token window is a drop in the bucket at that scale. The infinite dataset problem remains regardless of how large the context window becomes.

#### Retrieval infrastructure overhead

Graph-RAG and Agentic RAG both require an indexing pipeline: Graphiti's entity and edge extraction, Neo4j as the graph store, and an embedding model for semantic search. That infrastructure must be kept in sync with the source data. When a new 10-K is published, the graph must be re-indexed before the retrieval path can answer questions about it. Here that is around 20-30 seconds per document, manageable for batch pre-indexing, but at enterprise scale, where hundreds of filings land weekly, keeping the graph fresh becomes a continuous operational concern. More moving parts means more places for things to break: the extraction model, the graph write path, the embedding model, and the vector index all have to stay consistent.

Long Context carries none of this overhead: no graph, no embedding model, no re-sync pipeline. That is its main operational advantage, and it is real. The trade-off is that Long Context cannot scale beyond what fits in a single context window, while Graph-RAG and Agentic RAG scale to any corpus size at the cost of that maintenance burden.

---

## Future Work

The highest-impact change would be a stronger base model. The genuine model errors are mostly wrong-cell selection in multi-dimensional tables (IPG/2008, C/2010, PNC/2013), where stronger table reasoning would help. Full `gpt-5` would likely close several of those at 5-10x per-question cost. Worth measuring. A cheaper, orthogonal win is raising the 15-call tool cap and improving narrative ranking, which would recover the give-up cases (CB/2010, DRE/2013) where the value was present but the model ran out of calls.

Beyond that, the gaps I would close in order:

- Prompt caching for Long Context. The source document is fixed before the conversation begins and stable across all turns in a session, making it a natural fit for provider prefix caching (Anthropic's `cache_control` markers; OpenAI's automatic per-prefix caching). Only the new query turn would be billed at full input token rates, which would meaningfully reduce Long Context's per-query cost.
- Prompt evaluation system. The manual debug checklist works but does not scale. Integrating a tool like Braintrust, LangFuse, or Arize would instrument every LLM call as a traceable span, automate failure triage, and track accuracy across prompt versions to surface regressions before they reach the eval loop.
- Native Cypher path. Hybrid search works well for lookup questions but struggles with structural ones ("compare metric X across all years"). Letting the model emit Cypher directly would unlock that class.
- Cross-document reasoning. The graph is scoped per document today. Multi-document comparison requires relaxing `group_id` scoping or building a routing layer on top.
- `LLMProvider` Protocol. The system is OpenAI-only. One interface would make provider swapping a one-file change.

---

## How I worked

I used Claude Code as an execution accelerator and OpenSpec for change management. Each feature was scoped as a spec before any code was written, implemented against it, and archived on completion. The throughline was to interrogate the eval rather than treat it as a scoreboard: every number here comes from a committed run I can reproduce, and where a result was noisy or simply did not pay off, I followed the evidence and wrote it down.

### Where AI was used

The assistant handled volume work that would otherwise slow iteration: test fixtures, SDK boilerplate, eval log parsing scripts, and prompt drafts. During failure classification it cross-referenced each failing turn against the source document, proposed a bucket, and flagged potential annotation errors for me to verify.

Where AI was not used: architecture decisions (graph over chunked RAG, the `search_document` tool design, the two-evaluator setup), interpreting ambiguous results, and the final call on every failure classification. The assistant proposed; I checked and decided.

### Key calls

**The two-evaluator setup.** A deliberate choice, not a safety net. Running code and judge side-by-side and logging every disagreement gave a much clearer picture of where the system was actually failing versus where the code evaluator was too lenient. The sign-convention edge cases would not have surfaced without it.

**Failure classification.** The most time-consuming and most useful part. Every failing turn was triaged through a fixed four-step pipeline before any fix was proposed: read the model's chain-of-thought, verify the gold value exists in the source document, check whether the code evaluator was the problem, then check whether retrieval returned the right facts. Only after all four did I consider a prompt or code change. This order is codified in `docs/debug_checklist.md`. Working through all 50 failures this way settled on: 26 dataset-annotation bugs, 3 evaluator scale/sign gaps, 2 ambiguous phrasing, and only 19 genuine model errors, separating the raw 85.7% from the effective 93.1%.

Running this manually for 50 failures makes the need for a prompt evaluation system obvious: tools like Braintrust, LangFuse, or Arize would automate the triage pipeline and surface regressions across prompt changes without manual cross-referencing.

### What didn't work

Surfacing the retrieval reranker score on each fact so the model could weight a strong match over a weak one. In the Graph-RAG path it was accuracy-neutral (+0.9pp, within noise), so I kept it as an interpretable default. In the Agentic RAG path it was slightly negative (-1.4pp, edge of the noise band) and added prompt text plus per-result tokens for no gain, so I reverted it. A clean negative result is still a result.
