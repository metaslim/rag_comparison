## ADDED Requirements

### Requirement: Chain-of-Thought reasoning with `gpt-5-mini`

The agent SHALL use `gpt-5-mini` (a reasoning model) for the main question-answering call. The system prompt SHALL produce structured Chain-of-Thought output:

```
Step 1: Resolve references. What does "that" / "it" / "this" refer to?
Step 2: Look for the answer directly in the retrieved facts
Step 3: Compute from values if no direct answer
ANSWER: <numeric value or math expression>
```

#### Scenario: Single-turn direct lookup
- **WHEN** the user asks "what was net cash from operating activities in 2009?"
- **THEN** Step 1 notes there are no references to resolve, Step 2 surfaces the retrieved fact, and `ANSWER:` returns the value

#### Scenario: Multi-turn calculation from history
- **WHEN** the user asks "what is the difference?" after prior turns established two values
- **THEN** Step 1 resolves "the difference" to the two prior answers, Step 3 computes the difference, `ANSWER:` returns the result

### Requirement: Math expression evaluation in ANSWER

The `ANSWER:` field SHALL accept a math expression (e.g. `108 / 6197`, `1500 + 2000 - 100`), which the agent SHALL evaluate using `simpleeval` for exact arithmetic. The evaluated numeric result is stored in conversation history (not the original expression).

#### Scenario: LLM emits a math expression
- **WHEN** the LLM responds with `ANSWER: 30092 / 63884`
- **THEN** the agent evaluates to `0.471038`, stores that value in history, and returns it

### Requirement: Curated prompt rules

The system prompt SHALL contain a curated set of domain rules, each added in response to a concrete failure and regression-tested before keeping. Current rules cover:

1. Parenthetical % is the answer (`"decreased by 13.4%"` returns `0.134`)
2. Portion = subtotal ÷ total
3. Change direction = later year − earlier year
4. Prefer FINAL `total` row over intermediate subtotals
5. Program names by year (`"1995 program"` is a name, not a date)
6. Stockholder fluctuation = end − 100 (for tables starting at 100)
7. Total/aggregate price uses TABLE row over text breakdowns
8. Rule 12 exception → Rule 15 for `[year] program` columns

#### Scenario: Parenthetical percentage
- **WHEN** a fact contains `"decreased by $X (13.4%)"` and the question asks for the percentage change
- **THEN** the agent returns `0.134` per Rule 3 (parenthetical % is the answer)

### Requirement: Decimal-format enforcement across turns

The agent SHALL store all numeric answers as decimals, never percentages. This prevents scale drift across conversation turns.

#### Scenario: Percentage answer stored as decimal
- **WHEN** the answer is conceptually "22.35 percent"
- **THEN** the stored value in conversation history is `0.2235`, so subsequent turns computing `Q1 - X` use `0.2235`, not `22.35`

### Requirement: Two retrieval modes

The agent SHALL operate in one of two modes per record:

- **Mode 1 (graph search)**: triplets + episodes via Graphiti hybrid search. Primary mode for indexed records.
- **Mode 2 (raw document fallback)**: `pre_text + table JSON + post_text` passed inline to the LLM. Kicks in when the record is unindexed or graph search returns empty.

#### Scenario: Indexed record uses Mode 1
- **WHEN** the record's `group_id` has data in the graph
- **THEN** the agent retrieves facts via hybrid search and passes them to the LLM

#### Scenario: Un-indexed record falls back to Mode 2
- **WHEN** graph search returns an empty list
- **THEN** the agent builds context from `pre_text + table + post_text` directly (truncated to `MAX_CONTEXT_CHARS=12000` if needed) and still returns an answer

### Requirement: Query rewriting for reference resolution

The agent SHALL rewrite the search query before issuing it to the graph, using `gpt-4o-mini` at `temp=0`, given the conversation history. This resolves references like `"that"`, `"it"`, `"those"` so the graph search has concrete keywords.

#### Scenario: Follow-up question gets rewritten
- **WHEN** the user asks "and what about 2008?" after asking about 2007 net income
- **THEN** the rewritten query is `"net income 2008"` (not the raw `"and what about 2008?"`) so BM25 has meaningful tokens

### Requirement: Dependency injection (Protocol-based)

`FinancialAgent` SHALL depend on the `GraphStore` and `HistoryStore` Protocols defined in `src/lib/interfaces.py`, not on concrete classes. This makes the agent testable with mocks and swappable with alternative backends without touching the agent code.

#### Scenario: Test with mocks
- **WHEN** unit tests instantiate `FinancialAgent` with mock graph and mock store
- **THEN** the agent runs without any real Neo4j or OpenAI calls
