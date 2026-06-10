## ADDED Requirements

### Requirement: Six-tool surface for the agent

The agent SHALL expose exactly six tools to the LLM via OpenAI function-calling:
`search_facts`, `lookup_cell`, `list_columns_matching`, `list_rows_matching`, `compare`, `compute`. No other tools SHALL be exposed without an explicit spec change.

#### Scenario: Tool surface enumeration
- **WHEN** the agent constructs the tool list for an OpenAI call
- **THEN** exactly six tool definitions are passed, with JSON-schema arguments matching the Python signatures

### Requirement: `search_facts`: iterative hybrid retrieval

`search_facts(query: str) → list[str]` SHALL wrap `FinancialGraph.search` scoped to the current record's `group_id`. The model SHALL be able to call it multiple times per question to refine queries or decompose compound questions.

#### Scenario: Compound question decomposed into two searches
- **WHEN** the question is "what was revenue in 2007 and 2008?"
- **THEN** the model may call `search_facts("revenue 2007")` and `search_facts("revenue 2008")` separately, each returning targeted facts

#### Scenario: Thin initial results trigger a refined re-query
- **WHEN** `search_facts("operating cash")` returns fewer than 3 facts
- **THEN** the model may call `search_facts("cash from operations")` to retrieve alternative phrasing

### Requirement: `lookup_cell`: exact-coordinate lookup with structured ambiguity

`lookup_cell(column: str, row: str) → value | AmbiguousError(candidates: list[str]) | NotFoundError(closest: list[str])` SHALL perform a case-insensitive, whitespace-normalised lookup in `record.doc.table[column][row]`. Ambiguous matches SHALL return a structured `AmbiguousError` value (not raise) listing all candidate labels. No-match SHALL return `NotFoundError` with closest labels.

#### Scenario: Exact lookup succeeds
- **WHEN** the table has column `"2007"` with row `"net income"` = 100
- **AND** the model calls `lookup_cell(column="2007", row="net income")`
- **THEN** the tool returns `100` (as a number, not a string)

#### Scenario: Column ambiguity surfaces both candidates
- **WHEN** the table has columns `"2007"` AND `"2007 program"`
- **AND** the model calls `lookup_cell(column="2007", row="liability")`
- **THEN** the tool returns `AmbiguousError(candidates=["2007", "2007 program"])`
- **AND** the model can re-query with the exact label it meant

#### Scenario: Row not found returns suggestions
- **WHEN** the table has row `"balance at end of year"` but the model asks for `"closing balance"`
- **THEN** the tool returns `NotFoundError(closest=["balance at end of year"])`

### Requirement: `list_columns_matching`: schema probe for columns

`list_columns_matching(query: str) → list[str]` SHALL return every column label containing the query string (case-insensitive, normalised whitespace). Returns empty list when no match.

#### Scenario: Probe for year-related columns before lookup
- **WHEN** the question mentions "2007"
- **AND** the model calls `list_columns_matching("2007")`
- **THEN** the tool returns all columns containing "2007", e.g. `["2007", "2007 program", "december 31 2007"]`
- **AND** the model picks the right one before calling `lookup_cell`

### Requirement: `list_rows_matching`: schema probe for rows

`list_rows_matching(query: str) → list[str]` SHALL return every row label (across all columns) containing the query string. Returns empty list when no match.

#### Scenario: Probe for total row, avoid subtotals
- **WHEN** the question asks for "total"
- **AND** the table has rows `"subtotal segment a"`, `"subtotal segment b"`, `"total"`
- **AND** the model calls `list_rows_matching("total")`
- **THEN** the tool returns `["subtotal segment a", "subtotal segment b", "total"]`
- **AND** the model picks `"total"` (the final row) per the remaining prompt rule

### Requirement: `compare`: year-over-year, all four interpretations

`compare(metric: str, from_col: str, to_col: str) → {from: float, to: float, abs_change: float, pct_change: float}` SHALL call `lookup_cell` twice and return all four interpretations of the change. The model SHALL pick which interpretation matches the question shape.

#### Scenario: Standard year-over-year change
- **WHEN** the model calls `compare(metric="revenue", from_col="2007", to_col="2008")`
- **AND** the lookups resolve to 100 and 120
- **THEN** the tool returns `{from: 100, to: 120, abs_change: 20, pct_change: 0.20}`

#### Scenario: Stockholder fluctuation (Rule 6 equivalent)
- **WHEN** the model calls `compare(metric="value", from_col="initial 100", to_col="2010")` for a table starting at 100
- **AND** lookups resolve to 100 and 115
- **THEN** `abs_change=15` and `pct_change=0.15` are both returned; the model picks `abs_change` (per the question shape)

#### Scenario: Either side ambiguous propagates the error
- **WHEN** `compare(...)` triggers an `AmbiguousError` on the `from_col` lookup
- **THEN** the tool returns the `AmbiguousError` upward unchanged, so the model can refine

### Requirement: `compute`: mid-chain deterministic arithmetic

`compute(expression: str) → number` SHALL evaluate a math expression using `simpleeval` with the same allow-list as the existing `ANSWER:` evaluator. Each call SHALL be recorded in the tool trace so intermediate values are observable.

#### Scenario: Multi-step arithmetic with observable intermediates
- **WHEN** the question requires `(revenue_2008 - revenue_2007) / revenue_2007`
- **THEN** the model may call `compute("120 - 100")` → `20`, then `compute("20 / 100")` → `0.20`
- **AND** both intermediates appear in the tool trace

#### Scenario: Invalid expression raises so the model retries
- **WHEN** the model calls `compute("revenue / total")` (non-numeric)
- **THEN** the tool raises a `ValueError` surfaced to the model
- **AND** the model retries with numeric values

### Requirement: Hard cap at 15 tool calls per question

The agent SHALL enforce a maximum of 15 tool calls per question. After the cap, the agent SHALL inject a system message instructing the model to produce a final `ANSWER:`. If the model does not comply, the agent SHALL fall back to the single-pass CoT path using the facts already retrieved.

#### Scenario: Cap respected on simple question
- **WHEN** a single-turn lookup question consumes 2 tool calls
- **THEN** the agent produces an `ANSWER:` without exhausting the cap

#### Scenario: Cap enforced on runaway loop
- **WHEN** the model has made 15 tool calls without emitting `ANSWER:`
- **THEN** the agent appends a system message *"You have used your tool budget. Respond with `ANSWER:` using the facts you have."*
- **AND** if the model still doesn't comply, the single-pass fallback is used

### Requirement: Tool-call traces recorded per question

Every tool invocation SHALL be recorded in `data/eval_results.json` under a `tool_traces` field per question. Each trace entry SHALL include `tool_name`, `args` (JSON), `result` (JSON or error-type tag), and `latency_ms`.

#### Scenario: Trace survives the eval run
- **WHEN** a question completes with 3 tool calls
- **THEN** the corresponding entry in `data/eval_results.json` has a `tool_traces` array of length 3

#### Scenario: Loop-cap exhaustion is countable
- **WHEN** the eval summary is computed
- **THEN** the result includes a `loop_cap_exhausted` count (number of questions that hit the 15-call cap before answering)

### Requirement: Tools opt-in via `--tools` flag

The tool-calling path SHALL be opt-in via a `--tools` flag on `evaluate` and `chat`. The default behaviour SHALL remain the single-pass CoT path until validation completes.

#### Scenario: Default invocation uses single-pass
- **WHEN** the user runs `uv run main evaluate --limit=100`
- **THEN** the existing CoT path is used; no tool calls are made

#### Scenario: Opt-in invocation uses tool path
- **WHEN** the user runs `uv run main evaluate --limit=100 --tools`
- **THEN** the tool-loop path is used; traces are recorded

### Requirement: Early-termination guard for tool dependencies

The agent SHALL enforce a `search_table` -> `search_narrative` dependency in application code (the `prepareStep` pattern). Tool descriptions alone are too soft a signal; the dependency MUST be enforced before `ANSWER:` is accepted.

#### Scenario: Guard fires when search_narrative is missing
- **GIVEN** the model has called `search_table` this turn but not `search_narrative`
- **WHEN** the model emits an assistant message with no `tool_calls` (intends to commit to `ANSWER:`)
- **THEN** the loop appends a one-shot system message instructing the model to call `search_narrative`
- **AND** re-prompts the model for another step

#### Scenario: Guard fires at most once per turn
- **GIVEN** the guard has already fired once this turn
- **WHEN** the model emits another `ANSWER:` without calling `search_narrative`
- **THEN** the loop accepts the final answer (no infinite re-prompting)

#### Scenario: Guard does not fire when search_narrative was used
- **GIVEN** the model called `search_narrative` at any point this turn
- **WHEN** the model emits the final `ANSWER:` message
- **THEN** the answer is accepted without re-prompting (independent of whether `search_table` was used)

#### Scenario: Guard does not fire when no tools were called
- **GIVEN** the model emits `ANSWER:` without making any tool calls (pure history / pre-loaded table derivation)
- **THEN** the answer is accepted; the guard does NOT trigger an unnecessary `search_narrative` call

#### Scenario: Guard nudges are not persisted
- **GIVEN** the guard fired this turn
- **WHEN** the turn's message thread is saved via `save_messages`
- **THEN** the transient system nudge is filtered out (matched by the sentinel "before finalising your answer")
- **AND** the next turn's replay does not include the nudge
