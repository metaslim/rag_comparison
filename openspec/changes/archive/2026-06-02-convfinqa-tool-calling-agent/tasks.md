# Tasks (final status)

Goal: Tool-calling agent (Mode 3) with early-termination guard. **Shipped and measured: 88.3% raw / 97.2% effective on 100 records. +9 questions over Graph-RAG.**

## Final architecture (post-refactoring)

- `src/convfinqa/agent/tools.py` → `src/lib/answers/tool_calling/tools.py`
- `src/convfinqa/agent/tool_registry.py` → `src/lib/answers/tool_calling/tool_registry.py`
- `ToolRegistry` protocol → `src/lib/answers/tool_calling/interfaces.py`
- Tools: `search_table`, `search_narrative`, `compute` (3 tools, not 6 from original spec)
- Early-termination guard (prepareStep pattern) enforces `search_narrative` check before any `ANSWER:` that used `search_table`
- `ToolCallingStrategy` in `src/lib/answers/tool_calling/strategy.py` implements `AnswerStrategy`
- `answer_with_tools()` on `FinancialAgent`; fallback saves correct `TOOL_SYSTEM_PROMPT` context

## 1. Tool implementations

- [x] 1.1 `search_table(query, graph, record_id)`: hybrid BM25 + embedding graph search over table cells. Returns `[{column, row, value}]`.
- [x] 1.2 `search_narrative(query, graph, record_id)`: BM25 search over pre_text + post_text episodes. Returns `[{source, snippet}]`.
- [x] 1.3 `compute(expression)`: wraps `simpleeval`. Raises `ValueError` on bad input.
- [x] 1.4 All tools use `GraphSearcher` type (not `GraphStore`) since tools only read.

## 2. Tool registry & dispatch

- [x] 2.1 `ToolRegistry` Protocol in `src/lib/answers/tool_calling/interfaces.py`.
- [x] 2.2 `ConvFinQAToolRegistry` bound to a record (`IndexedRecord`) + graph (`GraphSearcher`).
- [x] 2.3 JSON schemas in `ALL_TOOL_SCHEMAS`.

## 3. Agent tool-loop

- [x] 3.1 `FinancialAgent.answer_with_tools()`: separate entry point.
- [x] 3.2 Hard cap at 15 tool calls. After cap: system nudge → fallback to single-pass if model still refuses.
- [x] 3.3 **Early-termination guard (prepareStep):** intercepts `ANSWER:` if `search_table` called without `search_narrative`. Fires once per turn; transient nudges filtered from persisted thread.
- [x] 3.4 Full message thread replay (tool_calls, tool results) persisted to SQLite.
- [x] 3.5 Fallback saves `TOOL_SYSTEM_PROMPT` + table as system context (not `SYSTEM_PROMPT`).

## 4. Prompt

- [x] 4.1 `TOOL_SYSTEM_PROMPT`: pre-loaded table grid + 16 domain rules + tool usage guidance.

## 5. CLI

- [x] 5.1 `evaluate --tools`: uses `ToolCallingStrategy`.
- [x] 5.2 `chat --tools`: same.
- [x] 5.3 Output mode suffix: `_mode_3.json` via `Path.with_stem()`.

## 6. Evaluation runner

- [x] 6.1 `EvaluationResult` has `tool_traces`, `loop_cap_exhausted`.
- [x] 6.2 Per-question tool trace logged (e.g. `tools=search_table -> search_narrative`).

## 7. Tests

- [x] 7.1 `tests/lib/answers/tool_calling/test_tools.py`: tool function unit tests.
- [x] 7.2 `tests/lib/answers/tool_calling/test_tool_registry.py`: dispatch tests.
- [x] 7.3 `tests/convfinqa/agent/test_tool_loop.py`: cap, fallback, guard, trace recording.
- [x] 7.4 231 total tests, 100% coverage, ruff clean.

## 8. Validation

- [x] 8.1 `evaluate --limit=100 --tools --judge`: 308/349 = 88.3% code, ~81.1% judge.
- [x] 8.2 vs Graph-RAG: +9 questions. Guard accounts for most of the gap (CB/2010: 0/5 → 5/5).
- [x] 8.3 Loop-cap exhausted: 0 questions on 100-record run.

## 9. Documentation

- [x] 9.1 REPORT.md: Mode 3 description, Agentic RAG vs other modes, early-termination guard rationale.
- [x] 9.2 README.md: `--tools` examples, mode descriptions.

## Status

**Complete.** 88.3% raw / 97.2% effective on 100 dev records. Guard eliminated source-routing failure class. CB/2010 went 0/5 → 5/5. Tool-calling mode available via `--tools` flag; Graph-RAG remains default.
