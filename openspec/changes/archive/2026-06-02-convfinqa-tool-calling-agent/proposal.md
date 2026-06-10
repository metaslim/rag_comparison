## Why

The current agent does *retrieve → CoT reason → answer* in a single LLM pass. This works (87.4% raw / 96.2% effective on 100 dev records) but the documented brittle failures are structural, not reasoning-quality issues:

- **Column/row ambiguity** (records 25, 29, 72, 99): model silently picks the wrong label when multiple candidates exist.
- **Convention errors** (record 56): model has to remember sign/baseline conventions from prompt rules.
- **Compound-question retrieval**: one query rewriter pass under-serves one side of "compare X and Y" questions.
- **Precision propagation**: mid-chain arithmetic happens in the model's head; only the final expression is evaluated deterministically.

Eight curated prompt rules currently paper over these. Each new rule risks regressing other records (REPORT documents *three regressions in three attempts*). That's the practical ceiling of prompt engineering on a non-deterministic model.

A tool-calling architecture *replaces 5 of those 8 prompt rules with structural, testable function APIs* and turns silent wrong choices into observable, recoverable failures. The model commits to exact row/column labels; ambiguous lookups raise with candidates; arithmetic steps are evaluated mid-chain.

## What Changes

- **New**: Tool-calling agent layer using OpenAI's function-calling API. Six tools cover retrieval, schema probing, exact lookup, year-over-year comparison, and mid-chain arithmetic.
- **New**: `src/convfinqa/agent/tools.py`: pure-function tool implementations and JSON-schema definitions, swappable behind a `ToolRegistry` Protocol.
- **Modified**: `FinancialAgent` gains a tool-loop branch (max 6 calls per question, then forced `ANSWER:`). The single-pass CoT path is preserved as a fallback for unindexed records (Mode 2).
- **Modified**: System prompt shrinks from 8 curated rules to 3 (only those that aren't replaceable by a tool). The other 5 become tool contracts.
- **Modified**: Evaluation runner records the tool-call trace per question alongside the final answer, so failures are diagnosable from logs.

## Capabilities

### New Capabilities

- `financial-agent-tools`: Six tools: `search_facts`, `lookup_cell`, `list_columns_matching`, `list_rows_matching`, `compare`, `compute`: with strict typed signatures, structured error returns for ambiguity, and a per-question loop cap.

### Modified Capabilities

- `financial-agent`: Adds a tool-loop branch alongside the existing single-pass CoT. Trims 5 prompt rules now enforced by tool contracts. Adds tool-trace logging for diagnosability.
- `evaluation`: Records tool-call traces in `data/eval_results.json` for every question. Loop-cap exhaustion counts as a failure mode separate from reasoning errors.

## Impact

- **Dependencies**: no new packages (`simpleeval`, `openai`, the graph already in place are sufficient).
- **Module additions**:
  - `src/convfinqa/agent/tools.py`: tool implementations + JSON schemas.
  - `src/lib/interfaces.py`: add `ToolRegistry` Protocol.
- **Module changes**:
  - `src/convfinqa/agent/agent.py`: tool-loop path; existing single-pass path retained for Mode 2.
  - `src/convfinqa/agent/prompts.py`: trim rules 3, 4, 5, 6, 12→15 exception (now tool contracts).
  - `src/convfinqa/evaluation/runner.py`: record tool traces.
- **Tests**: ~25 new unit tests for tools (pure functions, mockable graph). Existing 132 tests unchanged.
- **Backwards compatibility**: tool-calling is opt-in via `--tools` flag on `evaluate` and `chat`. Default behaviour is unchanged.
- **Expected accuracy delta**: +3-4 records on the brittle set (records 25, 29, 72, 99). No regression on currently-passing records because the single-pass path is the fallback.
- **Cost delta**: ~2× LLM calls per question on average (1 planning call + 1-4 tool resolutions + 1 final answer), each on `gpt-5-mini`. Latency goes from ~3s to ~7-10s per question.
