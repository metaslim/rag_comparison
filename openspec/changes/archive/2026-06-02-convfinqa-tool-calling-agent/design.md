## Context

The current agent is a *retrieve-then-reason* pipeline: query rewriter → graph search (Mode 1) or raw doc (Mode 2) → one LLM call with structured CoT and `simpleeval`-evaluated `ANSWER:`. It hits 87.4% raw / 96.2% effective on 100 dev records.

REPORT.md documents the remaining failures as falling into four shapes:

1. **Column/row ambiguity** (records 25, 29, 72, 99): multiple candidate labels exist; model picks silently.
2. **Convention errors** (record 56): sign/baseline conventions held by prompt rules; model occasionally forgets.
3. **Compound questions**: one query rewriter pass under-serves one half of "compare X and Y" shapes.
4. **Mid-chain arithmetic precision**: only the final expression is evaluated deterministically; intermediate steps live in the model's head.

Prompt engineering has hit its ceiling on a non-deterministic model. The next move is *architectural*: structural tools that turn the model's implicit choices into explicit, observable, recoverable function calls.

## Goals / Non-Goals

**Goals:**

- Replace 5 of 8 curated prompt rules with structural tool contracts (deterministic, testable, regression-resistant).
- Surface ambiguity as a *typed error* with candidates, not a silent wrong answer.
- Make mid-chain arithmetic observable so step-2 mistakes are visible in the trace.
- Preserve current behaviour as the default; tool-calling is opt-in (`--tools` flag) until validated.

**Non-Goals:**

- Replacing the single-pass CoT entirely. Mode 2 (raw document) still uses single-pass: tools are graph-backed.
- Free-form ReAct loops. The loop is hard-capped at 15 calls.
- Cross-record reasoning. Tools are scoped to the current record's `group_id`.
- New retrieval modes. Tools wrap existing search + table access; no new index types.

## Decisions

### D1: Six tools, scoped to documented failure modes

**Decision**: Ship exactly these six tools, no more:

| Tool | Signature | Replaces |
|---|---|---|
| `search_facts` | `(query: str) → list[fact: str]` | One-shot retrieval limitation (compound questions, missed facts) |
| `lookup_cell` | `(column: str, row: str) → value \| AmbiguousError(candidates)` | Rules 5, 12→15 |
| `list_columns_matching` | `(query: str) → list[str]` | Implicit half of Rule 5/12 |
| `list_rows_matching` | `(query: str) → list[str]` | Rule 4 |
| `compare` | `(metric: str, from_col: str, to_col: str) → {from, to, abs_change, pct_change}` | Rules 3, 6 |
| `compute` | `(expression: str) → number` | Mid-chain extension of `ANSWER:` |

**Rationale**: each tool maps directly to a documented failure mode in REPORT §"Error Analysis". Tools that *don't* address a real failure (cite, cypher, get_text) are deliberately excluded: YAGNI.

### D2: Errors are first-class signals, not exceptions

**Decision**: `lookup_cell` returns `AmbiguousError(candidates: list[str])` as a *structured value*, not a Python exception. The model sees the candidates in the next message and can refine its query.

**Rationale**: silent wrong answers are the current failure mode. Forcing the model to *see* the ambiguity and pick from a finite candidate list is the whole point of the tool. Raising would just kill the loop; returning a typed error keeps the model in control.

### D3: Loop discipline

**Decision**: hard cap of **15 tool calls per question**. After the cap, the agent prompt forces an `ANSWER:` response.

**Budget rationale**:
- 2-3 `list_*` calls to probe schema (often multiple disambiguations on the same question)
- 3-5 `lookup_cell` calls to retrieve values (compound questions, cross-year comparisons)
- 4-6 `compute` calls to derive intermediates step-by-step
- Headroom for one or two retries when `AmbiguousError` is returned

A tighter cap (e.g. 6) starved compound questions and exploratory schema probing. 15 is generous enough for the hardest ConvFinQA questions and still tight enough to prevent runaway loops on stuck reasoning.

**Failure mode tracking**: loop-cap exhaustion is logged as a distinct failure category in `data/eval_results.json` so we can see if it's a meaningful percentage (vs. wrong-tool-pick or wrong-arg errors).

### D4: Tool implementations are pure functions over the record

**Decision**: every tool takes the current `ConvFinQARecord` (and the graph for `search_facts`) and returns a deterministic result. No hidden state, no global side effects.

**Rationale**:
- Testable: each tool has unit tests with no mocks needed (except `search_facts`).
- Predictable: same inputs always produce same outputs. Good behaviour for an LLM tool surface.
- Mockable in production: a future `MockToolRegistry` for offline eval is trivial.

### D5: Tools opt-in via `--tools` flag

**Decision**: `evaluate --tools` and `chat --tools` enable tool-calling. Default behaviour is unchanged.

**Rationale**:
- Lets the existing 96.2% effective baseline stand untouched while the tool path is validated.
- Allows side-by-side A/B comparison (`evaluate --limit=100` vs `evaluate --limit=100 --tools`) in the same run.
- Easy rollback if tool-calling regresses anything.

Once validated (probably +3-4 records, no regressions), make it the default and remove the flag.

### D6: Tool contracts replace 5 prompt rules; 3 remain

**Decision**: trim system prompt from 8 rules to 3.

**Kept** (genuine domain knowledge, not coordinate-resolution):
- Rule 1 (Parenthetical % is the answer)
- Rule 2 (Portion = subtotal ÷ total)
- Rule 7 (Total/aggregate price uses TABLE row)

**Removed** (now enforced by tools):
- Rule 3 (Change = later − earlier) → `compare` returns all four interpretations
- Rule 4 (Prefer FINAL "total" row) → `list_rows_matching("total")` surfaces all candidates
- Rule 5 (Program names by year) → `list_columns_matching` shows both forms
- Rule 6 (Stockholder fluctuation = end − 100) → `compare(metric, baseline, end)` handles convention
- Rule 12 → Rule 15 exception → `lookup_cell` + `AmbiguousError` makes it impossible to pick silently

**Rationale**: each removed rule was a *workaround* for a structural problem (the model picking the wrong label). Once the structure surfaces ambiguity, the workaround is redundant: and worse, can conflict with the tool result.

### D7: Tool-loop fallback to single-pass on exhaustion

**Decision**: if the loop hits the 15-call cap or the model refuses to emit a final `ANSWER:`, fall through to the existing single-pass CoT path on the *raw retrieved facts* (whatever the tool calls accumulated).

**Rationale**: never silently fail. The single-pass path is proven (87.4% raw on 100 records); using it as fallback bounds the worst case to "current accuracy."

### D8: Tool-trace logging in eval results

**Decision**: `data/eval_results.json` gains a `tool_traces` field per question listing `(tool_name, args, result)` tuples.

**Rationale**: post-eval analysis becomes mechanical. "Which records hit the loop cap?" "Which tools fired most often?" "Where did the model pick the wrong column?" All answerable from logs, no re-running.

### D9: Early-termination guard for tool dependencies (prepareStep pattern)

**Decision**: before accepting a final `ANSWER:` message from the model, the agent loop inspects the tool trace. If `search_table` was called this turn but `search_narrative` was not, the loop injects a one-shot system message ("you have not checked narrative; a table row may hold a ratio while the dollar amount lives in pre/post text") and re-prompts the model. Fires at most once per turn. Transient nudges are filtered from the persisted message thread.

**Rationale**: dominant Mode 3 failure pattern (CB/2010 and similar) was the model calling `search_table`, getting a plausible-looking value, and committing without ever consulting narrative. Six rounds of prompt-side fixes (stronger tool descriptions, parallel-tool routing rules, value-type cues) did not flip this -- description-only guidance in the prompt is too soft a signal. The fix follows OpenAI's documented multi-tool sequencing pattern ("a multi-step conversation between your application and a model"): dependencies are enforced in application code, not prompts. The OpenAI SDK has no native `must_call=[X, Y]` flag, so we open-code the dependency. Equivalent to Vercel AI SDK's `prepareStep` hook.

**Measured impact**: Mode 3 aggregate moved from 81.4% (pre-guard) to 88.3% (post-guard) on the 100-record A/B; CB/2010 went 0/5 to 5/5. The guard is general (any record matching the trigger fires it), not record-specific.

**Persistence**: the guard's nudge is a transient prompt within one turn. The agent's `_is_control_flow_message` filter drops it from `save_messages` so it is not replayed on subsequent turns (where it would confuse the model on questions that do not match the pattern).

## Risks / Trade-offs

- **Latency**: tool-calling adds ~2× LLM calls per question. From ~3s to ~7-10s. Acceptable for an interactive agent; not for a high-throughput batch job.
- **Cost**: ~2× more API spend per evaluation run. Mitigated by `--tools` being opt-in until validated.
- **Tool-args correctness**: the model might emit `lookup_cell(column="2007 program", row="Total")` when the row is actually `"total"` (case-sensitive). Mitigated by tool implementations normalising whitespace and case before lookup, with the original spelling preserved in the trace.
- **Loop-cap underestimation**: 15 calls is generous but not unbounded. The fallback path bounds the risk to current accuracy. If loop-cap exhaustion shows up in >5% of evals during validation, audit the traces for stuck patterns before raising the cap further.
- **Backwards compatibility**: zero risk: tools are opt-in via `--tools` and the default path is byte-identical.

## Open Questions

- **Should `lookup_cell` accept fuzzy matches with a confidence score?** Probably not. The whole point is forcing exact commitment; fuzziness reintroduces the silent-wrong-pick failure mode. Reconsider only if validation shows the model routinely emits wrong-case or wrong-spacing args.
- **Should there be a `verify(claim, source)` tool for the table-vs-text conflict (record 25)?** Marginal: addresses one record, adds ceremony on the other ~95. Deferring to Future Work pending validation evidence.
