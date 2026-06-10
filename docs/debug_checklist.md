# Debug Checklist for Eval Failures

The standard methodology I follow for every failing record. Follow IN ORDER. Never propose a fix without running steps 1 through 4 first.

## Step 1: Check the model's reasoning (CoT output)

Run the failing question, log the full LLM response. Read its `Step 1 / Step 2 / Step 3` output to see exactly which path it took and why.

**Never propose a fix without reading the reasoning first.** Otherwise you're guessing.

## Step 2: Check if the gold value exists in the raw dataset

Look at `turn_program` in the dataset JSON:

- If it's a **literal** (e.g. `"53.14"`), search `pre_text`, `post_text`, and `table` for that value.
- If it's a **computation** (e.g. `"subtract(108.4, 71.0)"`), check the operands exist in the data.

**Key question: is the gold value COMPUTED or DIRECTLY STATED?** This determines whether the model needs to derive it or look it up.

If the gold value isn't derivable from any document field, it's a **dataset extraction error**. Mark unfixable.

## Step 3: Check if it's an evaluator issue

Run `is_correct(got, gold)` directly. The prediction might be semantically right but failing the metric (wrong scale, sign, percent format). If so, fix `src/lib/evaluation/metrics.py`, not the prompt.

## Step 4: Check if it's a search issue

Run `graph.search(record_id, question)` and look at the returned facts. If the relevant fact isn't there, the issue is retrieval. Investigate query rewriting, BM25 + cosine ranking, or the result limit.

## Step 5: Only after 1 through 4, fix prompt or code

If reasoning is wrong, the gold IS in the data, eval is correct, and search returned the right fact, then it's a prompt or agent-logic issue. Add a rule, regression-test it on previously-passing records before keeping.

## Anti-patterns to avoid

- Don't mark records as "TBD" or "skipped" without running the debug pipeline. Every failure must be triaged through steps 1 to 4 at minimum.
- Don't propose prompt rules without checking reasoning first. Brute-force prompt-stacking is brittle.
- Don't burn tokens by re-indexing the full dataset. Use the checkpoint resume; never drop the DB unless explicitly asked.
- Don't change prompts for format or scale issues. That's the evaluator's job in `metrics.py`.
- Don't change prompts for search issues. Fix retrieval first.

## Why this order

- Without reasoning, you guess. Guesses lead to brittle stacked rules.
- Dataset errors are unfixable. Spending time on them wastes tokens.
- Evaluator fixes are cheaper than prompt fixes (deterministic, no LLM cost).
- Search fixes are cheaper than prompt fixes (no per-query LLM cost).
- Prompt is the last resort because it affects all records, with the highest regression risk.
