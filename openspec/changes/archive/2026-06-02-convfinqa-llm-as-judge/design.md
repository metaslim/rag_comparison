## Context

The seven-path code evaluator (`is_correct` in `src/lib/evaluation/metrics.py`) catches most numeric-tolerance cases but has hard limits:

- It can't evaluate text answers ("yes" / "no") because the agent only produces numbers.
- It treats `0.358` and `35.8` as equivalent via the reverse-percent path, even when the question wording implies a raw subtraction: this is leniency that hides format bugs.
- It can't reason about question intent: "what was the change" vs "what was the percent change" need different gold-format expectations.

A small LLM judging `(question, gold, predicted)` triples can address all three. The key design choice is whether to *replace* the code evaluator or *augment* it.

## Goals / Non-Goals

**Goals:**

- Add a second evaluator that catches semantic-equivalence cases the code path misses.
- Run *side by side* with the code evaluator so disagreements surface, not hide.
- Make the judge model configurable via env var.
- Bound cost (one extra LLM call per question; opt-in flag).
- Fail closed: API errors return `correct=False` with a logged reason, never silently passing a wrong answer.

**Non-Goals:**

- Replacing the code evaluator. The code path is fast, free, and deterministic; the judge is slower and stochastic.
- Calibration against human raters. That's a measured-methodology task; this change ships the mechanism, not the calibration.
- Multiple judges or ensemble voting. YAGNI: one judge is enough to surface disagreements; if disagreements are noisy we can add a second model later.

## Decisions

### D1: Run side by side, log disagreements

**Decision**: The judge runs in addition to the code evaluator on every question when `--judge` is set. `EvaluationResult.correct` continues to reflect the code evaluator's count; `judge_correct` is a separate count. Cases where the two disagree go into `judge_disagreements` with full context.

**Rationale**: replacing the code evaluator would discard a deterministic signal in favour of a stochastic one. Augmenting preserves both and surfaces *interesting* cases (where the evaluators differ) for human review. The disagreement count is itself a useful metric: high disagreement means one of the evaluators is miscalibrated.

### D2: `gpt-4o-mini` at `temperature=0`

**Decision**: default judge model is `gpt-4o-mini` at `temperature=0`, configurable via `JUDGE_MODEL` env var.

**Rationale**:
- `gpt-4o-mini` is cheap (~$0.001 per judgment) and accurate enough for binary semantic comparison.
- `temperature=0` makes the judge as deterministic as the model allows: same input gives same verdict on the same day. (gpt-4o supports temp=0; gpt-5-mini does not.)
- A reasoning model would be overkill for binary equivalence; this isn't a CoT task.
- Configurable via env var so a future evaluator can swap to a stronger model without code changes.

### D3: `JudgeVerdict` carries reason and confidence

**Decision**: the judge returns a dataclass with `correct: bool`, `reason: str`, `confidence: "high" | "low"`.

**Rationale**:
- `reason` is what makes disagreements actionable. "Wrong magnitude" vs "Sign convention ambiguous" tell different stories.
- `confidence` lets the judge declare when it's making a close call. Low-confidence cases are prime candidates for human review.
- No need for floats or other complex confidence scales: `high`/`low` is enough signal and easy for the LLM to produce reliably via the system prompt.

### D4: Fail closed on API errors

**Decision**: if the judge LLM call raises (rate limit, network, malformed JSON), the function returns `JudgeVerdict(correct=False, reason="judge_error: <e>", confidence="low")`.

**Rationale**: a transient outage that silently turns a wrong answer into "correct" would be the worst outcome. Failing closed (always-False on error) means errors show up as judge_correct dropping, not as bogus correctness. The error reason is preserved so we can spot rate-limit storms in logs.

### D5: Pure function with dependency injection

**Decision**: `judge_answer(question, gold, predicted, client, model)` takes the OpenAI client as an explicit parameter rather than constructing one internally or reading from a global.

**Rationale**: testable without mocking module-level imports (unit tests pass a `MagicMock` directly); same client can be shared across many judge calls in a run; no hidden coupling to the global `settings`.

### D6: `--judge` opt-in flag

**Decision**: `evaluate --judge` enables the judge; default eval flow is unchanged. The two flags `--tools` and `--judge` are independent (combinable).

**Rationale**:
- The 96.2% effective baseline stands untouched until validated.
- The judge adds API cost (~$0.35 per full run); making it opt-in respects the user's budget choice.
- Combinable with `--tools` because the judge evaluates the *output*, not the path that produced it.

## Risks / Trade-offs

- **Judge bias**: the LLM may be systematically too lenient or too strict. Mitigation: log disagreements with reasons; a small sample of human review on the disagreement set is enough to calibrate, and is listed as remaining work in REPORT.md.
- **Cost**: ~$0.35 per full eval run with `--judge`. Acceptable for ad-hoc analysis, not for high-frequency CI.
- **Non-determinism**: even at `temperature=0`, the model can drift across runs. Mitigation: disagreements are logged with reasons, so flaky verdicts surface in the diff rather than getting lost.

## Open Questions

- **Should disagreements affect the displayed accuracy number?** Currently no: `result.accuracy` is the code-evaluator count, `result.judge_accuracy` is the judge count, both shown separately. Leaving this open until we've seen disagreement patterns on a 100-record run.
- **Multi-shot voting (ask the judge 3× and take majority)?** Not implementing now. YAGNI; revisit if single-shot variance is too high.
