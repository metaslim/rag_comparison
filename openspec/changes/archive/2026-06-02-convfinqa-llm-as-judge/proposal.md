## Why

The code-based evaluator in `src/lib/evaluation/metrics.py` chains seven numeric-tolerance paths to accept semantically-correct answers that differ in format (percentage vs decimal, scale 1B vs 1M, sign flips, inverse ratios). It reaches 96.2% effective accuracy on the 100-record sample.

The remaining gap is *semantic* equivalence the tolerance paths can't reach:

- Yes/No gold answers vs numeric outputs (record 72 Q2)
- Unit conversions across the question and prediction (record 95)
- Valid alternative interpretations of ambiguous questions

These were listed in REPORT.md Future Work item #2. This change ships that item.

## What Changes

- **New**: `src/lib/evaluation/llm_judge.py`: `judge_answer(question, gold, predicted, client, model)` returning `JudgeVerdict(correct: bool, reason: str, confidence: str)`. Default model `gpt-4o-mini` at `temperature=0` for determinism.
- **New**: `JUDGE_MODEL` env var (default `gpt-4o-mini`) added to `Settings` so the judge model is configurable without code changes.
- **Modified**: `EvaluationResult` (in `convfinqa/evaluation/runner.py`) gains `judge_correct: int`, `judge_disagreements: list[dict]`, and a `judge_accuracy` property.
- **Modified**: `run_evaluation` gains a `use_judge: bool` parameter. When True, the judge runs *side by side* with the code evaluator; disagreements are appended to `judge_disagreements` with the question, both answers, both verdicts, and the judge's reason.
- **Modified**: `evaluate` CLI gains a `--judge` flag.

## Capabilities

### New Capabilities

- `llm-as-judge`: semantic answer verification via a small LLM. Complements (does not replace) the code evaluator.

### Modified Capabilities

- `evaluation`: optional second evaluator runs alongside the code path; disagreements logged in `data/eval_results.json` for post-hoc analysis.

## Impact

- **Dependencies**: no new packages. Reuses the existing OpenAI SDK and the agent's `AsyncOpenAI` client.
- **Module additions**:
  - `src/lib/evaluation/llm_judge.py` (~70 LoC).
- **Module changes**:
  - `src/lib/evaluation/runner.py`: judge call site + new result fields.
  - `src/main.py`: `--judge` CLI flag.
  - `src/config/settings.py`: `JUDGE_MODEL` env var.
- **Tests**: 8 judge unit tests + 3 runner integration tests + 1 CLI smoke test. Total 12 new tests; existing tests unchanged.
- **Backwards compatibility**: zero impact when `--judge` is not passed (`use_judge` defaults to `False`). API costs are unchanged for the default eval flow.
- **Cost when enabled**: one extra LLM call per question (~$0.001 each on `gpt-4o-mini`). 349 questions × $0.001 ≈ **$0.35 per full eval run with `--judge`**.
- **Calibration is out of scope**: validating the judge against human raters is the part that needs careful methodology and is left for a follow-up (noted in REPORT.md).
