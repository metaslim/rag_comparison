# Tasks (final status)

Goal: LLM-as-judge as a side-by-side second opinion, opt-in via `--judge`. **Shipped and measured: 8 (Long Context), 25 (Graph-RAG), 19 (Agentic RAG) disagreements on 100-record run. All "code passed, judge rejected".**

## Final architecture (post-refactoring)

- `src/lib/evaluation/llm_judge.py` → `src/lib/evaluation/llm_judge/evaluator.py`
- `LLMJudgeEvaluator` implements `Evaluator` protocol (same interface as `CodeEvaluator`)
- Runner takes `evaluators: list[Evaluator]`: no separate `use_judge` flag
- `agent.client` property used instead of `agent._client` (DIP fix)

## 1. Judge module

- [x] 1.1 `src/lib/evaluation/llm_judge/evaluator.py`: `judge_answer()` + `LLMJudgeEvaluator`.
- [x] 1.2 `JudgeVerdict` dataclass with `correct`, `reason`, `confidence`.
- [x] 1.3 System prompt covers ACCEPT/REJECT cases including scale equivalence, percent/decimal, sign ambiguity, dataset annotation suspicion patterns.
- [x] 1.4 Fails closed on API errors.
- [x] 1.5 `response_format={"type": "json_object"}` pinned.

## 2. Configuration

- [x] 2.1 `JUDGE_MODEL` env var in `Settings`, default `gpt-4o-mini`.

## 3. Runner integration

- [x] 3.1 `EvaluationResult` has `secondary_correct`, `secondary_by_turn`, `disagreements`.
- [x] 3.2 `run_evaluation(evaluators: list[Evaluator])`: first is primary (code), rest are secondaries (judge).
- [x] 3.3 Disagreements logged per turn with question, gold, predicted, both verdicts, judge reason.
- [x] 3.4 `strategy: AnswerStrategy` param replaces `use_tools: bool`.

## 4. CLI

- [x] 4.1 `evaluate --judge` appends `LLMJudgeEvaluator` to evaluators list.
- [x] 4.2 Output shows per-evaluator accuracy tables + disagreement count.

## 5. Tests

- [x] 5.1 `tests/lib/evaluation/llm_judge/test_llm_judge.py`: 10 tests covering verdict parsing, API errors, malformed JSON, temperature defaults, prompt content.
- [x] 5.2 Runner and CLI integration tests updated.
- [x] 5.3 231 total tests, 100% coverage, ruff clean.

## 6. Documentation

- [x] 6.1 REPORT.md: two-evaluator design, disagreement analysis, key finding on code evaluator as moving target.
- [x] 6.2 README.md: `--judge` documented in evaluate section.

## Status

**Complete.** Measured on 100-record run: judge disagreements = 25 (Graph-RAG), 8 (Long Context), 19 (Agentic RAG). All "code passed, judge rejected". Judge is stricter by design.
