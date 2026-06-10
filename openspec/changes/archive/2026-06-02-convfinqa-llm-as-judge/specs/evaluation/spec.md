## MODIFIED Requirements

### Requirement: Evaluation result carries both evaluators when judge enabled

`EvaluationResult` SHALL track the code evaluator's count (`correct`) and the LLM judge's count (`judge_correct`) independently. A `judge_accuracy` property SHALL expose `judge_correct / total`.

#### Scenario: Both counts visible after a `--judge` run
- **WHEN** `evaluate --judge --limit=100` completes
- **THEN** `result.correct`, `result.judge_correct`, `result.judge_disagreements`, and `result.judge_accuracy` are all populated

#### Scenario: Judge counts stay zero when judge disabled
- **WHEN** `evaluate --limit=100` (no `--judge` flag)
- **THEN** `result.judge_correct == 0` and `result.judge_disagreements == []`

## ADDED Requirements

### Requirement: Cost-bounded judge invocation

When the judge is enabled, the runner SHALL invoke it exactly once per question (not once per record, not once per turn-position). A 349-question eval SHALL incur exactly 349 judge LLM calls.

#### Scenario: Call count matches question count
- **WHEN** `run_evaluation(use_judge=True)` processes N questions
- **THEN** `judge_answer` is awaited exactly N times
