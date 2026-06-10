## ADDED Requirements

### Requirement: Tool-trace logging per question

When the tool-calling path is used (`--tools` flag), the evaluation runner SHALL record a `tool_traces` array per question in `data/eval_results.json`. Each trace entry SHALL include `tool_name`, `args`, `result`, and `latency_ms`.

#### Scenario: Trace persists across runs
- **WHEN** `evaluate --tools --limit=100` completes
- **THEN** every question in `data/eval_results.json` has a `tool_traces` array (possibly empty for questions answered via fallback)

### Requirement: Loop-cap exhaustion counted as a failure mode

The `EvaluationResult` SHALL track `loop_cap_exhausted: int`: the count of questions where the agent hit the 15-tool-call cap before producing an `ANSWER:`. This SHALL be reported alongside overall accuracy and per-turn breakdowns.

#### Scenario: Final summary distinguishes failure modes
- **WHEN** the eval summary is printed
- **THEN** it includes a row "Loop-cap exhausted" with the count and percentage of total questions
- **AND** loop-cap exhaustion is NOT counted twice (i.e. an exhausted question that also produces a wrong final answer is counted only as a wrong answer)
