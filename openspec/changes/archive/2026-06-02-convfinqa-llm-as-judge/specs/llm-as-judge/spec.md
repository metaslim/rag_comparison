## ADDED Requirements

### Requirement: Judge function with structured verdict

The system SHALL expose `judge_answer(question, gold, predicted, client, model) -> JudgeVerdict`.

`JudgeVerdict` SHALL be a dataclass with fields:
- `correct: bool`
- `reason: str` (one-sentence rationale)
- `confidence: str`: `"high"` or `"low"`

#### Scenario: Equivalent answers verdict
- **WHEN** the question is "what was net income in 2007?", gold is `60.94`, predicted is `"60.94"`
- **THEN** the judge returns `correct=True` with a reason citing numeric equivalence

#### Scenario: Wrong-magnitude rejection
- **WHEN** gold is `100` and predicted is `"1000"` for the same question
- **THEN** the judge returns `correct=False` citing magnitude mismatch

### Requirement: Configurable judge model via env var

The judge model SHALL default to `gpt-4o-mini` and be overridable via the `JUDGE_MODEL` environment variable, exposed in `Settings` as `settings.judge_model`.

#### Scenario: Default model used when env unset
- **WHEN** `JUDGE_MODEL` is not set
- **THEN** `settings.judge_model == "gpt-4o-mini"`

#### Scenario: Env override applied
- **WHEN** `JUDGE_MODEL=gpt-4o` is exported
- **THEN** `settings.judge_model == "gpt-4o"` and the runner passes this to `judge_answer`

### Requirement: Determinism via temperature=0

The judge SHALL call the LLM with `temperature=0` so identical inputs produce identical verdicts within a model version.

#### Scenario: Temperature pinned
- **WHEN** `judge_answer(...)` is invoked
- **THEN** the OpenAI API call includes `temperature=0`

### Requirement: JSON response format

The judge SHALL request structured JSON via `response_format={"type": "json_object"}` and parse the response as `{"correct": bool, "reason": str, "confidence": str}`.

#### Scenario: Malformed JSON treated as failure
- **WHEN** the LLM returns non-JSON content
- **THEN** the judge returns `JudgeVerdict(correct=False, reason="judge_error: ...", confidence="low")`

### Requirement: Fail closed on API errors

If the OpenAI call raises (rate limit, network, timeout), the judge SHALL return `correct=False` with an error reason and `confidence="low"`. The exception SHALL NOT propagate to the runner.

#### Scenario: Rate-limit error treated as not-correct
- **WHEN** the OpenAI call raises `RuntimeError("rate limit exceeded")`
- **THEN** the judge returns `correct=False` with `reason` containing `"judge_error"` and `confidence="low"`

### Requirement: Side-by-side execution alongside code evaluator

When `use_judge=True`, the evaluation runner SHALL invoke `is_correct` (the seven-path code evaluator) AND `judge_answer` for every question, recording both verdicts independently in `EvaluationResult`.

#### Scenario: Both verdicts recorded
- **WHEN** `run_evaluation(use_judge=True)` processes a question
- **THEN** `result.correct` reflects the code evaluator
- **AND** `result.judge_correct` reflects the judge

### Requirement: Disagreement logging

When `is_correct` and `judge_answer` produce different verdicts on the same question, the runner SHALL append a structured entry to `result.judge_disagreements`:

```python
{
    "record_id": str,
    "turn": int,
    "question": str,
    "gold": str | float,
    "predicted": str,
    "code_correct": bool,
    "judge_correct": bool,
    "judge_reason": str,
}
```

#### Scenario: Code says wrong, judge says right
- **WHEN** the code evaluator marks `predicted="1.5 billion"` against `gold="1500000000"` as wrong (no scale path matches the wording)
- **AND** the judge marks it as semantically equivalent
- **THEN** an entry is appended to `judge_disagreements` with both verdicts and the judge's reason

#### Scenario: No disagreement, no log entry
- **WHEN** both evaluators agree
- **THEN** `judge_disagreements` is unchanged

### Requirement: Opt-in via `--judge` CLI flag

The judge path SHALL be opt-in via `evaluate --judge`. The default `evaluate` flow SHALL NOT invoke the judge and SHALL incur no extra API cost.

#### Scenario: Default invocation skips judge
- **WHEN** the user runs `uv run main evaluate --limit=100`
- **THEN** `judge_answer` is not called and `result.judge_correct` stays at `0`

#### Scenario: Opt-in invocation enables judge
- **WHEN** the user runs `uv run main evaluate --limit=100 --judge`
- **THEN** the judge is invoked for every question and the output shows `LLM judge: X/Y correct (Z%), N disagreement(s)`
