## ADDED Requirements

### Requirement: Pydantic v2 models match the ConvFinQA dataset schema

The system SHALL provide Pydantic v2 models in `src/convfinqa/models.py` that match the ConvFinQA dataset JSON structure exactly: `ConvFinQARecord`, `Document`, `Dialogue`, `Features`.

#### Scenario: Load valid record
- **WHEN** a valid JSON record is parsed into `ConvFinQARecord`
- **THEN** all fields are populated with correct types and no validation errors are raised

#### Scenario: Document fields accessible
- **WHEN** accessing `record.doc.table`
- **THEN** returns `dict[str, dict[str, float | str | int]]` keyed by column header then row label

#### Scenario: Dialogue fields aligned to turns
- **WHEN** accessing `record.dialogue.conv_questions`, `record.dialogue.conv_answers`, `record.dialogue.turn_program`, `record.dialogue.executed_answers`
- **THEN** the four lists have the same length (one entry per conversation turn)

### Requirement: Feature flags used by the evaluator

The `Features` model SHALL include at minimum `has_type2_question` (used by the eval runner to label conversations as `hybrid` vs `simple`). Other flags (`has_duplicate_columns`, `has_non_numeric_values`, `num_dialogue_turns`) are preserved as-is from the dataset.

#### Scenario: Hybrid vs simple labelling
- **WHEN** `record.features.has_type2_question` is `True`
- **THEN** the evaluator labels the conversation as `hybrid`; otherwise `simple`
