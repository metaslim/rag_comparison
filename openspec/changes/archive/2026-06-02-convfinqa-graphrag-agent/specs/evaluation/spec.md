## ADDED Requirements

### Requirement: Execution accuracy with tolerant numeric comparison

The system SHALL compare predicted answers to gold `executed_answers` using **5% relative tolerance** or **1e-3 absolute tolerance** (whichever is satisfied). The comparison chains seven normalisation paths before the final tolerance check.

#### Scenario: Correct within relative tolerance
- **WHEN** predicted is `0.1414` and gold is `0.14136`
- **THEN** marked as correct

#### Scenario: Wrong answer outside tolerance
- **WHEN** predicted is `180000` and gold is `206588`
- **THEN** marked as incorrect

### Requirement: Numeric normalisation paths

The system SHALL apply the following normalisation paths before the final tolerance check. Each is applied conditionally based on the magnitudes of `gold` and `pred`.

1. **Unit scale (1B / 1M / 1K, both directions)**: handles `880,000,000` ≡ `0.88` (billions gold).
2. **Multiply-by-scale**: handles `21.12` ≡ `20.7M` (raw value vs millions).
3. **Percent vs decimal**: if `gold < 1` and `pred > 1`, divide pred by 100.
4. **Reverse percent**: if `gold > 1` and `pred < 1`, multiply pred by 100.
5. **Reverse percent both-small (100x off)**: handles `0.0035` ≡ `0.35` percent points.
6. **Inverse ratio**: `1 / pred ≈ gold` for `X/Y` vs `Y/X` ambiguity.
7. **Sign flip**: accepts `-X` when gold is `X` (table stores losses negative).

#### Scenario: Percentage string normalised
- **WHEN** predicted is `"14.1%"` and gold is `0.14136`
- **THEN** normalised to `0.141` and marked correct

#### Scenario: Reverse percent (gold in percent format)
- **WHEN** predicted is `0.504` and gold is `51.2`
- **THEN** pred multiplied by 100 (= 50.4), within tolerance of 51.2, marked correct

#### Scenario: Sign flip (loss stored negative)
- **WHEN** predicted is `-1082` and gold is `1082`
- **THEN** sign-flip check passes, marked correct

### Requirement: Per-record auto-indexing flow

The `evaluate` command SHALL run **per-record**: for each dev record, check if the record is indexed in the graph, index it if not, then evaluate all its turns, then move to the next record. Index and eval logs interleave on stdout so the user sees real-time progress.

#### Scenario: Un-indexed record auto-indexed during eval
- **WHEN** `evaluate` reaches record 101 which is not in the graph
- **THEN** the runner indexes the record (logs "Indexing record N/total"), then evaluates its turns

#### Scenario: Indexed record skips indexing
- **WHEN** `evaluate` reaches record 1 which is already in the graph
- **THEN** the runner skips indexing and goes straight to evaluation

### Requirement: Evaluation report

The system SHALL output overall execution accuracy, accuracy by turn position (Q1, Q2, Q3, Q4+), and accuracy by conversation type (simple vs hybrid). Results are written to `data/eval_results.json`.

#### Scenario: Report generated
- **WHEN** evaluation completes
- **THEN** the CLI prints overall accuracy, per-turn breakdown, and simple-vs-hybrid breakdown using a Rich table

### Requirement: --limit and --no-index flags

The `evaluate` command SHALL accept:
- `--limit N` to evaluate only the first N records (e.g. `--limit=100` reproduces the REPORT.md headline figure).
- `--no-index` to skip per-record auto-indexing; un-indexed records fall back to Mode 2 silently.

#### Scenario: --limit reproduces published figure
- **WHEN** `uv run main evaluate --limit=100`
- **THEN** all 100 records run in Mode 1 (pre-indexed by the committed dump) and the accuracy matches the REPORT.md figure
