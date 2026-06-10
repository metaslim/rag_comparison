## ADDED Requirements

### Requirement: index command, split-aware and resumable

The system SHALL provide `uv run main index --split {dev,train,all}` that loads records from `data/convfinqa_dataset.json` and indexes them into Graphiti + Neo4j via `FinancialGraph.index_bulk`. Default split is `dev`. Indexing is checkpoint-resumable via `data/index_checkpoint.json`.

#### Scenario: Index dev split (default)
- **WHEN** `uv run main index`
- **THEN** all 421 dev records are indexed (already-indexed records are skipped via the checkpoint)

#### Scenario: Index all (train + dev)
- **WHEN** `uv run main index --split=all`
- **THEN** all 3,458 records are indexed

#### Scenario: Index is resumable
- **WHEN** indexing crashes mid-way and is re-run
- **THEN** the checkpoint skips already-completed records, avoiding double API spend

### Requirement: chat command with on-demand indexing

The system SHALL provide `uv run main chat <record_id>` that starts an interactive multi-turn chat session for a specific record.

If no `record_id` is given, the CLI SHALL list the first 20 records and prompt the user to enter one.

If the record is not yet indexed, the CLI SHALL prompt the user with three choices:
- `Y` (default): index now, then chat in Mode 1 (graph search).
- `N`: chat in Mode 2 (raw `pre_text + table + post_text` fallback).
- Once indexed, the record stays in Neo4j for future chats.

If the user types a record ID without the trailing `-N` suffix, the CLI SHALL fuzzy-match the base document and proceed with the matched record.

#### Scenario: Chat on a pre-indexed record
- **WHEN** `chat "Single_MRO/2007/page_134.pdf-1"` is run (record is in the committed dump)
- **THEN** interactive chat starts in Mode 1 without any indexing prompt

#### Scenario: Chat on an un-indexed record, opt into indexing
- **WHEN** `chat <unindexed-record>` is run, user answers `Y` to the prompt
- **THEN** the record is indexed (~20s), then chat starts in Mode 1

#### Scenario: Chat on an un-indexed record, decline indexing
- **WHEN** user answers `N`
- **THEN** chat starts in Mode 2 (raw document inline in the prompt)

#### Scenario: Fuzzy suffix match
- **WHEN** `chat "Single_MRO/2007/page_134.pdf"` (no `-N`)
- **THEN** the CLI matches the base document, picks the first record matching that base, and prints which exact record it picked

### Requirement: evaluate command with per-record auto-indexing

The system SHALL provide `uv run main evaluate` that runs all dev set records through the agent and prints an accuracy report.

The command SHALL accept:
- `--limit N`: evaluate only the first N records.
- `--no-index`: skip per-record auto-indexing.

Default behaviour is per-record auto-indexing: for each record, check if indexed → index if missing → evaluate all turns → next record. Indexing is checkpoint-resumable.

#### Scenario: Evaluate runs and prints report
- **WHEN** `evaluate` completes
- **THEN** overall accuracy, per-turn accuracy, simple-vs-hybrid accuracy, and total correct/total counts are printed

#### Scenario: Reproduce the published headline
- **WHEN** `uv run main evaluate --limit=100`
- **THEN** runs in pure Mode 1 (all 100 records covered by the committed dump) and reproduces the REPORT.md figure

#### Scenario: --no-index avoids the indexing step
- **WHEN** `evaluate --no-index` is run with un-indexed records present
- **THEN** un-indexed records fall back to Mode 2 silently; no API calls for indexing are made

### Requirement: Neo4j reachability check

All three CLI commands (`index`, `chat`, `evaluate`) SHALL check that Neo4j is reachable before doing any work, and SHALL print a clear error message if not. This logic is centralised in a single `_require_neo4j(graph)` helper to satisfy DRY.

#### Scenario: Neo4j is not running
- **WHEN** any command is invoked while the Neo4j container is stopped
- **THEN** the CLI prints `[red]Neo4j is not running. Start it with: docker-compose up -d[/red]` and exits cleanly
