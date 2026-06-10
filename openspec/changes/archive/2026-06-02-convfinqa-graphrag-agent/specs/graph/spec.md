## ADDED Requirements

### Requirement: Hybrid indexing, triplets for tables and episodes for text

The system SHALL index each record using two parallel strategies:
- `_index_table_triplets`: every table cell becomes a direct `add_triplet` write with no LLM call. Produces a controlled fact string and a typed edge.
- `add_episode_bulk`: pre_text + post_text are sentence-chunked (`Chonkie SentenceChunker`, `chunk_size=400`) and submitted to Graphiti for LLM extraction of entities and relationships.

#### Scenario: Table cell stored as exact triplet
- **WHEN** a cell `{"2007 program": {"liability at december 31 2008": 1.2}}` is indexed
- **THEN** a triplet edge is created with `fact="In column [2007 program], the row [liability at december 31 2008] has value 1.2"` and `valid_at=2007-01-01` (parsed from the column key)

#### Scenario: Narrative text indexed as episodes
- **WHEN** `pre_text` contains `"Analog Devices evaluates goodwill annually for impairment."`
- **THEN** the chunk is added via `add_episode_bulk` and Graphiti's LLM extracts entity-relationship triples around `Analog Devices`, `goodwill`, etc.

### Requirement: Explicit [column] and [row] markers in triplet facts

The system SHALL format triplet fact strings as `"In column [X], the row [Y] has value Z"`. The bracket markers SHALL be preserved literally so BM25 and the downstream LLM treat them as structural disambiguators.

#### Scenario: Column header looks like a year
- **WHEN** a table has columns `"2007 program"` and `"2008 program"`
- **THEN** the LLM treats them as category labels (program names) rather than year references, because the `[column]` marker disambiguates them from row-level date references like `[liability at december 31 2007]`

### Requirement: Stable UUIDs for entity dedup

The system SHALL compute deterministic UUIDs for entity nodes using `md5(group_id + "::" + entity_name)`. This ensures the same entity created from different cells dedupes to a single node.

#### Scenario: Same entity across cells
- **WHEN** `"net income"` appears as a row label for 2007, 2008, and 2009 in the same table
- **THEN** all three triplets share the same `net income` node UUID, producing one entity with three edges (not three separate entities)

### Requirement: _parse_year for temporal edges

The system SHALL extract a 4-digit year (1970 to 2039) from table column keys using `_parse_year()`, used to set `valid_at` on every triplet edge.

#### Scenario: Year parsed from verbose key
- **WHEN** `_parse_year("Year ended June 30, 2009")`
- **THEN** returns `2009`

#### Scenario: Concatenated date string
- **WHEN** `_parse_year("december 312016")`
- **THEN** returns `2016`

#### Scenario: Dollar amount is not a year
- **WHEN** `_parse_year("$ 2014")`
- **THEN** returns `None` (the edge uses the current timestamp as `valid_at`)

### Requirement: record_id_to_group_id deduplicates shared documents

The system SHALL map record IDs to `group_id` by stripping the trailing `-N` suffix and sanitising characters Graphiti rejects (`/`, `.`).

#### Scenario: Shared PDFs collapse to one group_id
- **WHEN** `record_id_to_group_id("Single_ADI/2016/page_61.pdf-3")`
- **THEN** returns `"Single_ADI-2016-page_61-pdf"` (same as for `-1`, `-2`, etc.)

#### Scenario: Multiple records dedupe to one indexed document
- **WHEN** records `Single_X/page_1.pdf-1`, `Single_X/page_1.pdf-2`, `Single_X/page_1.pdf-3` are indexed
- **THEN** the underlying document is indexed once and all three records share its facts via the same `group_id`

### Requirement: Hybrid search scoped per document

The system SHALL search via Graphiti's `COMBINED_HYBRID_SEARCH_RRF` recipe (BM25 + cosine + RRF rerank) with `limit=50`, scoped to a single document via `group_ids=[record's group_id]`.

#### Scenario: Document-scoped retrieval
- **WHEN** searching for `"net income 2009"` on record A
- **THEN** results contain only edges and episodes from record A's `group_id`, never from record B

### Requirement: Search query preprocessing

The system SHALL strip `&` and `/` characters from the search query string before passing it to Graphiti, to avoid BM25 tokenizer breaks (e.g. `"rd&e"`).

#### Scenario: Query with ampersand
- **WHEN** searching for `"research and development (rd&e) expenses"`
- **THEN** the query is rewritten to `"research and development (rd e) expenses"` before BM25 tokenization

### Requirement: Full chunk content returned alongside edges

The system SHALL include full episode chunk content (deduplicated by exact-match) in the search results, alongside the extracted edge facts. This compensates for Graphiti's edge extraction occasionally dropping values from multi-value sentences (e.g. `"$11M and $8M"` extracting only `"$11M"`).

#### Scenario: Edge extraction missed a value
- **WHEN** the source sentence is `"interest costs of $16.8 million and $9.4 million"` and Graphiti's extracted edge only mentions one of the values
- **THEN** the full chunk is also in the returned facts list, so the agent's LLM still sees both values

### Requirement: Checkpoint-resumable bulk indexing

The system SHALL persist the set of completed record IDs to `data/index_checkpoint.json` after each successful record. On restart, already-completed records are skipped.

#### Scenario: Resume after crash
- **WHEN** `index_bulk` is interrupted after indexing 50 of 100 records, then re-run
- **THEN** records 51 to 100 are indexed; records 1 to 50 are skipped without API calls
