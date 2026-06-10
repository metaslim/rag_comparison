## ADDED Requirements

### Requirement: Persist conversation turns to SQLite
The system SHALL store each conversation turn (conversation_id, record_id, turn, question, answer, timestamp) in a SQLite database at `data/conversations.db`.

#### Scenario: Save a turn
- **WHEN** a question and answer are saved for a conversation
- **THEN** a row is inserted with the correct conversation_id, record_id, turn number, question, and answer

### Requirement: Load conversation history
The system SHALL retrieve all turns for a given conversation_id ordered by turn number.

#### Scenario: Load history
- **WHEN** history is loaded for a conversation_id with 3 turns
- **THEN** returns a list of 3 dicts with `question` and `answer` keys in order

#### Scenario: Empty history
- **WHEN** history is loaded for an unknown conversation_id
- **THEN** returns an empty list

### Requirement: Format conversation history as LLM context
The system SHALL format loaded history into a structured string suitable for inclusion in a Claude prompt, showing each prior Q&A turn clearly.

#### Scenario: History formatted for LLM
- **WHEN** history with 3 turns is formatted
- **THEN** returns a string like:
  ```
  Q1: what was net cash in 2009? → 206588
  Q2: what about in 2008? → 181001
  Q3: what is the difference? → 25587
  ```

#### Scenario: Empty history formatted
- **WHEN** history is empty
- **THEN** returns an empty string

### Requirement: Generate unique conversation ID
The system SHALL generate a unique conversation_id when a new chat session starts.

#### Scenario: New conversation ID
- **WHEN** a new conversation is started
- **THEN** a UUID string is returned that is unique across calls
