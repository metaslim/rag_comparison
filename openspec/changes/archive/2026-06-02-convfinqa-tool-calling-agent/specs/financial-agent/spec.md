## MODIFIED Requirements

### Requirement: Curated prompt rules (trimmed to 3)

The system prompt for the tool-calling path SHALL contain only the 3 retained rules:

1. Parenthetical % is the answer (`"decreased by 13.4%"` returns `0.134`)
2. Portion = subtotal ÷ total
3. Total/aggregate price uses TABLE row over text breakdowns

The other 5 rules from the single-pass path (Rules 3, 4, 5, 6, 12→15 exception) SHALL be removed from the tool-path system prompt. Their behaviour SHALL be enforced by the tool contracts (`compare`, `list_rows_matching`, `list_columns_matching`, `lookup_cell` with `AmbiguousError`) defined in the `financial-agent-tools` capability.

#### Scenario: Rule 3 (sign convention) enforced by `compare`
- **WHEN** the question asks "what was the change in revenue from 2007 to 2008?"
- **THEN** the tool-path system prompt does NOT contain a "change = later − earlier" rule
- **AND** the model calls `compare(metric="revenue", from_col="2007", to_col="2008")` which returns both signed and unsigned change values

#### Scenario: Rule 5 (program names) enforced by `list_columns_matching`
- **WHEN** the table has both `"2007"` and `"2007 program"` columns
- **THEN** the tool-path system prompt does NOT contain a "program names by year" rule
- **AND** the model calls `list_columns_matching("2007")` to disambiguate before lookup

## ADDED Requirements

### Requirement: Tool-calling answer path

`FinancialAgent` SHALL expose `answer_with_tools(conv_id, record, question) → str` as a separate entry point from `answer()`. The tool path SHALL use the OpenAI function-calling API with the six tools defined in the `financial-agent-tools` capability.

#### Scenario: Tool path invoked when `--tools` flag is set
- **WHEN** `evaluate` or `chat` is run with `--tools`
- **THEN** every question is routed through `answer_with_tools`, not `answer`

#### Scenario: Single-pass path remains the default
- **WHEN** `evaluate` or `chat` is run without `--tools`
- **THEN** every question is routed through the existing `answer()` method

### Requirement: Tool-loop fallback to single-pass

When the tool loop exhausts its 15-call cap or the model refuses to emit `ANSWER:` after the cap-warning system message, the agent SHALL fall back to the existing single-pass CoT path using the facts already accumulated from tool calls.

#### Scenario: Loop-cap exhaustion triggers fallback
- **WHEN** the model has made 15 tool calls without producing `ANSWER:`
- **AND** the cap-warning system message also fails to elicit a final answer
- **THEN** the agent calls the existing single-pass path with the accumulated facts as context
- **AND** the conversation history records both the tool trace and the fallback result
