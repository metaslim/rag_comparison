"""Anti-drift guards for Mode 1 / Mode 3 prompts.

Both prompts compose shared constants (`_ANALYST_INTRO`, `_DOMAIN_RULES`,
`_OUTPUT_FORMAT_COT`) so the substantive financial-domain content can't
diverge. These tests fail loudly if someone copies a rule into one prompt
without going through the shared constant.
"""

from src.convfinqa.agent.prompts import (
    _ANALYST_INTRO,
    _DOMAIN_RULES,
    _OUTPUT_FORMAT_COT,
    SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
)


def test_analyst_intro_appears_in_both_prompts() -> None:
    assert _ANALYST_INTRO in SYSTEM_PROMPT
    assert _ANALYST_INTRO in TOOL_SYSTEM_PROMPT


def test_domain_rules_appear_in_both_prompts() -> None:
    """The 16 numbered RULES must be byte-identical in both prompts."""
    assert _DOMAIN_RULES in SYSTEM_PROMPT
    assert _DOMAIN_RULES in TOOL_SYSTEM_PROMPT


def test_output_format_cot_appears_in_both_prompts() -> None:
    assert _OUTPUT_FORMAT_COT in SYSTEM_PROMPT
    assert _OUTPUT_FORMAT_COT in TOOL_SYSTEM_PROMPT


def test_domain_rules_covers_known_conventions() -> None:
    """Spot-check the rules that have caused real failures historically.

    If any of these markers disappear, a previously battle-tested convention
    has been silently dropped from the rules. Adjust the test only after
    confirming the rule is genuinely no longer needed.
    """
    expected_markers = [
        "PARENTHETICAL PERCENTAGES",
        "CHANGE/DIFFERENCE DIRECTION",
        "VARIANCE IN FINANCIAL TABLES",
        "stockholder return tables",
        "COUNTING YEARS IN A PERIOD",
        "PREFER TOTALS",
        "PORTION/FRACTION",
        "PROGRAM NAMES BY YEAR",
        "UNIT / SCALE FROM NARRATIVE",
    ]
    for marker in expected_markers:
        assert marker in _DOMAIN_RULES, f"Missing rule marker: {marker!r}"


def test_tool_prompt_includes_three_tool_names() -> None:
    """Mode 3 prompt must reference each of the three tools by name."""
    for tool in ("search_document", "compute", "compare"):
        assert tool in TOOL_SYSTEM_PROMPT, f"Tool {tool!r} missing from TOOL_SYSTEM_PROMPT"


def test_single_pass_prompt_omits_tool_names() -> None:
    """Mode 1 prompt should not reference the Mode 3 tools by name.

    Mentioning them in Mode 1 would confuse the model -- it has no tools.
    """
    for tool in ("search_document", "compare"):
        assert f"`{tool}`" not in SYSTEM_PROMPT, f"Mode 1 leaks tool name {tool!r}"
