"""Tool implementations for the tool-calling agent path.

Three tools: one combined document search, one arithmetic evaluator,
one comparator for yes/no questions.

  1. ``search_document(query)``: parallel hybrid search over table cells AND
     narrative (pre/post text). Returns both result sets so the model can
     pick the value whose type matches the question.
  2. ``compute(reasoning, expression)``: math expression evaluator.
  3. ``compare(reasoning, expr_a, expr_b)``: returns "yes" if expr_a > expr_b.

The full table is pre-loaded into the initial user prompt so the model always
has it as baseline context.
"""

import asyncio

from src.lib.graph.interfaces import GraphSearcher
from src.lib.util.eval import eval_expression


# ---------------------------------------------------------------------------
# Tool 1: search_document (table + narrative in parallel)
# ---------------------------------------------------------------------------

async def search_document(
    query: str, graph: GraphSearcher, record_id: str
) -> dict:
    """Search table cells and narrative in parallel for the given query.

    Returns both result sets so the model can compare value types and pick
    the one that matches the question (e.g. dollar amount vs ratio).
    """
    table_results, narrative_results = await asyncio.gather(
        graph.search_table(record_id, query),
        graph.search_narrative(record_id, query),
        return_exceptions=True,
    )
    return {
        "table_cells": table_results if not isinstance(table_results, Exception) else [],
        "narrative_snippets": narrative_results if not isinstance(narrative_results, Exception) else [],
    }


SEARCH_DOCUMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "Search both table cells and narrative text (pre_text + post_text) "
            "in parallel. Returns: "
            "table_cells ([{'column': str, 'row': str, 'value': number}, ...]) and "
            "narrative_snippets ([{'source': 'pre'|'post', 'snippet': str}]). "
            "A table cell may hold a ratio or percentage-point figure while the "
            "dollar amount the question wants lives only in the narrative. "
            "Always inspect both result sets and pick the value whose type "
            "matches the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords describing the value to find, combining row/column "
                        "labels and any qualifiers. E.g. 'net income 2008', "
                        "'senior notes unamortized issuance costs 2015', "
                        "'prior period development 2009'. 5-15 words."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 2: compute
# ---------------------------------------------------------------------------

def compute(expression: str) -> float:
    """Evaluate a math expression using simpleeval.

    Raises ValueError on invalid input so the dispatcher can surface a
    structured error and the model can retry with a corrected expression.
    """
    return float(eval_expression(expression))


COMPUTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute",
        "description": (
            "Evaluate a math expression. Supports +, -, *, /, parentheses, and "
            "numeric literals only. Use for EVERY arithmetic step. "
            "IMPORTANT: before writing the expression, check the conversation "
            "history to confirm which prior-turn answer each value comes from. "
            "For example, if the question asks 'how much does the change represent "
            "in relation to the original', identify which turn gave the change and "
            "which turn gave the original, then write the expression using those "
            "exact numeric values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Before computing, state in one sentence what each value "
                        "in the expression represents and which prior turn it comes "
                        "from. E.g. '35.8 is the change from T3, 25.14 is the 2005 "
                        "value from T2, so I am computing 35.8 / 25.14.' "
                        "This ensures you use the correct values."
                    ),
                },
                "expression": {
                    "type": "string",
                    "description": (
                        "Any valid arithmetic expression using numeric literals, "
                        "+, -, *, /, **, and parentheses. Write out the full "
                        "expression with the actual numbers -- no variables. "
                        "Simple: '60.94 - 25.14'. "
                        "Ratio: '35.8 / 25.14'. "
                        "Percentage change: '(93 - 86) / 86'. "
                        "Nested: '(1636526 - 1580761) / 1580761'. "
                        "Multi-term: '503 + 471.4 + 382.1'. "
                        "Exponentiation: '1.05 ** 3'. "
                        "Chained: '(120 - 100) / 100 * 4.5'."
                    ),
                },
            },
            "required": ["reasoning", "expression"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 3: compare
# ---------------------------------------------------------------------------

def compare(expr_a: str, expr_b: str) -> str:
    """Return 'yes' if expr_a > expr_b, 'no' otherwise.

    Evaluates both expressions as arithmetic, then compares them.
    Use for questions like 'did X grow?', 'is A greater than B?'.
    """
    try:
        a = float(eval_expression(expr_a))
        b = float(eval_expression(expr_b))
    except ValueError as e:
        raise ValueError(f"Invalid expression in compare: {e}") from e
    return "yes" if a > b else "no"


COMPARE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compare",
        "description": (
            "Compare two numeric values and return 'yes' if the first is greater "
            "than the second, 'no' otherwise. Use for yes/no questions: "
            "'did X grow?', 'is A greater than B?', 'did revenue increase?'. "
            "Both arguments are arithmetic expressions evaluated to numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "State what expr_a and expr_b represent and which prior "
                        "turns or sources the values come from."
                    ),
                },
                "expr_a": {
                    "type": "string",
                    "description": "The first value (or expression). Returns 'yes' if this is larger.",
                },
                "expr_b": {
                    "type": "string",
                    "description": "The second value (or expression) to compare against.",
                },
            },
            "required": ["reasoning", "expr_a", "expr_b"],
        },
    },
}


# ---------------------------------------------------------------------------
# Schema bundle for OpenAI function-calling
# ---------------------------------------------------------------------------

ALL_TOOL_SCHEMAS = [
    SEARCH_DOCUMENT_SCHEMA,
    COMPUTE_SCHEMA,
    COMPARE_SCHEMA,
]
