"""Shared arithmetic evaluator wrapper over simpleeval."""

from simpleeval import InvalidExpression, simple_eval


def eval_expression(expression: str) -> int | float:
    """Evaluate an arithmetic expression and return a numeric result.

    Returns int when the expression evaluates to a whole number, float otherwise.
    Raises ValueError with a human-readable message on invalid input or
    non-numeric result so callers can surface a structured error to the model
    and let it retry with a corrected expression.
    """
    try:
        result = simple_eval(expression)
    except (InvalidExpression, Exception) as e:
        raise ValueError(f"Invalid expression {expression!r}: {e}") from e
    # Exclude bools explicitly: bool is a subclass of int in Python, so
    # isinstance(True, int) is True -- we don't want "True"/"False" as answers.
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(
            f"Expression {expression!r} did not evaluate to a number "
            f"(got {type(result).__name__})."
        )
    return result
