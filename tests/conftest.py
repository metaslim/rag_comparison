"""Shared pytest configuration and fixtures."""

from src.convfinqa.dataset.models import ConvFinQARecord


def make_record(
    rid: str = "Single_TEST/2009/page_1.pdf-1",
    table: dict | None = None,
    questions: list[str] | None = None,
    answers: list[float] | None = None,
    pre_text: str = "Some pre text.",
    post_text: str = "Some post text.",
) -> ConvFinQARecord:
    """Build a minimal valid ConvFinQARecord for tests."""
    if table is None:
        table = {"2009": {"net income": 100.0}}
    if questions is None:
        questions = ["what was net income?"]
    if answers is None:
        answers = [100.0]
    return ConvFinQARecord.model_validate({
        "id": rid,
        "doc": {
            "pre_text": pre_text,
            "post_text": post_text,
            "table": table,
        },
        "dialogue": {
            "conv_questions": questions,
            "conv_answers": [str(a) for a in answers],
            "turn_program": [str(a) for a in answers],
            "executed_answers": answers,
            "qa_split": [False] * len(answers),
        },
        "features": {
            "num_dialogue_turns": len(answers),
            "has_type2_question": False,
            "has_duplicate_columns": False,
            "has_non_numeric_values": False,
        },
    })
