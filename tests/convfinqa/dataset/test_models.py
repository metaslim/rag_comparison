"""Unit tests for convfinqa.models: Pydantic model parsing."""

from src.convfinqa.dataset.models import ConvFinQARecord, Dialogue, Document, Features

SAMPLE_RECORD = {
    "id": "Single_TEST/2009/page_1.pdf",
    "doc": {
        "pre_text": "Some context text.",
        "post_text": "More context text.",
        "table": {
            "2009": {"net income": 100.0, "revenue": 200.0},
            "2008": {"net income": 90.0, "revenue": 180.0},
        },
    },
    "dialogue": {
        "conv_questions": ["what was net income in 2009?", "what about 2008?"],
        "conv_answers": ["100", "90"],
        "turn_program": ["100", "90"],
        "executed_answers": [100.0, 90.0],
        "qa_split": [False, False],
    },
    "features": {
        "num_dialogue_turns": 2,
        "has_type2_question": False,
        "has_duplicate_columns": False,
        "has_non_numeric_values": False,
    },
}


def test_record_parses_correctly() -> None:
    record = ConvFinQARecord.model_validate(SAMPLE_RECORD)
    assert record.id == "Single_TEST/2009/page_1.pdf"
    assert record.doc.pre_text == "Some context text."
    assert record.doc.table["2009"]["net income"] == 100.0
    assert record.dialogue.conv_questions[0] == "what was net income in 2009?"
    assert record.dialogue.executed_answers[0] == 100.0
    assert record.features.num_dialogue_turns == 2


def test_document_fields() -> None:
    record = ConvFinQARecord.model_validate(SAMPLE_RECORD)
    assert isinstance(record.doc, Document)
    assert "2009" in record.doc.table
    assert "2008" in record.doc.table


def test_dialogue_fields() -> None:
    record = ConvFinQARecord.model_validate(SAMPLE_RECORD)
    assert isinstance(record.dialogue, Dialogue)
    assert len(record.dialogue.conv_questions) == 2
    assert len(record.dialogue.executed_answers) == 2


def test_features_fields() -> None:
    record = ConvFinQARecord.model_validate(SAMPLE_RECORD)
    assert isinstance(record.features, Features)
    assert record.features.has_type2_question is False
