"""Unit tests for convfinqa.data: dataset loading and record lookup."""

import json

import pytest

from src.convfinqa.dataset.loader import build_index, load_dataset
from src.convfinqa.dataset.models import ConvFinQARecord

SAMPLE_RECORD = {
    "id": "Single_TEST/2009/page_1.pdf",
    "doc": {
        "pre_text": "Context.",
        "post_text": "Post.",
        "table": {"2009": {"net income": 100.0}, "2008": {"net income": 90.0}},
    },
    "dialogue": {
        "conv_questions": ["Q?"],
        "conv_answers": ["100"],
        "turn_program": ["100"],
        "executed_answers": [100.0],
        "qa_split": [False],
    },
    "features": {
        "num_dialogue_turns": 1,
        "has_type2_question": False,
        "has_duplicate_columns": False,
        "has_non_numeric_values": False,
    },
}


def test_build_index() -> None:
    record = ConvFinQARecord.model_validate(SAMPLE_RECORD)
    index = build_index([record])
    assert "Single_TEST/2009/page_1.pdf" in index
    assert index["Single_TEST/2009/page_1.pdf"].id == "Single_TEST/2009/page_1.pdf"
    assert build_index([]).get("anything") is None


def test_load_dataset(tmp_path: pytest.fixture) -> None:
    dataset = {"train": [SAMPLE_RECORD], "dev": [SAMPLE_RECORD]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    result = load_dataset(path)
    assert "train" in result
    assert "dev" in result
    assert len(result["train"]) == 1
    assert isinstance(result["train"][0], ConvFinQARecord)
