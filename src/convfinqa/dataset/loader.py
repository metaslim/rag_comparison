"""Dataset loading and record lookup."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.convfinqa.dataset.models import ConvFinQARecord


def load_dataset(path: str | Path) -> dict[str, list[ConvFinQARecord]]:
    """Load the ConvFinQA dataset from JSON, returning train and dev splits."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        with open(path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file {path}: {e}") from e
    try:
        return {
            split: [ConvFinQARecord.model_validate(r) for r in records]
            for split, records in raw.items()
        }
    except ValidationError as e:
        raise ValueError(f"Dataset schema mismatch in {path}: {e}") from e


def build_index(records: list[ConvFinQARecord]) -> dict[str, ConvFinQARecord]:
    """Build an O(1) lookup index from record ID to record."""
    return {r.id: r for r in records}
