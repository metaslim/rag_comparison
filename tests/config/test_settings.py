"""Unit tests for src.config.settings module-level helpers."""

import pytest

from src.config.settings import _require


def test_require_returns_existing_env_var(monkeypatch) -> None:
    monkeypatch.setenv("MY_TEST_VAR_PRESENT", "some-value")
    assert _require("MY_TEST_VAR_PRESENT") == "some-value"


def test_require_raises_clear_error_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("MY_TEST_VAR_MISSING", raising=False)
    with pytest.raises(ValueError, match="MY_TEST_VAR_MISSING is not set"):
        _require("MY_TEST_VAR_MISSING")


def test_require_treats_empty_string_as_missing(monkeypatch) -> None:
    """Empty-string env var should be treated as unset (.env footgun guard)."""
    monkeypatch.setenv("MY_TEST_VAR_EMPTY", "")
    with pytest.raises(ValueError, match="MY_TEST_VAR_EMPTY is not set"):
        _require("MY_TEST_VAR_EMPTY")
