"""Unit tests for src.lib.util.logger.get_logger.

Pytest's logging plugin installs LogCaptureHandlers on the root logger before
every test, so `logger.hasHandlers()` returns True via propagation and
short-circuits `get_logger`'s setup branch. The helper below sidesteps that by
disabling propagation on the target logger BEFORE calling get_logger, so
hasHandlers() only checks the logger itself.
"""

import logging
from unittest.mock import patch

from src.lib.util.logger import get_logger


def _isolated_get_logger(name: str) -> logging.Logger:
    """Call get_logger after detaching the target from pytest's root handlers."""
    # Pre-create and isolate from root so hasHandlers() returns False.
    target = logging.getLogger(name)
    target.handlers = []
    target.propagate = False
    return get_logger(name)


def test_get_logger_returns_logger_with_handler() -> None:
    """First call on a fresh logger name configures a StreamHandler."""
    logger = _isolated_get_logger("test.logger.fresh.handler")
    assert isinstance(logger, logging.Logger)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_get_logger_is_idempotent() -> None:
    """Calling get_logger twice on the same name does not double-add handlers."""
    name = "test.logger.idempotent"
    first = _isolated_get_logger(name)
    handler_count = len(first.handlers)
    second = get_logger(name)
    assert second is first
    assert len(second.handlers) == handler_count


def test_get_logger_respects_settings_level() -> None:
    """The configured log level matches settings.log_level (uppercased)."""
    name = "test.logger.level"
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = False
    with patch("src.lib.util.logger.settings") as mock_settings:
        mock_settings.log_level = "DEBUG"
        mock_settings.log_format = "%(message)s"
        logger = get_logger(name)
    assert logger.level == logging.DEBUG


def test_get_logger_falls_back_to_info_on_invalid_level() -> None:
    """Unknown log_level strings fall back to INFO via getattr default."""
    name = "test.logger.invalid_level"
    logging.getLogger(name).handlers = []
    logging.getLogger(name).propagate = False
    with patch("src.lib.util.logger.settings") as mock_settings:
        mock_settings.log_level = "NOT_A_REAL_LEVEL"
        mock_settings.log_format = "%(message)s"
        logger = get_logger(name)
    assert logger.level == logging.INFO


def test_get_logger_does_not_propagate() -> None:
    """The configured logger has propagation disabled (leaf handler)."""
    logger = _isolated_get_logger("test.logger.no_propagate")
    assert logger.propagate is False
