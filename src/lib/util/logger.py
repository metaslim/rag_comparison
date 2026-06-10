"""Logger factory configured from `Settings`."""

import logging

from src.config import settings


def _suppress_noisy_third_party() -> None:
    """Silence third-party loggers that emit expected warnings we can't fix."""
    # Graphiti logs WARNING when episode LLM extraction creates a relation
    # whose target entity doesn't exist yet in the graph. This is a known
    # Graphiti internal behaviour -- data still lands correctly -- but the
    # warning pollutes the user-facing output on every indexing run.
    logging.getLogger("graphiti_core").setLevel(logging.ERROR)


_suppress_noisy_third_party()


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger for `name`.

    Pulls log level and format from `Settings` (which reads `LOG_LEVEL` and
    `LOG_FORMAT` from the environment). One source of truth for config,
    set via `.env` or runtime environment variables.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        formatter = logging.Formatter(settings.log_format)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger
