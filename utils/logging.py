"""Structured logging setup shared across the pipeline.

Usage:
    from utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Loaded %d trades", len(trades_df))
"""

import logging
import os
import sys

from config import config

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Logs must stay off stdout so callers can pipe a script's stdout
    # (e.g. JSON results) without status/info noise mixed in.
    handler = logging.StreamHandler(sys.stderr)

    if config.LOG_FORMAT == "json":
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        except ImportError:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")

    handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        
    root_logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name` (typically `__name__`)."""
    _configure()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Override the root logger's verbosity, e.g. from a CLI --log-level flag."""
    _configure()
    logging.getLogger().setLevel(level.upper())
