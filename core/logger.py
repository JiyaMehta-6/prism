"""Central logging configuration for PRISM.

A rotating file handler writes to ``logs/analysis.log`` while a mirrored
console handler keeps terminal output visible during development.
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

from core import config

_LOGGER_NAME = "prism"
_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger (idempotent).

    Logging must never be the reason the app dies: if the log file cannot be
    opened (locked, read-only install, missing permissions) PRISM degrades to
    console-only logging instead of raising at import time.
    """
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    if _configured:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    target = "console only"
    try:
        config.ensure_directories()
        file_handler = RotatingFileHandler(
            config.LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        target = os.path.abspath(config.LOG_PATH)
    except Exception as exc:  # noqa: BLE001 - file logging must never crash
        logger.warning("File logging unavailable (%s); using console only", exc)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)

    logger.propagate = False
    _configured = True
    logger.info("Logging initialised -> %s", target)
    return logger


def get_logger(suffix: Optional[str] = None) -> logging.Logger:
    """Return the application logger, optionally namespaced."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        setup_logging()
    if suffix:
        return logger.getChild(suffix)
    return logger


class StageTimer:
    """Context manager that logs elapsed wall time for a processing stage."""

    def __init__(self, name: str, logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.logger = logger or get_logger("perf")
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.start = time.perf_counter()
        self.logger.info("Stage started: %s", self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start
        if exc is None:
            self.logger.info("Stage finished: %s (%.2fs)", self.name, self.elapsed)
        else:
            self.logger.error("Stage failed: %s (%.2fs) - %s", self.name, self.elapsed, exc)
