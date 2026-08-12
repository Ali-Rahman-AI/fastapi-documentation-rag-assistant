"""
utils.py

Small shared helper functions. Kept separate from rag.py and app.py so
both files can reuse the same logger setup without duplicating code.
"""

import logging
import sys

from config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Create (or reuse) a logger with a simple, readable format.

    Example:
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this function is called more
    # than once for the same logger name (e.g. during reloads).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)

    return logger


def clean_question(raw_question: str) -> str:
    """Strip whitespace and collapse the question to a single line."""
    if raw_question is None:
        return ""
    return " ".join(raw_question.strip().split())
