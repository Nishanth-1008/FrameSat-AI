from __future__ import annotations
import logging
from app.logger import get_logger as _get_logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for infrastructure logging.
    Delegates to the core app.logger.
    """
    return _get_logger(name, log_to_file=True)
