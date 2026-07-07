from __future__ import annotations
from pathlib import Path

import logging
import sys

from app.shared.config.settings import get_settings

settings = get_settings()


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Every module should obtain its logger using this function.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt=(
            "[%(asctime)s] "
            "[%(levelname)s] "
            "[%(name)s] "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    logger.addHandler(console)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(
        log_dir / "framesat.log",
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger