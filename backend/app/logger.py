from __future__ import annotations
import logging
import sys
from pathlib import Path
from app.config import LOG_LEVEL


def get_logger(
    name: str, log_to_file: bool = False, file_path: Path | str | None = None
) -> logging.Logger:
    """
    Get a configured reusable logger.

    Args:
        name: Name of the logger.
        log_to_file: If True, outputs logs to a file in addition to console.
        file_path: Specific path for log files. Defaults to 'logs/framesat.log'.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    # Resolve log level
    level_str = LOG_LEVEL.upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # Create formatter with timestamp
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (sys.stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_to_file or file_path is not None:
        if file_path is None:
            file_path = Path("logs/framesat.log")
        else:
            file_path = Path(file_path)

        # Create parent directories for logs if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid propagating to root logger to prevent double logging
    logger.propagate = False

    return logger
