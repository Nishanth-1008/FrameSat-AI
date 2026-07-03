import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level, logging.INFO)

logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("FrameSat")
