"""
Logging setup: console plus a file in data/logs
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import LOG_FILE, LOG_LEVEL

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger that writes to stdout and to the log file"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file or LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


def log_performance(logger: logging.Logger, operation: str, duration_ms: float,
                    metadata: Optional[dict] = None):
    """Log how long an operation took"""
    meta_str = f" | {metadata}" if metadata else ""
    logger.info(f"{operation} took {duration_ms:.1f} ms{meta_str}")
