"""
Logging configuration for comprehensive monitoring
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import LOG_FILE, LOG_LEVEL


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logger with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional custom log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler with custom formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # File handler with detailed formatting
    file_log = log_file or LOG_FILE
    file_handler = logging.FileHandler(file_log)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, 
                   success: bool = True, metadata: Optional[dict] = None):
    """
    Log performance metrics for monitoring
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        success: Whether operation succeeded
        metadata: Additional metadata to log
    """
    status = "SUCCESS" if success else "FAILURE"
    meta_str = f" | Metadata: {metadata}" if metadata else ""
    
    log_msg = f"PERFORMANCE | {operation} | {status} | Duration: {duration_ms:.2f}ms{meta_str}"
    
    if success and duration_ms < 300:
        logger.info(log_msg)
    elif success:
        logger.warning(f"{log_msg} | Exceeds target latency")
    else:
        logger.error(log_msg)


# Create default logger
logger = setup_logger("renewable_energy")