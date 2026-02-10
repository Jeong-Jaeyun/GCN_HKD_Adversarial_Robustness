"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Path = None,
    format_str: str = None
) -> logging.Logger:
    """Setup logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        format_str: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format string
    if format_str is None:
        format_str = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Root logger for the project
root_logger = setup_logger('GCN_HKD', log_level=logging.INFO)
