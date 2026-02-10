
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Path = None,
    format_str: str = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)


    logger.handlers = []


    if format_str is None:
        format_str = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    formatter = logging.Formatter(format_str)


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


root_logger = setup_logger('GCN_HKD', log_level=logging.INFO)
