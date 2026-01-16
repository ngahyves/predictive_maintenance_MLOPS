import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

def get_logger(name: str):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 1. Defining the logger
    logger = logging.getLogger(name)
    
    # 2. Avoid duplicates
    if logger.hasHandlers():
        return logger

    # Setting the level of logs
    logger.setLevel(logging.DEBUG)

    # 3. Setting the format
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. Level of logging in the terminal
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter) # Correction typo setFormatter

    # 5. Rotating File Handler
    fh = RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=1_000_000, # 1 Mo par fichier
        backupCount=5       # Garde les 5 derniers fichiers
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 6. Add handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger