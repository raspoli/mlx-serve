"""
logging_config.py — configure structured logging with console + file output.

Console gets human-readable logs; file gets the same for post-mortem debugging.
Log files rotate automatically to avoid filling the disk.
"""

import logging
import logging.handlers
from pathlib import Path


def setup(log_dir: Path, level: int = logging.INFO) -> None:
    """Initialise root logger with console + rotating file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler — everything INFO+
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    # File handler — rotating, 10 MB per file, keep last 3
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "mlx-serve.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("hf_transfer").setLevel(logging.WARNING)
