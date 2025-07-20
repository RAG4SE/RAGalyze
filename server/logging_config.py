import logging
import os
from pathlib import Path

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    grey = "\x1b[90m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    format = "[RAGalyze][%(levelname_title)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset,
    }

    def format(self, record):
        record.levelname_title = record.levelname.title()
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging(format: str = None):
    """
    Configure logging for the application.
    Reads LOG_LEVEL and LOG_FILE_PATH from environment (defaults: INFO, logs/application.log).
    Ensures log directory exists, and configures both file and console handlers.
    """
    # Determine log directory and default file path
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get log level and file path from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file_path = Path(os.environ.get(
        "LOG_FILE_PATH", str(default_log_file)))

    # ensure log_file_path is within the project's logs directory to prevent path traversal
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(
            f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'"
        )
    # Ensure parent dirs exist for the log file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # File log (plain format)
    file_handler = logging.FileHandler(resolved_path)
    file_handler.setFormatter(logging.Formatter(
        format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    ))

    # Console log (plain format, for root logger)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        format or "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    ))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Add color handler to your project logger (e.g., api, RAGalyze) and prevent propagation to root logger
    project_logger = logging.getLogger("api")
    project_logger.handlers.clear()  # Prevent duplicate handlers
    color_handler = logging.StreamHandler()
    color_handler.setFormatter(ColorFormatter())
    project_logger.addHandler(color_handler)
    project_logger.propagate = False  # Prevent propagation to root logger to avoid duplicate output

    # Initial debug message to confirm configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Log level set to {log_level_str}, log file: {resolved_path}")
