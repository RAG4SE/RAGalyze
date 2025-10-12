"""
Logging utilities module.
Uses configuration variables for logging setup.
"""

from config import ENABLE_LOGGING, APP_NAME
from config.settings import LOG_LEVEL, LOG_FORMAT
from . import DATE_FORMAT

class Logger:
    """Custom logger using configuration variables."""

    def __init__(self, name):
        self.name = name
        self.enabled = ENABLE_LOGGING
        self.log_level = LOG_LEVEL
        self.log_format = LOG_FORMAT
        self.app_name = APP_NAME
        self.date_format = DATE_FORMAT

    def log(self, level, message):
        """Log message with formatting."""
        if not self.enabled:
            return

        # Format log message
        formatted_message = self.log_format.replace("%(name)s", self.name)
        formatted_message = formatted_message.replace("%(levelname)s", level.upper())
        formatted_message = formatted_message.replace("%(asctime)s", self._get_timestamp())
        formatted_message = formatted_message.replace("%(message)s", f"[{self.app_name}] {message}")

        print(formatted_message)

    def info(self, message):
        """Log info message."""
        self.log("info", message)

    def debug(self, message):
        """Log debug message."""
        if self.log_level.upper() in ["DEBUG"]:
            self.log("debug", message)

    def warning(self, message):
        """Log warning message."""
        self.log("warning", message)

    def error(self, message):
        """Log error message."""
        self.log("error", message)

    def _get_timestamp(self):
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().strftime(self.date_format)

def get_logger(name):
    """Get logger instance."""
    return Logger(name)

def setup_logging(app_config=None):
    """Setup logging with custom configuration."""
    logger = get_logger("setup")

    logger.info("Setting up logging configuration")
    logger.info(f"Log level: {LOG_LEVEL}")
    logger.info(f"Logging enabled: {ENABLE_LOGGING}")
    logger.info(f"App name: {APP_NAME}")

    if app_config:
        logger.info(f"Custom config provided: {type(app_config)}")
        for key, value in app_config.items():
            logger.info(f"  {key}: {value}")

    logger.info("Logging setup complete")
    return logger

class ServiceLogger:
    """Specialized logger for services."""

    def __init__(self, service_name):
        self.logger = get_logger(f"service.{service_name}")
        self.service_name = service_name

    def log_request(self, method, endpoint, user_id=None):
        """Log API request."""
        message = f"Request: {method} {endpoint}"
        if user_id:
            message += f" by user {user_id}"
        self.logger.info(message)

    def log_response(self, status_code, response_time=None):
        """Log API response."""
        message = f"Response: {status_code}"
        if response_time:
            message += f" ({response_time}ms)"
        self.logger.info(message)

    def log_error(self, error_type, error_message):
        """Log service error."""
        self.logger.error(f"{error_type}: {error_message}")

    def log_performance(self, operation, duration):
        """Log performance metrics."""
        self.logger.info(f"Performance: {operation} took {duration:.2f}s")