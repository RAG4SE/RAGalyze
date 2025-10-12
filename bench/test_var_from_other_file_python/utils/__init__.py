"""
Utilities module containing helper functions and utilities.
"""

from .helpers import format_price, validate_email
from .loggers import get_logger, setup_logging
from .validators import validate_user_data, validate_product_data

# Global utility variables
DEFAULT_CURRENCY = "USD"
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"