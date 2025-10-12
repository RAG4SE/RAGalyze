"""
Application settings module.
Variables here will be imported and used by other modules.
"""

# Timeouts and limits
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
CONNECTION_POOL_SIZE = 10
RATE_LIMIT_PER_MINUTE = 100

# Cache settings
CACHE_TTL = 3600  # seconds
CACHE_MAX_SIZE = 1000

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security settings
JWT_SECRET_KEY = "your-secret-key-here"
JWT_EXPIRATION_HOURS = 24
ALLOWED_ORIGINS = ["http://localhost:3000", "https://myapp.com"]