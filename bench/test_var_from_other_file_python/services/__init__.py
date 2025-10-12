"""
Services module containing business logic and external service integrations.
"""

from .auth import AuthService, TokenManager
from .api import APIService, RequestHandler
from .cache import CacheService

# Service-wide configuration
SERVICE_TIMEOUT = 60
MAX_RETRIES = 3
SERVICE_REGISTRY = {}