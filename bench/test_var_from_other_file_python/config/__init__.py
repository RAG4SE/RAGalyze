"""
Configuration module for the application.
Contains settings and constants used across the application.
"""

# Application settings
APP_NAME = "MyComplexApp"
VERSION = "2.1.0"
DEBUG = True
ENVIRONMENT = "development"

# Database settings
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "name": "myapp_db",
    "user": "admin",
    "password": "secret123"
}

# Feature flags
ENABLE_LOGGING = True
ENABLE_CACHING = False
ENABLE_MONITORING = True

# Import submodules
from .settings import *
from .database_settings import *