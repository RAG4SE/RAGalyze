"""
Database-specific settings.
These variables will be used by the database module.
"""

# Connection settings
DB_ENGINE = "postgresql"
DB_ECHO = False
DB_POOL_PRE_PING = True
DB_POOL_RECYCLE = 3600

# Performance settings
BATCH_SIZE = 1000
QUERY_TIMEOUT = 60
MAX_CONNECTIONS = 20
MIN_CONNECTIONS = 5

# Migration settings
MIGRATION_DIR = "migrations"
AUTO_MIGRATE = True