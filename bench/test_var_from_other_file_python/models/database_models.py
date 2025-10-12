"""
Database models module.
Uses database configuration variables.
"""

from config.database_settings import DB_ENGINE, DB_ECHO, MAX_CONNECTIONS, BATCH_SIZE
from config.settings import LOG_LEVEL, REQUEST_TIMEOUT

class BaseModel:
    """Base model class that all other models inherit from."""

    def __init__(self):
        self.db_engine = DB_ENGINE
        self.db_echo = DB_ECHO
        self.max_connections = MAX_CONNECTIONS
        self.batch_size = BATCH_SIZE
        self.log_level = LOG_LEVEL

    def save(self):
        """Save model to database."""
        if self.db_echo:
            print(f"[{self.log_level}] Saving {self.__class__.__name__} to {self.db_engine}")
        return True

    def delete(self):
        """Delete model from database."""
        if self.db_echo:
            print(f"[{self.log_level}] Deleting {self.__class__.__name__} from {self.db_engine}")
        return True

class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self):
        self.engine = DB_ENGINE
        self.max_connections = MAX_CONNECTIONS
        self.request_timeout = REQUEST_TIMEOUT
        self.current_connections = 0
        self.is_connected = False

    def connect(self):
        """Connect to database."""
        if self.current_connections >= self.max_connections:
            raise Exception("Max connections reached")

        print(f"Connecting to {self.engine} database...")
        print(f"Timeout: {self.request_timeout}s")
        self.is_connected = True
        self.current_connections += 1
        return True

    def disconnect(self):
        """Disconnect from database."""
        if self.is_connected:
            print(f"Disconnecting from {self.engine} database...")
            self.is_connected = False
            self.current_connections -= 1

    def execute_batch(self, operations):
        """Execute batch operations."""
        print(f"Executing {len(operations)} operations in batches of {BATCH_SIZE}")
        return True