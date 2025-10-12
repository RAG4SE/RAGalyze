"""
User model module.
Uses configuration variables from config module.
"""

from config import APP_NAME, DEBUG, DATABASE_CONFIG
from config.settings import LOG_LEVEL, JWT_EXPIRATION_HOURS
from . import CURRENT_USER_ID, SYSTEM_ADMIN_EMAIL
from .database_models import BaseModel

class User(BaseModel):
    def __init__(self, user_id, username, email, role="user"):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.app_name = APP_NAME  # Variable from config module
        self.is_debug = DEBUG      # Variable from config module

    def get_permissions(self):
        """Get user permissions based on role and configuration."""
        if self.role == "admin":
            return ["read", "write", "delete", "admin"]
        elif self.role == "moderator":
            return ["read", "write", "moderate"]
        else:
            return ["read"]

    def get_token_expiry(self):
        """Get token expiry time from config."""
        return JWT_EXPIRATION_HOURS * 3600  # Convert to seconds

    def is_system_admin(self):
        """Check if user is system admin."""
        return self.email == SYSTEM_ADMIN_EMAIL

class UserManager:
    def __init__(self):
        self.current_user_id = CURRENT_USER_ID
        self.db_config = DATABASE_CONFIG
        self.log_level = LOG_LEVEL

    def create_user(self, username, email):
        """Create a new user."""
        user_id = f"user_{len(self.users) + 1}"
        user = User(user_id, username, email)
        self.users[user_id] = user
        return user

    def get_current_user(self):
        """Get the current active user."""
        return self.users.get(self.current_user_id)

    def __init__(self):
        self.users = {}
        self.current_user_id = CURRENT_USER_ID