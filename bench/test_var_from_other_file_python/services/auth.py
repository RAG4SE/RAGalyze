"""
Authentication service module.
Uses configuration variables and models.
"""

from config.settings import JWT_SECRET_KEY, JWT_EXPIRATION_HOURS, LOG_LEVEL
from models.user import User, UserManager
from models import SYSTEM_ADMIN_EMAIL
from . import SERVICE_TIMEOUT, MAX_RETRIES

class TokenManager:
    """Manages JWT tokens using configuration variables."""

    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.expiration_hours = JWT_EXPIRATION_HOURS
        self.log_level = LOG_LEVEL

    def generate_token(self, user):
        """Generate token for user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "exp_hours": self.expiration_hours
        }
        print(f"[{self.log_level}] Generating token for user {user.username}")
        print(f"Token expires in {self.expiration_hours} hours")
        return f"token_{hash(str(payload))}"

    def validate_token(self, token):
        """Validate token."""
        print(f"[{self.log_level}] Validating token: {token[:20]}...")
        return True  # Simplified validation

class AuthService:
    """Authentication service using models and configuration."""

    def __init__(self):
        self.user_manager = UserManager()
        self.token_manager = TokenManager()
        self.service_timeout = SERVICE_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.system_admin_email = SYSTEM_ADMIN_EMAIL

    def authenticate(self, username, password):
        """Authenticate user and return token."""
        print(f"Authenticating user: {username}")
        print(f"Timeout: {self.service_timeout}s, Max retries: {self.max_retries}")

        # Find user (simplified)
        user = None
        for uid, u in self.user_manager.users.items():
            if u.username == username:
                user = u
                break

        if user:
            token = self.token_manager.generate_token(user)
            return {"user": user, "token": token}
        return None

    def is_system_admin(self, user):
        """Check if user is system admin."""
        return user.email == self.system_admin_email