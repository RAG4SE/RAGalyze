"""
Models module containing data models and business logic.
"""

from .user import User, UserManager
from .product import Product, ProductCategory
from .database_models import BaseModel, DatabaseManager

# Global variables used across modules
CURRENT_USER_ID = None
DEFAULT_TENANT_ID = "default_tenant"
SYSTEM_ADMIN_EMAIL = "admin@myapp.com"