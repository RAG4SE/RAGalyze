"""
Validation utilities module.
Uses configuration and model variables for validation.
"""

from config.settings import MAX_RETRIES, RATE_LIMIT_PER_MINUTE
from models.user import User
from models.product import Product, ProductCategory
from .helpers import validate_email

def validate_user_data(user_data):
    """Validate user data using User model constraints."""
    errors = []

    if not user_data:
        errors.append("User data is required")
        return {"valid": False, "errors": errors}

    # Required fields
    required_fields = ["username", "email", "role"]
    for field in required_fields:
        if field not in user_data:
            errors.append(f"Missing required field: {field}")

    # Validate email
    if "email" in user_data:
        if not validate_email(user_data["email"]):
            errors.append("Invalid email format")

    # Validate role
    if "role" in user_data:
        valid_roles = ["user", "admin", "moderator"]
        if user_data["role"] not in valid_roles:
            errors.append(f"Invalid role. Must be one of: {valid_roles}")

    # Validate username length
    if "username" in user_data:
        username = user_data["username"]
        if len(username) < 3:
            errors.append("Username must be at least 3 characters")
        elif len(username) > 50:
            errors.append("Username must be less than 50 characters")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_product_data(product_data):
    """Validate product data using Product model constraints."""
    errors = []

    if not product_data:
        errors.append("Product data is required")
        return {"valid": False, "errors": errors}

    # Required fields
    required_fields = ["name", "price", "category"]
    for field in required_fields:
        if field not in product_data:
            errors.append(f"Missing required field: {field}")

    # Validate price
    if "price" in product_data:
        price = product_data["price"]
        if not isinstance(price, (int, float)):
            errors.append("Price must be a number")
        elif price <= 0:
            errors.append("Price must be greater than 0")
        elif price > 100000:
            errors.append("Price seems too high")

    # Validate category
    if "category" in product_data:
        valid_categories = [
            ProductCategory.ELECTRONICS,
            ProductCategory.CLOTHING,
            ProductCategory.BOOKS,
            ProductCategory.HOME
        ]
        if product_data["category"] not in valid_categories:
            errors.append(f"Invalid category. Must be one of: {valid_categories}")

    # Validate name
    if "name" in product_data:
        name = product_data["name"]
        if len(name.strip()) < 1:
            errors.append("Product name cannot be empty")
        elif len(name) > 200:
            errors.append("Product name must be less than 200 characters")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_api_request(method, endpoint, user=None):
    """Validate API request parameters."""
    errors = []

    # Validate method
    valid_methods = ["GET", "POST", "PUT", "DELETE"]
    if method not in valid_methods:
        errors.append(f"Invalid HTTP method. Must be one of: {valid_methods}")

    # Validate endpoint format
    if not endpoint.startswith("/"):
        errors.append("Endpoint must start with '/'")

    # Rate limiting check (simplified)
    if user and hasattr(user, 'request_count'):
        if user.request_count > RATE_LIMIT_PER_MINUTE:
            errors.append(f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests per minute")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "max_retries": MAX_RETRIES
    }

def validate_auth_token(token, user):
    """Validate authentication token for user."""
    errors = []

    if not token:
        errors.append("Token is required")

    if not isinstance(user, User):
        errors.append("Valid user object is required")

    # Check token format (simplified)
    if token and not token.startswith("token_"):
        errors.append("Invalid token format")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

class ValidationSummary:
    """Summarize validation results."""

    @staticmethod
    def format_validation_result(result, item_type="item"):
        """Format validation result for display."""
        if result["valid"]:
            return f"✓ {item_type.title()} is valid"
        else:
            error_list = "\n  - ".join(result["errors"])
            return f"✗ {item_type.title()} validation failed:\n  - {error_list}"