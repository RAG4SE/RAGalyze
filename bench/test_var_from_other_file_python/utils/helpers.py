"""
Helper functions module.
Uses configuration variables and works with models.
"""

from config import APP_NAME, DEBUG
from config.settings import LOG_LEVEL
from models.product import Product
from models.user import User
from . import DEFAULT_CURRENCY, SUPPORTED_CURRENCIES

def format_price(price, currency=None):
    """Format price with currency."""
    curr = currency or DEFAULT_CURRENCY
    if curr not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {curr}")

    if DEBUG:
        print(f"[{LOG_LEVEL}] Formatting price: {price} {curr}")

    return f"{price:.2f} {curr}"

def validate_email(email):
    """Validate email format."""
    if DEBUG:
        print(f"[{LOG_LEVEL}] Validating email: {email}")

    # Simple email validation
    if "@" not in email or "." not in email:
        return False
    return True

def generate_user_summary(user):
    """Generate user summary using User object and config."""
    if not isinstance(user, User):
        raise ValueError("Expected User object")

    summary = f"""
    User Summary for {APP_NAME}
    ==========================
    ID: {user.user_id}
    Username: {user.username}
    Email: {user.email}
    Role: {user.role}
    Token Expiry: {user.get_token_expiry()} seconds
    Is System Admin: {user.is_system_admin()}
    Permissions: {', '.join(user.get_permissions())}
    """

    if DEBUG:
        print(f"[{LOG_LEVEL}] Generated user summary for {user.username}")

    return summary.strip()

def generate_product_summary(product):
    """Generate product summary using Product object."""
    if not isinstance(product, Product):
        raise ValueError("Expected Product object")

    should_cache = "Yes" if product.should_cache() else "No"

    summary = f"""
    Product Summary
    ===============
    ID: {product.product_id}
    Name: {product.name}
    Base Price: {format_price(product.base_price)}
    Final Price: {format_price(product.get_final_price())}
    Category: {product.category}
    Tenant ID: {product.tenant_id}
    Should Cache: {should_cache}
    Cache Key: {product.get_cache_key()}
    Monitoring Enabled: {product.monitoring_enabled}
    """

    if DEBUG:
        print(f"[{LOG_LEVEL}] Generated product summary for {product.name}")

    return summary.strip()

def calculate_discount_price(product, discount_percent):
    """Calculate discounted price for a product."""
    if not isinstance(product, Product):
        raise ValueError("Expected Product object")

    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")

    final_price = product.get_final_price()
    discount_amount = final_price * (discount_percent / 100)
    discounted_price = final_price - discount_amount

    if DEBUG:
        print(f"[{LOG_LEVEL}] Applied {discount_percent}% discount to {product.name}")
        print(f"[{LOG_LEVEL}] Original: {format_price(final_price)}, Discounted: {format_price(discounted_price)}")

    return discounted_price