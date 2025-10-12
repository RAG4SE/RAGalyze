"""
Product model module.
Uses variables from config and other model modules.
"""

from config import ENABLE_CACHING, ENABLE_MONITORING
from config.settings import BATCH_SIZE, CACHE_TTL
from . import DEFAULT_TENANT_ID
from .database_models import BaseModel

class ProductCategory:
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"

    # Configuration-driven pricing
    CATEGORY_MARKUP = {
        ELECTRONICS: 1.2,
        CLOTHING: 1.5,
        BOOKS: 1.3,
        HOME: 1.1
    }

class Product(BaseModel):
    def __init__(self, product_id, name, price, category, tenant_id=None):
        self.product_id = product_id
        self.name = name
        self.base_price = price
        self.category = category
        self.tenant_id = tenant_id or DEFAULT_TENANT_ID  # From models/__init__.py
        self.caching_enabled = ENABLE_CACHING
        self.monitoring_enabled = ENABLE_MONITORING
        self.batch_size = BATCH_SIZE
        self.cache_ttl = CACHE_TTL

    def get_final_price(self):
        """Calculate final price with category markup."""
        markup = ProductCategory.CATEGORY_MARKUP.get(self.category, 1.0)
        return self.base_price * markup

    def should_cache(self):
        """Check if product should be cached based on config."""
        return self.caching_enabled and self.price > 100

    def get_cache_key(self):
        """Generate cache key for this product."""
        return f"product:{self.tenant_id}:{self.product_id}"

    def get_monitoring_data(self):
        """Get monitoring data if enabled."""
        if self.monitoring_enabled:
            return {
                "product_id": self.product_id,
                "price": self.get_final_price(),
                "category": self.category,
                "tenant_id": self.tenant_id
            }
        return None