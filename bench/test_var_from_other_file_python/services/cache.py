"""
Cache service module.
Uses configuration variables for caching operations.
"""

from config import ENABLE_CACHING
from config.settings import CACHE_TTL, CACHE_MAX_SIZE
from models.product import Product

class CacheService:
    """Cache service using configuration variables."""

    def __init__(self):
        self.enabled = ENABLE_CACHING
        self.ttl = CACHE_TTL
        self.max_size = CACHE_MAX_SIZE
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}

    def get(self, key):
        """Get value from cache."""
        if not self.enabled:
            self.cache_stats["misses"] += 1
            return None

        if key in self.cache:
            self.cache_stats["hits"] += 1
            print(f"[CACHE] Hit for key: {key}")
            return self.cache[key]
        else:
            self.cache_stats["misses"] += 1
            print(f"[CACHE] Miss for key: {key}")
            return None

    def set(self, key, value, ttl=None):
        """Set value in cache."""
        if not self.enabled:
            return False

        # Check cache size limit
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        cache_ttl = ttl or self.ttl
        self.cache[key] = {
            "value": value,
            "expires_at": cache_ttl  # Simplified - would use actual timestamp
        }
        print(f"[CACHE] Set key: {key}, TTL: {cache_ttl}s")
        return True

    def cache_product(self, product):
        """Cache a product object."""
        if isinstance(product, Product):
            cache_key = product.get_cache_key()
            product_data = {
                "id": product.product_id,
                "name": product.name,
                "price": product.get_final_price(),
                "category": product.category
            }
            self.set(cache_key, product_data)
            return True
        return False

    def _evict_oldest(self):
        """Evict oldest cache entries."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            print(f"[CACHE] Evicted key: {oldest_key}")

    def get_stats(self):
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        return {
            "enabled": self.enabled,
            "ttl": self.ttl,
            "max_size": self.max_size,
            "current_size": len(self.cache),
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": f"{hit_rate:.2f}%"
        }