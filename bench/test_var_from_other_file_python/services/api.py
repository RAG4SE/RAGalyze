"""
API service module.
Uses configuration and models for API operations.
"""

from config import ENABLE_LOGGING, ENABLE_MONITORING
from config.settings import RATE_LIMIT_PER_MINUTE, REQUEST_TIMEOUT, ALLOWED_ORIGINS
from models.product import Product, ProductCategory
from models import CURRENT_USER_ID
from . import SERVICE_REGISTRY

class RequestHandler:
    """Handles API requests using configuration variables."""

    def __init__(self):
        self.rate_limit = RATE_LIMIT_PER_MINUTE
        self.timeout = REQUEST_TIMEOUT
        self.allowed_origins = ALLOWED_ORIGINS
        self.enable_logging = ENABLE_LOGGING
        self.enable_monitoring = ENABLE_MONITORING
        self.current_user_id = CURRENT_USER_ID

    def handle_request(self, method, endpoint, data=None):
        """Handle incoming API request."""
        if self.enable_logging:
            print(f"[API] {method} {endpoint}")
            print(f"[API] Rate limit: {self.rate_limit}/min")
            print(f"[API] Timeout: {self.timeout}s")
            print(f"[API] Current user: {self.current_user_id}")

        # Check rate limiting (simplified)
        if endpoint == "/products" and method == "GET":
            return self.get_products()
        elif endpoint == "/products" and method == "POST":
            return self.create_product(data)
        else:
            return {"error": "Endpoint not found"}

    def get_products(self):
        """Get products with monitoring."""
        products = [
            Product("p1", "Laptop", 999.99, ProductCategory.ELECTRONICS),
            Product("p2", "T-Shirt", 29.99, ProductCategory.CLOTHING),
            Product("p3", "Python Book", 49.99, ProductCategory.BOOKS)
        ]

        result = []
        for product in products:
            product_data = {
                "id": product.product_id,
                "name": product.name,
                "price": product.get_final_price(),
                "category": product.category
            }

            if self.enable_monitoring:
                monitoring_data = product.get_monitoring_data()
                if monitoring_data:
                    product_data["monitoring"] = monitoring_data

            result.append(product_data)

        return {"products": result}

    def create_product(self, data):
        """Create new product."""
        if not data:
            return {"error": "No data provided"}

        product = Product(
            data.get("id"),
            data.get("name"),
            data.get("price"),
            data.get("category")
        )

        return {"message": "Product created", "product": product.name}

class APIService:
    """Main API service."""

    def __init__(self):
        self.request_handler = RequestHandler()
        self.service_registry = SERVICE_REGISTRY
        self.service_registry["api"] = self

    def register_endpoint(self, endpoint, handler):
        """Register new endpoint."""
        self.service_registry[f"endpoint_{endpoint}"] = handler

    def process_request(self, method, endpoint, data=None):
        """Process API request."""
        return self.request_handler.handle_request(method, endpoint, data)