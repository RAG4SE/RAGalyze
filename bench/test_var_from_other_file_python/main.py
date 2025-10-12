#!/usr/bin/env python3
"""
Main test script to demonstrate cross-file variable usage.
"""

# Import variables from different modules
from config import APP_NAME, VERSION, DEBUG
from config.settings import LOG_LEVEL, REQUEST_TIMEOUT
from models.user import User, UserManager
from models.product import Product, ProductCategory
from services.auth import AuthService
from services.api import APIService
from services.cache import CacheService
from utils.helpers import format_price, generate_user_summary, generate_product_summary

def main():
    print(f"=== {APP_NAME} v{VERSION} ===")
    print(f"Debug Mode: {DEBUG}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Request Timeout: {REQUEST_TIMEOUT}s")
    print()

    # Create users using UserManager
    user_manager = UserManager()
    admin_user = user_manager.create_user("admin", "admin@myapp.com")
    admin_user.role = "admin"

    regular_user = user_manager.create_user("john_doe", "john@example.com")

    print("=== Users ===")
    print(generate_user_summary(admin_user))
    print()
    print(generate_user_summary(regular_user))
    print()

    # Create products
    laptop = Product("p1", "Laptop", 999.99, ProductCategory.ELECTRONICS)
    tshirt = Product("p2", "T-Shirt", 29.99, ProductCategory.CLOTHING)

    print("=== Products ===")
    print(generate_product_summary(laptop))
    print()
    print(generate_product_summary(tshirt))
    print()

    # Test authentication
    auth_service = AuthService()
    auth_result = auth_service.authenticate("admin", "password")
    if auth_result:
        print(f"Authentication successful for {auth_result['user'].username}")
        print(f"Token: {auth_result['token']}")
    print()

    # Test API service
    api_service = APIService()
    products_response = api_service.process_request("GET", "/products")
    print("=== API Response ===")
    for product in products_response["products"]:
        print(f"{product['name']}: {format_price(product['price'])}")
    print()

    # Test cache service
    cache_service = CacheService()
    cache_service.cache_product(laptop)
    cached_data = cache_service.get(laptop.get_cache_key())
    if cached_data:
        print(f"Cached product: {cached_data['name']}")

    print("\n=== Cache Stats ===")
    stats = cache_service.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()