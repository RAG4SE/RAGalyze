#!/usr/bin/env python3

"""
Test script to verify the enhanced _resolve_function_call method
can handle built-in functions like map.find(), vector.begin(), etc.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import ChainedExpressionTypeAnalyzerPipeline

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_builtin_function_resolution():
    """Test the built-in function resolution functionality."""

    # Test context with built-in function calls
    test_context = """
#include <map>
#include <vector>
#include <string>
#include <iostream>

int main() {
    std::map<std::string, int> my_map = {{"key1", 1}, {"key2", 2}};
    std::vector<int> my_vector = {1, 2, 3, 4, 5};
    std::string my_string = "hello world";

    // Test built-in function calls
    auto it = my_map.find("key1");
    auto size = my_vector.size();
    auto begin_it = my_vector.begin();
    auto c_ptr = my_string.c_str();
    bool is_empty = my_map.empty();

    return 0;
}
"""

    try:
        # Initialize the pipeline with debug mode
        pipeline = ChainedExpressionTypeAnalyzerPipeline(debug=True)

        # Test cases for built-in functions
        test_cases = [
            (
                'my_map.at("key1")',
                "std::map<std::string, int>",
                "int&",
            ),  # Should return iterator type
            (
                "my_vector.size()",
                "std::vector<int>",
                "std::size_t",
            ),  # Should return size_t
            (
                "my_vector.begin()",
                "std::vector<int>",
                "std::vector<int>::iterator",
            ),  # Should return iterator type
            (
                "my_string.c_str()",
                "std::string",
                "const char*",
            ),  # Should return const char*
            (
                "my_map.empty()",
                "std::map<std::string, int>",
                "bool",
            ),  # Should return bool
        ]

        logger.info("Testing built-in function resolution")

        pass_cnt = 0

        for function_call, class_name, expected_return_type in test_cases:
            logger.info(f"\nTesting: {function_call}() on {class_name}")

            # Create a component for testing
            component = {
                "name": function_call.split(".")[-1],  # Extract function name
                "type": "function_call",
                "operator": ".",
            }

            try:
                # Test the builtin_function_query method directly
                result = pipeline.builtin_function_query(
                    function_name=component["name"],
                    class_name=class_name,
                    context=test_context,
                )

                if result and result == expected_return_type:
                    logger.info(f"✓ SUCCESS: {function_call}() → {result}")
                    pass_cnt += 1
                else:
                    logger.error(f"✗ Expected: {expected_return_type}, got: {result}")

            except Exception as e:
                logger.error(f"✗ ERROR: Exception testing {function_call}(): {e}")

        logger.info(
            f"\nBuilt-in function resolution test completed, {pass_cnt}/{len(test_cases)} passed"
        )

    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_builtin_function_resolution()
    sys.exit(0 if success else 1)
