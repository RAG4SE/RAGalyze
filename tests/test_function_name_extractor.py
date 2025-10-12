#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import FunctionNameExtractor

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_function_name_extractor():
    """Test the FunctionNameExtractor with various function names from different programming languages."""

    extractor = FunctionNameExtractor(debug=True)

    # Test cases: (function_name, expected_base_name, expected_generic_params)
    test_cases = [
        # C++ Function Template Cases
        ("func<int, double>", "func", ["int", "double"]),
        ("MyClass::method<std::string, std::vector<int>>", "MyClass::method", ["std::string", "std::vector<int>"]),
        ("templateFunction<T>", "templateFunction", ["T"]),
        ("processItem<Type, Value>", "processItem", ["Type", "Value"]),
        ("calculateSum<NumType>(values)", "calculateSum", ["NumType"]),
        ("transform<InputType, OutputType>(data)", "transform", ["InputType", "OutputType"]),
        ("factoryFunc<T, U, V>()", "factoryFunc", ["T", "U", "V"]),
        ("std::make_unique<int[]>", "std::make_unique", ["int[]"]),
        ("container::insert<iterator, value_type>", "container::insert", ["iterator", "value_type"]),

        # Java Generic Method Cases
        ("GenericExample.<String>getFirstElement(words)", "getFirstElement", ["String"]),
        ("Collections.<T>emptyList()", "emptyList", ["T"]),
        ("Utils.<Integer, String>convertPair(pair)", "convertPair", ["Integer", "String"]),
        ("MyClass.<E>newInstance(args)", "newInstance", ["E"]),
        ("ArrayUtils.<T>copyOf(original, newLength)", "copyOf", ["T"]),
        ("Stream.<R>map(function)", "map", ["R"]),
        ("Optional.<T>ofNullable(value)", "ofNullable", ["T"]),
        ("Comparators.<T>naturalOrder()", "naturalOrder", ["T"]),
        ("processor.<Double>square(2.5)", "square", ["Double"]),
        # Python Generic Function Cases
        # >.< Python does not template arguments in function calls

    ]

    logger.info("=" * 80)
    logger.info("Testing FunctionNameExtractor with Multi-Language Support")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (function_name, expected_base, expected_params) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {function_name}")
        logger.info(f"Expected: base='{expected_base}', params={expected_params}")

        try:
            actual_base, actual_args = extractor(function_name)
            logger.info(f"Got: base='{actual_base}', params={actual_args}")

            # Check base name match
            base_match = actual_base == expected_base

            # Check params match (order matters for template parameters)
            params_match = len(actual_args) == len(expected_params)
            if params_match:
                for j, (expected_param, actual_param) in enumerate(zip(expected_params, actual_args)):
                    if expected_param != actual_param:
                        logger.warning(f"  Param {j} mismatch: expected '{expected_param}', got '{actual_param}'")
                        params_match = False
                        break

            if base_match and params_match:
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.warning("✗ Test %d FAILED:", i)
                if not base_match:
                    logger.warning(f"  Base name: expected '{expected_base}', got '{actual_base}'")
                if not params_match:
                    logger.warning(f"  Parameters: expected {expected_params}, got {actual_args}")
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FunctionNameExtractor Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting FunctionNameExtractor tests...")

    # Set up test configuration
    try:
        from ragalyze.configs import set_global_config_value
        set_global_config_value("repo_path", "./bench/call_maze_cpp")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
        logger.info("Configuration set successfully")
    except Exception as e:
        logger.error("Failed to set configuration: %s", e)
        return False

    # Run tests
    test_passed = test_function_name_extractor()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("FunctionNameExtractor tests: %s", "PASSED" if test_passed else "FAILED")
    logger.info("=" * 80)

    if test_passed:
        logger.info("All tests passed! ✓")
        return True
    else:
        logger.warning("Some tests failed. See details above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)