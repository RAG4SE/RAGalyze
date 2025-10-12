#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import LiteralTypeInferenceQuery
import logging

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_literal_type_inference():
    """Test the LiteralTypeInferenceQuery class with various literal expressions."""

    type_inferencer = LiteralTypeInferenceQuery(debug=True)

    # Test cases: (literal, expected_type)
    test_cases = [
        # String literals
        ('"hello world"', "string"),
        ('"test"', "string"),
        ("'single quoted'", "string"),
        ("'a'", "string"),  # Single character in quotes is string
        ('"123"', "string"),  # Numeric string
        ('""', "string"),  # Empty string
        ("''", "string"),  # Empty single quotes

        # Numeric literals - integers
        ("42", "int"),
        ("0", "int"),
        ("-17", "int"),
        ("1000000", "int"),

        # Numeric literals - floats
        ("3.14", "float"),
        ("0.0", "float"),
        ("-2.718", "float"),
        ("1.5e10", "float"),
        ("1.5E-10", "float"),

        # Boolean literals
        ("true", "bool"),
        ("false", "bool"),
        ("TRUE", "bool"),
        ("FALSE", "bool"),
        ("True", "bool"),
        ("False", "bool"),

        # Null/nil literals
        ("null", "null"),
        ("nil", "null"),
        ("None", "null"),
        ("NULL", "null"),
        ("NIL", "null"),

        # Character literals
        ("'a'", "char"),
        ("'1'", "char"),
        ("'@'", "char"),
        ("'\\n'", "char"),  # Escape sequence

        # Hexadecimal literals
        ("0xFF", "int"),
        ("0x0", "int"),
        ("0xDEADBEEF", "int"),
        ("0X123", "int"),

        # Binary literals
        ("0b1010", "int"),
        ("0b0", "int"),
        ("0B1111", "int"),

        # Octal literals
        ("0o123", "int"),
        ("0O456", "int"),
        ("0123", "int"),  # Legacy octal format

        # Complex numeric literals
        ("100L", "long"),  # Long integer
        ("3.14f", "float"),  # Float literal
        ("2.718281828459045", "double"),  # High precision

        # Special numeric formats
        ("1_000_000", "int"),  # Numeric separators
        ("0.5f", "float"),

        # Edge cases
        ("inf", "float"),  # Infinity
        ("nan", "float"),  # Not a number
        ("Infinity", "float"),

        # Unicode and escape sequences
        ('"\\u0041"', "string"),  # Unicode escape
        ('"\\n\\t\\r"', "string"),  # Escape sequences
        ("'\\''", "string"),  # Escaped quote

        # Raw strings
        ('r"raw string"', "string"),
        ("r'raw'", "string"),

        # Format strings (if applicable)
        ('f"hello {name}"', "string"),

        # Byte strings
        ('b"bytes"', "bytes"),

        # Complex numbers
        ("1+2j", "complex"),
        ("3.14j", "complex"),
    ]

    logger.info("=" * 80)
    logger.info("Testing LiteralTypeInferenceQuery")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (literal, expected) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {literal}")
        logger.info(f"Expected: {expected}")

        try:
            result = type_inferencer(literal)
            logger.info(f"Got: {result}")

            if result == expected:
                logger.info(f"✓ Test {i} PASSED")
                passed += 1
            else:
                logger.warning(f"✗ Test {i} FAILED")
                logger.warning(f"  Expected: {expected}")
                logger.warning(f"  Got: {result}")
                failed += 1

        except Exception as e:
            logger.error(f"✗ Test {i} ERROR: {e}")
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("LiteralTypeInferenceQuery Test Summary:")
    logger.info(f"Total tests: {len(test_cases)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 80)

    return failed == 0


def test_edge_cases():
    """Test edge cases for LiteralTypeInferenceQuery."""

    type_inferencer = LiteralTypeInferenceQuery(debug=True)

    # Edge cases that should handle gracefully
    edge_cases = [
        # Empty or whitespace
        ("", None),  # Empty string
        ("   ", None),  # Only whitespace

        # Invalid literals
        ("invalid_literal", None),  # Not a literal
        ("undefined", None),  # Not a standard literal

        # Malformed literals
        ('"unclosed string', None),  # Unclosed string
        ("123abc", None),  # Invalid number format
        ("0xG", None),  # Invalid hex

        # Very long literals
        ('"' + "a" * 1000 + '"', "string"),  # Long string
        ("123456789012345678901234567890", "int"),  # Long number

        # Special characters
        ("'\\x41'", "char"),  # Hex escape
        ("'\\u263A'", "char"),  # Unicode emoji

        # Zero values
        ("0", "int"),
        ("0.0", "float"),
        ("0x0", "int"),
        ("0b0", "int"),
        ("false", "bool"),

        # Maximum/minimum values
        ("2147483647", "int"),  # Max 32-bit int
        ("-2147483648", "int"),  # Min 32-bit int

        # Scientific notation edge cases
        ("1e308", "float"),  # Very large float
        ("1e-308", "float"),  # Very small float

        # Mixed formats
        ("0x1.8p1", "float"),  # Hex float
    ]

    logger.info("\n" + "=" * 80)
    logger.info("Testing LiteralTypeInferenceQuery Edge Cases")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (literal, expected) in enumerate(edge_cases, 1):
        logger.info(f"\nEdge Case {i}: {literal}")
        logger.info(f"Expected: {expected}")

        try:
            result = type_inferencer(literal)
            logger.info(f"Got: {result}")

            # For edge cases, None is acceptable if we can't determine the type
            if result == expected or (expected is None and result is None):
                logger.info(f"✓ Edge Case {i} PASSED")
                passed += 1
            else:
                logger.warning(f"✗ Edge Case {i} FAILED")
                logger.warning(f"  Expected: {expected}")
                logger.warning(f"  Got: {result}")
                failed += 1

        except Exception as e:
            logger.error(f"✗ Edge Case {i} ERROR: {e}")
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Edge Cases Test Summary:")
    logger.info(f"Total edge cases: {len(edge_cases)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 80)

    return failed == 0


def test_consistency():
    """Test that the LiteralTypeInferenceQuery produces consistent results."""

    type_inferencer = LiteralTypeInferenceQuery(debug=True)

    # Test cases that should produce consistent results across multiple calls
    consistency_tests = [
        '"hello"',
        "42",
        "3.14",
        "true",
        "null",
        "'a'",
        "0xFF",
        "0b1010",
    ]

    logger.info("\n" + "=" * 80)
    logger.info("Testing LiteralTypeInferenceQuery Consistency")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, literal in enumerate(consistency_tests, 1):
        logger.info(f"\nConsistency Test {i}: {literal}")

        try:
            # Call the inferencer multiple times
            results = []
            for _ in range(3):
                result = type_inferencer(literal)
                results.append(result)

            # Check if all results are the same
            if len(set(results)) == 1:
                logger.info(f"✓ Consistency Test {i} PASSED - All results: {results[0]}")
                passed += 1
            else:
                logger.warning(f"✗ Consistency Test {i} FAILED - Inconsistent results: {results}")
                failed += 1

        except Exception as e:
            logger.error(f"✗ Consistency Test {i} ERROR: {e}")
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Consistency Test Summary:")
    logger.info(f"Total consistency tests: {len(consistency_tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests for LiteralTypeInferenceQuery."""
    logger.info("Starting LiteralTypeInferenceQuery tests...")

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
    basic_tests_passed = test_literal_type_inference()
    edge_cases_passed = test_edge_cases()
    consistency_passed = test_consistency()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info(f"Basic tests: {'PASSED' if basic_tests_passed else 'FAILED'}")
    logger.info(f"Edge cases: {'PASSED' if edge_cases_passed else 'FAILED'}")
    logger.info(f"Consistency: {'PASSED' if consistency_passed else 'FAILED'}")
    logger.info("=" * 80)

    all_passed = basic_tests_passed and edge_cases_passed and consistency_passed

    if all_passed:
        logger.info("All tests passed! ✓")
        return True
    else:
        logger.warning("Some tests failed. See details above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)