#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import ExpressionSegmentationQuery
import logging

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_expression_segmentation():
    """Test the ExpressionSegmentationQuery class with various expressions."""

    segmenter = ExpressionSegmentationQuery(debug=True)

    # Test cases: (expression, expected_segments)
    test_cases = [
        # Basic arithmetic
        ("1 + 2", [("1", "literal"), ("2", "literal")]),
        ("a + b", [("a", "identifier"), ("b", "identifier")]),
        ("x * y + z", [("x", "identifier"), ("y", "identifier"), ("z", "identifier")]),
        # Function calls
        ("func() + method()", [("func()", "function_call"), ("method()", "function_call")]),
        ("obj.method() + func(x)", [("obj.method()", "function_call"), ("func(x)", "function_call")]),
        # Complex expressions with member access
        ("1 + a->f().g()->m", [("1", "literal"), ("a->f().g()->m", "function_call")]),
        ("arr[i] + len", [("arr[i]", "identifier"), ("len", "identifier")]),
        # Ternary operations
        ("x = true ? y : z", [("x", "identifier"), ("true", "literal"), ("y", "identifier"), ("z", "identifier")]),
        ("condition ? a : b", [("condition", "identifier"), ("a", "identifier"), ("b", "identifier")]),
        # Mixed literals and identifiers
        ("123 + variable", [("123", "literal"), ("variable", "identifier")]),
        ('"string" + name', [('"string"', "literal"), ("name", "identifier")]),
        ("true && false", [("true", "literal"), ("false", "literal")]),
        # Complex nested expressions
        ("obj.method().field + func(param)", [("obj.method().field", "identifier"), ("func(param)", "function_call")]),
        ("array[index]->method() + value", [("array[index]->method()", "function_call"), ("value", "identifier")]),
        # Single elements
        ("just_a_variable", [("just_a_variable", "identifier")]),
        ("42", [("42", "literal")]),
        ("function_call()", [("function_call()", "function_call")]),
        # Property access
        ("object.property + another.field", [("object.property", "identifier"), ("another.field", "identifier")]),
        # Array/pointer access
        ("ptr->member + array[index]", [("ptr->member", "identifier"), ("array[index]", "identifier")]),
        # Namespace expressions (should be treated as single identifiers)
        ("std::vector + my_namespace::value", [("std::vector", "identifier"), ("my_namespace::value", "identifier")]),
        # Complex chained calls
        ("a.b().c->d() + e.f().g", [("a.b().c->d()", "function_call"), ("e.f().g", "identifier")]),
    ]

    logger.info("=" * 80)
    logger.info("Testing ExpressionSegmentationQuery")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (expression, expected) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {expression}")
        logger.info(f"Expected: {expected}")

        try:
            result = segmenter(expression)  # context is unused
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
    logger.info("ExpressionSegmentationQuery Test Summary:")
    logger.info(f"Total tests: {len(test_cases)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 80)

    return failed == 0


def test_edge_cases():
    """Test edge cases for ExpressionSegmentationQuery."""

    segmenter = ExpressionSegmentationQuery(debug=True)

    # Edge cases
    edge_cases = [
        # Empty string
        ("", []),
        # Single operator (should return empty list or handle gracefully)
        ("+", []),
        # Only literals
        ("42", [("42", "literal")]),
        # Only identifiers
        ("variable", [("variable", "identifier")]),
        # Only function calls
        ("func()", [("func()", "function_call")]),
        # Complex nested parentheses
        ("func(a, b(c)) + method()", [("func(a, b(c))", "function_call"), ("method()", "function_call")]),
        # Multiple operators
        ("a + b - c * d / e", [("a", "identifier"), ("b", "identifier"), ("c", "identifier"), ("d", "identifier"), ("e", "identifier")]),
        # Logical operators
        ("a && b || c", [("a", "identifier"), ("b", "identifier"), ("c", "identifier")]),
        # Bitwise operators
        ("a & b | c ^ d", [("a", "identifier"), ("b", "identifier"), ("c", "identifier"), ("d", "identifier")]),
        # Comparison operators
        ("a == b != c > d", [("a", "identifier"), ("b", "identifier"), ("c", "identifier"), ("d", "identifier")]),
        # Assignment chains
        ("a = b = c", [("a", "identifier"), ("b", "identifier"), ("c", "identifier")]),
        # Complex member access
        ("obj->field->method().property", [("obj->field->method().property", "identifier")]),
    ]

    logger.info("\n" + "=" * 80)
    logger.info("Testing ExpressionSegmentationQuery Edge Cases")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (expression, expected) in enumerate(edge_cases, 1):
        logger.info(f"\nEdge Case {i}: {expression}")
        logger.info(f"Expected: {expected}")

        try:
            result = segmenter("", expression)
            logger.info(f"Got: {result}")

            if result == expected:
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


def main():
    """Run all tests."""
    logger.info("Starting ExpressionSegmentationQuery tests...")

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
    basic_tests_passed = test_expression_segmentation()
    edge_cases_passed = test_edge_cases()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info(f"Basic tests: {'PASSED' if basic_tests_passed else 'FAILED'}")
    logger.info(f"Edge cases: {'PASSED' if edge_cases_passed else 'FAILED'}")
    logger.info("=" * 80)

    all_passed = basic_tests_passed and edge_cases_passed

    if all_passed:
        logger.info("All tests passed! ✓")
        return True
    else:
        logger.warning("Some tests failed. See details above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
