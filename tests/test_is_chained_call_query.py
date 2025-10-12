#!/usr/bin/env python3
"""
Simple test for IsChainedCallQuery class.
Tests various expressions to determine if they are chained calls.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragalyze.agent import IsChainedCallQuery
from ragalyze.configs import set_global_config_value


def test_is_chained_call():
    """Test the IsChainedCallQuery with various expressions."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")

    # Create the query instance
    query = IsChainedCallQuery(debug=True)

    # Test cases: (expression, context, expected_is_chained)
    test_cases = [
        # Simple calls - should NOT be chained
        ("obj.method()", "", True),
        ("obj->method()", "", True),
        ("function_call()", "", False),
        ("variable.property", "", False),
        # Chained calls - should BE chained
        ("obj.method1().method2()", "", True),
        ("obj->method1()->method2()", "", True),
        ("obj.method().property.subproperty", "", True),
        ('engine.pipeline().stage("primary").tune(1.4)', "", True),
        ("engine.accessor()->compound.describe()", "", True),
        ("obj.method1().method2().method3().method4()", "", True),
        # Complex chained calls
        ("obj.method1().subobj.method2().property.method3()", "", True),
        ('engine.dashboard().latestReport.tag("tuning").summary()', "", True),
        # Edge cases
        ("", "", False),
        ("single_variable", "", False),
        ("obj.method1().method2().method3().method4().method5()", "", True),
    ]

    # Run tests
    passed = 0
    failed = 0

    print("=" * 60)
    print("Testing IsChainedCallQuery")
    print("=" * 60)

    for i, (expression, context, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{expression}'")
        try:
            result = query(expression, context)
            is_chained = result["is_chained_call"]
            confidence = result["confidence"]
            components = result["chain_components"]
            explanation = result["explanation"]

            print(f"  Expected: {expected}")
            print(f"  Actual: {is_chained} (confidence: {confidence})")
            print(f"  Components: {len(components)} found")
            print(f"  Explanation: {explanation}")

            if is_chained == expected:
                print("  ✓ PASSED")
                passed += 1
            else:
                print("  ✗ FAILED")
                failed += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. See details above.")


if __name__ == "__main__":
    test_is_chained_call()
