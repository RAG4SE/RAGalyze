#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import FetchCalleeInfoPipeline


def test_language_aware_parsing():
    """Test that language-aware parsing works correctly for different languages."""

    pipeline = FetchCalleeInfoPipeline(debug=True)

    # Test cases for different languages
    test_cases = [
        # C++ cases
        {
            "language": "cpp",
            "expression": "std::vector<int>::push_back(element)",
            "expected_components": [
                {"name": "std", "type": "namespace_access", "operator": "::"},
                {"name": "vector<int>", "type": "namespace_access", "operator": "::"},
                {"name": "push_back", "type": "function_call", "operator": "."},
            ],
        },
        {
            "language": "cpp",
            "expression": "namespace1::namespace2::class_instance.func1()->member->func2()",
            "expected_components": [
                {"name": "namespace1", "type": "namespace_access", "operator": "::"},
                {"name": "namespace2", "type": "namespace_access", "operator": "::"},
                {"name": "class_instance", "type": "instance", "operator": None},
                {"name": "func1", "type": "function_call", "operator": "."},
                {"name": "member", "type": "member_access", "operator": "->"},
                {"name": "func2", "type": "function_call", "operator": "->"},
            ],
        },
        {
            "language": "cpp",
            "expression": "A::B::f()",
            "expected_components": [
                {"name": "A", "type": "namespace_access", "operator": "::"},
                {"name": "B", "type": "namespace_access", "operator": "::"},
                {"name": "f", "type": "function_call", "operator": "."},
            ],
        },
        # Python cases
        {
            "language": "python",
            "expression": "namespace1.class_instance.func1().member1.func2()",
            "expected_components": [
                {"name": "namespace1", "type": "namespace_access", "operator": "."},
                {"name": "class_instance", "type": "instance", "operator": None},
                {"name": "func1", "type": "function_call", "operator": "."},
                {"name": "member1", "type": "member_access", "operator": "."},
                {"name": "func2", "type": "function_call", "operator": "."},
            ],
        },
        {
            "language": "python",
            "expression": "my_module.MyClass.my_method()",
            "expected_components": [
                {"name": "my_module", "type": "namespace_access", "operator": "."},
                {"name": "MyClass", "type": "namespace_access", "operator": "."},
                {"name": "my_method", "type": "function_call", "operator": "."},
            ],
        },
        # Java cases
        {
            "language": "java",
            "expression": "com.example.MyClass.myMethod()",
            "expected_components": [
                {"name": "com", "type": "namespace_access", "operator": "."},
                {"name": "example", "type": "namespace_access", "operator": "."},
                {"name": "MyClass", "type": "namespace_access", "operator": "."},
                {"name": "myMethod", "type": "function_call", "operator": "."},
            ],
        },
        # Solidity cases
        {
            "language": "solidity",
            "expression": "MyContract.myFunction()",
            "expected_components": [
                {"name": "MyContract", "type": "namespace_access", "operator": "."},
                {"name": "myFunction", "type": "function_call", "operator": "."},
            ],
        },
        # Generic cases (fallback)
        {
            "language": "unknown",
            "expression": "obj.method().member",
            "expected_components": [
                {"name": "obj", "type": "instance", "operator": None},
                {"name": "method", "type": "function_call", "operator": "."},
                {"name": "member", "type": "member_access", "operator": "."},
            ],
        },
    ]

    print("Testing language-aware chained expression parsing:")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['language']} - {test_case['expression']}")

        try:
            components = pipeline._parse_chained_expression(test_case["expression"])
            print(f"Parsed components:")
            for j, component in enumerate(components):
                print(
                    f"  {j+1}. Name: {component['name']}, Type: {component['type']}, Operator: {component['operator']}"
                )

            # Check if components match expected
            expected = test_case["expected_components"]
            if len(components) == len(expected):
                match = True
                for actual_comp, expected_comp in zip(components, expected):
                    if (
                        actual_comp["name"] != expected_comp["name"]
                        or actual_comp["type"] != expected_comp["type"]
                        or actual_comp["operator"] != expected_comp["operator"]
                    ):
                        match = False
                        break

                if match:
                    print(f"✓ Test {i} PASSED")
                    passed += 1
                else:
                    print(f"✗ Test {i} FAILED - Component mismatch")
                    failed += 1
            else:
                print(f"✗ Test {i} FAILED - Component count mismatch")
                print(f"  Expected: {len(expected)} components")
                print(f"  Got: {len(components)} components")
                failed += 1

        except Exception as e:
            print(f"✗ Test {i} ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Language-Aware Parsing Test Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_language_aware_parsing()
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. See details above.")
