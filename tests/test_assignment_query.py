#!/usr/bin/env python3
"""
Simple test for AssignmentQuery class.
Tests various variable assignment scenarios to verify the pipeline can correctly identify
the latest assignment of a variable from code context.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragalyze.agent import AssignmentQuery
from ragalyze.configs import set_global_config_value


def test_simple_assignments():
    """Test AssignmentQuery with simple variable assignments."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")

    # Create the query instance
    query = AssignmentQuery(debug=True)

    # Test cases: (context, variable_name, expected_assignment)
    test_cases = [
        # Basic assignments
        ("int x = 5;", "x", "5"),
        ('string name = "test";', "name", '"test"'),
        ("double value = 3.14;", "value", "3.14"),
        # Reassignments
        ("int x = 5; x = 10;", "x", "10"),
        ('string name = "test"; name = "updated";', "name", '"updated"'),
        ("double value = 3.14; value = 2.71;", "value", "2.71"),
        # Multiple variables
        ("int x = 5; int y = 10; x = 15;", "x", "15"),
        ("int x = 5; int y = 10; y = 20;", "y", "20"),
        # Complex expressions
        ("int x = 5 + 3;", "x", "5 + 3"),
        ('string name = "hello" + "world";', "name", '"hello" + "world"'),
        ("double result = calculate(a, b);", "result", "calculate(a, b)"),
        # Auto declarations (C++)
        ("auto x = 5;", "x", "5"),
        ('auto name = "test";', "name", '"test"'),
        ("auto ptr = &variable;", "ptr", "&variable"),
        # Var declarations (Java/JavaScript)
        ("var x = 5;", "x", "5"),
        ('var name = "test";', "name", '"test"'),
        ("const value = 3.14;", "value", "3.14"),
        ("let result = compute();", "result", "compute()"),
        # Function calls
        ("int result = function_call();", "result", "function_call()"),
        ("string output = obj.toString();", "output", "obj.toString()"),
        # No assignment cases
        ("int x;", "x", None),
        ("string name;", "name", None),
        ("extern int global_var;", "global_var", None),
    ]

    # Run tests
    passed = 0
    failed = 0

    print("=" * 60)
    print("Testing AssignmentQuery with Simple Assignments")
    print("=" * 60)

    for i, (context, variable_name, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: variable '{variable_name}' in context")
        print(f"Context: {context}")
        print(f"Expected assignment: {expected}")

        try:
            result = query(context, variable_name)

            if result == expected:
                print(f"  ✓ PASSED: got '{result}'")
                passed += 1
            else:
                print(f"  ✗ FAILED: expected '{expected}', got '{result}'")
                failed += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Simple Assignments Test Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return passed, failed


def test_complex_scenarios():
    """Test AssignmentQuery with more complex scenarios."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")

    # Create the query instance
    query = AssignmentQuery(debug=True)

    # Test cases with more complex contexts
    test_cases = [
        # Variable with scope
        (
            """
        void function() {
            int x = 5;
            if (condition) {
                x = 10;
            }
            x = 15;
        }
        """,
            "x",
            "15",
        ),
        # Variable in loops
        (
            """
        void function() {
            int sum = 0;
            for (int i = 0; i < 10; i++) {
                sum = sum + i;
            }
            sum = sum * 2;
        }
        """,
            "sum",
            "sum * 2",
        ),
        # Multiple assignments with different types
        (
            """
        void function() {
            auto x = 5;
            x = 3.14;
            x = "string";
        }
        """,
            "x",
            '"string"',
        ),
        # Variable assignment with function calls
        (
            """
        void function() {
            Engine engine;
            auto result = engine.process();
            result = engine.calculate();
        }
        """,
            "result",
            "engine.calculate()",
        ),
        # Chain assignments
        (
            """
        void function() {
            int x = 5;
            int y = x;
            int z = y;
            z = 10;
        }
        """,
            "z",
            "10",
        ),
        # Variable with multiple declarations
        (
            """
        void function() {
            int x = 1;
            {
                int x = 2;
                x = 3;
            }
            x = 4;
        }
        """,
            "x",
            "4",
        ),
        # Assignment with complex expressions
        (
            """
        void function() {
            auto config = new Configuration();
            auto result = config.builder()
                .setOption("opt1", "val1")
                .setOption("opt2", "val2")
                .build();
        }
        """,
            "result",
            'config.builder()\n                .setOption("opt1", "val1")\n                .setOption("opt2", "val2")\n                .build()',
        ),
        # Assignment from return value
        (
            """
        void function() {
            auto pipeline = engine.createPipeline();
            auto stage = pipeline.stage("primary");
            stage = pipeline.stage("secondary");
        }
        """,
            "stage",
            'pipeline.stage("secondary")',
        ),
        # Variable with initialization and reassignment
        (
            """
        void function() {
            std::vector<int> numbers = {1, 2, 3};
            numbers = {4, 5, 6};
        }
        """,
            "numbers",
            "{4, 5, 6}",
        ),
        # No assignment found cases
        (
            """
        void function() {
            int x;
            std::cout << x;
        }
        """,
            "x",
            None,
        ),
        (
            """
        void function() {
            extern int global_var;
            printf("%d", global_var);
        }
        """,
            "global_var",
            None,
        ),
    ]

    # Run tests
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Testing AssignmentQuery with Complex Scenarios")
    print("=" * 60)

    for i, (context, variable_name, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: variable '{variable_name}'")
        print(f"Expected assignment: {expected}")

        try:
            result = query(context, variable_name)

            if result == expected:
                print(f"  ✓ PASSED: got '{result}'")
                passed += 1
            else:
                print(f"  ✗ FAILED: expected '{expected}', got '{result}'")
                failed += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Complex Scenarios Test Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return passed, failed


def test_real_world_examples():
    """Test AssignmentQuery with real-world code examples."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")

    # Create the query instance
    query = AssignmentQuery(debug=True)

    # Real-world examples from call_maze_cpp
    test_cases = [
        # From call_maze_cpp main function
        (
            """
        int main() {
            using namespace callmaze;

            Engine engine;

            auto descriptor = engine.accessor()->compound.describe();
            std::cout << descriptor << '\\\\n';

            auto prepared = chain_value(
                tap(std::vector<double>{1.0, 4.0, 9.0, 16.0}, [](auto& vec) {
                    vec.push_back(25.0);
                    std::rotate(vec.begin(), vec.begin() + 1, vec.end());
                }),
                [](std::vector<double> vec) {
                    std::transform(vec.begin(), vec.end(), vec.begin(), [](double value) {
                        return std::sqrt(value);
                    });
                    return vec;
                },
                [](std::vector<double> vec) {
                    vec.erase(vec.begin());
                    return vec;
                });

            auto value = engine.process(prepared);

            auto report = engine.accessor()->compound.sample(value);
            std::cout << report.summary() << '\\\\n';

            auto tuningScore = engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize();
            std::cout << "Secondary tuning score: " << tuningScore << '\\\\n';

            return 0;
        }
        """,
            [
                ("descriptor", "engine.accessor()->compound.describe()"),
                (
                    "prepared",
                    "chain_value(\n                tap(std::vector<double>{1.0, 4.0, 9.0, 16.0}, [](auto& vec) {\n                    vec.push_back(25.0);\n                    std::rotate(vec.begin(), vec.begin() + 1, vec.end());\n                }),\n                [](std::vector<double> vec) {\n                    std::transform(vec.begin(), vec.end(), vec.begin(), [](double value) {\n                        return std::sqrt(value);\n                    });\n                    return vec;\n                },\n                [](std::vector<double> vec) {\n                    vec.erase(vec.begin());\n                    return vec;\n                })",
                ),
                ("value", "engine.process(prepared)"),
                ("report", "engine.accessor()->compound.sample(value)"),
                (
                    "tuningScore",
                    'engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize()',
                ),
            ],
        ),
        # Multiple variables with reassignments
        (
            """
        void example() {
            int x = 5;
            int y = 10;

            x = y + 5;
            y = x * 2;

            x = y - 3;
            y = x + 1;
        }
        """,
            [
                ("x", "y - 3"),
                ("y", "x + 1"),
            ],
        ),
    ]

    # Run tests
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Testing AssignmentQuery with Real-World Examples")
    print("=" * 60)

    for test_group_idx, (context, variable_tests) in enumerate(test_cases, 1):
        print(f"\nTest Group {test_group_idx}:")

        for var_idx, (variable_name, expected) in enumerate(variable_tests, 1):
            print(f"\n  Test {var_idx}: variable '{variable_name}'")
            print(f"  Expected assignment: {expected}")

            try:
                result = query(context, variable_name)

                if result == expected:
                    print(f"    ✓ PASSED")
                    passed += 1
                else:
                    print(f"    ✗ FAILED: expected '{expected}', got '{result}'")
                    failed += 1

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Real-World Examples Test Summary:")
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return passed, failed


def main():
    """Run all tests."""
    print("Testing AssignmentQuery")
    print("=" * 60)

    # Test simple assignments
    simple_passed, simple_failed = test_simple_assignments()

    # Test complex scenarios
    complex_passed, complex_failed = test_complex_scenarios()

    # Test real-world examples
    real_passed, real_failed = test_real_world_examples()

    # Overall summary
    total_passed = simple_passed + complex_passed + real_passed
    total_failed = simple_failed + complex_failed + real_failed
    total_tests = total_passed + total_failed

    print("\n" + "=" * 60)
    print("OVERALL TEST SUMMARY")
    print("=" * 60)
    print(f"Simple Assignments: {simple_passed} passed, {simple_failed} failed")
    print(f"Complex Scenarios: {complex_passed} passed, {complex_failed} failed")
    print(f"Real-World Examples: {real_passed} passed, {real_failed} failed")
    print("-" * 60)
    print(
        f"TOTAL: {total_passed} passed, {total_failed} failed out of {total_tests} tests"
    )
    print("=" * 60)

    if total_failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. See details above.")
        print("Note: Some failures may be due to LLM interpretation differences.")


if __name__ == "__main__":
    main()
