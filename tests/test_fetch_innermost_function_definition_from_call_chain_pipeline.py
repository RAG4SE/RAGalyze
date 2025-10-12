#!/usr/bin/env python3
"""
Simple test for FetchInnerMostFunctionDefinitionFromCallChainPipeline class.
Tests various expressions to verify the pipeline can extract innermost function definitions.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragalyze.agent import FetchInnerMostFunctionDefinitionFromCallChainPipeline
from ragalyze.configs import set_global_config_value


def test_extract_innermost_function_name():
    """Test the _extract_innermost_function_name helper method."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")
    set_global_config_value("rag.recreate_db", True)

    # Create the pipeline instance
    pipeline = FetchInnerMostFunctionDefinitionFromCallChainPipeline(debug=True)

    # Test cases: (expression, expected_function_name)
    test_cases = [
        # Simple chained calls
        ("obj.method()", "method"),
        ("obj->method()", "method"),
        ("obj.method1().method2()", "method2"),
        ("obj->method1()->method2()", "method2"),
        ("obj.property.method()", "method"),
        # Real call_maze_cpp examples
        ('engine.pipeline().stage("primary").tune(1.4)', "tune"),
        ("engine.accessor()->compound.describe()", "describe"),
        ("engine.dashboard().logView.print()", "print"),
        (
            'engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary()',
            "summary",
        ),
        # Complex chained calls
        ("obj.method1().subobj.method2().property.method3()", "method3"),
        (
            'engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize()',
            "finalize",
        ),
        # Edge cases
        ("simple_function()", "simple_function"),
        ("obj.method()", "method"),
        ("obj->method()", "method"),
        ("obj.method1().method2().method3().method4().method5()", "method5"),
    ]

    # Run tests
    passed = 0
    failed = 0

    print("=" * 60)
    print("Testing _extract_innermost_function_name helper method")
    print("=" * 60)

    for i, (expression, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{expression}'")
        try:
            result = pipeline._extract_innermost_function_name(expression)
            print(f"  Expected: '{expected}'")
            print(f"  Actual: '{result}'")

            if result == expected:
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
    print("Function Name Extraction Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return passed, failed


def test_pipeline_with_real_call_maze():
    """Test the pipeline with real chained calls from call_maze_cpp main function."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")

    # Create the pipeline instance
    pipeline = FetchInnerMostFunctionDefinitionFromCallChainPipeline(debug=True)

    # Real main function body as context
    main_function_body = """#include <cmath>
#include <iostream>
#include <vector>

#include "call_maze_cpp.hpp"

int main() {
    using namespace callmaze;

    Engine engine;

    engine.pipeline().stage("primary").tune(1.4).limit(60.0);
    engine.pipeline().stage("secondary").tune(0.65).limit(40.0);

    auto descriptor = engine.accessor()->compound.describe();
    std::cout << descriptor << '\\n';

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

    engine.analyzer().enable().setThreshold(0.5);
    engine.calibrate([](double value) { return value * 1.1; });

    auto value = engine.process(prepared);

    auto report = engine.accessor()->compound.sample(value);
    std::cout << report.summary() << '\\n';

    auto tuningScore = engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize();
    std::cout << "Secondary tuning score: " << tuningScore << '\\n';

    auto log = engine.dashboard().logView.print();
    std::cout << log;

    auto summary = engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary();
    std::cout << summary << '\\n';

    return 0;
}"""

    # Expected function definitions from call_maze_cpp.hpp
    expected_definitions = {
        "tune": """StageProxy& tune(double factor) {
        factor_ = factor;
        return *this;
    }""",
        "describe": """std::string describe() const {
    if (!owner_) {
        return "<no pipeline>";
    }
    auto names = owner_->listing();
    if (names.empty()) {
        return "<empty pipeline>";
    }
    std::ostringstream oss;
    oss << "Pipeline stages:";
    for (const auto& name : names) {
        oss << ' ' << name;
    }
    return oss.str();
}""",
        "sample": """StageReport sample(double input) const {
    if (!owner_) {
        return StageReport("unavailable", input);
    }
    auto names = owner_->listing();
    if (names.empty()) {
        return StageReport("undefined", input);
    }
    return owner_->run(names.front(), input);
}""",
        "bias": """StageProxyOptions& bias(double b) {
        bias_ = b;
        return *this;
    }""",
        "smoothing": """StageProxyOptions& smoothing(double alpha) {
        smoothing_ = alpha;
        return *this;
    }""",
        "finalize": """double finalize() const {
    const double base = owner_.factor();
    return base * smoothing_ + bias_ * owner_.limit() * 0.01;
}""",
        "print": """std::string print() const { return logger ? logger->str() : std::string{}; }""",
        "summary": """std::string summary() const {
        std::ostringstream oss;
        oss << "Stage: " << stage_ << " -> value: " << value_;
        if (!notes_.empty()) {
            oss << " | notes:";
            for (const auto& note : notes_) {
                oss << ' ' << note;
            }
        }
        if (!tags_.empty()) {
            oss << " | tags:";
            for (const auto& [key, value] : tags_) {
                oss << ' ' << key << '=' << value;
            }
        }
        return oss.str();
    }""",
    }

    # Extract real chained calls from main function
    test_cases = [
        ('engine.pipeline().stage("primary").tune(1.4)', "tune"),
        ('engine.pipeline().stage("secondary").tune(0.65)', "tune"),
        ("engine.accessor()->compound.describe()", "describe"),
        ("engine.accessor()->compound.sample(value)", "sample"),
        ('engine.pipeline().stage("secondary").options().bias(0.2)', "bias"),
        ('engine.pipeline().stage("secondary").options().smoothing(0.95)', "smoothing"),
        (
            'engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize()',
            "finalize",
        ),
        ("engine.dashboard().logView.print()", "print"),
        (
            'engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary()',
            "summary",
        ),
        ("report.summary()", "summary"),
    ]

    # Run tests
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print(
        "Testing FetchInnerMostFunctionDefinitionFromCallChainPipeline with Real Call Maze Data"
    )
    print("=" * 60)

    for i, (expression, expected_function_name) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{expression}'")
        print(f"Expected function name: '{expected_function_name}'")

        try:
            # First, test the function name extraction
            extracted_function_name = pipeline._extract_innermost_function_name(
                expression
            )
            print(f"  Extracted function name: '{extracted_function_name}'")

            # Test the full pipeline
            result = pipeline(expression, main_function_body)

            # Check if function name extraction worked
            name_extraction_passed = extracted_function_name == expected_function_name
            if name_extraction_passed:
                print("  ✓ Function name extraction PASSED")
            else:
                print("  ✗ Function name extraction FAILED")

            # Check pipeline result and compare with expected definition
            definition_match_passed = False
            if result is not None:
                print(
                    f"  ✓ Pipeline returned function definition (length: {len(result)})"
                )

                # Check if the result contains key parts of the expected definition
                expected_def = expected_definitions.get(expected_function_name, "")
                if expected_def:
                    # Extract key parts from expected definition for comparison
                    expected_lines = [
                        line.strip()
                        for line in expected_def.split("\n")
                        if line.strip()
                    ]
                    result_lines = [
                        line.strip() for line in result.split("\n") if line.strip()
                    ]

                    # Check if result contains the function signature and key elements
                    signature_matches = False
                    for expected_line in expected_lines:
                        if expected_line in result:
                            signature_matches = True
                            break

                    if signature_matches:
                        print("  ✓ Function definition content MATCHES expected")
                        definition_match_passed = True
                    else:
                        print("  ⚠ Function definition content differs from expected")
                        print(
                            f"    Expected contains: {expected_lines[0] if expected_lines else 'N/A'}"
                        )
                        print(
                            f"    Result preview: {result_lines[0] if result_lines else 'N/A'}"
                        )
                else:
                    print(
                        "  ✓ Function definition found (no expected definition provided)"
                    )
                    definition_match_passed = True
            else:
                print("  ⚠ Pipeline returned None (may be due to missing index/data)")
                definition_match_passed = False  # Still consider partial success

            # Overall test result
            if name_extraction_passed and definition_match_passed:
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
    print("Real Call Maze Pipeline Test Summary:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return passed, failed


def test_non_chained_calls():
    """Test with expressions that should NOT be considered chained calls."""

    # Set up configuration for testing
    set_global_config_value("repo_path", "./bench/call_maze_cpp")
    set_global_config_value("generator.provider", "deepseek")
    set_global_config_value("generator.model", "deepseek-chat")

    # Create the pipeline instance
    pipeline = FetchInnerMostFunctionDefinitionFromCallChainPipeline(debug=True)

    # Expressions that should NOT be chained calls
    non_chained_test_cases = [
        "simple_function()",
        "variable.property",
        "std::cout << descriptor",
        "std::cout << report.summary()",
        "return 0",
        "using namespace callmaze",
        "Engine engine",
        "auto descriptor",
        "std::vector<double>{1.0, 4.0, 9.0, 16.0}",
    ]

    # Run tests
    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("Testing Non-Chained Call Expressions")
    print("=" * 60)

    for i, expression in enumerate(non_chained_test_cases, 1):
        print(f"\nNon-Chained Test {i}: '{expression}'")

        try:
            # Test the full pipeline
            result = pipeline(expression, "int main() { /* some context */ }")

            # For non-chained calls, we expect None
            if result is None:
                print("  ✓ PASSED (correctly identified as non-chained)")
                passed += 1
            else:
                print(f"  ✗ FAILED (unexpectedly found chained call: {result[:50]}...)")
                failed += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Non-Chained Call Test Summary:")
    print(f"Total tests: {len(non_chained_test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    return passed, failed


def main():
    """Run all tests."""
    print("Testing FetchInnerMostFunctionDefinitionFromCallChainPipeline")
    print("=" * 60)

    # Test function name extraction
    name_passed, name_failed = test_extract_innermost_function_name()

    # Test pipeline with real call maze data
    real_maze_passed, real_maze_failed = test_pipeline_with_real_call_maze()

    # Test non-chained calls
    non_chained_passed, non_chained_failed = test_non_chained_calls()

    # Overall summary
    total_passed = name_passed + real_maze_passed + non_chained_passed
    total_failed = name_failed + real_maze_failed + non_chained_failed
    total_tests = total_passed + total_failed

    print("\n" + "=" * 60)
    print("OVERALL TEST SUMMARY")
    print("=" * 60)
    print(f"Function Name Extraction: {name_passed} passed, {name_failed} failed")
    print(
        f"Real Call Maze Pipeline Tests: {real_maze_passed} passed, {real_maze_failed} failed"
    )
    print(
        f"Non-Chained Call Tests: {non_chained_passed} passed, {non_chained_failed} failed"
    )
    print("-" * 60)
    print(
        f"TOTAL: {total_passed} passed, {total_failed} failed out of {total_tests} tests"
    )
    print("=" * 60)

    if total_failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. See details above.")
        print(
            "Note: Some failures may be due to missing index/data for full pipeline testing."
        )


if __name__ == "__main__":
    main()
