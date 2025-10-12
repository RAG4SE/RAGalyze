#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import FetchClassPipeline

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_fetch_class_pipeline():
    """Test the enhanced FetchClassPipeline with templated class names."""

    pipeline = FetchClassPipeline(debug=True)

    # Test cases: (class_name, should_find_definition, expected_namespaces, expected_base, expected_args)
    test_cases = [
        # These should work with the existing test data
        ("Engine", True, [], "", []),  # Should exist in the call_maze_cpp test data
        ("DataProcessor", True, [], "", []),  # Should exist in the test data
        ("DataMap<std::string, DataProcessor>", True, [], "DataMap", ["std::string", "DataProcessor"]),  # Should fallback to DataMap
        ("std::vector<DataProcessor>", False, ["std"], "vector", ["DataProcessor"]),  # Should fallback to vector
        ("List<Engine>", False, [], "List", ["Engine"]),  # Should fallback to List
        ("NonExistentClass", False, [], "", []),  # Should not find anything

        # Namespace-qualified test cases
        ("std::string", False, ["std"], "string", []),  # Should fallback to string
        ("boost::filesystem::path", False, ["boost", "filesystem"], "path", []),  # Should fallback to path
        ("namespace1::TestNamespace", False, ["namespace1"], "TestNamespace", []),  # Should fallback to TestNamespace

        # Complex nested cases
        ("Container<Map<KeyType, ValueType>>", False, [], "Container", ["Map<KeyType, ValueType>"]),  # Should fallback to Container
        ("OuterClass<InnerClass<Type>>", False, [], "OuterClass", ["InnerClass<Type>"]),  # Should fallback to OuterClass

        # Additional namespace cases
        ("my.package.MyClass", False, ["my", "package"], "MyClass", []),  # Java-style namespace
        ("com.example.util.DataStructure", False, ["com", "example", "util"], "DataStructure", []),  # Complex Java namespace
    ]

    logger.info("=" * 80)
    logger.info("Testing FetchClassPipeline with template and namespace support")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (class_name, should_find, expected_namespaces, expected_base, expected_args) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {class_name}")
        logger.info(f"Should find definition: {should_find}")
        logger.info(f"Expected: namespaces={expected_namespaces}, base={expected_base}, args={expected_args}")

        try:
            result = pipeline(class_name)

            # Check if result is a dictionary with expected structure
            if isinstance(result, dict):
                class_definition = result.get("class_definition")
                original_class_name = result.get("original_class_name")
                namespaces = result.get("namespaces", [])
                base_class_name = result.get("base_class_name")
                type_arguments = result.get("type_arguments", [])

                found = class_definition is not None and class_definition.strip() != "None"

                logger.info(f"Original: {original_class_name}")
                logger.info(f"Namespaces: {namespaces}")
                logger.info(f"Base: {base_class_name}")
                logger.info(f"Args: {type_arguments}")

                if found:
                    logger.info(f"✓ Found definition (length: {len(class_definition)} chars)")
                    logger.info(f"Preview: {class_definition[:200]}...")
                else:
                    logger.info("✗ No definition found")

                # Check if we found what we expected
                find_match = found == should_find

                # Check namespaces match
                namespaces_match = namespaces == expected_namespaces

                # Check base name match
                base_match = True
                if expected_base:
                    base_match = base_class_name.lower().replace(" ", "") == expected_base.lower().replace(" ", "")

                # Check args match
                args_match = True
                if expected_args:
                    args_match = len(type_arguments) == len(expected_args)
                    if args_match and len(expected_args) > 0:
                        expected_set = set(arg.lower().replace(" ", "") for arg in expected_args)
                        actual_set = set(arg.lower().replace(" ", "") for arg in type_arguments)
                        args_match = expected_set == actual_set

                if find_match and namespaces_match and base_match and args_match:
                    logger.info("✓ Test %d PASSED", i)
                    passed += 1
                else:
                    logger.warning("✗ Test %d FAILED:", i)
                    if not find_match:
                        logger.warning(f"  Find: expected {should_find}, got {found}")
                    if not namespaces_match:
                        logger.warning(f"  Namespaces: expected {expected_namespaces}, got {namespaces}")
                    if not base_match:
                        logger.warning(f"  Base: expected '{expected_base}', got '{base_class_name}'")
                    if not args_match:
                        logger.warning(f"  Args: expected {expected_args}, got {type_arguments}")
                    failed += 1
            elif result is None:
                # No result found - check if this was expected
                if not should_find:
                    logger.info("✓ Test %d PASSED (correctly returned None)", i)
                    passed += 1
                else:
                    logger.warning("✗ Test %d FAILED: Expected to find definition but got None", i)
                    failed += 1
            else:
                logger.warning("✗ Test %d FAILED: Expected dict result, got %s", i, type(result))
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FetchClassPipeline Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting FetchClassPipeline tests...")

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
    test_passed = test_fetch_class_pipeline()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("FetchClassPipeline tests: %s", "PASSED" if test_passed else "FAILED")
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