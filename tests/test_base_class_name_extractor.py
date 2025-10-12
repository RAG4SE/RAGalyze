#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import BaseClassNameExtractor

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_base_class_name_extractor():
    """Test the BaseClassNameExtractor with various templated class names."""

    extractor = BaseClassNameExtractor(debug=True)

    # Test cases: (input, expected_namespaces, expected_base, expected_args)
    test_cases = [
        # C++ templates
        ("DataMap<std::string, DataProcessor>", [], "DataMap", ["std::string", "DataProcessor"]),
        ("std::vector<double>", ["std"], "vector", ["double"]),
        ("std::map<int, std::string>", ["std"], "map", ["int", "std::string"]),
        ("List<int>", [], "List", ["int"]),
        ("Vector<Map<string, int>>", [], "Vector", ["Map<string, int>"]),
        ("Array<DataType, 10>", [], "Array", ["DataType", "10"]),
        ("MyClass<T, U>", [], "MyClass", ["T", "U"]),
        ("TemplateClass<Parameter1, Parameter2, Parameter3>", [], "TemplateClass", ["Parameter1", "Parameter2", "Parameter3"]),

        # Python generics
        ("List[int]", [], "List", ["int"]),
        ("Dict[str, int]", [], "Dict", ["str", "int"]),
        ("Optional[MyType]", [], "Optional", ["MyType"]),
        ("Union[str, int]", [], "Union", ["str", "int"]),
        ("Sequence[DataType]", [], "Sequence", ["DataType"]),

        # Java generics
        ("List<String>", [], "List", ["String"]),
        ("Map<Integer, String>", [], "Map", ["Integer", "String"]),
        ("Optional<User>", [], "Optional", ["User"]),
        ("GenericRepository<T extends Comparable<T>, ID>", [], "GenericRepository", ["T extends Comparable<T>", "ID"]),

        # Simple cases (should return empty args)
        ("MyClass", [], "MyClass", []),
        ("String", [], "String", []),
        ("User", [], "User", []),

        # Complex nested cases
        ("OuterClass<InnerClass<Type>>", [], "OuterClass", ["InnerClass<Type>"]),
        ("Container<Map<KeyType, ValueType>>", [], "Container", ["Map<KeyType, ValueType>"]),

        # Namespace-qualified cases
        ("namespace1::namespace2::Class<T1, T2>", ["namespace1", "namespace2"], "Class", ["T1", "T2"]),
        ("std::map<int, std::vector<std::string>>", ["std"], "map", ["int", "std::vector<std::string>"]),
        ("JavaPackage.InnerClass<String, Integer>", ["JavaPackage"], "InnerClass", ["String", "Integer"]),
        ("boost::filesystem::path", ["boost", "filesystem"], "path", []),
        ("my.package.Class<T extends Comparable<T>>", ["my", "package"], "Class", ["T extends Comparable<T>"]),
        ("outer::middle::inner::NestedClass<Type1, Type2>", ["outer", "middle", "inner"], "NestedClass", ["Type1", "Type2"]),
        ("com.example.util.DataStructure<Key, Value>", ["com", "example", "util"], "DataStructure", ["Key", "Value"]),

        # Edge cases
        ("global::MyClass", ["global"], "MyClass", []),
        ("very.long.namespace.hierarchy.Class", ["very", "long", "namespace", "hierarchy"], "Class", []),
        ("A.B.C<D.E.F>", ["A", "B"], "C", ["D.E.F"]),
    ]

    logger.info("=" * 80)
    logger.info("Testing BaseClassNameExtractor")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (input_name, expected_namespaces, expected_base, expected_args) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {input_name}")
        logger.info(f"Expected: namespaces={expected_namespaces}, base={expected_base}, args={expected_args}")

        try:
            result = extractor(input_name)
            logger.info(f"Got: {result}")

            # Check if result is a dictionary with expected structure
            if isinstance(result, dict):
                actual_namespaces = result.get("namespaces", [])
                actual_base = result.get("base_class_name", "")
                actual_args = result.get("type_arguments", [])

                # Check namespaces match
                namespaces_match = actual_namespaces == expected_namespaces

                # Check base name match
                base_match = actual_base.lower().replace(" ", "") == expected_base.lower().replace(" ", "")

                # Check args match (order might vary, so we check as sets)
                args_match = len(actual_args) == len(expected_args)
                if args_match:
                    # Convert to normalized sets for comparison
                    expected_set = set(arg.lower().replace(" ", "") for arg in expected_args)
                    actual_set = set(arg.lower().replace(" ", "") for arg in actual_args)
                    args_match = expected_set == actual_set

                if namespaces_match and base_match and args_match:
                    logger.info("✓ Test %d PASSED", i)
                    passed += 1
                else:
                    logger.warning("✗ Test %d FAILED:", i)
                    if not namespaces_match:
                        logger.warning(f"  Namespaces: expected {expected_namespaces}, got {actual_namespaces}")
                    if not base_match:
                        logger.warning(f"  Base name: expected '{expected_base}', got '{actual_base}'")
                    if not args_match:
                        logger.warning(f"  Args: expected {expected_args}, got {actual_args}")
                    failed += 1
            else:
                logger.warning("✗ Test %d FAILED: Expected dict result, got %s", i, type(result))
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BaseClassNameExtractor Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting BaseClassNameExtractor tests...")

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
    test_passed = test_base_class_name_extractor()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("BaseClassNameExtractor tests: %s", "PASSED" if test_passed else "FAILED")
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