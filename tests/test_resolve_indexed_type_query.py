import logging
from pathlib import Path
from ragalyze.configs import *

from ragalyze.agent import ResolveIndexedTypeQuery


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_resolve_indexed_type_query():
    """Test the ResolveIndexedTypeQuery with various container types and index expressions."""

    # Set up test fixtures
    try:
        set_global_config_value("repo_path", "./bench/call_maze_cpp")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
        logger.info("Instantiating ResolveIndexedTypeQuery")
        query = ResolveIndexedTypeQuery(debug=True)
    except Exception as e:
        logger.error("Failed to set up test fixtures: %s", e)
        return

    # Test cases: (container_type, index_expression, expected_result)
    test_cases = [
        # Basic array types
        ("vector<int>", "[0]", "int"),
        ("string[]", "[i]", "string"),
        ("int[]", "[5]", "int"),
        ("double*", "[0]", "double"),
        # Map types
        ("map<string, int>", '["key"]', "int"),
        ("unordered_map<string, double>", "[key]", "double"),
        # Nested containers
        ("vector<vector<string>>", "[0]", "vector<string>"),
        ("vector<vector<string>>", "[0][1]", "string"),
        ("map<string, vector<int>>", '["key"]', "vector<int>"),
        ("map<string, vector<int>>", '["key"][0]', "int"),
        # Template containers
        ("array<DataProcessor, 3>", "[0]", "DataProcessor"),
        ("vector<Sensor*>", "[5]", "Sensor*"),
        # Custom containers
        ("DataArray<DataProcessor>", "[0]", "DataProcessor"),
        ("DataMap<string, ConfigValue>", '["threshold"]', "ConfigValue"),
        # Edge cases
        ("MyClass", "[0]", None),  # Unknown if supports indexing
        ("vector<MyClass>", "[i]", "MyClass"),
        ("int", "[0]", None),  # Basic types don't support indexing
    ]

    # Run tests
    passed = 0
    failed = 0

    for i, (container_type, index_expression, expected) in enumerate(test_cases, 1):
        logger.info(
            "Test %d: Resolving index %s on type %s",
            i,
            index_expression,
            container_type,
        )
        try:
            result = query(container_type, index_expression, language="cpp")
            logger.info(
                "Query result for container '%s' with index '%s': %s (expected %s)",
                container_type,
                index_expression,
                result,
                expected,
            )

            if result == expected:
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.error(
                    "✗ Test %d FAILED: Expected '%s', got '%s'", i, expected, result
                )
                failed += 1
        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("=" * 50)
    logger.info("Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 50)

    if failed == 0:
        logger.info("All tests passed! ✓")
    else:
        logger.warning("Some tests failed. See details above.")


if __name__ == "__main__":
    test_resolve_indexed_type_query()
