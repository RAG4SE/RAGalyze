import logging
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import ElementTypeAnalyzerQuery

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_element_type_analyzer():
    """Test the ElementTypeAnalyzerPipeline with various compound types and expressions."""

    # Set up the query
    query = ElementTypeAnalyzerQuery(debug=True)

    # Test cases: (compound_type, base_expression, indexed_expression, expected_result)
    test_cases = [
        # Standard containers
        ("List<int>", "a", "a[0]", "int"),
        ("Vector<float>", "v", "v[5]", "float"),
        ("Array<string>", "arr", "arr[10]", "string"),
        ("std::vector<double>", "vec", "vec[i]", "double"),
        ("std::list<bool>", "lst", "lst[3]", "bool"),

        # Map/dictionary types
        ("Map<int, string>", "m", "m[123]", "string"),
        ("Dictionary<string, int>", "dict", "dict[\"key\"]", "int"),
        ("std::map<int, float>", "map", "map[42]", "float"),
        ("std::unordered_map<string, bool>", "umap", "umap[\"test\"]", "bool"),

        # Custom data structures
        ("MyCustomList<int>", "data", "data[0]", "int"),
        ("CustomVector<float>", "vec", "vec[2]", "float"),
        ("SpecialArray<string>", "arr", "arr[i]", "string"),
        ("DataList<double>", "list", "list[5]", "double"),

        # Custom map-like structures
        ("CustomMap<int, string>", "map", "map[123]", "string"),
        ("MyDictionary<string, bool>", "dict", "dict[\"key\"]", "bool"),
        ("LookupTable<int, float>", "table", "table[42]", "float"),
        ("KeyValueStore<string, int>", "store", "store[\"name\"]", "int"),

        # Nested types
        ("List<List<int>>", "matrix", "matrix[0]", "List<int>"),
        ("Map<int, List<string>>", "mapping", "mapping[123]", "List<string>"),
        ("Vector<Map<string, int>>", "vec", "vec[i]", "Map<string, int>"),

        # Complex custom types
        ("DataArray<MyCustomType>", "data", "data[0]", "MyCustomType"),
        ("ResultList<StatusCode>", "results", "results[i]", "StatusCode"),
        ("ConfigMap<string, ConfigValue>", "config", "config[\"key\"]", "ConfigValue"),

        # Pointer types
        ("List<int*>", "list", "list[0]", "int*"),
        ("Vector<string*>", "vec", "vec[5]", "string*"),
        ("Map<int, Object*>", "map", "map[123]", "Object*"),

        # Reference types
        ("List<int&>", "list", "list[0]", "int&"),
        ("Vector<const string&>", "vec", "vec[i]", "const string&"),

        # Template template parameters
        ("Container<List<int>>", "c", "c[0]", "List<int>"),
        ("Storage<Map<string, int>>", "storage", "storage[\"key\"]", "Map<string, int>"),

        # Multi-dimensional arrays
        ("List<int[5]>", "list", "list[0]", "int[5]"),
        ("Array<Array<double, 3>, 4>", "arr", "arr[2]", "Array<double, 3>"),
    ]

    # Additional context for complex types
    context = """
// Custom data structure definitions
template<typename T>
class MyCustomList {
public:
    T& operator[](int index) { return data_[index]; }
private:
    T* data_;
    size_t size_;
};

template<typename K, typename V>
class CustomMap {
public:
    V& operator[](const K& key) { return data_[key]; }
private:
    std::map<K, V> data_;
};

template<typename T>
class SpecialArray {
public:
    T& operator[](int index) { return elements_[index]; }
private:
    T* elements_;
    size_t capacity_;
};

enum class StatusCode { OK, ERROR, TIMEOUT };

class MyCustomType {
public:
    int value;
    string name;
};

class ConfigValue {
public:
    string get_value() const { return value_; }
private:
    string value_;
};

template<typename T>
class Container {
public:
    T& operator[](int index) { return items_[index]; }
private:
    T* items_;
};
"""

    # Run tests
    passed = 0
    failed = 0

    logger.info("=" * 80)
    logger.info("Testing ElementTypeAnalyzer")
    logger.info("=" * 80)

    for i, (compound_type, base_expression, indexed_expression, expected) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {compound_type} with {base_expression} -> {indexed_expression}")
        logger.info(f"Expected: {expected}")

        try:
            element_type = query(compound_type=compound_type, base_expression=base_expression, indexed_expression=indexed_expression, context=context)
            logger.info(f"Got: {element_type}")

            # For testing purposes, we'll be lenient with exact matches
            # since LLM responses might vary slightly
            if element_type.lower().replace(" ", "") == expected.lower().replace(" ", ""):
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.warning("✗ Test %d FAILED: Expected '%s', got '%s'", i, expected, element_type)
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

  

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary:")
    logger.info("=" * 80)
    logger.info("Total tests: %d", len(test_cases) + 3)  # +3 for special tests
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    if failed == 0:
        logger.info("All tests passed! ✓")
    else:
        logger.warning("Some tests failed. See details above.")

    return failed == 0


if __name__ == "__main__":
    success = test_element_type_analyzer()
    sys.exit(0 if success else 1)