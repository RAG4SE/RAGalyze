#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import TypeParameterSubstitutionQuery

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_type_parameter_substitution_query():
    """Test the TypeParameterSubstitutionQuery with various compound type expressions."""

    substitutor = TypeParameterSubstitutionQuery(debug=True)

    # Test cases: (type_expression, type_mapping, expected_result)
    test_cases = [
        # C++ template types
        ("vector<T>", {"T": "int"}, "vector<int>"),
        ("list<T>", {"T": "string"}, "list<string>"),
        ("map<K, V>", {"K": "string", "V": "int"}, "map<string, int>"),
        ("unordered_map<Key, Value>", {"Key": "int", "Value": "string"}, "unordered_map<int, string>"),
        ("set<T>", {"T": "double"}, "set<double>"),
        ("array<T, N>", {"T": "float", "N": "10"}, "array<float, 10>"),

        # Nested C++ types
        ("vector<vector<T>>", {"T": "int"}, "vector<vector<int>>"),
        ("map<string, vector<int>>", {}, "map<string, vector<int>"),  # No mapping
        ("vector<map<K, V>>", {"K": "string", "V": "int"}, "vector<map<string, int>>"),
        ("Optional<vector<T>>", {"T": "User"}, "Optional<vector<User>>"),
        ("Result<vector<T>, Error>", {"T": "Data"}, "Result<vector<Data>, Error>"),
        ("unique_ptr<T[]>", {"T": "char"}, "unique_ptr<char[]>"),
        ("shared_ptr<MyClass<T>>", {"T": "User"}, "shared_ptr<MyClass<User>>"),
        ("variant<T, U>", {"T": "int", "U": "string"}, "variant<int, string>"),
        ("tuple<T, U, V>", {"T": "int", "U": "string", "V": "double"}, "tuple<int, string, double>"),

        # Complex C++ cases
        ("std::vector<std::map<T, U>>", {"T": "string", "U": "int"}, "std::vector<std::map<string, int>>"),
        ("boost::variant<T, boost::optional<U>>", {"T": "int", "U": "string"}, "boost::variant<int, boost::optional<string>>"),
        ("future<vector<optional<T>>>", {"T": "Data"}, "future<vector<optional<Data>>>"),
        ("expected<T, error_code>", {"T": "Result"}, "expected<Result, error_code>"),

        # Java-style generics
        ("List<T>", {"T": "String"}, "List<String>"),
        ("Map<K, V>", {"K": "String", "V": "Integer"}, "Map<String, Integer>"),
        ("Optional<T>", {"T": "User"}, "Optional<User>"),
        ("Stream<T>", {"T": "Integer"}, "Stream<Integer>"),
        ("List<Map<String, Integer>>", {}, "List<Map<String, Integer>>"),  # No mapping
        ("Collection<T>", {"T": "Object"}, "Collection<Object>"),
        ("Iterable<T>", {"T": "Number"}, "Iterable<Number>"),
        ("Function<T, R>", {"T": "String", "R": "Integer"}, "Function<String, Integer>"),
        ("BiFunction<T, U, R>", {"T": "String", "U": "Integer", "R": "Boolean"}, "BiFunction<String, Integer, Boolean>"),
        ("Predicate<T>", {"T": "User"}, "Predicate<User>"),
        ("Supplier<T>", {"T": "Data"}, "Supplier<Data>"),
        ("Consumer<T>", {"T": "Event"}, "Consumer<Event>"),
        ("CompletableFuture<Optional<T>>", {"T": "Result"}, "CompletableFuture<Optional<Result>>"),

        # Java wildcards and bounds (should preserve wildcards)
        ("List<? extends T>", {"T": "Number"}, "List<? extends Number>"),
        ("Map<? super K, ? extends V>", {"K": "String", "V": "Integer"}, "Map<? super String, ? extends Integer>"),
        ("List<T extends Number>", {"T": "Integer"}, "List<Integer>"),  # LLM should handle this correctly

        # Python-style generics
        ("List[T]", {"T": "int"}, "List[int]"),
        ("Dict[K, V]", {"K": "str", "V": "int"}, "Dict[str, int]"),
        ("Optional[T]", {"T": "User"}, "Optional[User]"),
        ("Tuple[T, U]", {"T": "int", "U": "str"}, "Tuple[int, str]"),
        ("Set[T]", {"T": "float"}, "Set[float]"),
        ("Union[T, U]", {"T": "int", "U": "str"}, "Union[int, str]"),
        ("Callable[[T], U]", {"T": "str", "U": "int"}, "Callable[[str], int]"),
        ("Iterator[T]", {"T": "Data"}, "Iterator[Data]"),
        ("Generator[T, None, None]", {"T": "Value"}, "Generator[Value, None, None]"),
        ("Sequence[T]", {"T": "int"}, "Sequence[int]"),
        ("Mapping[K, V]", {"K": "str", "V": "int"}, "Mapping[str, int]"),
        ("MutableSequence[T]", {"T": "float"}, "MutableSequence[float]"),
        ("AbstractSet[T]", {"T": "str"}, "AbstractSet[str]"),
        ("Container[T]", {"T": "object"}, "Container[object]"),
        ("Awaitable[T]", {"T": "Result"}, "Awaitable[Result]"),
        ("AsyncIterator[T]", {"T": "Item"}, "AsyncIterator[Item]"),
        ("Coroutine[Any, Any, T]", {"T": "Value"}, "Coroutine[Any, Any, Value]"),
        ("Deque[T]", {"T": "int"}, "Deque[int]"),
        ("Counter[T]", {"T": "str"}, "Counter[str]"),
        ("DefaultDict[K, V]", {"K": "str", "V": "int"}, "DefaultDict[str, int]"),
        ("OrderedDict[K, V]", {"K": "str", "V": "int"}, "OrderedDict[str, int]"),
        ("ChainMap[K, V]", {"K": "str", "V": "int"}, "ChainMap[str, int]"),
        ("frozenset[T]", {"T": "int"}, "frozenset[int]"),
        ("Protocol[T]", {"T": "Data"}, "Protocol[Data]"),
        ("Generic[T]", {"T": "Type"}, "Generic[Type]"),
        ("Type[T]", {"T": "User"}, "Type[User]"),
        ("ClassVar[T]", {"T": "int"}, "ClassVar[int]"),
        ("Final[T]", {"T": "str"}, "Final[str]"),

        # C# style generics
        ("List<T>", {"T": "string"}, "List<string>"),
        ("Dictionary<K, V>", {"K": "string", "V": "int"}, "Dictionary<string, int>"),
        ("Nullable<T>", {"T": "int"}, "Nullable<int>"),
        ("Task<T>", {"T": "Result"}, "Task<Result>"),
        ("IEnumerable<T>", {"T": "User"}, "IEnumerable<User>"),
        ("ICollection<T>", {"T": "Data"}, "ICollection<Data>"),
        ("IQueryable<T>", {"T": "Entity"}, "IQueryable<Entity>"),
        ("Tuple<T, U>", {"T": "int", "U": "string"}, "Tuple<int, string>"),
        ("ValueTuple<T, U, V>", {"T": "int", "U": "string", "V": "bool"}, "ValueTuple<int, string, bool>"),
        ("Func<T, R>", {"T": "string", "R": "int"}, "Func<string, int>"),
        ("Action<T>", {"T": "Event"}, "Action<Event>"),
        ("Predicate<T>", {"T": "User"}, "Predicate<User>"),
        ("Lazy<T>", {"T": "Data"}, "Lazy<Data>"),
        ("Observable<T>", {"T": "Event"}, "Observable<Event>"),
        ("Maybe<T>", {"T": "Value"}, "Maybe<Value>"),
        ("Either<T, U>", {"T": "Error", "U": "Success"}, "Either<Error, Success>"),

        # Partial substitution cases
        ("Result<T, Error>", {"T": "Data"}, "Result<Data, Error>"),  # Only T mapped
        ("Tuple<T, U, V>", {"T": "int", "V": "bool"}, "Tuple<int, U, bool>"),  # Partial mapping
        ("Optional<T>", {}, "Optional<T>"),  # Empty mapping
        ("vector<ClassT<T>>", {"T": "int"}, "vector<ClassT<int>>"),  # Don't replace T in ClassT
        ("MyTemplateClass<T, OtherClass<U>>", {"T": "int", "U": "string"}, "MyTemplateClass<int, OtherClass<string>>"),  # Mixed substitution

        # Edge cases
        ("int", {"T": "int"}, "int"),  # No type parameters
        ("string", {"T": "string"}, "string"),  # No type parameters
        ("void", {"T": "void"}, "void"),  # No type parameters
        ("MyClass", {"T": "int"}, "MyClass"),  # No type parameters
        ("vector", {"T": "int"}, "vector"),  # Missing template parameter
        ("", {"T": "int"}, ""),  # Empty string

        # Real-world complex examples
        ("vector<map<string, vector<optional<T>>>>", {"T": "User"}, "vector<map<string, vector<optional<User>>>>"),
        ("unordered_map<K, vector<map<V, W>>>", {"K": "string", "V": "int", "W": "bool"}, "unordered_map<string, vector<map<int, bool>>>"),
        ("Result<vector<T>, map<U, V>>", {"T": "Data", "U": "Key", "V": "Value"}, "Result<vector<Data>, map<Key, Value>>"),
        ("CompletableFuture<Optional<Map<K, List<V>>>>", {"K": "string", "V": "int"}, "CompletableFuture<Optional<Map<string, List<int>>>>"),
        ("Function<Supplier<T>, Consumer<U>>", {"T": "Data", "U": "Event"}, "Function<Supplier<Data>, Consumer<Event>>"),
        ("vector<map<set<T>, vector<U>>>", {"T": "int", "U": "string"}, "vector<map<set<int>, vector<string>>>"),
        ("optional<tuple<map<K, V>, list<T>>>", {"K": "string", "V": "int", "T": "float"}, "optional<tuple<map<string, int>, list<float>>>"),

        # Language-specific edge cases
        ("std::unique_ptr<T[]>", {"T": "char"}, "std::unique_ptr<char[]>"),  # C++ array syntax
        ("List<? super T>", {"T": "Number"}, "List<? super Number>"),  # Java wildcard
        ("Map.Entry<K, V>", {"K": "String", "V": "Integer"}, "Map.Entry<String, Integer>"),  # Java nested class
        ("Optional[Union[T, None]]", {"T": "int"}, "Optional[Union[int, None]]"),  # Python Union
        ("Tuple[T, ...]", {"T": "int"}, "Tuple[int, ...]"),  # Python variadic
        ("Nullable<T>", {"T": "DateTime"}, "Nullable<DateTime>"),  # C# nullable

        # Multi-language mixed scenarios
        ("vector<T>", {"T": "java.lang.String"}, "vector<java.lang.String>"),  # C++ with Java type
        ("List[cpp::vector<T>]", {"T": "int"}, "List[cpp::vector<int>]"),  # Python with C++ type
        ("Map<T, List<U>>", {"T": "String", "U": "Integer"}, "Map<String, List<Integer>>"),  # Java generics
        ("Dictionary<K, List<V>>", {"K": "string", "V": "int"}, "Dictionary<string, List<int>>"),  # C# generics
    ]

    logger.info("=" * 80)
    logger.info("Testing TypeParameterSubstitutionQuery with Multi-Language Support")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (type_expression, type_mapping, expected_result) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {type_expression}")
        logger.info(f"Mapping: {type_mapping}")
        logger.info(f"Expected: {expected_result}")

        try:
            actual_result = substitutor(type_expression, type_mapping)
            logger.info(f"Got: {actual_result}")

            # Check if result matches expected
            if actual_result == expected_result:
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.warning("✗ Test %d FAILED:", i)
                logger.warning(f"  Expected: {expected_result}")
                logger.warning(f"  Got: {actual_result}")
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TypeParameterSubstitutionQuery Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting TypeParameterSubstitutionQuery tests...")

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
    test_passed = test_type_parameter_substitution_query()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("TypeParameterSubstitutionQuery tests: %s", "PASSED" if test_passed else "FAILED")
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