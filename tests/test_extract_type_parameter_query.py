#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import ExtractTypeParameterQuery

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_extract_type_parameter_query():
    """Test the ExtractTypeParameterQuery with various compound type expressions."""

    extractor = ExtractTypeParameterQuery(debug=True)

    # Test cases: (type_expression, expected_type_parameters, [optional_known_type_parameters])
    test_cases = [
        # Simple template types
        ("vector<T>", ["T"]),
        ("list<T>", ["T"]),
        ("map<K, V>", ["K", "V"]),
        ("unordered_map<Key, Value>", ["Key", "Value"]),
        ("set<T>", ["T"]),
        ("array<T, N>", ["T", "N"]),

        # Nested template types
        ("vector<vector<T>>", ["T"]),
        ("map<string, vector<int>>", []),
        ("vector<map<K, V>>", ["K", "V"]),
        ("Optional<vector<T>>", ["T"]),
        ("Result<vector<T>, Error>", ["T", "Error"]),

        # Complex types with type parameters
        ("unique_ptr<T>", ["T"]),
        ("shared_ptr<MyClass<T>>", ["T"]),
        ("optional<T>", ["T"]),
        ("variant<T, U>", ["T", "U"]),
        ("tuple<T, U, V>", ["T", "U", "V"]),

        # Standard library types with concrete types
        ("vector<int>", []),
        ("map<string, int>", []),
        ("list<double>", []),
        ("unordered_map<string, vector<int>>", []),

        # Java-style generics
        ("List<T>", ["T"]),
        ("Map<K, V>", ["K", "V"]),
        ("Optional<T>", ["T"]),
        ("Stream<T>", ["T"]),
        ("List<Map<String, Integer>>", []),
        ("Collection<T>", ["T"]),
        ("Iterable<T>", ["T"]),
        ("Comparator<T>", ["T"]),
        ("Consumer<T>", ["T"]),
        ("Supplier<T>", ["T"]),
        ("Function<T, R>", ["T", "R"]),
        ("BiFunction<T, U, R>", ["T", "U", "R"]),
        ("Predicate<T>", ["T"]),
        ("UnaryOperator<T>", ["T"]),
        ("BinaryOperator<T>", ["T"]),
        ("Callable<V>", ["V"]),
        ("Runnable", []),  # No type parameters
        ("Future<T>", ["T"]),
        ("CompletableFuture<T>", ["T"]),
        ("OptionalInt", []),  # Primitive specializations
        ("OptionalLong", []),
        ("OptionalDouble", []),
        ("Pair<L, R>", ["L", "R"]),
        ("Triple<A, B, C>", ["A", "B", "C"]),
        ("Map.Entry<K, V>", ["K", "V"]),
        ("Set<T>", ["T"]),
        ("Queue<T>", ["T"]),
        ("Deque<T>", ["T"]),
        ("BlockingQueue<T>", ["T"]),
        ("ConcurrentMap<K, V>", ["K", "V"]),
        ("NavigableMap<K, V>", ["K", "V"]),
        ("NavigableSet<T>", ["T"]),
        ("SortedMap<K, V>", ["K", "V"]),
        ("SortedSet<T>", ["T"]),
        ("ArrayList<T>", ["T"]),
        ("HashMap<K, V>", ["K", "V"]),
        ("HashSet<T>", ["T"]),
        ("LinkedList<T>", ["T"]),
        ("TreeMap<K, V>", ["K", "V"]),
        ("TreeSet<T>", ["T"]),
        ("WeakHashMap<K, V>", ["K", "V"]),
        ("EnumMap<K extends Enum<K>, V>", ["K", "V"]),
        ("IdentityHashMap<K, V>", ["K", "V"]),
        ("LinkedHashMap<K, V>", ["K", "V"]),
        ("LinkedHashSet<T>", ["T"]),
        ("PriorityQueue<T>", ["T"]),
        ("ArrayDeque<T>", ["T"]),
        ("CopyOnWriteArrayList<T>", ["T"]),
        ("CopyOnWriteArraySet<T>", ["T"]),
        ("ConcurrentSkipListMap<K, V>", ["K", "V"]),
        ("ConcurrentSkipListSet<T>", ["T"]),
        ("DelayQueue<T extends Delayed>", ["T"]),
        ("SynchronousQueue<E>", ["E"]),
        ("LinkedBlockingQueue<E>", ["E"]),
        ("ArrayBlockingQueue<E>", ["E"]),
        ("PriorityBlockingQueue<E>", ["E"]),
        (" ConcurrentHashMap<K, V>", ["K", "V"]),
        ("Hashtable<K, V>", ["K", "V"]),
        ("Properties", []),  # No type parameters
        ("Vector<E>", ["E"]),
        ("Stack<E>", ["E"]),
        ("Dictionary<K, V>", ["K", "V"]),
        ("BitSet", []),  # No type parameters

        # C# style generics
        ("List<T>", ["T"]),
        ("Dictionary<K, V>", ["K", "V"]),
        ("Nullable<T>", ["T"]),
        ("Task<T>", ["T"]),
        ("IEnumerable<T>", ["T"]),

        # Python-style generics
        ("List[T]", ["T"]),
        ("Dict[K, V]", ["K", "V"]),
        ("Optional[T]", ["T"]),
        ("Tuple[T, U]", ["T", "U"]),
        ("Set[T]", ["T"]),
        ("Union[T, U]", ["T", "U"]),
        ("Callable[[T], U]", ["T", "U"]),
        ("Iterator[T]", ["T"]),
        ("Generator[T, None, None]", ["T"]),
        ("Sequence[T]", ["T"]),
        ("Mapping[K, V]", ["K", "V"]),
        ("MutableSequence[T]", ["T"]),
        ("AbstractSet[T]", ["T"]),
        ("Container[T]", ["T"]),
        ("Awaitable[T]", ["T"]),
        ("AsyncIterator[T]", ["T"]),
        ("Coroutine[Any, Any, T]", ["T"]),
        ("Deque[T]", ["T"]),
        ("Counter[T]", ["T"]),
        ("DefaultDict[K, V]", ["K", "V"]),
        ("OrderedDict[K, V]", ["K", "V"]),
        ("ChainMap[K, V]", ["K", "V"]),
        ("frozenset[T]", ["T"]),
        ("NamedTuple", []),  # Concrete type
        ("TypedDict", []),   # Concrete type
        ("Protocol[T]", ["T"]),
        ("Generic[T]", ["T"]),
        ("Type[T]", ["T"]),
        ("ClassVar[T]", ["T"]),
        ("Final[T]", ["T"]),
        ("Literal['a', 'b']", []),  # Literal types with concrete values
        ("Any", []),                # Special type
        ("NoReturn", []),           # Special type
        ("Never", []),              # Special type

        # Mixed and complex cases
        ("std::vector<std::map<T, U>>", ["T", "U"]),
        ("boost::variant<T, boost::optional<U>>", ["T", "U"]),
        ("MyTemplateClass<T, U>::NestedType", ["T", "U"]),
        ("vector<std::shared_ptr<MyClass<T>>>", ["T"]),

        # Edge cases
        ("int", []),
        ("string", []),
        ("void", []),
        ("MyClass", []),
        ("vector", []),  # Missing template parameter
        ("map<", []),   # Incomplete template
        ("", []),       # Empty string

        # Real-world examples
        ("vector<std::string>", []),
        ("unordered_map<int, vector<string>>", []),
        ("optional<reference_wrapper<T>>", ["T"]),
        ("future<vector<optional<T>>>", ["T"]),
        ("expected<T, error_code>", ["T", "error_code"]),

        # Template template parameters
        ("template<typename T> class Container", ["T"]),
        ("template<class U> class Allocator", ["U"]),

        # Complex nested types
        ("vector<map<string, vector<optional<T>>>>", ["T"]),
        ("unordered_map<K, vector<map<V, W>>>", ["K", "V", "W"]),
        ("Result<vector<T>, map<U, V>>", ["T", "U", "V"]),

        # Java complex nested generics
        ("Map<String, List<T>>", ["T"]),
        ("List<Set<Map.Entry<K, V>>>", ["K", "V"]),
        ("CompletableFuture<Optional<T>>", ["T"]),
        ("Stream<Optional<Map<K, List<V>>>>", ["K", "V"]),
        ("Function<Supplier<T>, Consumer<U>>", ["T", "U"]),
        ("Map<K extends Comparable<K>, V>", ["K", "V"]),
        ("List<? extends T>", ["T"]),
        ("Map<? super K, ? extends V>", ["K", "V"]),
        ("List<T extends Number & Comparable<T>>", ["T"]),
        ("Class<? extends T>", ["T"]),
        ("Optional<? extends T>", ["T"]),
        ("Collection<? super T>", ["T"]),
        ("Comparator<? super T>", ["T"]),
        ("Map.Entry<? extends K, ? extends V>", ["K", "V"]),
        ("List<? extends List<T>>", ["T"]),
        ("Map<? extends K, ? extends List<V>>", ["K", "V"]),
        ("Stream<? extends T>", ["T"]),
        ("Iterable<? extends T>", ["T"]),
        ("Iterator<? extends T>", ["T"]),
        ("Collection<? super T>", ["T"]),
        ("List<? super T>", ["T"]),
        ("Set<? super T>", ["T"]),
        ("Queue<? super T>", ["T"]),
        ("Deque<? super T>", ["T"]),
        ("BlockingQueue<? super T>", ["T"]),
        ("Map<? super K, ? extends V>", ["K", "V"]),
        ("BiPredicate<? super T, ? extends U>", ["T", "U"]),
        ("Function<? super T, ? extends R>", ["T", "R"]),
        ("Supplier<? extends T>", ["T"]),
        ("Consumer<? super T>", ["T"]),
        ("Predicate<? super T>", ["T"]),
        ("UnaryOperator<? super T>", ["T"]),
        ("BinaryOperator<? super T>", ["T"]),

        # Function types with type parameters
        ("function<T(U)>", ["T", "U"]),
        ("std::function<T(const U&)>", ["T", "U"]),

        # Pointer and reference types
        ("T*", ["T"]),
        ("T&", ["T"]),
        ("const T*", ["T"]),
        ("std::unique_ptr<T[]>", ["T"]),
        ("shared_ptr<T&>", ["T"]),

        # Const/volatile qualified
        ("const vector<T>&", ["T"]),
        ("volatile map<K, V>*", ["K", "V"]),
        ("const std::optional<T>&", ["T"]),

        # Multiple layers of nesting
        ("vector<map<set<T>, vector<U>>>", ["T", "U"]),
        ("optional<tuple<map<K, V>, list<T>>>", ["K", "V", "T"]),

        # Test cases with known type parameters
        ("vector<T>", ["T"], ["T"]),  # With known type parameters
        ("map<K, V>", ["K", "V"], ["K", "V", "W"]),  # With extra known parameters
        ("Optional<SomeType>", [], ["T", "U"]),  # SomeType is not a known parameter
        ("Result<T, Error>", ["T", "Error"], ["T"]),  # Only T is known
        ("vector<map<K, V>>", ["K", "V"], ["K", "V", "T"]),  # Nested with known parameters
        ("list<int>", [], ["T"]),  # No type parameters despite known ones
        ("tuple<T, U, V>", ["T", "U", "V"], ["T", "U", "V", "W"]),  # All known
        ("SomeType<T>", ["T"], ["T"]),  # Simple known parameter

    ]

    logger.info("=" * 80)
    logger.info("Testing ExtractTypeParameterQuery with Compound Type Expressions")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        if len(test_case) == 2:
            type_expression, expected_params = test_case
            known_params = None
        else:
            type_expression, expected_params, known_params = test_case

        logger.info(f"\nTest {i}: {type_expression}")
        logger.info(f"Expected: {expected_params}")
        if known_params:
            logger.info(f"Known parameters: {known_params}")

        try:
            if known_params:
                actual_params = extractor(type_expression, known_params)
            else:
                actual_params = extractor(type_expression)
            logger.info(f"Got: {actual_params}")

            # Check if result matches expected
            if actual_params == expected_params:
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.warning("✗ Test %d FAILED:", i)
                logger.warning(f"  Expected: {expected_params}")
                logger.warning(f"  Got: {actual_params}")
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ExtractTypeParameterQuery Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting ExtractTypeParameterQuery tests...")

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
    test_passed = test_extract_type_parameter_query()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("ExtractTypeParameterQuery tests: %s", "PASSED" if test_passed else "FAILED")
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