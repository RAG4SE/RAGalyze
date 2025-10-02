#!/usr/bin/env python3

"""Tests for :class:`FetchReturnTypeQuery`."""

from __future__ import annotations

from typing import Iterable, List
from unittest import TestCase

from ragalyze.agent import FetchReturnTypeQuery


class FetchReturnTypeQueryTests(TestCase):
    def setUp(self) -> None:
        self.query = FetchReturnTypeQuery(debug=False)

    def test_empty_snippet_returns_none(self) -> None:
        self.assertIsNone(self.query("   \n  "))

    def test_cpp_simple_function(self) -> None:
        snippet = "int add(int a, int b);"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "int")

    def test_cpp_method_with_qualifiers(self) -> None:
        snippet = "const std::string& MyClass::name() const { return value_; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "const std::string&")

    def test_java_static_method(self) -> None:
        snippet = "public static Optional<String> find(String key) { /* ... */ }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "Optional<String>")

    def test_python_with_annotation(self) -> None:
        snippet = "def transform(data: dict) -> list:\n    return list(data.values())"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "list")

    def test_python_constructor_returns_none(self) -> None:
        snippet = "class Example:\n    def __init__(self, value):\n        self.value = value"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "None")

    def test_solidity_function(self) -> None:
        snippet = "function totalSupply() external view returns (uint256) { return supply; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")


    # More C++ test cases
    def test_cpp_template_function(self) -> None:
        snippet = "template<typename T> T max(T a, T b) { return (a > b) ? a : b; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "T")

    def test_cpp_function_pointer_return(self) -> None:
        snippet = "int (*get_comparator())(const void*, const void*) { return compare_func; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "int (*)(const void*, const void*)")

    def test_cpp_reference_return(self) -> None:
        snippet = "std::vector<int>& get_data() { return data_vector; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "std::vector<int>&")

    def test_cpp_pointer_return(self) -> None:
        snippet = "int* create_array(int size) { return new int[size]; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "int*")

    def test_cpp_void_return(self) -> None:
        snippet = "void clear_data() { data_vector.clear(); }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "void")

    def test_cpp_auto_return_type(self) -> None:
        snippet = "auto calculate() -> decltype(result) { return result; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "decltype(result)")

    def test_cpp_lambda_return(self) -> None:
        snippet = "auto get_lambda() { return [](int x) { return x * 2; }; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "auto")

    def test_cpp_namespace_return(self) -> None:
        snippet = "mylib::utils::Result get_result() { return mylib::utils::Result::SUCCESS; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "mylib::utils::Result")

    def test_cpp_constexpr_function(self) -> None:
        snippet = "constexpr int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "constexpr int")

    def test_cpp_noexcept_function(self) -> None:
        snippet = "int divide(int a, int b) noexcept { return b != 0 ? a / b : 0; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "int")

    # More Java test cases
    def test_java_generic_return(self) -> None:
        snippet = "public <T> List<T> filter(List<T> items, Predicate<T> predicate) { /* ... */ }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "List<T>")

    def test_java_array_return(self) -> None:
        snippet = "public String[] split(String text, String delimiter) { return text.split(delimiter); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "String[]")

    def test_java_primitive_array_return(self) -> None:
        snippet = "public int[] getNumbers() { return new int[]{1, 2, 3}; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "int[]")

    def test_java_wildcard_generic_return(self) -> None:
        snippet = "public List<?> getUnknownList() { return unknownItems; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "List<?>")

    def test_java_bounded_generic_return(self) -> None:
        snippet = "public <T extends Number> T process(T number) { return number; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "T")

    def test_java_stream_return(self) -> None:
        snippet = "public Stream<String> streamLines() { return lines.stream(); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "Stream<String>")

    def test_java_completable_future_return(self) -> None:
        snippet = "public CompletableFuture<Integer> calculateAsync() { /* ... */ }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "CompletableFuture<Integer>")

    def test_java_void_return(self) -> None:
        snippet = "public void clearCache() { cache.clear(); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "void")

    def test_java_boolean_return(self) -> None:
        snippet = "public boolean isValid() { return valid; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "boolean")

    # More Python test cases
    def test_python_no_annotation_implicit_return(self) -> None:
        snippet = "def add_numbers(a, b):\n    return a + b"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    def test_python_no_annotation_implicit_none(self) -> None:
        snippet = "def log_message(message):\n    print(message)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "None")

    def test_python_union_types(self) -> None:
        snippet = "def get_data() -> Union[str, int]: return random.choice(['hello', 42])"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Union[str, int]")

    def test_python_optional_type(self) -> None:
        snippet = "def find_user(user_id: int) -> Optional[User]: return db.query(User).get(user_id)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Optional[User]")

    def test_python_generator_return(self) -> None:
        snippet = "def fibonacci(n: int) -> Iterator[int]: a, b = 0, 1; yield from itertools.islice(lambda: a, b, n)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Iterator[int]")

    def test_python_async_generator(self) -> None:
        snippet = "async def stream_data() -> AsyncIterator[DataChunk]: while True: yield await get_chunk()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "AsyncIterator[DataChunk]")

    def test_python_coroutine_return(self) -> None:
        snippet = "async def fetch_data(url: str) -> Dict[str, Any]: return await http_get(url)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Dict[str, Any]")

    def test_python_nested_generics(self) -> None:
        snippet = "def process_items() -> List[Dict[str, Union[int, str]]]: return [{'id': 1, 'name': 'test'}]"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "List[Dict[str, Union[int, str]]]")

    def test_python_literal_types(self) -> None:
        snippet = "def get_status() -> Literal['pending', 'completed', 'failed']: return current_status"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Literal['pending', 'completed', 'failed']")

    def test_python_callable_return(self) -> None:
        snippet = "def get_processor() -> Callable[[str], bool]: return lambda x: x.isupper()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Callable[[str], bool]")

    def test_python_tuple_return(self) -> None:
        snippet = "def get_coordinates() -> Tuple[float, float]: return (x, y)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Tuple[float, float]")

    def test_python_any_return(self) -> None:
        snippet = "def deserialize(data: bytes) -> Any: return pickle.loads(data)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Any")

    def test_python_type_variable(self) -> None:
        snippet = "T = TypeVar('T'); def identity(x: T) -> T: return x"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "T")

    def test_python_protocol_return(self) -> None:
        snippet = "def get_serializer() -> Serializer: return JSONSerializer()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Serializer")

    def test_python_class_method_return(self) -> None:
        snippet = "@classmethod\ndef create_instance(cls) -> 'MyClass': return cls()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "'MyClass'")

    def test_python_property_return(self) -> None:
        snippet = "@property\ndef value(self) -> int: return self._value"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    def test_python_context_manager_return(self) -> None:
        snippet = "def open_file(path: str) -> ContextManager[TextIOWrapper]: return open(path, 'r')"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "ContextManager[TextIOWrapper]")

    def test_python_decorated_function_return(self) -> None:
        snippet = "@lru_cache(maxsize=128)\ndef expensive_computation(x: int) -> int: return x ** 2"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    # Edge cases
    def test_empty_function(self) -> None:
        snippet = "def empty(): pass"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "None")

    def test_constructor_return(self) -> None:
        snippet = "class Example:\n    def __init__(self):\n        self.value = 42"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "None")

    def test_destructor_return(self) -> None:
        snippet = "class Example:\n    def __del__(self):\n        cleanup()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "None")

    def test_lambda_function(self) -> None:
        snippet = "lambda x, y: x + y"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "auto")

    def test_function_with_multiple_returns(self) -> None:
        snippet = "def get_value(flag):\n    if flag:\n        return 42\n    else:\n        return \"hello\""
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int | str")

    def test_nested_function(self) -> None:
        snippet = "def outer():\n    def inner():\n        return 42\n    return inner()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    def test_recursive_function(self) -> None:
        snippet = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    def test_function_with_default_args(self) -> None:
        snippet = "def greet(name=\"World\"):\n    return f\"Hello, {name}\""
        result = self.query(snippet, language="python")
        self.assertEqual(result, "str")

    def test_function_with_varargs(self) -> None:
        snippet = "def sum_all(*args):\n    return sum(args)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    def test_function_with_kwargs(self) -> None:
        snippet = "def print_info(**kwargs):\n    return str(kwargs)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "str")

    def test_function_with_annotations_and_defaults(self) -> None:
        snippet = "def connect(host: str, port: int = 8080) -> bool:\n    return True"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "bool")

    def test_abstract_method(self) -> None:
        snippet = "from abc import ABC, abstractmethod\n\nclass Base(ABC):\n    @abstractmethod\n    def method(self) -> str:\n        pass"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "str")

    def test_static_method(self) -> None:
        snippet = "class Math:\n    @staticmethod\n    def add(a: int, b: int) -> int:\n        return a + b"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "int")

    def test_property_with_setter(self) -> None:
        snippet = "class Person:\n    @property\n    def name(self) -> str:\n        return self._name\n    @name.setter\n    def name(self, value: str):\n        self._name = value"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "str")

    # Additional C++ test cases
    def test_cpp_move_return_type(self) -> None:
        snippet = "std::vector<int>&& get_rvalue_ref() { return std::move(data); }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "std::vector<int>&&")

    def test_cpp_const_method_return(self) -> None:
        snippet = "const std::string& getName() const { return name_; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "const std::string&")

    def test_cpp_explicit_instantiation_return(self) -> None:
        snippet = "template<> std::string Parser<std::string>::parse() { return \"\"; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "std::string")

    def test_cpp_decltype_return(self) -> None:
        snippet = "auto getValue() -> decltype(value) { return value; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "decltype(value)")

    def test_cpp_function_pointer_return(self) -> None:
        snippet = "int (*get_callback())(int, double) { return callback_func; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "int (*)(int, double)")

    def test_cpp_smart_pointer_return(self) -> None:
        snippet = "std::unique_ptr<Object> create_object() { return std::make_unique<Object>(); }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "std::unique_ptr<Object>")

    def test_cpp_constexpr_return(self) -> None:
        snippet = "constexpr double calculate() const { return 3.14159; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "constexpr double")

    def test_cpp_noexcept_return(self) -> None:
        snippet = "int divide(int a, int b) noexcept { return b != 0 ? a / b : 0; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "int")

    def test_cpp_virtual_destructor_return(self) -> None:
        snippet = "virtual ~BaseClass() = default;"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "void")

    def test_cpp_operator_overload_return(self) -> None:
        snippet = "Matrix Matrix::operator+(const Matrix& other) const { /* ... */ }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "Matrix")

    def test_cpp_conversion_operator_return(self) -> None:
        snippet = "operator bool() const { return valid; }"
        result = self.query(snippet, language="cpp")
        self.assertEqual(result, "bool")

    # Additional Java test cases
    def test_java_varargs_return(self) -> None:
        snippet = "public String concat(String... parts) { return String.join(\"\", parts); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "String")

    def test_java_wildcard_return(self) -> None:
        snippet = "public List<?> getAnyList() { return new ArrayList<>(); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "List<?>")

    def test_java_array_return(self) -> None:
        snippet = "public int[] getNumbers() { return new int[]{1, 2, 3}; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "int[]")

    def test_java_generic_method_return(self) -> None:
        snippet = "public <T extends Comparable<T>> T max(T a, T b) { return a.compareTo(b) > 0 ? a : b; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "T")

    def test_java_stream_return(self) -> None:
        snippet = "public Stream<String> streamLines() { return lines.stream(); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "Stream<String>")

    def test_java_optional_return(self) -> None:
        snippet = "public Optional<User> findById(int id) { return Optional.ofNullable(users.get(id)); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "Optional<User>")

    def test_java_completable_future_return(self) -> None:
        snippet = "public CompletableFuture<Result> processAsync() { /* ... */ }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "CompletableFuture<Result>")

    def test_java_primitive_return(self) -> None:
        snippet = "public long getTimestamp() { return System.currentTimeMillis(); }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "long")

    def test_java_enum_return(self) -> None:
        snippet = "public Status getStatus() { return currentStatus; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "Status")

    def test_java_annotation_method_return(self) -> None:
        snippet = "@Override\npublic String toString() { return \"User{\" + \"name='\" + name + '\\'' + \"}\"; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "String")

    def test_java_synchronized_method_return(self) -> None:
        snippet = "public synchronized int getCounter() { return counter; }"
        result = self.query(snippet, language="java")
        self.assertEqual(result, "int")

    # Additional Python test cases
    def test_python_async_function_return(self) -> None:
        snippet = "async def fetch_data(url: str) -> Dict[str, Any]: return await http_get(url)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Dict[str, Any]")

    def test_python_generator_return(self) -> None:
        snippet = "def fibonacci(n: int) -> Iterator[int]: a, b = 0, 1; yield from itertools.islice(lambda: a, b, n)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Iterator[int]")

    def test_python_union_types_return(self) -> None:
        snippet = "def get_data() -> Union[str, int]: return random.choice(['hello', 42])"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Union[str, int]")

    def test_python_optional_return(self) -> None:
        snippet = "def find_user(user_id: int) -> Optional[User]: return db.query(User).get(user_id)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Optional[User]")

    def test_python_literal_types_return(self) -> None:
        snippet = "def get_status() -> Literal['pending', 'completed', 'failed']: return current_status"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Literal['pending', 'completed', 'failed']")

    def test_python_callable_return(self) -> None:
        snippet = "def get_processor() -> Callable[[str], bool]: return lambda x: x.isupper()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Callable[[str], bool]")

    def test_python_type_var_return(self) -> None:
        snippet = "T = TypeVar('T'); def identity(x: T) -> T: return x"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "T")

    def test_python_nested_generics_return(self) -> None:
        snippet = "def process_items() -> List[Dict[str, Union[int, str]]]: return [{'id': 1, 'name': 'test'}]"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "List[Dict[str, Union[int, str]]]")

    def test_python_protocol_return(self) -> None:
        snippet = "def get_serializer() -> Serializer: return JSONSerializer()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Serializer")

    def test_python_context_manager_return(self) -> None:
        snippet = "def open_file(path: str) -> ContextManager[TextIOWrapper]: return open(path, 'r')"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "ContextManager[TextIOWrapper]")

    def test_python_any_return(self) -> None:
        snippet = "def deserialize(data: bytes) -> Any: return pickle.loads(data)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "Any")

    def test_python_class_method_return(self) -> None:
        snippet = "@classmethod\ndef create_instance(cls) -> 'MyClass': return cls()"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "'MyClass'")

    def test_python_static_method_return(self) -> None:
        snippet = "@staticmethod\ndef from_json(json_str: str) -> 'DataObject': return json.loads(json_str)"
        result = self.query(snippet, language="python")
        self.assertEqual(result, "'DataObject'")

    # Solidity test cases
    def test_solidity_function_return(self) -> None:
        snippet = "function totalSupply() external view returns (uint256) { return supply; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")

    def test_solidity_payable_function_return(self) -> None:
        snippet = "function deposit() external payable returns (bool) { return true; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "bool")

    def test_solidity_pure_function_return(self) -> None:
        snippet = "function add(uint256 a, uint256 b) external pure returns (uint256) { return a + b; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")

    def test_solidity_view_function_return(self) -> None:
        snippet = "function balanceOf(address account) external view returns (uint256) { return balances[account]; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")

    def test_solidity_multiple_return_values(self) -> None:
        snippet = "function getUser(uint256 id) external view returns (string memory, address, bool) { return (users[id].name, users[id].wallet, users[id].active); }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "(string memory, address, bool)")

    def test_solidity_struct_return(self) -> None:
        snippet = "function getUser(uint256 id) external view returns (User memory) { return users[id]; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "User memory")

    def test_solidity_array_return(self) -> None:
        snippet = "function getUsers() external view returns (User[] memory) { return userList; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "User[] memory")

    def test_solidity_mapping_return(self) -> None:
        snippet = "function getAllBalances() external view returns (mapping(address => uint256) memory) { return balances; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "mapping(address => uint256) memory")

    def test_solidity_bytes_return(self) -> None:
        snippet = "function getHash() external pure returns (bytes32) { return keccak256(\"hello\"); }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "bytes32")

    def test_solidity_string_return(self) -> None:
        snippet = "function getName() external view returns (string memory) { return name; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "string memory")

    def test_solidity_address_return(self) -> None:
        snippet = "function getOwner() external view returns (address) { return owner; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "address")

    def test_solidity_boolean_return(self) -> None:
        snippet = "function isActive() external view returns (bool) { return active; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "bool")

    def test_solidity_enum_return(self) -> None:
        snippet = "function getStatus() external view returns (Status) { return currentStatus; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "Status")

    def test_solidity_internal_function_return(self) -> None:
        snippet = "function internalAdd(uint256 a, uint256 b) internal pure returns (uint256) { return a + b; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")

    def test_solidity_private_function_return(self) -> None:
        snippet = "function privateCalculate() private pure returns (uint256) { return 42; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")

    def test_solidity_override_function_return(self) -> None:
        snippet = "function supportsInterface(bytes4 interfaceId) external view override returns (bool) { return interfaceId == type(IERC721).interfaceId; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "bool")

    def test_solidity_virtual_function_return(self) -> None:
        snippet = "function beforeTransfer(address from, address to, uint256 amount) internal virtual returns (bool) { return true; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "bool")

    def test_solidity_modifier_return(self) -> None:
        snippet = "modifier onlyOwner() { require(msg.sender == owner); _; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "void")

    def test_solidity_constructor_return(self) -> None:
        snippet = "constructor(string memory _name, string memory _symbol) { name = _name; symbol = _symbol; }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "void")

    def test_solidity_fallback_function_return(self) -> None:
        snippet = "fallback() external payable { }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "void")

    def test_solidity_receive_function_return(self) -> None:
        snippet = "receive() external payable { }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "void")

    def test_solidity_library_function_return(self) -> None:
        snippet = "function toAsciiString(bytes32 x) internal pure returns (string memory) { bytes memory bytesString = new bytes(32); /* ... */ }"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "string memory")

    def test_solidity_interface_function_return(self) -> None:
        snippet = "function transfer(address to, uint256 amount) external returns (bool);"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "bool")

    def test_solidity_abstract_function_return(self) -> None:
        snippet = "function calculateBonus(uint256 amount) public virtual returns (uint256);"
        result = self.query(snippet, language="solidity")
        self.assertEqual(result, "uint256")


if __name__ == "__main__":
    import unittest
    unittest.main()