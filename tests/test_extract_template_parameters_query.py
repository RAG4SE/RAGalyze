#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import ExtractClassTemplateParametersQuery, ExtractFunctionTemplateParametersQuery

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_extract_class_template_parameters():
    """Test the ExtractClassTemplateParametersQuery with various class definitions from different languages."""

    query = ExtractClassTemplateParametersQuery(debug=True)

    # Test cases: (language, class_definition, expected_parameters, description)
    test_cases = [
        # C++ Template Cases
        ("C++", """
template<typename T>
class Vector {
private:
    T* data;
    size_t size;
public:
    Vector() : data(nullptr), size(0) {}
    void push_back(const T& value);
    T& operator[](size_t index);
};
""", ["T"], "Simple C++ template with typename"),

        ("C++", """
template<class T, class U>
class Pair {
private:
    T first;
    U second;
public:
    Pair(const T& t, const U& u) : first(t), second(u) {}
    T getFirst() const { return first; }
    U getSecond() const { return second; }
};
""", ["T", "U"], "C++ template with multiple class parameters"),

        ("C++", """
template<typename Key, typename Value, typename Hash = std::hash<Key>>
class HashMap {
private:
    struct Node {
        Key key;
        Value value;
        Node* next;
    };
    std::vector<std::vector<Node>> buckets;
    Hash hash_function;
public:
    HashMap(size_t bucket_count = 16) : buckets(bucket_count), hash_function() {}
    void insert(const Key& key, const Value& value);
    Value& operator[](const Key& key);
};
""", ["Key", "Value", "Hash"], "C++ template with default parameter"),

        ("C++", """
template<typename T, std::size_t N = 10>
class FixedArray {
private:
    T data[N];
public:
    FixedArray() = default;
    T& operator[](std::size_t index) { return data[index]; }
    constexpr std::size_t size() const { return N; }
};
""", ["T", "N"], "C++ template with non-type parameter"),

        ("C++", """
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
class Calculator {
public:
    T add(T a, T b) { return a + b; }
    T multiply(T a, T b) { return a * b; }
};
""", ["T"], "C++20 template with concept constraint"),

        ("C++", """
template<template<typename> class Container, typename T>
class Adapter {
private:
    Container<T> container;
public:
    void add(const T& item) { container.push_back(item); }
    T& get(size_t index) { return container[index]; }
};
""", ["Container", "T"], "C++ template template parameter"),

        ("C++", """
template<typename... Args>
class Tuple {
public:
    Tuple(Args... args) : values(args...) {}
    template<std::size_t I>
    auto& get() { return std::get<I>(values); }
private:
    std::tuple<Args...> values;
};
""", ["Args"], "C++ variadic template"),

        # Java Generic Cases
        ("Java", """
public class Box<T> {
    private T content;

    public Box() {
        this.content = null;
    }

    public void setContent(T content) {
        this.content = content;
    }

    public T getContent() {
        return content;
    }
}
""", ["T"], "Simple Java generic class"),

        ("Java", """
public class Map<K, V> {
    private java.util.List<Entry<K, V>> entries = new java.util.ArrayList<>();

    public void put(K key, V value) {
        entries.add(new Entry<>(key, value));
    }

    public V get(K key) {
        for (Entry<K, V> entry : entries) {
            if (entry.getKey().equals(key)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private static class Entry<K, V> {
        private final K key;
        private final V value;

        public Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() { return key; }
        public V getValue() { return value; }
    }
}
""", ["K", "V"], "Java generic with nested generic class"),

        ("Java", """
public class Repository<T extends Comparable<T>> {
    private java.util.List<T> items = new java.util.ArrayList<>();

    public void add(T item) {
        items.add(item);
        Collections.sort(items);
    }

    public T find(T item) {
        int index = Collections.binarySearch(items, item);
        return index >= 0 ? items.get(index) : null;
    }
}
""", ["T"], "Java generic with bounded type parameter"),

        ("Java", """
public class Pair<T, U> {
    private final T first;
    private final U second;

    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }

    public T getFirst() { return first; }
    public U getSecond() { return second; }

    public static <T, U> Pair<T, U> of(T first, U second) {
        return new Pair<>(first, second);
    }
}
""", ["T", "U"], "Java generic with static generic method"),

        ("Java", """
public class Container<T super Number> {
    private T[] items;

    @SuppressWarnings("unchecked")
    public Container(int size) {
        items = (T[]) new Object[size];
    }

    public void set(int index, T item) {
        items[index] = item;
    }

    public T get(int index) {
        return items[index];
    }
}
""", ["T"], "Java generic with lower bound wildcard"),

        # Python Generic Cases
        ("Python", """
from typing import TypeVar, Generic, List

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self.items: List[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> T:
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()

    def is_empty(self) -> bool:
        return len(self.items) == 0
""", ["T"], "Python generic with TypeVar"),

        ("Python", """
from typing import TypeVar, Generic, Dict, Optional

K = TypeVar('K')
V = TypeVar('V')

class Dictionary(Generic[K, V]):
    def __init__(self):
        self._data: Dict[K, V] = {}

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get(self, key: K) -> Optional[V]:
        return self._data.get(key)

    def remove(self, key: K) -> bool:
        return self._data.pop(key, None) is not None
""", ["K", "V"], "Python generic with multiple TypeVars"),

        ("Python", """
from typing import TypeVar, Generic, List
from numbers import Number

T = TypeVar('T', bound=Number)

class Statistics(Generic[T]):
    def __init__(self, data: List[T]):
        self.data = data

    def mean(self) -> float:
        return sum(self.data) / len(self.data)

    def sum(self) -> T:
        return sum(self.data)
""", ["T"], "Python generic with bound TypeVar"),

        ("Python", """
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')
C = TypeVar('C', bound='Comparable')

class Sorter(Generic[T]):
    def __init__(self, data: List[T]):
        self.data = data

    def sort(self, key=None):
        return sorted(self.data, key=key)

class ComparableSorter(Generic[C]):
    def __init__(self, data: List[C]):
        self.data = data

    def sort(self):
        return sorted(self.data)
""", ["T", "C"], "Python generic with Protocol bound"),

        # C# Generic Cases
        ("C#", """
using System.Collections.Generic;

public class Container<T>
{
    private T[] items;
    private int count;

    public Container(int capacity)
    {
        items = new T[capacity];
        count = 0;
    }

    public void Add(T item)
    {
        if (count < items.Length)
        {
            items[count++] = item;
        }
    }

    public T Get(int index)
    {
        if (index < 0 || index >= count)
        {
            throw new IndexOutOfRangeException();
        }
        return items[index];
    }
}
""", ["T"], "Simple C# generic class"),

        ("C#", """
using System.Collections.Generic;

public class Dictionary<TKey, TValue>
{
    private struct Entry
    {
        public TKey Key;
        public TValue Value;
        public int Next;
    }

    private Entry[] entries;
    private int[] buckets;

    public Dictionary()
    {
        entries = new Entry[16];
        buckets = new int[16];
        for (int i = 0; i < buckets.Length; i++)
        {
            buckets[i] = -1;
        }
    }

    public void Add(TKey key, TValue value)
    {
        // Implementation omitted
    }

    public bool TryGetValue(TKey key, out TValue value)
    {
        // Implementation omitted
        value = default(TValue);
        return false;
    }
}
""", ["TKey", "TValue"], "C# generic with multiple parameters"),

        ("C#", """
using System;

public class Repository<T> where T : IComparable<T>
{
    private List<T> items = new List<T>();

    public void Add(T item)
    {
        items.Add(item);
        items.Sort();
    }

    public T Find(T item)
    {
        int index = items.BinarySearch(item);
        return index >= 0 ? items[index] : default(T);
    }

    public IEnumerable<T> GetAll()
    {
        return items;
    }
}
""", ["T"], "C# generic with constraint"),

        ("C#", """
using System;

public class Factory<T> where T : new()
{
    public T Create()
    {
        return new T();
    }

    public T[] CreateArray(int size)
    {
        T[] array = new T[size];
        for (int i = 0; i < size; i++)
        {
            array[i] = new T();
        }
        return array;
    }
}
""", ["T"], "C# generic with new() constraint"),

        ("C#", """
using System;
using System.Collections.Generic;

public class Nullable<T> where T : struct
{
    private readonly T value;
    private readonly bool hasValue;

    public Nullable(T value)
    {
        this.value = value;
        this.hasValue = true;
    }

    public bool HasValue => hasValue;
    public T Value => hasValue ? value : throw new InvalidOperationException();

    public static implicit operator Nullable<T>(T value)
    {
        return new Nullable<T>(value);
    }
}
""", ["T"], "C# generic with struct constraint"),

        # Edge Cases and Complex Templates
        ("C++", """
class NonTemplateClass {
private:
    int value;
public:
    NonTemplateClass() : value(0) {}
    int getValue() const { return value; }
};
""", [], "Non-template class"),

        ("C++", """
template<>
class SpecializedVector<bool> {
private:
    unsigned int* data;
    size_t size;
public:
    SpecializedVector(size_t size = 0);
    bool get(size_t index) const;
    void set(size_t index, bool value);
};
""", [], "Template specialization"),

        ("C++", """
template<typename T, template<typename> class Allocator = std::allocator>
class CustomVector {
private:
    Allocator<T> allocator;
    T* data;
    size_t size;
    size_t capacity;
public:
    CustomVector() : data(nullptr), size(0), capacity(0) {}
    void push_back(const T& value);
    T& operator[](size_t index);
};
""", ["T", "Allocator"], "Complex template with template template parameter"),

        ("Java", """
public class RawTypeContainer {
    private Object[] items;

    public RawTypeContainer(int capacity) {
        items = new Object[capacity];
    }

    public void set(int index, Object item) {
        items[index] = item;
    }

    public Object get(int index) {
        return items[index];
    }
}
""", [], "Raw type (non-generic) class"),

        ("Java", """
public class MultipleBounds<T extends Number & Comparable<T>> {
    private T value;

    public MultipleBounds(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public int compareTo(T other) {
        return Double.compare(value.doubleValue(), other.doubleValue());
    }
}
""", ["T"], "Java generic with multiple bounds"),

        ("Python", """
from typing import Generic, TypeVar, List, Dict, Any

T = TypeVar('T')
S = TypeVar('S')

class ComplexContainer(Generic[T, S]):
    def __init__(self, items: List[T], mapping: Dict[S, T]):
        self.items = items
        self.mapping = mapping

    def add_item(self, item: T) -> None:
        self.items.append(item)

    def get_by_key(self, key: S) -> T:
        return self.mapping.get(key)

    def find_items(self, predicate: callable) -> List[T]:
        return [item for item in self.items if predicate(item)]

    def transform(self, transformer: callable) -> List[Any]:
        return [transformer(item) for item in self.items]
""", ["T", "S"], "Python generic with callable parameters"),

        # Rust-style (for future reference)
        ("Rust", """
pub struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
""", ["T"], "Rust generic struct"),

        # Solidity Cases (Solidity does not support templates/generics)
        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleStorage {
    uint256 private storedData;

    function set(uint256 x) public {
        storedData = x;
    }

    function get() public view returns (uint256) {
        return storedData;
    }
}
""", [], "Solidity contract - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Token {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }
}
""", [], "Solidity token contract - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }
}
""", [], "Solidity library - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
""", [], "Solidity interface - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

struct User {
    address wallet;
    string name;
    uint256 createdAt;
}

contract UserRegistry {
    mapping(address => User) public users;

    function registerUser(address _wallet, string memory _name) public {
        require(users[_wallet].wallet == address(0), "User already exists");
        users[_wallet] = User({
            wallet: _wallet,
            name: _name,
            createdAt: block.timestamp
        });
    }
}
""", [], "Solidity struct and contract - no template support"),
    ]

    logger.info("=" * 80)
    logger.info("Testing ExtractTemplateParametersQuery with Multi-Language Support")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (language, class_definition, expected_params, description) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {language} - {description}")
        logger.info(f"Expected parameters: {expected_params}")

        try:
            result = query(class_definition)
            logger.info(f"Got parameters: {result}")

            # Check if result matches expected parameters
            if set(result) == set(expected_params) and len(result) == len(expected_params):
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.warning("✗ Test %d FAILED:", i)
                logger.warning(f"  Expected: {expected_params}")
                logger.warning(f"  Got: {result}")
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ExtractTemplateParametersQuery Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def test_extract_function_template_parameters():
    """Test the ExtractFunctionTemplateParametersQuery with various function definitions from different languages."""

    query = ExtractFunctionTemplateParametersQuery(debug=True)

    # Test cases: (language, function_definition, expected_parameters, description)
    test_cases = [
        # C++ Function Template Cases
        ("C++", """
template<typename T>
void printValue(T value) {
    std::cout << value << std::endl;
}
""", ["T"], "Simple C++ function template"),

        ("C++", """
template<class T, class U>
T createObject(U input) {
    T result = static_cast<T>(input);
    return result;
}
""", ["T", "U"], "C++ function template with multiple parameters"),

        ("C++", """
template<typename Iter>
void sortContainer(Iter begin, Iter end) {
    std::sort(begin, end);
}
""", ["Iter"], "C++ function template with iterator parameter"),

        ("C++", """
template<typename Ret, typename... Args>
Ret callFunction(Args... args) {
    // Implementation omitted
    return Ret();
}
""", ["Ret", "Args"], "C++ variadic function template"),

        ("C++", """
template<template<typename> class Container, typename T>
void processContainer(Container<T>& container) {
    for (auto& item : container) {
        // Process item
    }
}
""", ["Container", "T"], "C++ function template with template template parameter"),

        ("C++", """
template<typename T = int>
T getDefault() {
    return T();
}
""", ["T"], "C++ function template with default parameter"),

        ("C++", """
void regularFunction(int param) {
    return param * 2;
}
""", [], "Non-template function"),

        # Java Generic Function Cases
        ("Java", """
public <T> void printValue(T value) {
    System.out.println(value);
}
""", ["T"], "Simple Java generic method"),

        ("Java", """
public static <T> T createInstance(Class<T> clazz) throws Exception {
    return clazz.newInstance();
}
""", ["T"], "Java static generic method"),

        ("Java", """
public <T extends Comparable<T>> T findMax(List<T> list) {
    if (list.isEmpty()) return null;
    T max = list.get(0);
    for (T item : list) {
        if (item.compareTo(max) > 0) {
            max = item;
        }
    }
    return max;
}
""", ["T"], "Java generic method with bounded type parameter"),

        ("Java", """
public <K, V> void putAll(Map<K, V> source, Map<K, V> target) {
    target.putAll(source);
}
""", ["K", "V"], "Java generic method with multiple parameters"),

        ("Java", """
public void regularMethod(String param) {
    System.out.println(param);
}
""", [], "Non-generic Java method"),

        # C# Generic Function Cases
        ("C#", """
public T Create<T>() where T : new()
{
    return new T();
}
""", ["T"], "C# generic method with constraint"),

        ("C#", """
public void Process<T, U>(T item, U other) where T : IComparable
{
    if (item.CompareTo(default(T)) > 0)
    {
        // Process item
    }
}
""", ["T", "U"], "C# generic method with multiple parameters and constraint"),

        ("C#", """
public static TResult Convert<T, TResult>(T value)
{
    // Type conversion logic
    return default(TResult);
}
""", ["T", "TResult"], "C# static generic method"),

        ("C#", """
public void RegularMethod(string param)
{
    Console.WriteLine(param);
}
""", [], "Non-generic C# method"),

        # Python Generic Function Cases (type hints)
        ("Python", """
from typing import TypeVar

T = TypeVar('T')

def print_value(value: T) -> None:
    print(value)
""", ["T"], "Python function with TypeVar hint"),

        ("Python", """
from typing import TypeVar, List

T = TypeVar('T')
K = TypeVar('K')

def transform(items: List[T], key_func: callable) -> List[K]:
    return [key_func(item) for item in items]
""", ["T", "K"], "Python function with multiple TypeVar hints"),

        ("Python", """
from typing import TypeVar
from numbers import Number

T = TypeVar('T', bound=Number)

def calculate_sum(values: list[T]) -> T:
    return sum(values)
""", ["T"], "Python function with bounded TypeVar"),

        ("Python", """
def regular_function(param: int) -> int:
    return param * 2
""", [], "Non-generic Python function"),

        # Complex Function Templates
        ("C++", """
template<typename Func, typename... Args>
auto call_function(Func&& func, Args&&... args) -> decltype(func(args...)) {
    return func(std::forward<Args>(args)...);
}
""", ["Func", "Args"], "C++ perfect forwarding function template"),

        ("Java", """
public static <E> void swap(List<E> list, int i, int j) {
    E temp = list.get(i);
    list.set(i, list.get(j));
    list.set(j, temp);
}
""", ["E"], "Java generic swap method"),

        ("C#", """
public static IEnumerable<TResult> Select<T, TResult>(
    this IEnumerable<T> source,
    Func<T, TResult> selector
) {
    foreach (T item in source)
    {
        yield return selector(item);
    }
}
""", ["T", "TResult"], "C# extension method with generics"),

        # Edge Cases
        ("C++", """
template<>
void specializedFunction<int>(int value) {
    // Specialized for int
}
""", [], "Template specialization (no template parameters)"),

        ("Java", """
public <T> void genericMethod(T value) {
    this.<String>helperMethod("test");
}

private <S> void helperMethod(S value) {
    System.out.println(value);
}
""", ["T"], "Java generic method calling another generic method"),

        # Solidity Function Cases (Solidity does not support template/generic functions)
        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MathOperations {
    function add(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }

    function subtract(uint256 a, uint256 b) public pure returns (uint256) {
        require(a >= b, "Underflow error");
        return a - b;
    }
}
""", [], "Solidity mathematical functions - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TokenOperations {
    function transfer(address recipient, uint256 amount) public returns (bool) {
        // Transfer logic
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        // Approval logic
        return true;
    }
}
""", [], "Solidity token functions - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library ArrayUtils {
    function sum(uint256[] memory arr) internal pure returns (uint256) {
        uint256 total = 0;
        for (uint i = 0; i < arr.length; i++) {
            total += arr[i];
        }
        return total;
    }

    function find(uint256[] memory arr, uint256 value) internal pure returns (int256) {
        for (uint i = 0; i < arr.length; i++) {
            if (arr[i] == value) {
                return int256(i);
            }
        }
        return -1;
    }
}
""", [], "Solidity library functions - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UserManagement {
    struct User {
        address wallet;
        string name;
        uint256 createdAt;
    }

    function createUser(address wallet, string memory name) public returns (User memory) {
        return User({
            wallet: wallet,
            name: name,
            createdAt: block.timestamp
        });
    }

    function updateUser(User memory user, string memory newName) public pure returns (User memory) {
        user.name = newName;
        return user;
    }
}
""", [], "Solidity functions with custom types - no template support"),

        ("Solidity", """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Events {
    event UserRegistered(address indexed user, string name, uint256 timestamp);
    event Transaction(address indexed from, address indexed to, uint256 value);

    function registerUser(address user, string memory name) public {
        emit UserRegistered(user, name, block.timestamp);
    }

    function emitTransaction(address from, address to, uint256 value) public {
        emit Transaction(from, to, value);
    }
}
""", [], "Solidity event functions - no template support"),
    ]

    logger.info("=" * 80)
    logger.info("Testing ExtractFunctionTemplateParametersQuery with Multi-Language Support")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (language, function_definition, expected_params, description) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {language} - {description}")
        logger.info(f"Expected parameters: {expected_params}")

        try:
            result = query(function_definition)
            logger.info(f"Got parameters: {result}")

            # Check if result matches expected parameters
            if set(result) == set(expected_params) and len(result) == len(expected_params):
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.warning("✗ Test %d FAILED:", i)
                logger.warning(f"  Expected: {expected_params}")
                logger.warning(f"  Got: {result}")
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ExtractFunctionTemplateParametersQuery Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting template parameter extraction tests...")

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
    class_test_passed = test_extract_class_template_parameters()
    function_test_passed = test_extract_function_template_parameters()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("ExtractClassTemplateParametersQuery tests: %s", "PASSED" if class_test_passed else "FAILED")
    logger.info("ExtractFunctionTemplateParametersQuery tests: %s", "PASSED" if function_test_passed else "FAILED")
    logger.info("=" * 80)

    if class_test_passed and function_test_passed:
        logger.info("All tests passed! ✓")
        return True
    else:
        logger.warning("Some tests failed. See details above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)