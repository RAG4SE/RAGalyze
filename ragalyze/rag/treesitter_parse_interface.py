#!/usr/bin/env python3
from re import M


try:
    import ragalyze.rag.treesitter_parse as treesitter_parse
except ImportError:
    import treesitter_parse
"""
Tree-sitter AST Parser

This module provides Python bindings to parse code into AST nodes
for multiple programming languages including C, C++, Python, and Java.
"""


def set_debug_mode(level: int):
    """
    Set the debug mode for the C extension.

    Args:
        level (int): Debug level (0 = off, 1 = on)
    """
    treesitter_parse.set_debug_mode(level)


# Test cases
test_codes_bm25_funcs = [
    # python
    (
        "python",
        "def add(a, b): return a + b",
        sorted(
            [
                "[FUNCDEF]add",
                "[IDENTIFIER]a",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]b",
            ]
        ),
    ),
    (
        "python",
        "def call_add(): x = add(1, 2) return x",
        sorted(["[CALL]add", "[FUNCDEF]call_add", "[IDENTIFIER]x", "[IDENTIFIER]x"]),
    ),
    (
        "python",
        "def mul(a, b): return a * b",
        sorted(
            [
                "[FUNCDEF]mul",
                "[IDENTIFIER]a",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]b",
            ]
        ),
    ),
    (
        "python",
        "def sub(a, b): return a - b",
        sorted(
            [
                "[FUNCDEF]sub",
                "[IDENTIFIER]a",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]b",
            ]
        ),
    ),
    (
        "python",
        "def div(a, b): return a / b",
        sorted(
            [
                "[FUNCDEF]div",
                "[IDENTIFIER]a",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]b",
            ]
        ),
    ),
    (
        "python",
        "def main(): print(add(3, 4))",
        sorted(["[FUNCDEF]main", "[CALL]print", "[CALL]add"]),
    ),
    ("python", "class Calculator: pass", sorted(["[CLASS]Calculator"])),
    (
        "python",
        "def main(s): s.a.b.c()",
        sorted(
            [
                "[CALL]c",
                "[FUNCDEF]main",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]s",
                "[IDENTIFIER]s",
            ]
        ),
    ),
    (
        "python",
        "def complex_call(): obj.method1().method2(param1, param2)",
        sorted(
            [
                "[CALL]method1",
                "[CALL]method2",
                "[FUNCDEF]complex_call",
                "[IDENTIFIER]obj",
                "[IDENTIFIER]param1",
                "[IDENTIFIER]param2",
            ]
        ),
    ),
    (
        "python",
        "def nested_call(): outer(inner(a, b), c)",
        sorted(
            [
                "[CALL]inner",
                "[CALL]outer",
                "[FUNCDEF]nested_call",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]c",
            ]
        ),
    ),
    (
        "python",
        "class TestClass: def method(self): self.attr.method_call()",
        sorted(
            [
                "[CALL]method_call",
                "[CLASS]TestClass",
                "[FUNCDEF]method",
                "[IDENTIFIER]attr",
                "[IDENTIFIER]self",
                "[IDENTIFIER]self",
            ]
        ),
    ),
    (
        "python",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
        sorted(
            [
                "[FUNCDEF]fibonacci",
                "[IDENTIFIER]n",
                "[IDENTIFIER]n",
                "[IDENTIFIER]n",
                "[IDENTIFIER]n",
                "[IDENTIFIER]n",
                "[CALL]fibonacci",
                "[CALL]fibonacci",
            ]
        ),
    ),  # Recursive function
    (
        "python",
        "class DataProcessor:\n    def __init__(self, data):\n        self.data = data\n    def process(self):\n        return self.data.map(lambda x: x * 2)",
        sorted(
            [
                "[CLASS]DataProcessor",
                "[FUNCDEF]__init__",
                "[FUNCDEF]process",
                "[IDENTIFIER]data",
                "[IDENTIFIER]data",
                "[IDENTIFIER]data",
                "[IDENTIFIER]data",
                "[IDENTIFIER]self",
                "[IDENTIFIER]self",
                "[IDENTIFIER]self",
                "[IDENTIFIER]self",
                "[IDENTIFIER]x",
                "[IDENTIFIER]x",
                "[CALL]map",
            ]
        ),
    ),  # Class with methods
    (
        "python",
        'import math\nresult = math.sqrt(16)\nprint(f"Result: {result}")',
        sorted(
            [
                "[CALL]sqrt",
                "[CALL]print",
                "[PKG]math",
                "[IDENTIFIER]math",
                "[IDENTIFIER]result",
                "[IDENTIFIER]result",
            ]
        ),
    ),  # Module import and usage
    (
        "python",
        "def decorator(func): def wrapper(*args, **kwargs): return func(*args, **kwargs) return wrapper",
        sorted(
            [
                "[CALL]func",
                "[FUNCDEF]decorator",
                "[FUNCDEF]wrapper",
                "[IDENTIFIER]args",
                "[IDENTIFIER]args",
                "[IDENTIFIER]func",
                "[IDENTIFIER]kwargs",
                "[IDENTIFIER]kwargs",
                "[IDENTIFIER]wrapper",
            ]
        ),
    ),  # Decorator pattern
    # ('python', 'a = 1', sorted(['[IDENTIFIER]a'])),
    # Additional Python test cases for comprehensive coverage
    ("python", "x = 1", sorted(["[IDENTIFIER]x"])),  # Simple variable assignment
    (
        "python",
        "y = x + 2",
        sorted(["[IDENTIFIER]x", "[IDENTIFIER]y"]),
    ),  # Variable assignment with expression
    (
        "python",
        "for i in range(10): print(i)",
        sorted(["[CALL]print", "[CALL]range", "[IDENTIFIER]i", "[IDENTIFIER]i"]),
    ),  # For loop
    (
        "python",
        "while condition: break",
        sorted(["[IDENTIFIER]condition"]),
    ),  # While loop
    (
        "python",
        "if x > 0: print('positive')",
        sorted(["[CALL]print", "[IDENTIFIER]x"]),
    ),  # If statement
    (
        "python",
        "try:\n    risky_operation()\nexcept Exception as e:\n    pass",
        sorted(["[CALL]risky_operation", "[IDENTIFIER]e"]),
    ),  # Try-except
    (
        "python",
        "with open('file.txt') as f: content = f.read()",
        sorted(
            [
                "[CALL]open",
                "[CALL]read",
                "[IDENTIFIER]content",
                "[IDENTIFIER]f",
                "[IDENTIFIER]f",
            ]
        ),
    ),  # With statement
    ("python", "from datetime import datetime", sorted([])),  # From import
    ("python", "import os, sys", sorted(["[PKG]os", "[PKG]sys"])),  # Multiple import
    (
        "python",
        "list_comprehension = [i*2 for i in range(5)]",
        sorted(
            [
                "[CALL]range",
                "[IDENTIFIER]i",
                "[IDENTIFIER]i",
                "[IDENTIFIER]list_comprehension",
            ]
        ),
    ),  # List comprehension
    (
        "python",
        "dict_comp = {k: v for k, v in zip(keys, values)}",
        sorted(
            [
                "[CALL]zip",
                "[IDENTIFIER]dict_comp",
                "[IDENTIFIER]k",
                "[IDENTIFIER]k",
                "[IDENTIFIER]keys",
                "[IDENTIFIER]v",
                "[IDENTIFIER]v",
                "[IDENTIFIER]values",
            ]
        ),
    ),  # Dict comprehension
    (
        "python",
        "generator = (x for x in range(10))",
        sorted(
            ["[CALL]range", "[IDENTIFIER]generator", "[IDENTIFIER]x", "[IDENTIFIER]x"]
        ),
    ),  # Generator expression
    (
        "python",
        "async def async_func(): await some_coroutine()",
        sorted(["[CALL]some_coroutine", "[FUNCDEF]async_func"]),
    ),  # Async function
    (
        "python",
        "class ChildClass(ParentClass): def method(self): super().method()",
        sorted(
            [
                "[CALL]super",
                "[CALL]method",
                "[CLASS]ChildClass",
                "[FUNCDEF]method",
                "[IDENTIFIER]ParentClass",
                "[IDENTIFIER]self",
            ]
        ),
    ),  # Inheritance with super()
    (
        "python",
        "class Example:\n    @property\n    def name(self):\n        return self._name",
        sorted(
            [
                "[CALL]property",
                "[CLASS]Example",
                "[FUNCDEF]name",
                "[IDENTIFIER]_name",
                "[IDENTIFIER]self",
                "[IDENTIFIER]self",
            ]
        ),
    ),  # Property decorator
    (
        "python",
        "numbers = [1, 2, 3]; total = sum(numbers)",
        sorted(
            [
                "[CALL]sum",
                "[IDENTIFIER]numbers",
                "[IDENTIFIER]numbers",
                "[IDENTIFIER]total",
            ]
        ),
    ),  # List and built-in function
    (
        "python",
        "def default_args(a, b=10, c=None): return a + b + c",
        sorted(
            [
                "[FUNCDEF]default_args",
                "[IDENTIFIER]a",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]b",
                "[IDENTIFIER]c",
                "[IDENTIFIER]c",
            ]
        ),
    ),  # Function with default arguments
    (
        "python",
        "def type_hints(x: int, y: str) -> bool: return True",
        sorted(["[FUNCDEF]type_hints", "[IDENTIFIER]x", "[IDENTIFIER]y"]),
    ),  # Function with type hints
    (
        "python",
        "lambda_example = lambda x, y: x + y",
        sorted(
            [
                "[IDENTIFIER]lambda_example",
                "[IDENTIFIER]x",
                "[IDENTIFIER]x",
                "[IDENTIFIER]y",
                "[IDENTIFIER]y",
            ]
        ),
    ),  # Lambda assignment
    (
        "python",
        "global_var = 100; def use_global(): global global_var; global_var += 1",
        sorted(
            [
                "[FUNCDEF]use_global",
                "[IDENTIFIER]global_var",
                "[IDENTIFIER]global_var",
                "[IDENTIFIER]global_var",
            ]
        ),
    ),  # Global variable
    (
        "python",
        "assert condition, 'Error message'",
        sorted(["[IDENTIFIER]condition"]),
    ),  # Assert statement
    ("python", "raise ValueError('Invalid value')", sorted([])),  # Raise exception
    ("python", "del temp_var", sorted(["[IDENTIFIER]temp_var"])),  # Delete statement
    ("python", "pass", sorted([])),  # Pass statement
    ("python", "break", sorted([])),  # Break statement
    ("python", "continue", sorted([])),  # Continue statement
    # cpp
    (
        "cpp",
        "class TestClass { int arr  [100]; public: void method(); };",
        sorted(["[CLASS]TestClass", "[FUNCDECL]method", "[VARDECL]arr"]),
    ),
    (
        "cpp",
        "class TestClass { public: void method() {} };",
        sorted(["[CLASS]TestClass", "[FUNCDEF]method"]),
    ),
    (
        "cpp",
        "int* foo(S* s) { return s->add(); }",
        sorted(["[CALL]add", "[FUNCDEF]foo", "[IDENTIFIER]s", "[VARDECL]s"]),
    ),
    (
        "cpp",
        "int*& foo(S* s) { return s->add(); }",
        sorted(["[CALL]add", "[FUNCDEF]foo", "[IDENTIFIER]s", "[VARDECL]s"]),
    ),
    (
        "cpp",
        "int const foo(S* s) { return s.add(); }",
        sorted(["[CALL]add", "[FUNCDEF]foo", "[IDENTIFIER]s", "[VARDECL]s"]),
    ),
    (
        "cpp",
        "int const& foo(S* s) { return add(); }",
        sorted(["[CALL]add", "[FUNCDEF]foo", "[VARDECL]s"]),
    ),
    # 测试C++关键字不会被误识别为函数调用
    (
        "cpp",
        "for (int i = 0; i < 10; i++) { int j; print(i); }",
        sorted(
            [
                "[CALL]print",
                "[VARDECL]i",
                "[VARDECL]j",
                "[IDENTIFIER]i",
                "[IDENTIFIER]i",
                "[IDENTIFIER]i",
            ]
        ),
    ),
    (
        "cpp",
        "std::cout << obj->method();",
        sorted(["[CALL]method", "[IDENTIFIER]cout", "[IDENTIFIER]obj"]),
    ),  # C++ pointer call
    (
        "cpp",
        "std::cout << obj->method1()->method2();",
        sorted(
            ["[CALL]method1", "[CALL]method2", "[IDENTIFIER]cout", "[IDENTIFIER]obj"]
        ),
    ),  # C++ pointer call
    (
        "cpp",
        "int function_name(int[] param) {}",
        sorted(["[FUNCDEF]function_name", "[VARDECL]param"]),
    ),  # C++ prototype
    ("cpp", "void method();", sorted(["[FUNCDECL]method"])),  # C++ prototype
    (
        "cpp",
        "void Animal::method(int x);",
        sorted(["[FUNCDECL]Animal::method", "[FUNCDECL]method", "[VARDECL]x"]),
    ),  # C++ prototype
    (
        "cpp",
        "CodeTransform::operator()();",
        sorted(["[CALL]CodeTransform::operator()", "[CALL]operator()"]),
    ),
    (
        "cpp",
        "void CodeTransform::operator()();",
        sorted(["[FUNCDECL]CodeTransform::operator()", "[FUNCDECL]operator()"]),
    ),
    ("cpp", "C::f();", sorted(["[CALL]f", "[CALL]C::f"])),
    ("cpp", "C::f() {}", sorted(["[FUNCDEF]f", "[FUNCDEF]C::f"])),
    ("cpp", "int const*& f();", sorted(["[FUNCDECL]f"])),
    ("cpp", "int const* f();", sorted(["[FUNCDECL]f"])),
    ("cpp", "int const& f();", sorted(["[FUNCDECL]f"])),
    (
        "cpp",
        "BuiltinFunctionForEVM const& builtin(BuiltinHandle const& _handle) const override;",
        sorted(["[FUNCDECL]builtin", "[VARDECL]_handle"]),
    ),  # C++ method with override - CORRECT SYNTAX
    ("cpp", "int& i = j;", sorted(["[VARDECL]i", "[IDENTIFIER]j"])),
    (
        "cpp",
        """
541: void CodeTransform::operator()(Break const& _break)
542: {
543: 	yulAssert(!m_context->forLoopStack.empty(), "Invalid break-statement. Requires surrounding for-loop in code generation.");
544: 	m_assembly.setSourceLocation(originLocationOf(_break));
545:
546: 	Context::JumpInfo const& jump = m_context->forLoopStack.top().done;
547: 	m_assembly.appendJumpTo(jump.label, appendPopUntil(jump.targetStackHeight));
548: }
     """,
        sorted(
            [
                "[CALL]appendJumpTo",
                "[CALL]appendPopUntil",
                "[CALL]empty",
                "[CALL]originLocationOf",
                "[CALL]setSourceLocation",
                "[CALL]top",
                "[CALL]yulAssert",
                "[FUNCDEF]CodeTransform::operator()",
                "[FUNCDEF]operator()",
                "[IDENTIFIER]_break",
                "[IDENTIFIER]jump",
                "[IDENTIFIER]jump",
                "[IDENTIFIER]m_assembly",
                "[IDENTIFIER]m_assembly",
                "[IDENTIFIER]m_context",
                "[IDENTIFIER]m_context",
                "[VARDECL]_break",
                "[VARDECL]jump",
            ]
        ),
    ),
    (
        "cpp",
        "template<typename T> class Container { T value; public: Container(T v) : value(v) {} T get() { return value; } };",
        sorted(
            [
                "[CLASS]Container",
                "[FUNCDEF]Container",
                "[FUNCDEF]get",
                "[IDENTIFIER]value",
                "[VARDECL]value",
                "[VARDECL]v",
            ]
        ),
    ),  # Template class
    (
        "cpp",
        "std::vector<int> numbers = {1, 2, 3}; auto it = numbers.begin();",
        sorted(
            ["[VARDECL]numbers", "[VARDECL]it", "[IDENTIFIER]numbers", "[CALL]begin"]
        ),
    ),  # STL containers
    (
        "cpp",
        "class Singleton { private: static Singleton* instance; Singleton() {} public: static Singleton* getInstance() { if (!instance) instance = new Singleton(); return instance; } };",
        sorted(
            [
                "[CLASS]Singleton",
                "[FUNCDEF]Singleton",
                "[FUNCDEF]getInstance",
                "[IDENTIFIER]instance",
                "[IDENTIFIER]instance",
                "[IDENTIFIER]instance",
                "[OBJECTCREATION]Singleton",
                "[VARDECL]instance",
            ]
        ),
    ),  # Singleton pattern
    (
        "cpp",
        "class Base { public: virtual void method() {} }; class Derived : public Base { public: void method() override {} };",
        sorted(["[CLASS]Base", "[CLASS]Derived", "[FUNCDEF]method", "[FUNCDEF]method"]),
    ),  # Inheritance and virtual methods
    (
        "cpp",
        """\
int const& EVMDialect::builtin(BuiltinHandle const& _handle) const
{
}
        """,
        sorted(
            ["[FUNCDEF]EVMDialect::builtin", "[FUNCDEF]builtin", "[VARDECL]_handle"]
        ),
    ),
    (
        "cpp",
        "A* a = new A(1, a);",
        sorted(["[OBJECTCREATION]A", "[VARDECL]a", "[IDENTIFIER]a"]),
    ),
    # Additional C++ test cases for comprehensive coverage
    (
        "cpp",
        "namespace mylib { class MyClass { public: void func() {} }; }",
        sorted(["[CLASS]MyClass", "[FUNCDEF]func"]),
    ),  # Namespace
    (
        "cpp",
        "int main() { int x = 42; return x; }",
        sorted(["[FUNCDEF]main", "[VARDECL]x", "[IDENTIFIER]x"]),
    ),  # Basic function with variables
    (
        "cpp",
        "int sum(int a, int b) { return a + b; }",
        sorted(
            [
                "[FUNCDEF]sum",
                "[VARDECL]a",
                "[VARDECL]b",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
            ]
        ),
    ),  # Function with parameters
    (
        "cpp",
        "const int MAX_SIZE = 100;",
        sorted(["[VARDECL]MAX_SIZE"]),
    ),  # Constant declaration
    ("cpp", "static int counter = 0;", sorted(["[VARDECL]counter"])),  # Static variable
    (
        "cpp",
        "extern int global_var;",
        sorted(["[VARDECL]global_var"]),
    ),  # External variable
    (
        "cpp",
        "struct Point { int x, y; };",
        sorted(["[CLASS]Point", "[VARDECL]x", "[VARDECL]y"]),
    ),  # Struct declaration
    (
        "cpp",
        "union Data { int i; float f; };",
        sorted(["[CLASS]Data"]),
    ),  # Union declaration
    (
        "cpp",
        "enum Color { RED, GREEN, BLUE };",
        sorted(
            [
                "[IDENTIFIER]BLUE",
                "[IDENTIFIER]Color",
                "[IDENTIFIER]GREEN",
                "[IDENTIFIER]RED",
            ]
        ),
    ),  # Enum
    ("cpp", "typedef int Integer;", sorted([])),  # Typedef
    ("cpp", "using String = std::string;", sorted([])),  # Using declaration
    (
        "cpp",
        "template<class T> T max(T a, T b) { return a > b ? a : b; }",
        sorted(
            [
                "[FUNCDEF]max",
                "[VARDECL]a",
                "[VARDECL]b",
                "[IDENTIFIER]a",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[IDENTIFIER]b",
            ]
        ),
    ),  # Template function
    (
        "cpp",
        "template<typename T> class Vector { T* data; public: Vector() : data(nullptr) {} };",
        sorted(["[CLASS]Vector", "[FUNCDEF]Vector", "[VARDECL]data"]),
    ),  # Template class with constructor
    (
        "cpp",
        "auto lambda = [](int x) { return x * 2; };",
        sorted(["[VARDECL]lambda", "[IDENTIFIER]x", "[VARDECL]x"]),
    ),  # Lambda expression
    (
        "cpp",
        "void process(std::function<int(int)> func) { func(42); }",
        sorted(["[FUNCDEF]process", "[VARDECL]func", "[CALL]func"]),
    ),  # std::function
    ("cpp", "int arr[10] = {1, 2, 3};", sorted(["[VARDECL]arr"])),  # Array declaration
    (
        "cpp",
        "int* ptr = &value;",
        sorted(["[VARDECL]ptr", "[IDENTIFIER]value"]),
    ),  # Pointer declaration
    (
        "cpp",
        "int& ref = value;",
        sorted(["[VARDECL]ref", "[IDENTIFIER]value"]),
    ),  # Reference declaration
    (
        "cpp",
        "if (condition) { result = true; } else { result = false; }",
        sorted(["[IDENTIFIER]condition", "[IDENTIFIER]result", "[IDENTIFIER]result"]),
    ),  # If-else statement
    (
        "cpp",
        "while (count < 10) { count++; }",
        sorted(["[IDENTIFIER]count", "[IDENTIFIER]count"]),
    ),  # While loop
    (
        "cpp",
        "do { count++; } while (count < 10);",
        sorted(["[IDENTIFIER]count", "[IDENTIFIER]count"]),
    ),  # Do-while loop
    (
        "cpp",
        "for (int i = 0; i < n; i++) { sum += i; }",
        sorted(
            [
                "[VARDECL]i",
                "[IDENTIFIER]i",
                "[IDENTIFIER]i",
                "[IDENTIFIER]i",
                "[IDENTIFIER]n",
                "[IDENTIFIER]sum",
            ]
        ),
    ),  # For loop
    (
        "cpp",
        "for (auto item : container) { process(item); }",
        sorted(
            [
                "[VARDECL]item",
                "[IDENTIFIER]container",
                "[CALL]process",
                "[IDENTIFIER]item",
            ]
        ),
    ),  # Range-based for loop
    (
        "cpp",
        "switch (option) { case 1: break; case 2: break; default: break; }",
        sorted(["[IDENTIFIER]option"]),
    ),  # Switch statement
    (
        "cpp",
        "try { risky_operation(); } catch (const std::exception& e) { handle_error(e); }",
        sorted(
            [
                "[CALL]risky_operation",
                "[CALL]handle_error",
                "[VARDECL]e",
                "[IDENTIFIER]e",
            ]
        ),
    ),  # Try-catch
    (
        "cpp",
        "class MyClass { public: MyClass() = default; ~MyClass() = default; private: int data; };",
        sorted(
            ["[CLASS]MyClass", "[FUNCDEF]MyClass", "[FUNCDEF]~MyClass", "[VARDECL]data"]
        ),
    ),  # Constructor and destructor
    (
        "cpp",
        "class MyClass { public: explicit MyClass(int x); private: int value; };",
        sorted(["[CLASS]MyClass", "[FUNCDECL]MyClass", "[VARDECL]value"]),
    ),  # Explicit constructor
    (
        "cpp",
        "class Base { public: virtual ~Base() = 0; }; class Derived : public Base { public: ~Derived() override {} };",
        sorted(
            ["[CLASS]Base", "[CLASS]Derived", "[FUNCDEF]~Base", "[FUNCDEF]~Derived"]
        ),
    ),  # Pure virtual destructor
    (
        "cpp",
        "template<typename T> void func(T param) { } template<> void func<int>(int param) { }",
        sorted(["[FUNCDEF]func", "[FUNCDEF]func", "[VARDECL]param", "[VARDECL]param"]),
    ),  # Template specialization
    (
        "cpp",
        "constexpr int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }",
        sorted(
            [
                "[FUNCDEF]factorial",
                "[VARDECL]n",
                "[IDENTIFIER]n",
                "[IDENTIFIER]n",
                "[IDENTIFIER]n",
                "[CALL]factorial",
            ]
        ),
    ),  # Constexpr function
    (
        "cpp",
        "inline int fast_add(int a, int b) { return a + b; }",
        sorted(
            [
                "[FUNCDEF]fast_add",
                "[VARDECL]a",
                "[VARDECL]b",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
            ]
        ),
    ),  # Inline function
    # ('cpp', "void thread_func() { std::thread t([]{ std::cout << \"Hello\"; }); t.join(); }", sorted(['[FUNCDEF]thread_func', '[CALL]cout', '[IDENTIFIER]t', '[VARDECL]t', '[CALL]join'])),  # Thread with lambda
    (
        "cpp",
        "auto ptr = std::make_unique<int>(42);",
        sorted(
            ["[VARDECL]ptr", "[CALL]make_unique<int>", "[CALL]std::make_unique<int>"]
        ),
    ),  # Smart pointer
    (
        "cpp",
        "std::shared_ptr<MyClass> shared = std::make_shared<MyClass>();",
        sorted(
            [
                "[VARDECL]shared",
                "[CALL]make_shared<MyClass>",
                "[CALL]std::make_shared<MyClass>",
            ]
        ),
    ),  # Shared pointer
    (
        "cpp",
        'std::vector<std::string> names = {"Alice", "Bob"};',
        sorted(["[VARDECL]names"]),
    ),  # Nested template
    ("cpp", "decltype(x) y = x;", sorted(["[VARDECL]y", "[IDENTIFIER]x"])),  # Decltype
    # ('cpp', "if constexpr (std::is_integral_v<T>) { }", sorted([])),  # If constexpr
    (
        "cpp",
        "Concept auto func = [](auto x) requires std::integral<decltype(x)> { return x; };",
        sorted(["[VARDECL]func", "[IDENTIFIER]x", "[IDENTIFIER]x", "[VARDECL]x"]),
    ),  # Concept and requires
    ("cpp", "std::string s;", sorted(["[VARDECL]s"])),
    (
        "cpp",
        "explicit operator bool() const { return *this != invalid(); }",
        sorted(["[CALL]invalid"]),
    ),
    (
        "cpp",
        """
bool TerminationFinder::containsNonContinuingFunctionCall(Expression const& _expr)
     """,
        sorted(
            [
                "[FUNCDECL]TerminationFinder::containsNonContinuingFunctionCall",
                "[FUNCDECL]containsNonContinuingFunctionCall",
                "[VARDECL]_expr",
            ]
        ),
    ),
    (
        "cpp",
        """
bool TerminationFinder::containsNonContinuingFunctionCall(Expression const& _expr) {
     """,
        sorted(
            [
                "[FUNCDEF]TerminationFinder::containsNonContinuingFunctionCall",
                "[FUNCDEF]containsNonContinuingFunctionCall",
                "[VARDECL]_expr",
            ]
        ),
    ),
    (
        "cpp",
        """
void MovableChecker::operator()(Identifier const& _identifier)
{
    SideEffectsCollector sc;
	sc(_identifier);
	m_variableReferences.emplace(_identifier.name);
}
        """,
        sorted(
            ['[CALL]emplace', '[CALL]sc', '[FUNCDEF]MovableChecker::operator()', '[FUNCDEF]operator()', '[IDENTIFIER]_identifier', '[IDENTIFIER]_identifier', '[IDENTIFIER]m_variableReferences', '[VARDECL]_identifier', '[VARDECL]sc']
        ),
    ),
    # java
    (
        "java",
        'public static void main(String[] args) { System.out.println("Hello"); }',
        sorted(
            [
                "[CALL]println",
                "[FUNCDEF]main",
                "[IDENTIFIER]System",
                "[IDENTIFIER]out",
                "[VARDECL]args",
            ]
        ),
    ),
    (
        "java",
        "public static int getValue();",
        sorted(["[FUNCDECL]getValue"]),
    ),  # Java prototype
    (
        "java",
        "public class Calculator { }",
        sorted(["[CLASS]Calculator"]),
    ),  # Java class
    (
        "java",
        "public class Calculator { public Calculator() { } }",
        sorted(["[CLASS]Calculator", "[FUNCDEF]Calculator"]),
    ),  # Java constructor
    (
        "java",
        "private void processData() { helper.transform(); }",
        sorted(["[CALL]transform", "[FUNCDEF]processData", "[IDENTIFIER]helper"]),
    ),  # Instance method call
    (
        "java",
        "public static void main() { obj.method1().method2(); }",
        sorted(["[CALL]method1", "[CALL]method2", "[FUNCDEF]main", "[IDENTIFIER]obj"]),
    ),  # Chained calls
    (
        "java",
        "interface Drawable { void draw(); }",
        sorted(["[CLASS]Drawable", "[FUNCDECL]draw"]),
    ),  # Java interface
    (
        "java",
        "public int add(int a, int b) { return Math.max(a, b); }",
        sorted(
            [
                "[CALL]max",
                "[FUNCDEF]add",
                "[IDENTIFIER]Math",
                "[IDENTIFIER]a",
                "[IDENTIFIER]b",
                "[VARDECL]a",
                "[VARDECL]b",
            ]
        ),
    ),  # Math utility call
    (
        "java",
        'List<String> list = new ArrayList<>(); list.add("item");',
        sorted(
            [
                "[CALL]add",
                "[IDENTIFIER]list",
                "[VARDECL]list",
                "[OBJECTCREATION]ArrayList<>",
            ]
        ),
    ),  # Generic types and method call
    ("java", "String[] s;", sorted(["[VARDECL]s"])),
    ("java", "String select1(int id);", sorted(["[FUNCDECL]select1", "[VARDECL]id"])),
    (
        "java",
        "System.out.println(Arrays.toString(data));",
        sorted(
            [
                "[CALL]println",
                "[CALL]toString",
                "[IDENTIFIER]Arrays",
                "[IDENTIFIER]System",
                "[IDENTIFIER]data",
                "[IDENTIFIER]out",
            ]
        ),
    ),  # Static method calls
    (
        "java",
        "public void setUp() { super.setUp(); this.initialize(); }",
        sorted(["[FUNCDEF]setUp", "[CALL]setUp", "[CALL]initialize"]),
    ),  # java treesitter regulates that the type of `this` is this, and the type of `super` is super
    (
        "java",
        "public class Service { private Helper helper = new Helper(a); }",
        sorted(
            [
                "[CLASS]Service",
                "[IDENTIFIER]a",
                "[OBJECTCREATION]Helper",
                "[VARDECL]helper",
            ]
        ),
    ),  # Field declaration
    (
        "java",
        "public enum Color { RED, GREEN, BLUE }",
        sorted(
            [
                "[IDENTIFIER]BLUE",
                "[IDENTIFIER]Color",
                "[IDENTIFIER]GREEN",
                "[IDENTIFIER]RED",
            ]
        ),
    ),  # Enum
    (
        "java",
        "public class Generics<T> { private T data; public Generics(T data) { this.data = data; } public T getData() { return data; } }",
        sorted(
            [
                "[CLASS]Generics",
                "[FUNCDEF]Generics",
                "[FUNCDEF]getData",
                "[IDENTIFIER]data",
                "[IDENTIFIER]data",
                "[IDENTIFIER]data",
                "[IDENTIFIER]this",
                "[VARDECL]data",
                "[VARDECL]data",
            ]
        ),
    ),  # Generic class
    (
        "java",
        'public class StreamExample { public static void main(String[] args) { List<String> list = Arrays.asList("a", "b"); list.stream().filter(s -> s.startsWith("a")).forEach(System.out::println); } }',
        sorted(
            [
                "[CLASS]StreamExample",
                "[FUNCDEF]main",
                "[VARDECL]list",
                "[VARDECL]args",
                "[IDENTIFIER]list",
                "[CALL]asList",
                "[CALL]stream",
                "[CALL]filter",
                "[CALL]startsWith",
                "[CALL]forEach",
                "[CALL]println",
                "[IDENTIFIER]Arrays",
                "[IDENTIFIER]System",
                "[IDENTIFIER]out",
                "[IDENTIFIER]s",
                "[IDENTIFIER]s",
            ]
        ),
    ),  # Stream API
    (
        "java",
        'interface Runnable { void run(); } class Task implements Runnable { public void run() { System.out.println("Running"); } }',
        sorted(
            [
                "[CLASS]Runnable",
                "[CLASS]Task",
                "[FUNCDECL]run",
                "[FUNCDEF]run",
                "[CALL]println",
                "[IDENTIFIER]System",
                "[IDENTIFIER]out",
            ]
        ),
    ),  # Interface implementation
    (
        "java",
        'public class ExceptionExample { public void riskyMethod() throws IOException { try { Files.readAllBytes(Paths.get("file.txt")); } catch (IOException e) { e.printStackTrace(); } } }',
        sorted(
            [
                "[CLASS]ExceptionExample",
                "[FUNCDEF]riskyMethod",
                "[CALL]readAllBytes",
                "[CALL]get",
                "[CALL]printStackTrace",
                "[VARDECL]e",
                "[IDENTIFIER]Files",
                "[IDENTIFIER]Paths",
                "[IDENTIFIER]e",
            ]
        ),
    ),  # Exception handling
    (
        "java",
        """
void encoding1() {
	try (SqlSession sqlSession = sqlSessionFactory.openSession()) {
		EncodingMapper mapper = sqlSession.getMapper(EncodingMapper.class);
		String answer = mapper.select1();
		assertEquals("Mara\u00f1\u00f3n", answer);
	}
}
     """,
        sorted(
            [
                "[CALL]assertEquals",
                "[CALL]getMapper",
                "[CALL]openSession",
                "[CALL]select1",
                "[FUNCDEF]encoding1",
                "[IDENTIFIER]answer",
                "[IDENTIFIER]mapper",
                "[IDENTIFIER]sqlSession",
                "[IDENTIFIER]sqlSession",
                "[IDENTIFIER]sqlSessionFactory",
                "[VARDECL]answer",
                "[VARDECL]mapper",
            ]
        ),
    ),  #!WARNING: one of the [IDENTIFIER]sqlSession should be [VARDECL]sqlSession
    # MyBatis mapper test cases - comprehensive SQL-related XML test cases
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectById" resultType="com.example.model.User">SELECT id, name, age FROM user WHERE id = #{id}</select></mapper>',
        sorted(["[FUNCDEF]selectById"]),
    ),  # Basic SELECT with WHERE clause
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><insert id="insert" useGeneratedKeys="true" keyProperty="id">INSERT INTO user(name, age) VALUES(#{name}, #{age})</insert></mapper>',
        sorted(["[FUNCDEF]insert"]),
    ),  # INSERT with useGeneratedKeys
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><update id="updateAgeById">UPDATE user SET age = #{age} WHERE id = #{id}</update></mapper>',
        sorted(["[FUNCDEF]updateAgeById"]),
    ),  # UPDATE with WHERE clause
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><delete id="deleteById">DELETE FROM user WHERE id = #{id}</delete></mapper>',
        sorted(["[FUNCDEF]deleteById"]),
    ),  # DELETE with WHERE clause
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><sql id="baseColumns">id, name, age</sql><select id="selectAll" resultType="com.example.model.User">SELECT <include refid="baseColumns"/> FROM user</select></mapper>',
        sorted(["[FUNCDEF]baseColumns", "[FUNCDEF]selectAll"]),
    ),  # SQL fragment with include
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectByAgeRange" resultType="com.example.model.User">SELECT * FROM user WHERE age BETWEEN #{minAge} AND #{maxAge}</select></mapper>',
        sorted(["[FUNCDEF]selectByAgeRange"]),
    ),  # SELECT with BETWEEN
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithInClause" resultType="com.example.model.User">SELECT * FROM user WHERE id IN <foreach item="item" index="index" collection="ids" open="(" separator="," close=")">#{item}</foreach></select></mapper>',
        sorted(["[FUNCDEF]selectWithInClause"]),
    ),  # SELECT with foreach
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithIf" resultType="com.example.model.User">SELECT * FROM user<if test="name != null">WHERE name = #{name}</if></select></mapper>',
        sorted(["[FUNCDEF]selectWithIf"]),
    ),  # SELECT with if condition
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithChoose" resultType="com.example.model.User">SELECT * FROM user<choose><when test="name != null">WHERE name = #{name}</when><otherwise>WHERE age > 18</otherwise></choose></select></mapper>',
        sorted(["[FUNCDEF]selectWithChoose"]),
    ),  # SELECT with choose/when/otherwise
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithTrim" resultType="com.example.model.User">SELECT * FROM user<trim prefix="WHERE" prefixOverrides="AND |OR "><if test="name != null">AND name = #{name}</if><if test="age != null">AND age = #{age}</if></trim></select></mapper>',
        sorted(["[FUNCDEF]selectWithTrim"]),
    ),  # SELECT with trim
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><update id="updateWithSet">UPDATE user<set><if test="name != null">name = #{name},</if><if test="age != null">age = #{age},</if></set>WHERE id = #{id}</update></mapper>',
        sorted(["[FUNCDEF]updateWithSet"]),
    ),  # UPDATE with set
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><insert id="batchInsert">INSERT INTO user(name, age) VALUES<foreach item="item" index="index" collection="list" separator=",">(#{item.name}, #{item.age})</foreach></insert></mapper>',
        sorted(["[FUNCDEF]batchInsert"]),
    ),  # Batch insert
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithJoin" resultType="com.example.model.Order">SELECT o.*, u.name as userName FROM orders o JOIN user u ON o.user_id = u.id</select></mapper>',
        sorted(["[FUNCDEF]selectWithJoin"]),
    ),  # SELECT with JOIN
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithGroupBy" resultType="com.example.model.Stat">SELECT age, COUNT(*) as count FROM user GROUP BY age HAVING COUNT(*) > 1</select></mapper>',
        sorted(["[FUNCDEF]selectWithGroupBy"]),
    ),  # SELECT with GROUP BY and HAVING
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><sql id="userBase">id, name, age</sql><sql id="userContact">email, phone</sql><select id="selectWithMultipleIncludes" resultType="com.example.model.User">SELECT <include refid="userBase"/>, <include refid="userContact"/> FROM user</select></mapper>',
        sorted(
            [
                "[FUNCDEF]userBase",
                "[FUNCDEF]userContact",
                "[FUNCDEF]selectWithMultipleIncludes",
            ]
        ),
    ),  # Multiple SQL fragments
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithSubquery" resultType="com.example.model.User">SELECT * FROM user WHERE id IN (SELECT user_id FROM orders WHERE amount > 1000)</select></mapper>',
        sorted(["[FUNCDEF]selectWithSubquery"]),
    ),  # SELECT with subquery
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><select id="selectWithUnion" resultType="com.example.model.User">SELECT * FROM user WHERE age < 20 UNION SELECT * FROM user WHERE age > 60</select></mapper>',
        sorted(["[FUNCDEF]selectWithUnion"]),
    ),  # SELECT with UNION
    (
        "xml",
        '<mapper namespace="com.example.mapper.UserMapper"><insert id="insertWithSelectKey"><selectKey keyProperty="id" resultType="long" order="BEFORE">SELECT NEXTVAL(\'user_seq\')</selectKey>INSERT INTO user(id, name, age) VALUES(#{id}, #{name}, #{age})</insert></mapper>',
        sorted(["[FUNCDEF]insertWithSelectKey"]),
    ),  # INSERT with selectKey
    (
        "xml",
        """
<?xml version="1.0" encoding="UTF-8"?>
<!--

       Copyright 2009-2022 the original author or authors.

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at

          https://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.

-->
<!DOCTYPE mapper
    PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
    "https://mybatis.org/dtd/mybatis-3-mapper.dtd">

<mapper namespace="org.apache.ibatis.submitted.encoding.EncodingMapper">
  <select id="select1" resultType="string">
    select lastName from names
  </select>

  <select id="select2" resultType="string">
    select 'Marañón' from names
  </select>
</mapper>

     """,
        sorted(["[FUNCDEF]select1", "[FUNCDEF]select2"]),
    ),
    # # 测试JavaScript关键字不会被误识别为函数调用
    # ('javascript', "function test() { if (true) { return; } }", sorted(["[FUNCDEF]test"])),
    # 测试更复杂的函数调用
    # 测试不同语言的函数调用
    #     ('javascript', "obj.property.method();", sorted(["[CALL]method"])),  # JavaScript/Python style
    #     ('javascript', "document.getElementById('test').addEventListener('click', handler);", sorted(["[CALL]getElementById", "[CALL]addEventListener"])),  # Complex JS call
    #     # 测试类方法调用
    #     # Tiny OCaml example
    #     ('ocaml', 'let add a b = a + b', sorted(['a', 'a', 'add', 'b', 'b', 'let'])),
    #     # XML test case - testing tag parsing
    #     # ('xml', '<function name="test"><call>execute</call></function>', sorted(['call', 'execute', 'function', 'name', 'test'])),
    #     ('xml', """
    # <mapper namespace="org.apache.ibatis.submitted.multidb.MultiDbMapper">
    #     <select id="select1" resultType="string" parameterType="int">
    #         select
    #         name from common where id=#{value}
    #     </select>
    # </mapper>
    #      """, sorted([])),
]


def parse_code(code, language):
    """
    Parse code with the specified language using the C extension.

    Args:
        code (str): The code to parse
        language (str): The programming language ('python', 'cpp', 'java', etc.)

    Returns:
        str: A string representation of the parsed AST
    """
    try:
        # Import the C extension - try both absolute and relative imports
        return treesitter_parse.parse_code_with_treesitter(code, language)
    except ImportError:
        # Fallback implementation if C extension is not available
        raise RuntimeError("C extension treesitter_parse is not available.")


def tokenize_for_bm25(code, language, file_path):
    """
    Tokenize code for BM25 search with [FUNCDEF] and [CALL] prefixes.

    Args:
        code (str): The code to tokenize
        language (str, optional): The programming language. If None, auto-detected.

    Returns:
        tuple: (tokens, positions) where tokens is a list of tokens with prefixes
               and positions is a list of (line_number, column_number) tuples
    """
    try:
        # Try to import the C extension - try both absolute and relative imports
        return treesitter_parse.tokenize_for_bm25(code, language, file_path)
    except ImportError:
        raise RuntimeError("C extension treesitter_parse is not available.")


def main():
    """Test the Tree-sitter parser with the provided test cases."""
    print("Tree-sitter AST Parser Test")
    print("==========================")

    set_debug_mode(1)

    # # For each test code, parse with different languages
    # for i, (lang, code, expected_tokens) in enumerate(test_codes_bm25_funcs):  # Only first 5 for brevity
    #     print(f"\n--- Test Case {i + 1} ---")
    #     print(f"Code: {code}")

    #     # Try parsing with different languages
    #     result = parse_code(code, lang)
    #     print(f"Parsing result: {result}")

    # print("\n" + "="*50)
    # print("BM25 Tokenization Test")
    # print("==========================")

    for i, (lang, code, expected_tokens) in enumerate(test_codes_bm25_funcs):
        print(f"\n--- BM25 Test Case {i + 1} ---")
        print(f"Code: {code}")
        tokens = tokenize_for_bm25(code, lang, "")
        tokens = [token[0] for token in tokens]
        tokens = sorted(tokens)
        assert (
            tokens == expected_tokens
        ), f"Expected: {expected_tokens}, but got: {tokens}"
        print(f"BM25 tokens: {tokens}")

    print("All tests passed!!!!!")


if __name__ == "__main__":
    main()
