#!/usr/bin/env python3
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
    ('python', "def add(a, b): return a + b", sorted(["[FUNC]add"])),
    ('python', "def call_add(): x = add(1, 2) return x", sorted(["[FUNC]call_add", "[CALL]add"])),
    ('python', "def mul(a, b): return a * b", sorted(["[FUNC]mul"])),
    ('python', "def sub(a, b): return a - b", sorted(["[FUNC]sub"])),
    ('python', "def div(a, b): return a / b", sorted(["[FUNC]div"])),
    ('python', "def main(): print(add(3, 4))", sorted(["[FUNC]main", "[CALL]print", "[CALL]add"])),
    ('python', "class Calculator: pass", sorted(["[CLASS]Calculator"])),
    ('cpp', "int const& foo(S* s) { return s->add(); }", sorted(["[FUNC]foo", "[CALL]add"])),
    # 测试C++关键字不会被误识别为函数调用
    ('cpp', "for (int i = 0; i < 10; i++) { print(i); }", sorted(["[CALL]print"])),
    # 测试Java关键字不会被误识别为函数调用
    ('java', "public static void main(String[] args) { System.out.println(\"Hello\"); }", sorted(["[FUNC]main", "[CALL]println",])),
    # 测试JavaScript关键字不会被误识别为函数调用
    ('javascript', "function test() { if (true) { return; } }", sorted(["[FUNC]test"])),
    # 测试更复杂的函数调用
    ('python', "def complex_call(): obj.method1().method2(param1, param2)", sorted(["[FUNC]complex_call", "[CALL]method1", "[CALL]method2"])),
    ('python', "def nested_call(): outer(inner(a, b), c)", sorted(["[FUNC]nested_call", "[CALL]outer", "[CALL]inner"])),
    ('python', "class TestClass: def method(self): self.attr.method_call()", sorted(["[CLASS]TestClass", "[FUNC]method", "[CALL]method_call"])),
    # 测试不同语言的函数调用
    ('cpp', "std::cout << obj->method();", sorted(["[CALL]method"])),  # C++ pointer call
    ('cpp', "std::cout << obj->method1()->method2();", sorted(["[CALL]method1", "[CALL]method2"])),  # C++ pointer call
    ('javascript', "obj.property.method();", sorted(["[CALL]method"])),  # JavaScript/Python style
    ('javascript', "document.getElementById('test').addEventListener('click', handler);", sorted(["[CALL]getElementById", "[CALL]addEventListener"])),  # Complex JS call
    # 测试额外的边缘情况
    ('cpp', "int function_name(param) {}", sorted(["[FUNC]function_name"])),  # C++ prototype
    ('cpp', "void method();", sorted(["[FUNC]method"])),  # C++ prototype
    ('java', "public static int getValue();", sorted(["[FUNC]getValue"])),  # Java prototype
    ('cpp', "CodeTransform::operator()", sorted(["[CALL]CodeTransform::operator()", "[CALL]operator()"])),
    ('cpp', "void CodeTransform::operator()();", sorted(["[FUNC]CodeTransform::operator()"])),
    # 测试类方法调用
    # ('cpp', "std::vector<int>::push_back(value)", sorted(["[CALL]push_back"])),  # Another C++ class method call
    ('cpp', 'BuiltinFunctionForEVM const& builtin(BuiltinHandle const& _handle) const override;', sorted(["[FUNC]builtin"])),  # C++ method with override - CORRECT SYNTAX
    ('cpp', """
541: void CodeTransform::operator()(Break const& _break)
542: {
543: 	yulAssert(!m_context->forLoopStack.empty(), "Invalid break-statement. Requires surrounding for-loop in code generation.");
544: 	m_assembly.setSourceLocation(originLocationOf(_break));
545: 
546: 	Context::JumpInfo const& jump = m_context->forLoopStack.top().done;
547: 	m_assembly.appendJumpTo(jump.label, appendPopUntil(jump.targetStackHeight));
548: }
     """, sorted(['[CALL]appendJumpTo', '[CALL]appendPopUntil', '[CALL]empty', '[CALL]originLocationOf', '[CALL]setSourceLocation', '[CALL]top', '[CALL]yulAssert', '[FUNC]CodeTransform::operator()'])),
    # Tiny OCaml example
    ('ocaml', 'let add a b = a + b', sorted(['a', 'a', 'add', 'b', 'b', 'let'])),
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

def tokenize_for_bm25(code, language=None):
    """
    Tokenize code for BM25 search with [FUNC] and [CALL] prefixes.
    
    Args:
        code (str): The code to tokenize
        language (str, optional): The programming language. If None, auto-detected.
        
    Returns:
        list: List of tokens with prefixes like [FUNC]function_name, [CALL]function_name
    """
    try:
        # Try to import the C extension - try both absolute and relative imports
        return treesitter_parse.tokenize_for_bm25(code, language)
    except ImportError:
        raise RuntimeError("C extension treesitter_parse is not available.")

def main():
    """Test the Tree-sitter parser with the provided test cases."""
    print("Tree-sitter AST Parser Test")
    print("==========================")
    
    set_debug_mode(1)
    
    # For each test code, parse with different languages
    for i, (lang, code, expected_tokens) in enumerate(test_codes_bm25_funcs):  # Only first 5 for brevity
        print(f"\n--- Test Case {i + 1} ---")
        print(f"Code: {code}")
        
        # Try parsing with different languages
        result = parse_code(code, lang)
        print(f"Parsing result: {result}")

    print("\n" + "="*50)
    print("BM25 Tokenization Test")
    print("==========================")
    
    for i, (lang, code, expected_tokens) in enumerate(test_codes_bm25_funcs):
        print(f"\n--- BM25 Test Case {i + 1} ---")
        print(f"Code: {code}")
        tokens = tokenize_for_bm25(code, lang)
        tokens = sorted(tokens)
        assert tokens == expected_tokens, f"Expected: {expected_tokens}, but got: {tokens}"
        print(f"BM25 tokens: {tokens}")

if __name__ == "__main__":
    main()