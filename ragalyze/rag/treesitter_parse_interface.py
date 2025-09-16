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
    # python
    ('python', "def add(a, b): return a + b", sorted(['[FUNCDEF]add', '[IDENTIFIER]a', '[IDENTIFIER]a', '[IDENTIFIER]b', '[IDENTIFIER]b'])),
    ('python', "def call_add(): x = add(1, 2) return x", sorted(['[CALL]add', '[FUNCDEF]call_add', '[IDENTIFIER]x', '[IDENTIFIER]x'])),
    ('python', "def mul(a, b): return a * b", sorted(['[FUNCDEF]mul', '[IDENTIFIER]a', '[IDENTIFIER]a', '[IDENTIFIER]b', '[IDENTIFIER]b'])),
    ('python', "def sub(a, b): return a - b", sorted(['[FUNCDEF]sub', '[IDENTIFIER]a', '[IDENTIFIER]a', '[IDENTIFIER]b', '[IDENTIFIER]b'])),
    ('python', "def div(a, b): return a / b", sorted(['[FUNCDEF]div', '[IDENTIFIER]a', '[IDENTIFIER]a', '[IDENTIFIER]b', '[IDENTIFIER]b'])),
    ('python', "def main(): print(add(3, 4))", sorted(["[FUNCDEF]main", "[CALL]print", "[CALL]add"])),
    ('python', "class Calculator: pass", sorted(["[CLASS]Calculator"])),
    ('python', "def main(s): s.a.b.c()", sorted(['[CALL]c', '[FUNCDEF]main', '[IDENTIFIER]a', '[IDENTIFIER]b', '[IDENTIFIER]s', '[IDENTIFIER]s'])),
    ('python', "def complex_call(): obj.method1().method2(param1, param2)", sorted(['[CALL]method1', '[CALL]method2', '[FUNCDEF]complex_call', '[IDENTIFIER]obj', '[IDENTIFIER]param1', '[IDENTIFIER]param2'])),
    ('python', "def nested_call(): outer(inner(a, b), c)", sorted(['[CALL]inner', '[CALL]outer', '[FUNCDEF]nested_call', '[IDENTIFIER]a', '[IDENTIFIER]b', '[IDENTIFIER]c'])),
    ('python', "class TestClass: def method(self): self.attr.method_call()", sorted(['[CALL]method_call', '[CLASS]TestClass', '[FUNCDEF]method', '[IDENTIFIER]attr', '[IDENTIFIER]self', '[IDENTIFIER]self'])),
    # ('python', 'a = 1', sorted(['[IDENTIFIER]a'])),

    # cpp
    ('cpp', "class TestClass { int arr  [100]; public: void method(); };", sorted(["[CLASS]TestClass", '[FUNCDECL]method', '[VARDECL]arr'])),
    ('cpp', "class TestClass { public: void method() {} };", sorted(["[CLASS]TestClass", "[FUNCDEF]method"])),
    ('cpp', "int* foo(S* s) { return s->add(); }", sorted(['[CALL]add', '[FUNCDEF]foo', '[IDENTIFIER]s', '[VARDECL]s'])),
    ('cpp', "int*& foo(S* s) { return s->add(); }", sorted(['[CALL]add', '[FUNCDEF]foo', '[IDENTIFIER]s', '[VARDECL]s'])),
    ('cpp', "int const foo(S* s) { return s.add(); }", sorted(['[CALL]add', '[FUNCDEF]foo', '[IDENTIFIER]s', '[VARDECL]s'])),
    ('cpp', "int const& foo(S* s) { return add(); }", sorted(['[CALL]add', '[FUNCDEF]foo', '[VARDECL]s'])),
    # 测试C++关键字不会被误识别为函数调用
    ('cpp', "for (int i = 0; i < 10; i++) { int j; print(i); }", sorted(['[CALL]print', '[VARDECL]i', '[VARDECL]j', '[IDENTIFIER]i', '[IDENTIFIER]i', '[IDENTIFIER]i'])),
    ('cpp', "std::cout << obj->method();", sorted(['[CALL]method', '[IDENTIFIER]cout', '[IDENTIFIER]obj'])),  # C++ pointer call
    ('cpp', "std::cout << obj->method1()->method2();", sorted(['[CALL]method1', '[CALL]method2', '[IDENTIFIER]cout', '[IDENTIFIER]obj'])),  # C++ pointer call
    ('cpp', "int function_name(int[] param) {}", sorted(["[FUNCDEF]function_name", '[VARDECL]param'])),  # C++ prototype
    ('cpp', "void method();", sorted(["[FUNCDECL]method"])),  # C++ prototype
    ('cpp', "void Animal::method(int x);", sorted(["[FUNCDECL]Animal::method", "[FUNCDECL]method", "[VARDECL]x"])),  # C++ prototype
    ('cpp', "CodeTransform::operator()();", sorted(["[CALL]CodeTransform::operator()", "[CALL]operator()"])),
    ('cpp', "void CodeTransform::operator()();", sorted(["[FUNCDECL]CodeTransform::operator()", "[FUNCDECL]operator()"])),
    ('cpp', "C::f();", sorted(["[CALL]f", "[CALL]C::f"])),
    ('cpp', "C::f() {}", sorted(["[FUNCDEF]f", "[FUNCDEF]C::f"])),
    ('cpp', "int const*& f();", sorted(["[FUNCDECL]f"])),
    ('cpp', "int const* f();", sorted(["[FUNCDECL]f"])),
    ('cpp', "int const& f();", sorted(["[FUNCDECL]f"])),
    ('cpp', 'BuiltinFunctionForEVM const& builtin(BuiltinHandle const& _handle) const override;', sorted(["[FUNCDECL]builtin", '[VARDECL]_handle'])),  # C++ method with override - CORRECT SYNTAX
    ('cpp', "int& i = j;", sorted(["[VARDECL]i", "[IDENTIFIER]j"])),
    ('cpp', """
541: void CodeTransform::operator()(Break const& _break)
542: {
543: 	yulAssert(!m_context->forLoopStack.empty(), "Invalid break-statement. Requires surrounding for-loop in code generation.");
544: 	m_assembly.setSourceLocation(originLocationOf(_break));
545: 
546: 	Context::JumpInfo const& jump = m_context->forLoopStack.top().done;
547: 	m_assembly.appendJumpTo(jump.label, appendPopUntil(jump.targetStackHeight));
548: }
     """, sorted(['[CALL]appendJumpTo', '[CALL]appendPopUntil', '[CALL]empty', '[CALL]originLocationOf', '[CALL]setSourceLocation', '[CALL]top', '[CALL]yulAssert', '[FUNCDEF]CodeTransform::operator()', '[FUNCDEF]operator()', '[IDENTIFIER]_break', '[IDENTIFIER]jump', '[IDENTIFIER]jump', '[IDENTIFIER]m_assembly', '[IDENTIFIER]m_assembly', '[IDENTIFIER]m_context', '[IDENTIFIER]m_context', '[VARDECL]_break', '[VARDECL]jump'])),
    ('cpp', """\
int const& EVMDialect::builtin(BuiltinHandle const& _handle) const
{
}
        """,
        sorted(['[FUNCDEF]EVMDialect::builtin', '[FUNCDEF]builtin', '[VARDECL]_handle'])),

    # # 测试Java关键字不会被误识别为函数调用
    # ('java', "public static void main(String[] args) { System.out.println(\"Hello\"); }", sorted(["[FUNCDEF]main", "[CALL]println",])),
    # # 测试JavaScript关键字不会被误识别为函数调用
    # ('javascript', "function test() { if (true) { return; } }", sorted(["[FUNCDEF]test"])),
    # 测试更复杂的函数调用
    # 测试不同语言的函数调用
#     ('javascript', "obj.property.method();", sorted(["[CALL]method"])),  # JavaScript/Python style
#     ('javascript', "document.getElementById('test').addEventListener('click', handler);", sorted(["[CALL]getElementById", "[CALL]addEventListener"])),  # Complex JS call
#     # 测试额外的边缘情况
#     ('java', "public static int getValue();", sorted(["[FUNCDEF]getValue"])),  # Java prototype
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

def tokenize_for_bm25(code, language=None):
    """
    Tokenize code for BM25 search with [FUNCDEF] and [CALL] prefixes.
    
    Args:
        code (str): The code to tokenize
        language (str, optional): The programming language. If None, auto-detected.
        
    Returns:
        list: List of tokens with prefixes like [FUNCDEF]function_name, [CALL]function_name
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
        tokens = tokenize_for_bm25(code, lang)
        tokens = sorted(tokens)
        assert tokens == expected_tokens, f"Expected: {expected_tokens}, but got: {tokens}"
        print(f"BM25 tokens: {tokens}")
    
    print("All tests passed!!!!!")

if __name__ == "__main__":
    main()