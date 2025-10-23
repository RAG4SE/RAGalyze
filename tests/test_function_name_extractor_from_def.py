#!/usr/bin/env python3
"""
Test script for FunctionNameExtractorFromDef class.
Tests function name extraction from various programming language function definitions.
"""

import sys
import os

# Add the parent directory to the path to import ragalyze modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragalyze.agent import FunctionNameExtractorFromDef


def test_function_name_extractor():
    """Test the FunctionNameExtractorFromDef with various function definitions."""

    extractor = FunctionNameExtractorFromDef(debug=True)

    # Test cases for different programming languages
    test_cases = [
        # C++ examples
        ("std::string MyClass::calculate(int value, const std::string& name) const", "calculate"),
        ("void processData()", "processData"),
        ("static int Helper::getCount()", "getCount"),
        ("template<typename T> T GenericClass::process(T item)", "process"),

        # Java examples
        ("public List<String> processData(String input, int count) throws IOException", "processData"),
        ("private void calculateSum()", "calculateSum"),
        ("static boolean isValid(String value)", "isValid"),
        ("public <T> T getValue(Class<T> type)", "getValue"),

        # Python examples
        ("def calculate_total(items: List[int], discount: float = 0.1) -> float:", "calculate_total"),
        ("def process_data():", "process_data"),
        ("@staticmethod def helper_method(value):", "helper_method"),
        ("async def fetch_data(url: str) -> dict:", "fetch_data"),

        # TypeScript examples
        ("function processData<T>(data: T[], callback: (item: T) => boolean): T[]", "processData"),
        ("const calculateSum = (a: number, b: number): number =>", "calculateSum"),
        ("function greet(name?: string): void", "greet"),

        # JavaScript examples
        ("function calculateSum(a, b) { return a + b; }", "calculateSum"),
        ("const processData = function(items, callback) { }", "processData"),
        ("async function fetchData(url) { }", "fetchData"),

        # Go examples
        ("func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request)", "handleRequest"),
        ("func calculateSum(a, b int) int", "calculateSum"),
        ("func (c Client) Connect() error", "Connect"),

        # Rust examples
        ("impl MyStruct { fn process_data(&self, data: &[u8]) -> Result<(), Error> }", "process_data"),
        ("fn calculate_sum(a: i32, b: i32) -> i32", "calculate_sum"),
        ("impl<T> Container<T> { fn new() -> Self { }", "new"),

        # Edge cases
        ("", ""),
        ("invalid function definition", ""),
        ("just some random text", ""),
    ]

    print("=" * 80)
    print("Testing FunctionNameExtractorFromDef")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, (function_def, expected_name) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {function_def[:60]}{'...' if len(function_def) > 60 else ''}")
        print(f"Expected: '{expected_name}'")

        try:
            result = extractor(function_def)
            print(f"Got:      '{result}'")

            if result == expected_name:
                print("✅ PASS")
                passed += 1
            else:
                print("❌ FAIL")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1

        print("-" * 40)

    print(f"\n{'=' * 80}")
    print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")
    print("=" * 80)

    return failed == 0


def test_with_real_world_examples():
    """Test with more complex real-world function definitions."""

    extractor = FunctionNameExtractorFromDef(debug=False)

    real_world_cases = [
        # Complex C++ example
        ("template<typename ReturnType, typename... Args> std::function<ReturnType(Args...)> std::bind(ReturnType(*func)(Args...), Args&&... args)", "bind"),

        # Complex Java example with generics
        ("public <T extends Comparable<T>> List<T> sortAndFilter(Collection<T> items, Predicate<T> filter) throws IllegalArgumentException", "sortAndFilter"),

        # Complex Python example with decorators
        ("@classmethod @validator('email') def validate_email(cls, v):", "validate_email"),

        # Complex TypeScript example
        ("export function createMiddleware<T extends Record<string, any>>(config: MiddlewareConfig<T>): MiddlewareFunction<T>", "createMiddleware"),

        # Go with multiple return values
        ("func (db *Database) Transaction(ctx context.Context, fn func(*Tx) error) (err error)", "Transaction"),

        # Rust with lifetimes
        ("fn process_ref_data<'a, 'b>(input: &'a str, output: &'b mut String) -> Result<(), ParseError>", "process_ref_data"),
    ]

    print("\n" + "=" * 80)
    print("Testing with Real-World Complex Examples")
    print("=" * 80)

    for i, (function_def, _) in enumerate(real_world_cases, 1):
        print(f"\nComplex Test {i}:")
        print(f"Function: {function_def}")

        try:
            result = extractor(function_def)
            print(f"Extracted name: '{result}'")
            print("✅ Successfully extracted (manual verification needed)")
        except Exception as e:
            print(f"❌ ERROR: {e}")

        print("-" * 40)


if __name__ == "__main__":
    print("Starting FunctionNameExtractorFromDef tests...")

    # Run basic tests
    success = test_function_name_extractor()

    # Run real-world examples
    test_with_real_world_examples()

    # Exit with appropriate code
    sys.exit(0 if success else 1)