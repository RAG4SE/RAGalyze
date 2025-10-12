#!/usr/bin/env python3

"""Tests for :class:`AnalyzeCallExpressionPipeline`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest import TestCase

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.configs import set_global_config_value
from ragalyze.agent import AnalyzeCallExpressionPipeline


class AnalyzeCallExpressionPipelineTests(TestCase):
    """Test cases for AnalyzeCallExpressionPipeline with diverse function call types."""

    def setUp(self) -> None:
        """Set up test configuration and pipeline instance."""
        self.query = AnalyzeCallExpressionPipeline(debug=False)

        # Configure test settings
        try:
            set_global_config_value("generator.provider", "deepseek")
            set_global_config_value("generator.model", "deepseek-chat")
        except Exception:
            # Skip configuration if it fails
            pass

    def validate_common_fields(self, result, callee_name, call_expr):
        """Common validation for all test results."""
        # Check all required fields are present
        required_fields = ["call_type", "qualified_names", "search_terms", "callee_name", "call_expression"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")

        # Validate callee name and expression
        self.assertEqual(result["callee_name"], callee_name)
        self.assertEqual(result["call_expression"], call_expr)

        # Validate qualified names
        self.assertIsInstance(result["qualified_names"], list)
        self.assertIn(callee_name, result["qualified_names"])

        # Validate search terms
        self.assertIsInstance(result["search_terms"], list)
        self.assertIn(callee_name, result["search_terms"])

    def test_cpp_simple_function_call(self) -> None:
        """Test simple function call analysis in C++."""
        callee_name = "add"
        call_expr = "add(5, 10)"
        caller_body = """
        int add(int a, int b) {
            return a + b;
        }

        int main() {
            int result = add(5, 10);
            return result;
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Validate common fields
        self.validate_common_fields(result, callee_name, call_expr)

        # Validate call type
        self.assertEqual(result["call_type"], "simple_function")

        # Validate parameter count
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 2)

        # Validate return type inference
        self.assertIn("return_type", result)
        self.assertEqual(result["return_type"], "int")

        # Check for optional fields that should be absent for simple functions
        self.assertNotIn("class_name", result)
        self.assertNotIn("method_name", result)
        self.assertNotIn("template_arguments", result)
        self.assertNotIn("namespace", result)

        # Check language detection
        self.assertIn("language", result)
        self.assertEqual(result["language"], "cpp")

    def test_cpp_member_function_call(self) -> None:
        """Test member function call analysis in C++."""
        callee_name = "push_back"
        call_expr = "myVector.push_back(42)"
        caller_body = """
        #include <vector>

        class Container {
            std::vector<int> myVector;
        public:
            void addElement(int value) {
                myVector.push_back(42);
            }
        };
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Check all required fields
        required_fields = ["call_type", "qualified_names", "search_terms", "callee_name", "call_expression"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")

        # Validate call type
        self.assertEqual(result["call_type"], "member_method")

        # Validate callee name and expression
        self.assertEqual(result["callee_name"], callee_name)
        self.assertEqual(result["call_expression"], call_expr)

        # Validate qualified names - should contain vector-related names
        self.assertIsInstance(result["qualified_names"], list)
        self.assertTrue(any("vector" in name.lower() for name in result["qualified_names"]))
        self.assertIn("push_back", result["qualified_names"])

        # Validate search terms
        self.assertIsInstance(result["search_terms"], list)
        self.assertIn("push_back", result["search_terms"])

        # Check class_name and method_name fields for member methods
        self.assertIn("class_name", result)
        self.assertIn("method_name", result)
        self.assertEqual(result["method_name"], "push_back")
        self.assertIsNotNone(result["class_name"])

        # Validate parameter count
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 1)

    def test_cpp_template_function_call(self) -> None:
        """Test template function call analysis in C++."""
        callee_name = "max"
        call_expr = "std::max<int>(a, b)"
        caller_body = """
        #include <algorithm>

        template<typename T>
        T process(T a, T b) {
            return std::max<int>(a, b);
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Check all required fields
        required_fields = ["call_type", "qualified_names", "search_terms", "callee_name", "call_expression"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")

        # Validate call type - could be template_function or static_method
        self.assertIn(result["call_type"], ["template_function", "static_method"])

        # Validate callee name and expression
        self.assertEqual(result["callee_name"], callee_name)
        self.assertEqual(result["call_expression"], call_expr)

        # Validate qualified names - should contain std::max
        self.assertIsInstance(result["qualified_names"], list)
        self.assertTrue(any("std::max" in name for name in result["qualified_names"]))
        self.assertIn("max", result["qualified_names"])

        # Validate search terms
        self.assertIsInstance(result["search_terms"], list)
        self.assertIn("max", result["search_terms"])

        # Check template-specific fields
        self.assertIn("template_arguments", result)
        self.assertIsInstance(result["template_arguments"], list)
        self.assertIn("int", result["template_arguments"])

        # Check namespace for std functions
        self.assertIn("namespace", result)
        self.assertEqual(result["namespace"], "std")

        # Validate parameter count
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 2)

    def test_cpp_constructor_call(self) -> None:
        """Test constructor call analysis in C++."""
        callee_name = "MyClass"
        call_expr = "MyClass(10, \"test\")"
        caller_body = """
        class MyClass {
            int value;
            std::string name;
        public:
            MyClass(int v, const std::string& n) : value(v), name(n) {}
        };

        void create() {
            auto obj = MyClass(10, "test");
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Check all required fields
        required_fields = ["call_type", "qualified_names", "search_terms", "callee_name", "call_expression"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")

        # Validate call type
        self.assertEqual(result["call_type"], "constructor")

        # Validate callee name and expression
        self.assertEqual(result["callee_name"], callee_name)
        self.assertEqual(result["call_expression"], call_expr)

        # Validate qualified names
        self.assertIsInstance(result["qualified_names"], list)
        self.assertIn("MyClass", result["qualified_names"])

        # Validate search terms
        self.assertIsInstance(result["search_terms"], list)
        self.assertIn("MyClass", result["search_terms"])

        # Check constructor-specific fields
        self.assertIn("class_name", result)
        self.assertEqual(result["class_name"], "MyClass")
        self.assertIn("constructor_name", result)
        self.assertEqual(result["constructor_name"], "MyClass")

        # Validate parameter count
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 2)

        # Check that method_name is None for constructors
        if "method_name" in result:
            self.assertIsNone(result["method_name"])

    def test_cpp_static_method_call(self) -> None:
        """Test static method call analysis in C++."""
        callee_name = "getInstance"
        call_expr = "Singleton::getInstance()"
        caller_body = """
        class Singleton {
        private:
            static Singleton* instance;
        public:
            static Singleton* getInstance() {
                if (!instance) {
                    instance = new Singleton();
                }
                return instance;
            }
        };

        void useSingleton() {
            auto* singleton = Singleton::getInstance();
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "static_method")

    def test_java_method_call(self) -> None:
        """Test method call analysis in Java."""
        callee_name = "add"
        call_expr = "list.add(\"item\")"
        caller_body = """
        import java.util.ArrayList;
        import java.util.List;

        public class Example {
            public void process() {
                List<String> list = new ArrayList<>();
                list.add("item");
            }
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "member_method")

    def test_java_static_method_call(self) -> None:
        """Test static method call analysis in Java."""
        callee_name = "valueOf"
        call_expr = "Integer.valueOf(\"42\")"
        caller_body = """
        public class Converter {
            public int convert() {
                return Integer.valueOf("42");
            }
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "static_method")

    def test_java_generic_method_call(self) -> None:
        """Test generic method call analysis in Java."""
        callee_name = "emptyList"
        call_expr = "Collections.emptyList<String>()"
        caller_body = """
        import java.util.Collections;
        import java.util.List;

        public class GenericExample {
            public List<String> getEmpty() {
                return Collections.emptyList<String>();
            }
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertIn(result.get("call_type"), ["static_method", "template_function"])

    def test_python_function_call(self) -> None:
        """Test function call analysis in Python."""
        callee_name = "print"
        call_expr = "print(\"Hello, World!\")"
        caller_body = """
        def main():
            print("Hello, World!")

        if __name__ == "__main__":
            main()
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "simple_function")

    def test_python_method_call(self) -> None:
        """Test method call analysis in Python."""
        callee_name = "append"
        call_expr = "my_list.append(42)"
        caller_body = """
        def process_data():
            my_list = []
            my_list.append(42)
            return my_list
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "member_method")

    def test_python_static_method_call(self) -> None:
        """Test static method call analysis in Python."""
        callee_name = "from_string"
        call_expr = "MyClass.from_string(\"test\")"
        caller_body = """
        class MyClass:
            @staticmethod
            def from_string(s: str):
                return MyClass(s)

            def __init__(self, value):
                self.value = value
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "static_method")

    def test_solidity_function_call(self) -> None:
        """Test function call analysis in Solidity."""
        callee_name = "transfer"
        call_expr = "token.transfer(to, amount)"
        caller_body = """
        contract TokenContract {
            IERC20 public token;

            function withdraw(address to, uint256 amount) public {
                token.transfer(to, amount);
            }
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "member_method")

    def test_solidity_external_call(self) -> None:
        """Test external call analysis in Solidity."""
        callee_name = "withdraw"
        call_expr = "targetProtocol.withdraw(token, amount)"
        caller_body = """
        contract AttackContract {
            IProtocol public targetProtocol;

            function launchAttack(address token, uint256 amount) public {
                targetProtocol.withdraw(token, amount);
            }
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "member_method")

    def test_function_pointer_call(self) -> None:
        """Test function pointer call analysis in C++."""
        callee_name = "func_ptr"
        call_expr = "func_ptr(10, 20)"
        caller_body = """
        #include <functional>

        void process(std::function<int(int, int)> func_ptr) {
            int result = func_ptr(10, 20);
            std::cout << result << std::endl;
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertIn(result.get("call_type"), ["function_pointer", "simple_function"])

    def test_lambda_call(self) -> None:
        """Test lambda function call analysis in C++."""
        callee_name = "lambda"
        call_expr = "lambda(x)"
        caller_body = """
        #include <algorithm>
        #include <vector>

        void processVector(std::vector<int>& vec) {
            auto lambda = [](int x) { return x * 2; };
            std::transform(vec.begin(), vec.end(), vec.begin(), lambda);
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

    def test_macro_call(self) -> None:
        """Test macro call analysis in C++."""
        callee_name = "MAX"
        call_expr = "MAX(a, b)"
        caller_body = """
        #define MAX(a, b) ((a) > (b) ? (a) : (b))

        int findMax(int a, int b) {
            return MAX(a, b);
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

    def test_nested_member_call(self) -> None:
        """Test nested member function call analysis."""
        callee_name = "get"
        call_expr = "config.getSettings().getTimeout()"
        caller_body = """
        class Settings {
            int timeout;
        public:
            int getTimeout() { return timeout; }
        };

        class Config {
            Settings settings;
        public:
            Settings& getSettings() { return settings; }
        };

        void testConfig(Config& config) {
            int timeout = config.getSettings().getTimeout();
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("call_type"), "member_method")

    def test_template_with_multiple_args(self) -> None:
        """Test template function with multiple template arguments."""
        callee_name = "make_pair"
        call_expr = "std::make_pair<int, std::string>(42, \"test\")"
        caller_body = """
        #include <utility>
        #include <string>

        std::pair<int, std::string> createPair() {
            return std::make_pair<int, std::string>(42, "test");
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)
        self.assertIn(result.get("call_type"), ["template_function", "static_method"])

    def test_operator_call(self) -> None:
        """Test operator call analysis in C++."""
        callee_name = "operator[]"
        call_expr = "vec[0]"
        caller_body = """
        #include <vector>

        void processVector(std::vector<int>& vec) {
            int first = vec[0];
            vec[0] = 42;
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

    def test_chain_call(self) -> None:
        """Test chained method calls."""
        callee_name = "add"
        call_expr = "builder.add(1).add(2).add(3)"
        caller_body = """
        class Builder {
        public:
            Builder& add(int value) {
                values.push_back(value);
                return *this;
            }
        private:
            std::vector<int> values;
        };

        void createBuilder() {
            auto result = Builder().add(1).add(2).add(3);
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Validate common fields
        self.validate_common_fields(result, callee_name, call_expr)

        # Validate call type
        self.assertEqual(result["call_type"], "member_method")

        # Validate method-specific fields
        self.assertIn("class_name", result)
        self.assertIn("method_name", result)
        self.assertEqual(result["method_name"], "add")

        # Validate chain-specific fields
        self.assertIn("is_chained_call", result)
        self.assertTrue(result["is_chained_call"], "Should identify as chained call")

        # Check that return type allows chaining
        self.assertIn("return_type", result)
        self.assertTrue("Builder" in result["return_type"] or "reference" in result["return_type"].lower(),
                       "Return type should allow chaining")

        # Validate parameter count
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 1)

        # Check for chain position information
        self.assertIn("chain_position", result)
        self.assertIsInstance(result["chain_position"], int)
        self.assertGreater(result["chain_position"], 0, "Chain position should be positive")

    def test_empty_call_expression(self) -> None:
        """Test handling of empty call expression."""
        callee_name = "test"
        call_expr = ""
        caller_body = "void test() {}"

        result = self.query(callee_name, call_expr, caller_body)
        # Should handle gracefully or return None
        self.assertIsNotNone(result)  # Pipeline should handle empty input

        # Even with empty input, basic structure should be maintained
        self.assertIn("call_type", result)
        self.assertIn("callee_name", result)
        self.assertEqual(result["callee_name"], callee_name)

    def test_malformed_call_expression(self) -> None:
        """Test handling of malformed call expression."""
        callee_name = "test"
        call_expr = "test(unclosed_paren"
        caller_body = "void test() {}"

        result = self.query(callee_name, call_expr, caller_body)
        # Should handle malformed input gracefully
        self.assertIsNotNone(result)

        # Should still maintain basic structure
        self.assertIn("call_type", result)
        self.assertIn("callee_name", result)
        self.assertEqual(result["callee_name"], callee_name)
        self.assertEqual(result["call_expression"], call_expr)

        # Should have some error indication
        self.assertIn("parse_error", result)
        self.assertTrue(result["parse_error"], "Should indicate parse error for malformed input")

    def test_unknown_function_call(self) -> None:
        """Test handling of unknown/undefined function calls."""
        callee_name = "nonexistent_function"
        call_expr = "nonexistent_function(42)"
        caller_body = """
        void existing_function() {
            nonexistent_function(42); // Function doesn't exist
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Should maintain basic structure
        self.validate_common_fields(result, callee_name, call_expr)

        # Should indicate function is not found
        self.assertIn("function_found", result)
        self.assertFalse(result["function_found"], "Should indicate function not found")

        # Should still provide basic analysis
        self.assertIn("call_type", result)
        self.assertEqual(result["call_type"], "simple_function")
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 1)

    def test_complex_template_expression(self) -> None:
        """Test complex template expression with nested types."""
        callee_name = "find_if"
        call_expr = "std::find_if<std::vector<int>::iterator>(vec.begin(), vec.end(), predicate)"
        caller_body = """
        #include <algorithm>
        #include <vector>
        #include <functional>

        void findElement(std::vector<int>& vec, std::function<bool(int)> predicate) {
            auto it = std::find_if<std::vector<int>::iterator>(vec.begin(), vec.end(), predicate);
        }
        """

        result = self.query(callee_name, call_expr, caller_body)
        self.assertIsNotNone(result)

        # Check all required fields
        required_fields = ["call_type", "qualified_names", "search_terms", "callee_name", "call_expression"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")

        # Validate call type
        self.assertIn(result["call_type"], ["template_function", "static_method"])

        # Validate callee name and expression
        self.assertEqual(result["callee_name"], callee_name)
        self.assertEqual(result["call_expression"], call_expr)

        # Validate qualified names - should contain std::find_if
        self.assertIsInstance(result["qualified_names"], list)
        self.assertTrue(any("std::find_if" in name for name in result["qualified_names"]))
        self.assertIn("find_if", result["qualified_names"])

        # Validate search terms
        self.assertIsInstance(result["search_terms"], list)
        self.assertIn("find_if", result["search_terms"])

        # Check template-specific fields
        self.assertIn("template_arguments", result)
        self.assertIsInstance(result["template_arguments"], list)
        self.assertTrue(any("vector" in arg and "iterator" in arg for arg in result["template_arguments"]))

        # Check namespace for std functions
        self.assertIn("namespace", result)
        self.assertEqual(result["namespace"], "std")

        # Validate parameter count
        self.assertIn("parameter_count", result)
        self.assertEqual(result["parameter_count"], 3)

        # Check that complex template arguments are parsed correctly
        complex_template_args = [arg for arg in result["template_arguments"] if "vector" in arg and "iterator" in arg]
        self.assertTrue(len(complex_template_args) > 0, "Complex template arguments should be parsed")

        # Validate function-specific fields
        self.assertIn("is_algorithm", result)
        self.assertTrue(result["is_algorithm"], "find_if should be identified as an STL algorithm")

        # Check return type inference
        self.assertIn("return_type", result)
        self.assertTrue(any("iterator" in result["return_type"].lower() for _ in ["iterator"] if result["return_type"]),
                       "Return type should indicate iterator type")


def run_comprehensive_tests():
    """Run all tests and provide detailed output."""
    import unittest

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(AnalyzeCallExpressionPipelineTests)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)