#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for FetchCallExpressionsQuery functionality.
Tests various function call formats with diverse patterns and edge cases.

Stop-on-Failure Feature:
This test class includes a stop-on-failure mechanism. When enabled, if any test
fails, all subsequent tests in the class will be skipped. This is useful for
debugging when you want to focus on the first failure without being overwhelmed
by subsequent failures.

To enable/disable this feature, modify the 'stop_on_failure' class attribute
in setUpClass(). By default, it is set to True.

When enabled, failed tests will show as "FAILED" and subsequent tests will
show as "SKIPPED" with a message indicating they were skipped due to previous
failure.
"""

import sys
import os
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ragalyze.agent import FetchCallExpressionsQuery


class TestFetchCallExpressionsQuery(unittest.TestCase):
    """Test cases for FetchCallExpressionsQuery class."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.stop_on_failure = True  # Set to False to disable stop-on-failure behavior
        cls.test_failed = False

    def setUp(self):
        """Set up test fixtures before each test method."""
        if self.test_failed and self.stop_on_failure:
            self.skipTest("Skipping remaining tests due to previous failure")
        self.query = FetchCallExpressionsQuery(debug=False)

    def fail(self, msg=None):
        """Override fail method to set failure flag."""
        TestFetchCallExpressionsQuery.test_failed = True
        super().fail(msg)

    def assertEqual(self, first, second, msg=None):
        """Override assertEqual to catch failures."""
        try:
            super().assertEqual(first, second, msg)
        except AssertionError:
            TestFetchCallExpressionsQuery.test_failed = True
            raise

    def assertIn(self, member, container, msg=None):
        """Override assertIn to catch failures."""
        try:
            super().assertIn(member, container, msg)
        except AssertionError:
            TestFetchCallExpressionsQuery.test_failed = True
            raise

    def assertTrue(self, expr, msg=None):
        """Override assertTrue to catch failures."""
        try:
            super().assertTrue(expr, msg)
        except AssertionError:
            TestFetchCallExpressionsQuery.test_failed = True
            raise

    def assertGreater(self, a, b, msg=None):
        """Override assertGreater to catch failures."""
        try:
            super().assertGreater(a, b, msg)
        except AssertionError:
            TestFetchCallExpressionsQuery.test_failed = True
            raise

    def assertGreaterEqual(self, a, b, msg=None):
        """Override assertGreaterEqual to catch failures."""
        try:
            super().assertGreaterEqual(a, b, msg)
        except AssertionError:
            TestFetchCallExpressionsQuery.test_failed = True
            raise

    def assertIsInstance(self, obj, cls, msg=None):
        """Override assertIsInstance to catch failures."""
        try:
            super().assertIsInstance(obj, cls, msg)
        except AssertionError:
            TestFetchCallExpressionsQuery.test_failed = True
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Reset failure flag for next test run
        cls.test_failed = False

    def test_empty_function_body(self):
        """Test that empty function body returns empty list."""
        result = self.query("", "test_function")
        self.assertEqual(result, [])

        result = self.query("   ", "test_function")
        self.assertEqual(result, [])

    def test_intentional_failure_demo(self):
        """Demonstration of stop-on-failure behavior (comment out to disable)."""
        # Uncomment the next line to test stop-on-failure behavior
        # self.assertEqual(1, 2, "This test intentionally fails to demonstrate stop-on-failure")
        pass

    def test_basic_function_calls(self):
        """Test basic function call extraction."""
        function_body = """
        void test_function() {
            int x = func(1, 2);
            float y = calculate(x, 3.14);
            string result = format_output(x, y);
        }
        """

        result = self.query(function_body, "test_function")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "func(1, 2)", "current_function": "test_function", "function_body": function_body},
            {"call_expr": "calculate(x, 3.14)", "current_function": "test_function", "function_body": function_body},
            {"call_expr": "format_output(x, y)", "current_function": "test_function", "function_body": function_body}
        ]

        self.assertEqual(len(result), len(expected_calls))

        # Check each expected call is present
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_member_function_calls(self):
        """Test member function call extraction."""
        function_body = """
        void test_method_calls() {
            obj.method(1);
            ptr->method(2);
            data.items.clear();
            this->process();
        }
        """

        result = self.query(function_body, "test_method_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "obj.method(1)", "current_function": "test_method_calls", "function_body": function_body},
            {"call_expr": "ptr->method(2)", "current_function": "test_method_calls", "function_body": function_body},
            {"call_expr": "data.items.clear()", "current_function": "test_method_calls", "function_body": function_body},
            {"call_expr": "this->process()", "current_function": "test_method_calls", "function_body": function_body}
        ]

        self.assertEqual(len(result), len(expected_calls))

        # Check each expected call is present
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

        # Check for different member access patterns
        has_dot_access = any('.' in call["call_expr"] for call in result)
        has_arrow_access = any('->' in call["call_expr"] for call in result)
        self.assertTrue(has_dot_access, "Should find dot access calls")
        self.assertTrue(has_arrow_access, "Should find arrow access calls")

    def test_namespace_calls(self):
        """Test namespace qualified function calls."""
        function_body = """
        void test_namespace_calls() {
            std::cout << "Hello";
            std::vector<int> vec;
            vec.push_back(42);
            MyNamespace::helper_func(x);
            boost::format("Value: %1%") % value;
        }
        """

        result = self.query(function_body, "test_namespace_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "std::cout << \"Hello\"", "current_function": "test_namespace_calls", "function_body": function_body},
            {"call_expr": "std::vector<int> vec", "current_function": "test_namespace_calls", "function_body": function_body},
            {"call_expr": "vec.push_back(42)", "current_function": "test_namespace_calls", "function_body": function_body},
            {"call_expr": "MyNamespace::helper_func(x)", "current_function": "test_namespace_calls", "function_body": function_body},
            {"call_expr": "boost::format(\"Value: %1%\") % value", "current_function": "test_namespace_calls", "function_body": function_body}
        ]

        self.assertEqual(len(result), len(expected_calls))

        # Check each expected call is present
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

        # Should find namespace qualified calls
        namespace_found = any('::' in call["call_expr"] for call in result)
        self.assertTrue(namespace_found, "Should find namespace qualified calls")

    def test_operator_calls(self):
        """Test operator function calls."""
        function_body = """
        void test_operator_calls() {
            obj.operator()(x);
            stream << "data";
            array.operator[](index);
            MyString str = "test";
            str.operator+(other);
            result = a + b;
            cout << value;
        }
        """

        result = self.query(function_body, "test_operator_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "obj.operator()(x)", "current_function": "test_operator_calls", "function_body": function_body},
            {"call_expr": "stream << \"data\"", "current_function": "test_operator_calls", "function_body": function_body},
            {"call_expr": "array.operator[](index)", "current_function": "test_operator_calls", "function_body": function_body},
            {"call_expr": "str.operator+(other)", "current_function": "test_operator_calls", "function_body": function_body}
        ]

        # Should find operator calls (check if any expected calls are found)
        operator_found = any(expected in result for expected in expected_calls)
        self.assertTrue(operator_found, "Should find operator calls")

        # Should find explicit operator calls
        explicit_operators = [call for call in result if "operator" in call["call_expr"]]
        self.assertGreaterEqual(len(explicit_operators), 1, "Should find explicit operator calls")

    def test_constructor_calls(self):
        """Test constructor calls."""
        function_body = """
        void test_constructor_calls() {
            MyClass obj(1, 2);
            std::vector<int> vec{1, 2, 3};
            std::string str("hello");
            auto data = std::make_unique<Data>(value);
            std::shared_ptr<Resource> ptr(new Resource());
        }
        """

        result = self.query(function_body, "test_constructor_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "MyClass obj(1, 2)", "current_function": "test_constructor_calls", "function_body": function_body},
            {"call_expr": "std::vector<int> vec{1, 2, 3}", "current_function": "test_constructor_calls", "function_body": function_body},
            {"call_expr": "std::string str(\"hello\")", "current_function": "test_constructor_calls", "function_body": function_body},
            {"call_expr": "std::make_unique<Data>(value)", "current_function": "test_constructor_calls", "function_body": function_body},
            {"call_expr": "std::shared_ptr<Resource> ptr(new Resource())", "current_function": "test_constructor_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_template_function_calls(self):
        """Test template function calls."""
        function_body = """
        void test_template_calls() {
            auto result = func<int>(42);
            process<std::string>(value);
            data.emplace_back<std::vector<int>>(items);
            auto ptr = std::make_unique<MyClass>(arg1, arg2);
            std::sort(vec.begin(), vec.end());
        }
        """

        result = self.query(function_body, "test_template_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "func<int>(42)", "current_function": "test_template_calls", "function_body": function_body},
            {"call_expr": "process<std::string>(value)", "current_function": "test_template_calls", "function_body": function_body},
            {"call_expr": "data.emplace_back<std::vector<int>>(items)", "current_function": "test_template_calls", "function_body": function_body},
            {"call_expr": "std::make_unique<MyClass>(arg1, arg2)", "current_function": "test_template_calls", "function_body": function_body},
            {"call_expr": "std::sort(vec.begin(), vec.end())", "current_function": "test_template_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_chained_calls(self):
        """Test chained method calls."""
        function_body = """
        void test_chained_calls() {
            obj.a().b().c(1);
            auto value = parser.get_document().get_root().get_value();
            db.connect().query("SELECT * FROM table");
            result = obj.get_item(0).get_property("name").to_string();
        }
        """

        result = self.query(function_body, "test_chained_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        # The query breaks down chained calls into individual method calls
        expected_calls = [
            {"call_expr": "obj.a()", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "obj.a().b()", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "obj.a().b().c(1)", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "parser.get_document()", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "parser.get_document().get_root()", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "parser.get_document().get_root().get_value()", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "db.connect()", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "db.connect().query(\"SELECT * FROM table\")", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "obj.get_item(0)", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "obj.get_item(0).get_property(\"name\")", "current_function": "test_chained_calls", "function_body": function_body},
            {"call_expr": "obj.get_item(0).get_property(\"name\").to_string()", "current_function": "test_chained_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_function_pointers_and_callbacks(self):
        """Test function pointer and callback calls."""
        function_body = """
        void test_function_pointers() {
            void (*func_ptr)(int) = &callback;
            func_ptr(42);
            std::function<void(int)> lambda = [](int x) { return x * 2; };
            lambda(21);
            signal.connect(&Handler::on_event);
            event_handler.fire();
        }
        """

        result = self.query(function_body, "test_function_pointers")

        # Should find function pointer/callback calls
        self.assertGreaterEqual(len(result), 1)

    def test_smart_pointers_and_memory_management(self):
        """Test smart pointer and memory management calls."""
        function_body = """
        void test_memory_management() {
            auto ptr = std::make_unique<MyClass>(arg1, arg2);
            auto shared = std::make_shared<Data>(value);
            ptr->do_something();
            auto raw_ptr = ptr.get();
            ptr.reset();
            shared.reset(new Data(new_value));
            auto obj = new MyClass();
            delete obj;
        }
        """

        result = self.query(function_body, "test_memory_management")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "std::make_unique<MyClass>(arg1, arg2)", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "std::make_shared<Data>(value)", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "ptr->do_something()", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "ptr.get()", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "ptr.reset()", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "shared.reset(new Data(new_value))", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "new MyClass()", "current_function": "test_memory_management", "function_body": function_body},
            {"call_expr": "delete obj", "current_function": "test_memory_management", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_static_member_functions(self):
        """Test static member function calls."""
        function_body = """
        void test_static_calls() {
            MyClass::static_method();
            Utility::helper_func(1, 2);
            Math::calculate_distance(p1, p2);
            Logger::get_instance().log("message");
        }
        """

        result = self.query(function_body, "test_static_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "MyClass::static_method()", "current_function": "test_static_calls", "function_body": function_body},
            {"call_expr": "Utility::helper_func(1, 2)", "current_function": "test_static_calls", "function_body": function_body},
            {"call_expr": "Math::calculate_distance(p1, p2)", "current_function": "test_static_calls", "function_body": function_body},
            {"call_expr": "Logger::get_instance().log(\"message\")", "current_function": "test_static_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_virtual_function_calls(self):
        """Test virtual function calls."""
        function_body = """
        void test_virtual_calls() {
            Base* ptr = new Derived();
            ptr->virtual_method();
            ptr->override_me();
            obj.abstract_function();
            interface.implement_me();
        }
        """

        result = self.query(function_body, "test_virtual_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "ptr->virtual_method()", "current_function": "test_virtual_calls", "function_body": function_body},
            {"call_expr": "ptr->override_me()", "current_function": "test_virtual_calls", "function_body": function_body},
            {"call_expr": "obj.abstract_function()", "current_function": "test_virtual_calls", "function_body": function_body},
            {"call_expr": "interface.implement_me()", "current_function": "test_virtual_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_overloaded_functions(self):
        """Test overloaded function calls."""
        function_body = """
        void test_overloaded_calls() {
            process(42);           // process(int)
            process(3.14);         // process(double)
            process("hello");      // process(const char*)
            calculate(a, b);       // calculate(int, int)
            calculate(x, y, z);    // calculate(int, int, int)
            obj.display();         // display()
            obj.display(1);        // display(int)
        }
        """

        result = self.query(function_body, "test_overloaded_calls")

        # Should find overloaded function calls
        self.assertGreaterEqual(len(result), 2)

    def test_lambda_calls(self):
        """Test lambda function calls."""
        function_body = """
        void test_lambda_calls() {
            auto lambda = [](int x) { return x * 2; };
            auto result = lambda(21);

            std::function<int(int)> func = [](int x) { return x + 1; };
            func(10);

            auto lambda_with_capture = [value](int x) { return x + value; };
            lambda_with_capture(5);
        }
        """

        result = self.query(function_body, "test_lambda_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "lambda(21)", "current_function": "test_lambda_calls", "function_body": function_body},
            {"call_expr": "func(10)", "current_function": "test_lambda_calls", "function_body": function_body},
            {"call_expr": "lambda_with_capture(5)", "current_function": "test_lambda_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_std_library_calls(self):
        """Test standard library function calls."""
        function_body = """
        void test_std_calls() {
            std::vector<int> vec{1, 2, 3, 4, 5};
            std::sort(vec.begin(), vec.end());
            auto it = std::find(vec.begin(), vec.end(), 3);
            std::transform(vec.begin(), vec.end(), vec.begin(), [](int x) { return x * 2; });
            std::cout << "Result: " << vec.size() << std::endl;
            std::string str = "hello";
            std::replace(str.begin(), str.end(), 'l', 'L');
        }
        """

        result = self.query(function_body, "test_std_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "std::vector<int> vec{1, 2, 3, 4, 5}", "current_function": "test_std_calls", "function_body": function_body},
            {"call_expr": "std::sort(vec.begin(), vec.end())", "current_function": "test_std_calls", "function_body": function_body},
            {"call_expr": "std::find(vec.begin(), vec.end(), 3)", "current_function": "test_std_calls", "function_body": function_body},
            {"call_expr": "std::transform(vec.begin(), vec.end(), vec.begin(), [](int x) { return x * 2; })", "current_function": "test_std_calls", "function_body": function_body},
            {"call_expr": "std::cout << \"Result: \" << vec.size() << std::endl", "current_function": "test_std_calls", "function_body": function_body},
            {"call_expr": "std::string str = \"hello\"", "current_function": "test_std_calls", "function_body": function_body},
            {"call_expr": "std::replace(str.begin(), str.end(), 'l', 'L')", "current_function": "test_std_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_exception_handling_calls(self):
        """Test exception handling related calls."""
        function_body = """
        void test_exception_calls() {
            try {
                risky_operation();
                auto result = might_throw(arg);
                if (result.empty()) {
                    throw std::runtime_error("Empty result");
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                logger.error("Exception occurred: {}", e.what());
            } catch (...) {
                std::cerr << "Unknown error occurred" << std::endl;
            }
        }
        """

        result = self.query(function_body, "test_exception_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "risky_operation()", "current_function": "test_exception_calls", "function_body": function_body},
            {"call_expr": "might_throw(arg)", "current_function": "test_exception_calls", "function_body": function_body},
            {"call_expr": "result.empty()", "current_function": "test_exception_calls", "function_body": function_body},
            {"call_expr": "std::cerr << \"Error: \" << e.what()", "current_function": "test_exception_calls", "function_body": function_body},
            {"call_expr": "logger.error(\"Exception occurred: {}\", e.what())", "current_function": "test_exception_calls", "function_body": function_body},
            {"call_expr": "std::cerr << \"Unknown error occurred\"", "current_function": "test_exception_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_file_io_calls(self):
        """Test file I/O related calls."""
        function_body = """
        void test_file_calls() {
            std::ifstream file("data.txt");
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    std::cout << line << std::endl;
                }
                file.close();
            }

            std::ofstream outfile("output.txt");
            outfile << "Hello World" << std::endl;
            outfile.close();
        }
        """

        result = self.query(function_body, "test_file_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "std::ifstream file(\"data.txt\")", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "file.is_open()", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "std::getline(file, line)", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "std::cout << line << std::endl", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "file.close()", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "std::ofstream outfile(\"output.txt\")", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "outfile << \"Hello World\" << std::endl", "current_function": "test_file_calls", "function_body": function_body},
            {"call_expr": "outfile.close()", "current_function": "test_file_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_threading_calls(self):
        """Test threading and concurrency calls."""
        function_body = """
        void test_threading_calls() {
            std::thread t1(worker_function, arg1);
            std::thread t2(&Class::method, this, arg2);

            {
                std::lock_guard<std::mutex> lock(mtx);
                shared_resource++;
            }

            auto future = std::async(std::launch::async, task_func, data);
            future.wait();
            auto result = future.get();

            t1.join();
            t2.join();
        }
        """

        result = self.query(function_body, "test_threading_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "std::thread t1(worker_function, arg1)", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "std::thread t2(&Class::method, this, arg2)", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "std::lock_guard<std::mutex> lock(mtx)", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "std::async(std::launch::async, task_func, data)", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "future.wait()", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "future.get()", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "t1.join()", "current_function": "test_threading_calls", "function_body": function_body},
            {"call_expr": "t2.join()", "current_function": "test_threading_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_container_operations(self):
        """Test container operations."""
        function_body = """
        void test_container_calls() {
            std::vector<int> vec;
            vec.push_back(1);
            vec.insert(vec.begin(), 0);
            vec.erase(vec.begin());
            vec.pop_back();
            vec.clear();

            std::map<int, std::string> mapping;
            mapping.insert({1, "one"});
            mapping.erase(1);
            mapping.clear();

            auto it = vec.find(42);
            if (it != vec.end()) {
                vec.erase(it);
            }
        }
        """

        result = self.query(function_body, "test_container_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "vec.push_back(1)", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.insert(vec.begin(), 0)", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.erase(vec.begin())", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.pop_back()", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.clear()", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "mapping.insert({1, \"one\"})", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "mapping.erase(1)", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "mapping.clear()", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.find(42)", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.end()", "current_function": "test_container_calls", "function_body": function_body},
            {"call_expr": "vec.erase(it)", "current_function": "test_container_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_algorithm_calls(self):
        """Test STL algorithm calls."""
        function_body = """
        void test_algorithm_calls() {
            std::vector<int> vec{1, 2, 3, 4, 5};
            std::sort(vec.begin(), vec.end());
            std::reverse(vec.begin(), vec.end());
            std::for_each(vec.begin(), vec.end(), [](int x) { std::cout << x << " "; });
            auto count = std::count(vec.begin(), vec.end(), 3);
            bool exists = std::binary_search(vec.begin(), vec.end(), 4);
            auto min_val = std::min_element(vec.begin(), vec.end());
            auto max_val = std::max_element(vec.begin(), vec.end());
        }
        """

        result = self.query(function_body, "test_algorithm_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "std::sort(vec.begin(), vec.end())", "current_function": "test_algorithm_calls", "function_body": function_body},
            {"call_expr": "std::reverse(vec.begin(), vec.end())", "current_function": "test_algorithm_calls", "function_body": function_body},
            {"call_expr": "std::for_each(vec.begin(), vec.end(), [](int x) { std::cout << x << \" \"; })", "current_function": "test_algorithm_calls", "function_body": function_body},
            {"call_expr": "std::count(vec.begin(), vec.end(), 3)", "current_function": "test_algorithm_calls", "function_body": function_body},
            {"call_expr": "std::binary_search(vec.begin(), vec.end(), 4)", "current_function": "test_algorithm_calls", "function_body": function_body},
            {"call_expr": "std::min_element(vec.begin(), vec.end())", "current_function": "test_algorithm_calls", "function_body": function_body},
            {"call_expr": "std::max_element(vec.begin(), vec.end())", "current_function": "test_algorithm_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_macro_calls(self):
        """Test macro calls."""
        function_body = """
        void test_macro_calls() {
            MACRO(arg1, arg2);
            DEBUG_LOG("Error: %s", message);
            ASSERT(condition, "message");
            printf("Value: %d\n", value);
            MAX(a, b);
            MIN(x, y);
        }
        """

        result = self.query(function_body, "test_macro_calls")

        # Expected results based on actual FetchCallExpressionsQuery output
        expected_calls = [
            {"call_expr": "MACRO(arg1, arg2)", "current_function": "test_macro_calls", "function_body": function_body},
            {"call_expr": "DEBUG_LOG(\"Error: %s\", message)", "current_function": "test_macro_calls", "function_body": function_body},
            {"call_expr": "ASSERT(condition, \"message\")", "current_function": "test_macro_calls", "function_body": function_body},
            {"call_expr": "printf(\"Value: %d\\n\", value)", "current_function": "test_macro_calls", "function_body": function_body},
            {"call_expr": "MAX(a, b)", "current_function": "test_macro_calls", "function_body": function_body},
            {"call_expr": "MIN(x, y)", "current_function": "test_macro_calls", "function_body": function_body}
        ]

        # Check that expected calls are found
        for expected in expected_calls:
            self.assertIn(expected, result, f"Expected call {expected} not found in results")

    def test_nested_calls(self):
        """Test nested function calls."""
        function_body = """
        void test_nested_calls() {
            int result = func1(func2(func3(1), 2), 3);
            process(inner(most_inner(arg1, arg2), arg3), arg4);
            auto value = calculate(transform(data.begin(), data.end(), [](int x) { return x * 2; }));
        }
        """

        result = self.query(function_body, "test_nested_calls")

        # Should find nested calls
        self.assertGreaterEqual(len(result), 1)

    def test_complex_arguments(self):
        """Test function calls with complex arguments."""
        function_body = """
        void test_complex_args() {
            process(std::move(data));
            func(std::make_pair(1, "value"));
            callback([this](int x) { return this->process(x); });
            setup({{"key1", "value1"}, {"key2", "value2"}});
            calculate(std::vector<int>{1, 2, 3, 4, 5});
        }
        """

        result = self.query(function_body, "test_complex_args")

        # Should find calls with complex arguments
        self.assertGreaterEqual(len(result), 1)

    def test_result_structure_validation(self):
        """Test that results have the correct structure."""
        function_body = """
        void test_structure() {
            simple_function(1, 2);
            obj.method_call("arg");
        }
        """

        result = self.query(function_body, "test_structure")

        # Should have at least some results
        self.assertGreater(len(result), 0)

        # Validate structure of each result
        for call in result:
            # Required fields
            self.assertIn("call_expr", call)
            self.assertIn("current_function", call)
            self.assertIn("function_body", call)

            # Call expr should be a string (new format)
            call_expr = call["call_expr"]
            self.assertIsInstance(call_expr, str)
            self.assertIsInstance(call["current_function"], str)
            self.assertIsInstance(call["function_body"], str)

            # Non-empty call expression
            self.assertTrue(call_expr.strip())


def callable_name_pattern(name):
    """Helper to identify potential lambda/function callable names."""
    return any(pattern in name.lower() for pattern in ['lambda', 'func', 'callback'])


if __name__ == "__main__":
    unittest.main()