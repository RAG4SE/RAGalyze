#!/usr/bin/env python3

import sys
from pathlib import Path
import os

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragalyze.agent import FetchCalleeInfoPipeline


class MockDocument:
    def __init__(self, meta_data):
        self.meta_data = meta_data


def test_namespace_parsing():
    """Test that namespace parsing works in chained expressions across different languages."""

    pipeline = FetchCalleeInfoPipeline(debug=True)

    # C++ test cases with :: namespace parsing
    cpp_test_cases = [
        "A::B::f()",
        "std::vector<int>::push_back(element)",
        "my_namespace::my_class::my_method()",
        "outer::inner::func()",
        "A::B()->C()",
        "ns1::ns2::obj.method()",
        "global::utils::helper::process()",
        "boost::filesystem::path::filename()",
        "std::map<int, std::string>::find(key)",
        "mylib::collections::List<int>::add(item)",
    ]

    # Python test cases with . namespace parsing
    python_test_cases = [
        "my_module.MyClass.my_method()",
        "package.subpackage.module.function()",
        "django.db.models.Model.objects.filter()",
        "numpy.array.dtype.fields",
        "pandas.DataFrame.groupby().agg()",
        "my_project.utils.helpers.format_data()",
        "from_import.module.Class.static_method()",
    ]

    # Java test cases with . namespace parsing
    java_test_cases = [
        "com.example.MyClass.myMethod()",
        "java.util.List<String>.stream()",
        "org.springframework.web.bind.annotation.RestController",
        "android.content.Context.getSystemService()",
        "java.lang.Math.max(a, b)",
        "com.company.project.service.UserService.findById()",
    ]

    # Solidity test cases with . namespace parsing
    solidity_test_cases = [
        "MyContract.myFunction()",
        "ERC20.balanceOf(address)",
        "Ownable.owner.transfer(amount)",
        "SafeMath.add(uint256, uint256)",
        "ContractA.ContractB.method()",
    ]

    # Test all language cases
    all_test_cases = [
        ("cpp", cpp_test_cases),
        ("python", python_test_cases),
        ("java", java_test_cases),
        ("solidity", solidity_test_cases),
    ]

    for language, test_cases in all_test_cases:
        print(f"\n\nTesting {language} namespace parsing:")
        print("=" * 50)
        pipeline.doc = MockDocument({"programming_language": language})

        for i, expression in enumerate(test_cases, 1):
            print(f"\n{language} Test {i}: {expression}")
            try:
                components = pipeline._parse_chained_expression(expression)
                print(f"Parsed components:")
                for j, component in enumerate(components):
                    print(
                        f"  {j+1}. Name: {component['name']}, Type: {component['type']}"
                    )
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    test_namespace_parsing()
