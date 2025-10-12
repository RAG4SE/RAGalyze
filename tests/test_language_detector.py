#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.agent import LanguageDetector

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_language_detector():
    """Test the LanguageDetector with various code snippets."""

    detector = LanguageDetector(debug=True)

    # Test cases: (code_snippet, expected_language)
    test_cases = [
        # Python
        (
'''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))''',
            "python"
        ),

        # C++
        (
'''#include <iostream>
#include <vector>

std::vector<int> fibonacci(int n) {
    std::vector<int> fib(n+1);
    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i <= n; i++) {
        fib[i] = fib[i-1] + fib[i-2];
    }

    return fib;
}

int main() {
    auto result = fibonacci(10);
    for (int num : result) {
        std::cout << num << " ";
    }
    return 0;
}''',
            "cpp"
        ),

        # JavaScript
        (
'''function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

const result = fibonacci(10);
console.log(result);

// Arrow function version
const fibArrow = (n) => {
    if (n <= 1) return n;
    return fibArrow(n-1) + fibArrow(n-2);
};''',
            "javascript"
        ),

        # Java
        (
'''public class Fibonacci {
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n-1) + fibonacci(n-2);
    }

    public static void main(String[] args) {
        int result = fibonacci(10);
        System.out.println("Fibonacci: " + result);
    }
}''',
            "java"
        ),

        # HTML
        (
'''<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a test page.</p>
    <div id="content">
        <button onclick="alert('Hello!')">Click me</button>
    </div>
</body>
</html>''',
            "html"
        ),

        # CSS
        (
'''body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h1 {
    color: #333;
    text-align: center;
}''',
            "css"
        ),

        # JSON
        (
'''{
    "name": "Test Project",
    "version": "1.0.0",
    "description": "A test project",
    "main": "index.js",
    "scripts": {
        "start": "node index.js",
        "test": "jest"
    },
    "dependencies": {
        "express": "^4.17.1",
        "lodash": "^4.17.21"
    },
    "devDependencies": {
        "jest": "^27.0.0"
    }
}''',
            "json"
        ),

        # SQL
        (
'''SELECT u.id, u.name, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
GROUP BY u.id, u.name, u.email
HAVING COUNT(o.id) > 0
ORDER BY order_count DESC
LIMIT 10;''',
            "sql"
        ),

        # Shell/Bash
        (
'''#!/bin/bash

# Simple script to process files
for file in *.txt; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        wc -l "$file" >> line_counts.txt
    fi
done

echo "Processing complete!"''',
            "shell"
        ),

        # Go
        (
'''package main

import "fmt"

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

func main() {
    result := fibonacci(10)
    fmt.Printf("Fibonacci: %d\\n", result)
}''',
            "go"
        ),

        # Rust
        (
'''fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n-1) + fibonacci(n-2)
}

fn main() {
    let result = fibonacci(10);
    println!("Fibonacci: {}", result);
}''',
            "rust"
        ),

        # TypeScript
        (
'''interface User {
    id: number;
    name: string;
    email: string;
}

function fibonacci(n: number): number {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

const result: number = fibonacci(10);
console.log(result);''',
            "typescript"
        ),

        # Empty case
        ("", "unknown"),
    ]

    logger.info("=" * 80)
    logger.info("Testing LanguageDetector")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (code_snippet, expected_language) in enumerate(test_cases, 1):
        snippet_preview = code_snippet[:100] + "..." if len(code_snippet) > 100 else code_snippet
        logger.info(f"\nTest {i}: Expected language '{expected_language}'")
        logger.info(f"Snippet preview: {snippet_preview}")

        try:
            result = detector(code_snippet)
            logger.info(f"Result: {result}")

            if isinstance(result, dict):
                detected_language = result.get("language", "unknown")
                confidence = result.get("confidence", "low")
                reasoning = result.get("reasoning", "")

                if detected_language.lower() == expected_language.lower():
                    logger.info(f"✓ Test {i} PASSED - Language: {detected_language} (Confidence: {confidence})")
                    passed += 1
                else:
                    logger.warning(f"✗ Test {i} FAILED:")
                    logger.warning(f"  Expected: {expected_language}")
                    logger.warning(f"  Got: {detected_language}")
                    logger.warning(f"  Confidence: {confidence}")
                    logger.warning(f"  Reasoning: {reasoning}")
                    failed += 1
            else:
                logger.warning(f"✗ Test {i} FAILED: Expected dict result, got {type(result)}")
                failed += 1

        except Exception as e:
            logger.error(f"✗ Test {i} ERROR: {e}")
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("LanguageDetector Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def main():
    """Run all tests."""
    logger.info("Starting LanguageDetector tests...")

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
    test_passed = test_language_detector()

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("Overall Test Summary:")
    logger.info("LanguageDetector tests: %s", "PASSED" if test_passed else "FAILED")
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