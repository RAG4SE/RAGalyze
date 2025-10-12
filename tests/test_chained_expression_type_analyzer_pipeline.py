import logging
from pathlib import Path
from ragalyze.configs import *

from ragalyze.agent import ChainedExpressionTypeAnalyzerPipeline


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_chained_expression_type_analyzer_pipeline():
    """Test the ChainedCallReturnTypePipeline with various chained call expressions including indexing support."""

    # Set up test fixtures
    try:
        set_global_config_value("repo_path", "./bench/call_maze_cpp")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
        logger.info("Reading source context from bench/call_maze_cpp/src/main.cpp")
        main_context = Path("bench/call_maze_cpp/src/main.cpp").read_text()
        logger.info("Instantiating ChainedExpressionTypeAnalyzerPipeline")
        pipeline = ChainedExpressionTypeAnalyzerPipeline(debug=True)
    except Exception as e:
        logger.error("Failed to set up test fixtures: %s", e)
        return

    # Test cases: (expression, expected_result)
    test_cases = [
        # Original test cases
        (
            'engine.pipeline().stage("primary").tune(1.4).limit(60.0)',
            "StageProxy",
        ),
        (
            'engine.pipeline().stage("secondary").tune(0.65).limit(40.0)',
            "StageProxy",
        ),
        ("engine.accessor()->compound.describe()", "std::string"),
        ("engine.analyzer().enable().setThreshold(0.5)", "Analyzer"),
        ("engine.accessor()->compound.sample(value)", "StageReport"),
        (
            'engine.pipeline().stage("secondary").options().bias(0.2).smoothing(0.95).finalize()',
            "double",
        ),
        ("engine.dashboard().logView.print()", "std::string"),
        (
            'engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary()',
            "std::string",
        ),
        (
            'engine.dashboard().latestReport.tag("tuning", std::to_string(tuningScore)).summary()',
            "std::string",
        ),
        ("baseObj->getName()", "std::string"),
        ("derivedObjAsBase->getName()", "std::string"),
        ("derivedObjDirect->getName()", "std::string"),
        ("castObj->getDerivedValue()", "double"),
        ("mapProcessor.isProcessed()", "bool"),
    ]

    # Test _parse_chained_expression method with index access
    logger.info("\n" + "=" * 80)
    logger.info("Testing _parse_chained_expression with Index Access")
    logger.info("=" * 80)

    parse_test_cases = [
        # Index access patterns from main.cpp
        ("engine.data()[0]", "Simple indexing with integer"),
        ("engine.sensors()[5]", "Simple indexing with integer"),
        ("engine.config()['threshold']", "String key indexing"),
        ("engine.results()[i]", "Variable indexing"),
        ("engine.metrics()[2][0]", "Nested integer indexing"),
        ("engine.maps()['key']", "Map string key indexing"),
        ("engine.arrays()[1][3]", "Multi-dimensional array indexing"),
        ("engine.collections()[0]['name']", "Mixed numeric and string indexing"),
        # Traditional patterns for comparison
        ("engine.pipeline().stage('primary')", "Traditional function chaining"),
        ("engine.accessor()->compound", "Pointer access"),
        ("obj.member", "Simple member access"),
    ]

    parse_passed = 0
    parse_failed = 0

    for i, (expression, description) in enumerate(parse_test_cases, 1):
        logger.info(f"\nParse Test {i}: {expression}")
        logger.info(f"Description: {description}")

        try:
            components = pipeline._parse_chained_expression(expression)
            logger.info(f"Parsed components: {components}")

            # Validate the components
            if components and len(components) > 0:
                # Check that we have an instance component
                if components[0]["type"] != "instance":
                    logger.error(
                        "✗ Parse Test %d FAILED: Missing instance component", i
                    )
                    parse_failed += 1
                    continue

                # Check index access components have proper structure
                has_index_access = False
                valid_structure = True

                for comp in components[1:]:  # Skip the instance
                    if comp["type"] == "index_access":
                        has_index_access = True
                        if "index_value" not in comp or comp["operator"] != "[":
                            valid_structure = False
                            logger.error(
                                "✗ Parse Test %d FAILED: Invalid index_access structure",
                                i,
                            )
                            break

                if valid_structure:
                    logger.info("✓ Parse Test %d PASSED", i)
                    parse_passed += 1
                else:
                    parse_failed += 1
            else:
                logger.error("✗ Parse Test %d FAILED: No components parsed", i)
                parse_failed += 1

        except Exception as e:
            logger.error("✗ Parse Test %d ERROR: %s", i, e)
            parse_failed += 1

    # Run main pipeline tests
    passed = 0
    failed = 0

    for i, (expression, expected) in enumerate(test_cases, 1):
        logger.info("Test %d: Analyzing expression: %s", i, expression)
        try:
            result = pipeline(expression, main_context)
            logger.info(
                "Pipeline result for expression '%s': %s (expected %s)",
                expression,
                result,
                expected,
            )

            if result == expected:
                logger.info("✓ Test %d PASSED", i)
                passed += 1
            else:
                logger.error(
                    "✗ Test %d FAILED: Expected '%s', got '%s'", i, expected, result
                )
                failed += 1
        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("=" * 80)
    logger.info("Test Summary:")
    logger.info(
        "Parse tests: %d total, %d passed, %d failed",
        len(parse_test_cases),
        parse_passed,
        parse_failed,
    )
    logger.info(
        "Pipeline tests: %d total, %d passed, %d failed",
        len(test_cases),
        passed,
        failed,
    )
    logger.info("=" * 80)

    if failed == 0 and parse_failed == 0:
        logger.info("All tests passed! ✓")
    else:
        logger.warning("Some tests failed. See details above.")


def test_type_parameter_scenarios():
    """Test the ChainedExpressionTypeAnalyzerPipeline with type parameter scenarios from main.cpp."""

    # Set up test fixtures
    try:
        set_global_config_value("repo_path", "./bench/call_maze_cpp")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
        logger.info("Reading source context from bench/call_maze_cpp/src/main.cpp")
        main_context = Path("bench/call_maze_cpp/src/main.cpp").read_text()
        logger.info("Instantiating ChainedExpressionTypeAnalyzerPipeline")
        pipeline = ChainedExpressionTypeAnalyzerPipeline(debug=True)
    except Exception as e:
        logger.error("Failed to set up test fixtures: %s", e)
        return

    # Type parameter test cases: (expression, expected_result)
    type_param_test_cases = [
        # Basic template function calls
        ('engine.computeAggregate(intData, "sum")', "int"),
        ('engine.computeAggregate(doubleData, "average")', "double"),
        ('engine.computeAggregate(intData, "max")', "int"),
        ('engine.computeAggregate(doubleData, "min")', "double"),
        # Template function calls with explicit type parameters
        ("engine.transformValues<int, int>(intData, lambda)", "std::vector<int>"),
        (
            "engine.transformValues<double, double>(doubleData, lambda)",
            "std::vector<double>",
        ),
        (
            "engine.transformValues<int, std::string>(intData, lambda)",
            "std::vector<std::string>",
        ),
        ("engine.filterValues<int>(intData, lambda)", "std::vector<int>"),
        ("engine.filterValues<double>(doubleData, lambda)", "std::vector<double>"),
        (
            "engine.filterValues<std::string>(operations, lambda)",
            "std::vector<std::string>",
        ),
        # Complex nested template calls from the additions
        (
            "engine.transformValues<int, int>(intData, [](int x) { return x * 2; })",
            "std::vector<int>",
        ),
        (
            "engine.filterValues<int>(processedInts, [](int x) { return x > 10; })",
            "std::vector<int>",
        ),
        ('engine.computeAggregate(filteredProcessed, "sum")', "int"),
        (
            "engine.transformValues<std::string, int>(names, [](const std::string& name) { return name.length(); })",
            "std::vector<int>",
        ),
        (
            "engine.filterValues<std::string>(names, [](const std::string& name) { return name.length() > 4; })",
            "std::vector<std::string>",
        ),
        ('engine.computeAggregate(nameLengths, "sum")', "int"),
        # Mixed type conversion templates
        (
            "engine.transformValues<int, double>(intData, [](int x) { return static_cast<double>(x); })",
            "std::vector<double>",
        ),
        (
            "engine.transformValues<double, std::string>(doubleData, [](double x) { return std::to_string(x); })",
            "std::vector<std::string>",
        ),
        # Template functions with different return types
        (
            'engine.transformValues<std::string, std::string>(names, [](const std::string& name) { return "name_" + name; })',
            "std::vector<std::string>",
        ),
        (
            "engine.filterValues<std::string>(names, [](const std::string& name) { return name.find('a') != std::string::npos; })",
            "std::vector<std::string>",
        ),
        # More complex template scenarios
        ('engine.computeAggregate(nameLengths, "average")', "double"),
        ('engine.computeAggregate(nameLengths, "count")', "int"),
        (
            "engine.transformValues<double, int>(doubleData, [](double x) { return static_cast<int>(std::round(x)); })",
            "std::vector<int>",
        ),
        (
            "engine.filterValues<int>(intData, [](int x) { return x % 3 == 0; })",
            "std::vector<int>",
        ),
        # Edge cases with template parameters
        (
            "engine.transformValues<int, bool>(intData, [](int x) { return x > 5; })",
            "std::vector<bool>",
        ),
        (
            "engine.filterValues<std::string>(names, [](const std::string& name) { return name.empty(); })",
            "std::vector<std::string>",
        ),
        ('engine.computeAggregate(std::vector<int>{1, 2, 3}, "product")', "int"),
    ]

    # Run type parameter tests
    logger.info("\n" + "=" * 80)
    logger.info("Testing Type Parameter Scenarios")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for i, (expression, expected) in enumerate(type_param_test_cases, 1):
        logger.info("\nType Parameter Test %d: %s", i, expression)
        logger.info("Expected: %s", expected)

        try:
            result = pipeline(expression, main_context)
            logger.info("Got: %s", result)

            if result == expected:
                logger.info("✓ Type Parameter Test %d PASSED", i)
                passed += 1
            else:
                logger.error(
                    "✗ Type Parameter Test %d FAILED: Expected '%s', got '%s'",
                    i,
                    expected,
                    result,
                )
                failed += 1

        except Exception as e:
            logger.error("✗ Type Parameter Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Type Parameter Scenarios Test Summary:")
    logger.info("Total tests: %d", len(type_param_test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    if failed == 0:
        logger.info("All type parameter tests passed! ✓")
    else:
        logger.warning("Some type parameter tests failed. See details above.")

    return failed == 0


if __name__ == "__main__":
    test_chained_expression_type_analyzer_pipeline()
    # test_type_parameter_scenarios()
