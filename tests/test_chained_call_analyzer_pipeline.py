import logging
from pathlib import Path
from ragalyze.configs import *

from ragalyze.agent import ChainedCallAnalyzerPipeline


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_chained_call_analyzer_pipeline():
    """Test the ChainedCallAnalyzerPipeline with various chained call expressions."""

    # Set up test fixtures
    try:
        set_global_config_value("repo_path", "./bench/call_maze")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
        logger.info("Reading source context from bench/call_maze/src/main.cpp")
        main_context = Path("bench/call_maze/src/main.cpp").read_text()
        logger.info("Instantiating ChainedCallAnalyzerPipeline")
        pipeline = ChainedCallAnalyzerPipeline(debug=True)
    except Exception as e:
        logger.error("Failed to set up test fixtures: %s", e)
        return

    # Test cases: (expression, expected_result)
    test_cases = [
        (
            "engine.pipeline().stage(\"primary\").tune(1.4).limit(60.0)",
            "StageProxy",
        ),
        (
            "engine.pipeline().stage(\"secondary\").tune(0.65).limit(40.0)",
            "StageProxy",
        ),
        ("engine.accessor()->compound.describe()", "PipelineCompound"),
        ("engine.analyzer().enable().setThreshold(0.5)", "Analyzer"),
        ("engine.accessor()->compound.sample(value)", "PipelineCompound"),
        (
            "engine.pipeline().stage(\"secondary\").options().bias(0.2).smoothing(0.95).finalize()",
            "StageProxyOptions",
        ),
        ("engine.dashboard().logView.print()", "LogView"),
        (
            "engine.dashboard().latestReport.tag(\"tuning\", std::to_string(tuningScore)).summary()",
            "StageReport",
        ),
    ]

    # Run tests
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
                logger.error("✗ Test %d FAILED: Expected '%s', got '%s'", i, expected, result)
                failed += 1
        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("=" * 50)
    logger.info("Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 50)

    if failed == 0:
        logger.info("All tests passed! ✓")
    else:
        logger.warning("Some tests failed. See details above.")


if __name__ == "__main__":
    test_chained_call_analyzer_pipeline()
