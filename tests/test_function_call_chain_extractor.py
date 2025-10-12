#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ragalyze.configs import set_global_config_value
from ragalyze.agent import FunctionCallChainExtractor, CallChain, FunctionCall

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_function_call_chain_extractor():
    """Test the FunctionCallChainExtractor with various complex scenarios."""

    # Set up test configuration
    try:
        set_global_config_value("repo_path", "./bench/solidity_vulnerable_contracts")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
        logger.info("Configuration set successfully")
    except Exception as e:
        logger.error("Failed to set configuration: %s", e)
        return False

    # Initialize the extractor
    extractor = FunctionCallChainExtractor(debug=True, max_depth=10)

    # Test cases: (function_name, context_file, description)
    test_cases = [
        # Test Case 1: VulnerableDeFiProtocol withdraw function
        (
            "withdraw",
            "VulnerableDeFiProtocol.sol",
            "Test reentrancy vulnerability in withdraw function",
        ),
        # Test Case 2: MaliciousPriceOracle updatePrice function
        (
            "updatePrice",
            "MaliciousPriceOracle.sol",
            "Test price manipulation vulnerability",
        ),
        # Test Case 3: ComplexToken transfer function
        ("transfer", "ComplexToken.sol", "Test reentrancy in token transfer"),
        # Test Case 4: AttackContract launchComplexAttack
        (
            "launchComplexAttack",
            "AttackContract.sol",
            "Test complex multi-vector attack",
        ),
        # Test Case 5: VulnerableDeFiProtocol borrow function
        (
            "borrow",
            "VulnerableDeFiProtocol.sol",
            "Test complex borrowing with external calls",
        ),
        # Test Case 6: MaliciousPriceOracle flashLoanPriceManipulation
        (
            "flashLoanPriceManipulation",
            "MaliciousPriceOracle.sol",
            "Test flash loan price manipulation",
        ),
        # Test Case 7: ComplexToken flashLoan
        ("flashLoan", "ComplexToken.sol", "Test flash loan vulnerability"),
        # Test Case 8: AttackContract drainFunds
        ("drainFunds", "AttackContract.sol", "Test fund draining attack"),
    ]

    # Run tests
    passed = 0
    failed = 0

    logger.info("=" * 80)
    logger.info("Testing FunctionCallChainExtractor")
    logger.info("=" * 80)

    for i, (function_name, context_file, description) in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {function_name} from {context_file}")
        logger.info(f"Description: {description}")

        try:
            # Read the context
            context_path = Path(f"./bench/solidity_vulnerable_contracts/{context_file}")
            if not context_path.exists():
                logger.error(f"Context file not found: {context_path}")
                failed += 1
                continue

            with open(context_path, "r") as f:
                context = f.read()

            # Extract call chain
            call_chain = extractor.extract_call_chain(
                function_name, context, str(context_path)
            )

            if call_chain:
                logger.info("✓ Test %d PASSED - Call chain extracted successfully", i)
                logger.info(f"  Entry function: {call_chain.entry_function}")
                logger.info(f"  Language: {call_chain.language}")
                logger.info(f"  Total calls: {len(call_chain.calls)}")
                logger.info(f"  Depth: {call_chain.depth}")

                # Display call chain summary
                summary = extractor.get_call_chain_summary(call_chain)
                logger.info("  Call Chain Summary:")
                for line in summary.split("\n"):
                    if line.strip():
                        logger.info(f"    {line}")

                # Display dependencies
                dependencies = extractor.get_function_dependencies(call_chain)
                if dependencies:
                    logger.info(f"  Dependencies: {', '.join(sorted(dependencies))}")

                # Display statistics
                stats = extractor.get_call_statistics(call_chain)
                logger.info(f"  Statistics: {stats}")

                # Display visualization
                viz = extractor.visualize_call_chain(call_chain)
                logger.info("  Visualization:")
                for line in viz.split("\n"):
                    if line.strip():
                        logger.info(f"    {line}")

                passed += 1
            else:
                logger.error("✗ Test %d FAILED - No call chain extracted", i)
                failed += 1

        except Exception as e:
            logger.error("✗ Test %d ERROR: %s", i, e)
            failed += 1

    # Summary
    logger.info("=" * 80)
    logger.info("Test Summary:")
    logger.info("Total tests: %d", len(test_cases))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    if failed == 0:
        logger.info("All tests passed! ✓")
        return True
    else:
        logger.warning("Some tests failed. See details above.")
        return False


def test_cross_contract_analysis():
    """Test cross-contract call chain analysis."""

    try:
        set_global_config_value("repo_path", "./bench/solidity_vulnerable_contracts")
        set_global_config_value("generator.provider", "deepseek")
        set_global_config_value("generator.model", "deepseek-chat")
    except Exception as e:
        logger.error("Failed to set configuration: %s", e)
        return False

    extractor = FunctionCallChainExtractor(debug=True, max_depth=4)

    # Read all contract files
    contracts_dir = Path("./bench/solidity_vulnerable_contracts")
    all_contexts = {}

    for contract_file in contracts_dir.glob("*.sol"):
        if contract_file.name != "README.md":
            with open(contract_file, "r") as f:
                all_contexts[contract_file.stem] = f.read()

    logger.info("=" * 80)
    logger.info("Testing Cross-Contract Call Chain Analysis")
    logger.info("=" * 80)

    # Test cross-contract interactions
    cross_contract_tests = [
        {
            "entry_function": "launchComplexAttack",
            "entry_contract": "AttackContract",
            "expected_contracts": [
                "AttackContract",
                "VulnerableDeFiProtocol",
                "MaliciousPriceOracle",
            ],
        },
        {
            "entry_function": "flashLoanPriceManipulation",
            "entry_contract": "MaliciousPriceOracle",
            "expected_contracts": ["MaliciousPriceOracle", "VulnerableDeFiProtocol"],
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(cross_contract_tests, 1):
        logger.info(f"\nCross-Contract Test {i}: {test['entry_function']}")
        logger.info(f"Entry contract: {test['entry_contract']}")
        logger.info(f"Expected contracts: {', '.join(test['expected_contracts'])}")

        try:
            # Get entry context
            entry_context = all_contexts.get(test["entry_contract"])
            if not entry_context:
                logger.error(f"Entry contract not found: {test['entry_contract']}")
                failed += 1
                continue

            # Extract call chain
            call_chain = extractor.extract_call_chain(
                test["entry_function"], entry_context, f"{test['entry_contract']}.sol"
            )

            if call_chain:
                # Analyze which contracts are involved
                involved_contracts = _analyze_involved_contracts(call_chain)
                logger.info(f"Involved contracts: {involved_contracts}")

                # Check if expected contracts are detected
                detected_contracts = set()
                for expected in test["expected_contracts"]:
                    if any(
                        expected.lower() in contract.lower()
                        for contract in involved_contracts
                    ):
                        detected_contracts.add(expected)

                if detected_contracts:
                    logger.info("✓ Cross-Contract Test %d PASSED", i)
                    logger.info(
                        f"  Detected contracts: {', '.join(detected_contracts)}"
                    )
                    passed += 1
                else:
                    logger.warning(
                        "✗ Cross-Contract Test %d PARTIAL - Limited contract detection",
                        i,
                    )
                    failed += 1
            else:
                logger.error(
                    "✗ Cross-Contract Test %d FAILED - No call chain extracted", i
                )
                failed += 1

        except Exception as e:
            logger.error("✗ Cross-Contract Test %d ERROR: %s", i, e)
            failed += 1

    logger.info("=" * 80)
    logger.info("Cross-Contract Test Summary:")
    logger.info("Total tests: %d", len(cross_contract_tests))
    logger.info("Passed: %d", passed)
    logger.info("Failed: %d", failed)
    logger.info("=" * 80)

    return failed == 0


def _analyze_involved_contracts(call_chain: CallChain) -> List[str]:
    """Analyze which contracts are involved in a call chain."""

    contracts = set()

    def analyze_calls(calls: List[FunctionCall]):
        for call in calls:
            # Check if call expression contains contract references
            expression = call.call_expression.lower()

            # Look for common Solidity contract patterns
            if any(
                keyword in expression
                for keyword in ["protocol", "oracle", "token", "attack"]
            ):
                contracts.add(call.function_name)

            # Check for external contract calls
            if "." in expression and "(" in expression:
                # Extract contract name from calls like "contract.function()"
                parts = expression.split(".")
                if len(parts) > 1:
                    contract_part = parts[0].strip()
                    if contract_part and contract_part not in ["this", "super"]:
                        contracts.add(contract_part)

            # Analyze nested calls
            if hasattr(call, "nested_calls") and call.nested_calls:
                analyze_calls(call.nested_calls)

    if call_chain and call_chain.calls:
        analyze_calls(call_chain.calls)

    return list(contracts)


def main():
    """Run all tests."""
    logger.info("Starting FunctionCallChainExtractor tests...")

    # Test basic functionality
    basic_test_passed = test_function_call_chain_extractor()

    # # # Test cross-contract analysis
    # cross_contract_test_passed = test_cross_contract_analysis()

    # Overall summary
    logger.info("=" * 80)
    logger.info("Overall Test Summary:")
    logger.info(
        "Basic functionality tests: %s", "PASSED" if basic_test_passed else "FAILED"
    )
    logger.info(
        "Vulnerability pattern tests: %s",
        "PASSED" if vulnerability_test_passed else "FAILED",
    )
    logger.info(
        "Cross-contract tests: %s", "PASSED" if cross_contract_test_passed else "FAILED"
    )
    logger.info("=" * 80)

    all_passed = (
        basic_test_passed and vulnerability_test_passed and cross_contract_test_passed
    )

    if all_passed:
        logger.info("All tests passed! ✓")
        return True
    else:
        logger.warning("Some tests failed. See details above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
