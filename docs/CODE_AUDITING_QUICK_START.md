# Code Auditing Quick Start

This guide shows how to explore call chains and run the higher-level auditing workflow that was added to `ragalyze.agent`.

## Prerequisites

- Python 3.9+
- Repository dependencies installed (e.g. `pip install -e .` from the repo root)
- BM25 / FAISS indexes generated for your codebase, as required by the existing RAG pipelines

> Tip: if you only want to experiment with the call graph logic, you can still run the examples below. Missing indexes will simply result in fewer hits.

## Inspect Call Chains From An Entry Function

```python
from ragalyze.agent import CallChainRetrievalAgent

if __name__ == "__main__":
    retriever = CallChainRetrievalAgent(debug=True)

    entry_function = "MyNamespace::Main"  # Replace with your real entry symbol
    call_chains = retriever.retrieve_call_chains(entry_function, max_depth=8)

    for chain in call_chains:
        print(f"Call chain starting at {chain.entry_function} (depth={chain.depth}):")
        for call in chain.functions:
            print(f"  • {call.function_name}")
        print()
```

### What You Get Back

- `CallChain.entry_function`: the root symbol you asked for
- `CallChain.functions`: ordered `FunctionCall` objects for each hop (includes context snippets, best-effort locations, etc.)
- `CallChain.context`: metadata such as the retrieved entry definition and later analysis results
- `CallChain.language`: heuristic language detection based on the entry definition

The retriever internally:

1. Uses `FetchFunctionDefinitionFromCallAgent` to resolve function bodies
2. Extracts nested call expressions (including macro expansions and local functions)
3. Recursively walks callees until reaching `max_depth`, skipping cycles

If a definition cannot be found, the chain is truncated gracefully so you still see partial paths.

## Full Audit Workflow

```python
from ragalyze.agent import CodeAuditingAgent

if __name__ == "__main__":
    auditor = CodeAuditingAgent(debug=True)

    audit = auditor.audit_function("MyNamespace::Main", max_depth=6)

    print(f"Entry: {audit.entry_function}")
    print(f"Call chains discovered: {len(audit.call_chains)}")
    print(f"Potential vulnerabilities: {len(audit.vulnerabilities)}")
    print(f"Suggested tests: {len(audit.test_cases)}")

    for vuln in audit.vulnerabilities:
        print(f"[{vuln.severity.value.upper()}] {vuln.description}")
```

The orchestrator runs:

1. Call-chain discovery (as shown above)
2. Lightweight security heuristics (`CallChainAnalysisAgent`)
3. Test case scaffolding (`TestCaseGenerationAgent`)

The returned `AuditResult` contains everything needed to surface findings in a UI, dump to JSON, or drive further automation.

## Next Steps

- Integrate the retriever into your own agent flow by reusing its `CallChain` objects
- Expand vulnerability heuristics or plug in static analyzers where the class stubs are defined (see `ragalyze/agent.py`)
- Feed the generated test case descriptions into your favourite test harness to bootstrap coverage

Happy auditing!
