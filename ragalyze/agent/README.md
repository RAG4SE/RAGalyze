# Function Call Chain Agent

The Function Call Chain Agent is a module within the RAGalyze system that builds bidirectional function call chains using advanced code analysis techniques. It leverages BM25 tokenization with special prefixes (`[CALL]` and `[FUNC]`) and the existing RAG components for retrieval and analysis.

## Features

1. **Forward Chain Analysis**: Starting from a function f, find all functions that f calls
2. **Backward Chain Analysis**: Starting from a function f, find all functions that call f
3. **Bidirectional Chain Building**: Combine both forward and backward analysis for complete call chain visualization
4. **Integration with BM25 Tokenization**: Leverages the special `[CALL]` and `[FUNC]` prefixes from BM25 tokenization
5. **Prompt Integration**: Uses the adalflow Prompt class for customizable analysis prompts

## Architecture

The agent consists of several key components:

- `FunctionCallChainBuilder`: Core logic for building forward and backward chains
- `FunctionCallChainAgent`: High-level interface for chain analysis
- `Visualization`: Tools for visualizing call chains
- `Utils`: Utility functions for exporting and formatting chains

## Usage

```python
from ragalyze.agent.main import FunctionCallChainAgent

# Initialize the agent with a repository path
agent = FunctionCallChainAgent("/path/to/your/repo")

# Analyze a single function
chain = agent.analyze_function("main")

# Visualize the chain
visualization = agent.visualize_chain(chain)
print(visualization)

# Analyze multiple functions
chains = agent.analyze_multiple_functions(["func1", "func2", "func3"])
multi_viz = agent.visualize_chains(chains)
print(multi_viz)
```

## Key Components

### FunctionCallChain Dataclass

Represents a function call chain with the following attributes:
- `target_function`: The function being analyzed
- `callers`: Functions that call the target function
- `callees`: Functions called by the target function
- `forward_chain`: Same as callees
- `backward_chain`: Same as callers

### FunctionCallChainBuilder

The core builder class that:
1. Uses BM25 queries with `[FUNC]function_name` to find documents containing function definitions
2. Extracts `[CALL]` prefixed tokens to identify function calls
3. Uses the `FIND_FUNCTION_CALL_TEMPLATE` prompt to find caller functions
4. Builds bidirectional chains combining both analyses

### FunctionCallChainAgent

High-level interface that:
1. Provides a simplified API for chain analysis
2. Integrates with adalflow's Prompt class for customizable prompts
3. Offers visualization capabilities
4. Supports both single and multiple function analysis

## Integration with BM25 Tokenization

The agent leverages the special tokenization approach where:
- Function definitions are prefixed with `[FUNC]`
- Function calls are prefixed with `[CALL]`

This allows for efficient retrieval and analysis using the BM25 index.

## Example Output

```
Function Call Chain Analysis for: main
==================================================

Functions that CALL this function:
----------------------------------------
  ← process_data
  ← handle_request

TARGET FUNCTION: main
  ↓

Functions CALLED by this function:
----------------------------------------
  → initialize_system
  → process_data
  → cleanup_resources
```

## Testing

The module includes comprehensive unit tests in `test_agent.py` that mock the RAG system responses to verify functionality.