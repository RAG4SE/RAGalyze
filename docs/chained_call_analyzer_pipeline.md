# ChainedCallAnalyzerPipeline

## Overview

The `ChainedCallAnalyzerPipeline` class is designed to analyze chained call expressions across multiple programming languages (C++, Java, Python). It resolves the type of complex chained expressions by iteratively applying type analysis queries to determine the final return type of the innermost function call.

## Purpose

This class handles complex chained call patterns that are common in object-oriented programming:
- Simple member access chains: `a.b.c()`
- Mixed function calls and member access: `a.b().c()`
- Pointer access with function calls: `a->b().c()`
- Pure pointer access chains: `a->b->c`

By analyzing these chains step by step, the pipeline can determine the exact class type of the final function call return value.

## API Specification

### Class Definition

```python
class ChainedCallAnalyzerPipeline:
    """Analyzes chained call expressions to determine the final return type."""

    def __init__(self, debug: bool = False):
        """
        Initialize the pipeline with optional debug logging.

        Args:
            debug: Enable debug logging for troubleshooting
        """
        self.debug = debug

    def __call__(self, expression: str, context: str) -> str:
        """
        Analyze a chained call expression and return the final class name.

        Args:
            expression: The chained call expression to analyze (e.g., "a.b.c()")
            context: The source code context containing the expression

        Returns:
            The class name of the innermost function call's return type
        """
```

## Supported Call Patterns

### 1. Simple Member Access Chains
```
a.b.c()          # Direct member access
obj.field.method() # Field access followed by method call
```

### 2. Mixed Function Calls and Member Access
```
a.b().c()        # Function call return used for member access
getFoo().bar()   # Method chaining
```

### 3. Pointer Access Patterns
```
a->b().c()       # Pointer dereference followed by method call
a->b->c          # Pure pointer access chain
```

### 4. Complex Nested Patterns
```
a.b().c->d()     # Mixed member and pointer access
getA()->b().c()  # Function calls with pointer access
```

## Implementation Approach

### Core Algorithm

1. **Parse Expression**: Break down the chained expression into individual components
2. **Initial Type Resolution**: Use `IsClassInstanceQuery` to determine the type of the outermost instance
3. **Iterative Resolution**: For each component in the chain:
   - If the component is a member function call:
     - Use `FetchClassPipeline` to get the class definition of the current instance
     - Use `ExtractMemberDefinitionFromClassQuery` to get the member definition
     - Use `FetchReturnTypeQuery` to get the return type of the member function, which is the class definition for the next iteration
   - If the component is a member variable access:
     - Use `FetchMemberVarQuery` to resolve the return type, which is the class definition for the next iteration
4. **Final Type Extraction**: Return the class name of the innermost function call

### Key Components

#### Call Expression Parser
- Identifies access operators (., ->)
- Separates function calls from member access
- Handles nested parentheses and complex expressions

#### Type Resolution Pipeline
- Leverages existing `IsClassInstanceQuery` for initial type detection
- Uses `FetchMemberVarQuery` for member variable type extraction
- Applies `FetchReturnTypeQuery` for function return type analysis

#### Error Handling
- Gracefully handles incomplete or malformed expressions
- Provides meaningful error messages for type resolution failures
- Supports debug logging for troubleshooting

## Usage Examples

### Basic Usage

```python
analyzer = ChainedCallAnalyzerPipeline(debug=True)

# Simple member access chain
result = analyzer("a.b.c()", context_code)
print(result)  # Output: "ReturnTypeOfC"

# Mixed function calls and member access
result = analyzer("a.b().c()", context_code)
print(result)  # Output: "ReturnTypeOfC"

# Pointer access patterns
result = analyzer("a->b().c()", context_code)
print(result)  # Output: "ReturnTypeOfC"
```

### Integration with Existing Code

```python
# Use within the ragalyze agent framework
pipeline = ChainedCallAnalyzerPipeline()
final_type = pipeline(expression, source_context)

# The result can be used for:
# - Type checking and validation
# - Code completion suggestions
# - Static analysis reports
```

## Dependencies

The `ChainedCallAnalyzerPipeline` depends on the following existing classes:
- `IsClassInstanceQuery`: For determining the initial class type
- `FetchMemberVarQuery`: For resolving member variable types
- `FetchReturnTypeQuery`: For analyzing function return types

## Language Support

### C++
- Supports both `.` and `->` operators
- Handles pointer and reference types
- Compatible with template types and namespaces

### Java
- Supports `.` operator for all access
- Handles object references and method calls
- Compatible with generic types

### Python
- Supports `.` operator for attribute access
- Handles method calls and property access
- Compatible with dynamic typing scenarios

## Error Handling

The pipeline provides robust error handling for:
- **Invalid Expressions**: Malformed syntax or incomplete chains
- **Type Resolution Failures**: When type information cannot be determined
- **Missing Context**: Insufficient context information for analysis
- **Unsupported Patterns**: Language-specific constructs not yet supported

## Debug Features

When `debug=True`, the pipeline provides:
- Step-by-step resolution logging
- Intermediate type information
- Detailed error messages
- Performance metrics

## Performance Considerations

- Each chain component requires separate query execution
- Complex chains may involve multiple type resolution steps
- Caching mechanisms can be implemented for repeated analysis
- Parallel processing opportunities for independent chain segments

## Future Enhancements

1. **Template/Generic Support**: Enhanced handling of templated types
2. **Namespace Resolution**: Support for namespace-qualified names
3. **Inheritance Analysis**: Consider inheritance hierarchies in type resolution
4. **Performance Optimization**: Caching and parallel processing improvements
5. **Additional Languages**: Support for JavaScript, C#, and other languages