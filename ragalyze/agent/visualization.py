"""
Visualization module for function call chains.
"""

from typing import List, Dict
from .function_chain_builder import FunctionCallChain


def visualize_call_chain(chain: FunctionCallChain) -> str:
    """Create a text-based visualization of the function call chain.
    
    Args:
        chain: FunctionCallChain object to visualize
        
    Returns:
        String representation of the call chain
    """
    result = []
    result.append(f"Function Call Chain Analysis for: {chain.target_function}")
    result.append("=" * 50)
    
    # Backward chain (functions that call this function)
    if chain.backward_chain:
        result.append("\nFunctions that CALL this function:")
        result.append("-" * 40)
        for caller in chain.backward_chain:
            result.append(f"  ← {caller}")
    else:
        result.append("\nNo functions found that call this function")
    
    # Target function
    result.append(f"\nTARGET FUNCTION: {chain.target_function}")
    result.append("  ↓")
    
    # Forward chain (functions this function calls)
    if chain.forward_chain:
        result.append("\nFunctions CALLED by this function:")
        result.append("-" * 40)
        for callee in chain.forward_chain:
            result.append(f"  → {callee}")
    else:
        result.append("\nThis function does not call any other functions")
    
    return "\n".join(result)


def visualize_multiple_chains(chains: List[FunctionCallChain]) -> str:
    """Create a visualization of multiple function call chains.
    
    Args:
        chains: List of FunctionCallChain objects to visualize
        
    Returns:
        String representation of all call chains
    """
    result = []
    result.append("Multiple Function Call Chain Analysis")
    result.append("=" * 50)
    
    for chain in chains:
        result.append(visualize_call_chain(chain))
        result.append("\n" + "-" * 50 + "\n")
    
    return "\n".join(result)