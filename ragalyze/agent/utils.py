"""
Utility functions for the function call chain agent.
"""

import json
from typing import List, Dict, Any
from .function_chain_builder import FunctionCallChain


def export_chain_to_json(chain: FunctionCallChain, filepath: str) -> None:
    """Export a function call chain to a JSON file.
    
    Args:
        chain: FunctionCallChain object to export
        filepath: Path to the output JSON file
    """
    chain_dict = {
        "target_function": chain.target_function,
        "callers": chain.callers,
        "callees": chain.callees,
        "forward_chain": chain.forward_chain,
        "backward_chain": chain.backward_chain
    }
    
    with open(filepath, 'w') as f:
        json.dump(chain_dict, f, indent=2)


def export_chains_to_json(chains: List[FunctionCallChain], filepath: str) -> None:
    """Export multiple function call chains to a JSON file.
    
    Args:
        chains: List of FunctionCallCallChain objects to export
        filepath: Path to the output JSON file
    """
    chains_list = []
    for chain in chains:
        chain_dict = {
            "target_function": chain.target_function,
            "callers": chain.callers,
            "callees": chain.callees,
            "forward_chain": chain.forward_chain,
            "backward_chain": chain.backward_chain
        }
        chains_list.append(chain_dict)
    
    with open(filepath, 'w') as f:
        json.dump(chains_list, f, indent=2)


def format_chain_as_markdown(chain: FunctionCallChain) -> str:
    """Format a function call chain as Markdown.
    
    Args:
        chain: FunctionCallChain object to format
        
    Returns:
        Markdown formatted string
    """
    md_lines = []
    md_lines.append(f"# Function Call Chain: {chain.target_function}")
    md_lines.append("")
    
    # Backward chain
    md_lines.append("## Functions that call this function")
    if chain.backward_chain:
        md_lines.append("")
        for caller in chain.backward_chain:
            md_lines.append(f"- `{caller}`")
    else:
        md_lines.append("")
        md_lines.append("No calling functions found.")
    
    md_lines.append("")
    
    # Forward chain
    md_lines.append("## Functions called by this function")
    if chain.forward_chain:
        md_lines.append("")
        for callee in chain.forward_chain:
            md_lines.append(f"- `{callee}`")
    else:
        md_lines.append("")
        md_lines.append("No called functions found.")
    
    return "\n".join(md_lines)