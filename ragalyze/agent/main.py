"""
Main module for the function call chain agent.
Provides a high-level interface for building and analyzing function call chains.
"""

from typing import List, Optional, Union
from .function_chain_builder import FunctionCallChainBuilder, FunctionCallChain
from .visualization import visualize_call_chain, visualize_multiple_chains
from ragalyze.prompts import FIND_FUNCTION_CALL_TEMPLATE
import adalflow as adal


class FunctionCallChainAgent:
    """Main agent for building and analyzing function call chains."""
    
    def __init__(self, repo_path: str):
        """Initialize the agent with a repository path.
        
        Args:
            repo_path: Path to the repository to analyze
        """
        self.chain_builder = FunctionCallChainBuilder(repo_path)
        # Leverage the Prompt class from adalflow
        self.prompt_template = FIND_FUNCTION_CALL_TEMPLATE
    
    def analyze_function(self, function_name: str) -> FunctionCallChain:
        """Analyze a single function and build its call chain.
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            FunctionCallChain object with the analysis results
        """
        return self.chain_builder.build_bidirectional_chain(function_name)
    
    def analyze_multiple_functions(self, function_names: List[str]) -> List[FunctionCallChain]:
        """Analyze multiple functions and build their call chains.
        
        Args:
            function_names: List of function names to analyze
            
        Returns:
            List of FunctionCallChain objects with the analysis results
        """
        chains = []
        for func_name in function_names:
            chain = self.analyze_function(func_name)
            chains.append(chain)
        return chains
    
    def get_function_context(self, function_name: str):
        """Get the context documents for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            List of documents containing the function
        """
        return self.chain_builder.get_function_context(function_name)
    
    def visualize_chain(self, chain: FunctionCallChain) -> str:
        """Visualize a single function call chain.
        
        Args:
            chain: FunctionCallChain object to visualize
            
        Returns:
            String representation of the call chain
        """
        return visualize_call_chain(chain)
    
    def visualize_chains(self, chains: List[FunctionCallChain]) -> str:
        """Visualize multiple function call chains.
        
        Args:
            chains: List of FunctionCallChain objects to visualize
            
        Returns:
            String representation of all call chains
        """
        return visualize_multiple_chains(chains)
    
    def generate_custom_prompt(self, task_description: str, function_name: str) -> str:
        """Generate a custom prompt using the adalflow Prompt class.
        
        Args:
            task_description: Description of the task
            function_name: Name of the function to analyze
            
        Returns:
            Formatted prompt string
        """
        # Using the Prompt class from adalflow to create a custom prompt
        custom_prompt = adal.Prompt(
            template=r"""
{{task_description}}

Function to analyze: {{function_name}}

Please provide a detailed analysis of this function's call chain.
"""
        )
        return custom_prompt.call(
            task_description=task_description,
            function_name=function_name
        )