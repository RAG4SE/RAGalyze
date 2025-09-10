"""
Example usage of the function call chain agent.
"""

from ragalyze.agent.main import FunctionCallChainAgent


def main():
    # Initialize the agent with a repository path
    repo_path = "./example_repo"  # Replace with actual repository path
    agent = FunctionCallChainAgent(repo_path)
    
    # Analyze a single function
    function_name = "example_function"
    chain = agent.analyze_function(function_name)
    
    # Visualize the chain
    visualization = agent.visualize_chain(chain)
    print(visualization)
    
    # Analyze multiple functions
    function_names = ["function_a", "function_b", "function_c"]
    chains = agent.analyze_multiple_functions(function_names)
    
    # Visualize all chains
    multi_visualization = agent.visualize_chains(chains)
    print(multi_visualization)
    
    # Generate a custom prompt using adalflow's Prompt class
    custom_prompt = agent.generate_custom_prompt(
        "Find all functions that call or are called by this function",
        "main"
    )
    print(f"Custom prompt: {custom_prompt}")


if __name__ == "__main__":
    main()