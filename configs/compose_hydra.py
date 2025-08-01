#!/usr/bin/env python3
"""
Simple example of loading all YAML configurations from the configs folder using Hydra.
This example works with the existing YAML files without requiring dataclass definitions.
"""

import os
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

#! Though the following imports are not directly used, they are stored in globals() and will be used implicitly. So DO NOT REMOVE THEM!!
from clients.huggingface_embedder_client import HuggingfaceClient, HuggingfaceEmbedder
from clients.dashscope_client import DashScopeClient, DashScopeEmbedder
from adalflow import GoogleGenAIClient

def load_all_configs_programmatically():
    """
    Load all YAML configurations programmatically using Hydra's compose API.
    """
    
    # Get absolute path to configs directory
    config_dir = os.path.abspath("configs")
    
    # Initialize Hydra with the configs directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Get list of available config files
        config_files = [f.stem for f in Path(config_dir).glob("*.yaml")]
        
        # Load each configuration
        all_configs = {}
        for config_name in config_files:
            try:
                cfg = compose(config_name=config_name)
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                all_configs[config_name] = cfg_dict
            except Exception as e:
                raise

        load_generator_config(all_configs)
        load_knowledge_config(all_configs)
        
        return all_configs

PROVIDER_NAME_TO_CLASS = {
    "google": GoogleGenAIClient,
    "dashscope": DashScopeClient,
}

def load_generator_config(configs: dict):
    # Add client classes to each provider
    assert "provider" in configs["generator"], "generator config must contain 'provider'"
    assert "model" in configs["generator"], "generator config must contain 'model'"
    assert "model_kwargs" in configs["generator"], "generator config must contain 'model_kwargs'"
    assert "temperature" in configs["generator"]["model_kwargs"], "generator config must contain 'temperature'"
    assert "top_p" in configs["generator"]["model_kwargs"], "generator config must contain 'top_p'"
    
    configs["generator"]["model_client"] = PROVIDER_NAME_TO_CLASS[configs["generator"]["provider"]]
                
                
def load_knowledge_config(configs: dict):
    # Process client classes
    
    class_name = configs["knowledge"]["embedder"]["client_class"]
    assert class_name in globals(), f"load_knowledge_config: {class_name} not in globals()  {globals()}"
    configs["knowledge"]["embedder"]["model_client"] = globals()[class_name]
    code_understanding_config = configs["knowledge"]["code_understanding"]
    # Get client class
    client_class = code_understanding_config.get("client_class")

    # Map client class to actual class
    model_client = globals().get(client_class)
    if not model_client:
        raise ValueError(f"Unknown client class: {client_class}")

    # Set model client
    code_understanding_config["model_client"] = model_client

def demonstrate_config_access(configs):
    """
    Demonstrate different ways to access configuration values.
    """
    # Access generator config
    if "generator" in configs:
        gen_cfg = configs["generator"]
        
        # Dictionary-style access
        if "providers" in gen_cfg and "dashscope" in gen_cfg["providers"]:
            dashscope_cfg = gen_cfg["providers"]["dashscope"]
            print(f"Dashscope default model: {dashscope_cfg.get('default_model', 'Not specified')}")
            print(f"Dashscope client class: {dashscope_cfg.get('client_class', 'Not specified')}")
        
        # Dot notation access
        if hasattr(gen_cfg, 'providers') and hasattr(gen_cfg.providers, 'openai'):
            openai_cfg = gen_cfg.providers.openai
            print(f"OpenAI default model: {openai_cfg.get('default_model', 'Not specified')}")
            if hasattr(openai_cfg, 'models') and hasattr(openai_cfg.models, 'gpt-4o'):
                gpt4o_cfg = getattr(openai_cfg.models, 'gpt-4o')
                print(f"GPT-4o temperature: {gpt4o_cfg.get('temperature', 'Not specified')}")
    
    # Access embedder config
    if "embedder" in configs:
        emb_cfg = configs["knowledge"]
        print("\n--- Embedder Config Access ---")
        print(f"Batch size: {emb_cfg.get('batch_size', 'Not specified')}")
        print(f"Model: {emb_cfg.get('model', 'Not specified')}")
        print(f"Dimensions: {emb_cfg.get('dimensions', 'Not specified')}")
    
    # Access repo config
    if "repo" in configs:
        repo_cfg = configs["repo"]
        print("\n--- Repository Config Access ---")
        print(f"Max size: {repo_cfg.get('max_size', 'Not specified')}")
        if "excluded_dirs" in repo_cfg:
            print(f"Excluded directories: {repo_cfg['excluded_dirs']}")
        if "code_extensions" in repo_cfg:
            print(f"Code extensions: {repo_cfg['code_extensions']}")
    
    # Access code understanding config
    if "code_understanding" in configs:
        code_cfg = configs["knowledge"]["code_understanding"]
        print("\n--- Code Understanding Config Access ---")
        if "retriever" in code_cfg:
            print(f"Retriever top K: {code_cfg['retriever'].get('top_k', 'Not specified')}")
        if "hybrid_search" in code_cfg:
            print(f"Hybrid search enabled: {code_cfg['hybrid_search'].get('enabled', 'Not specified')}")


def main():
    """
    Main function demonstrating different ways to load all configs.
    """
    print("Hydra All Configs Example")
    print("=" * 50)
    
    try:
        # Load all configurations
        all_configs = load_all_configs_programmatically()
        print(all_configs)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()