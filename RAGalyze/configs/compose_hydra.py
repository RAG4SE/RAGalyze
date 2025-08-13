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
from RAGalyze.clients.huggingface_embedder_client import HuggingfaceClient, HuggingfaceEmbedder
from RAGalyze.clients.dashscope_client import DashScopeClient, DashScopeEmbedder
from adalflow import GoogleGenAIClient

def load_all_configs():
    """
    Load all YAML configurations programmatically using Hydra's compose API.
    """
    
    # Get absolute path to configs directory
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(f"config_dir: {config_dir}")
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


def main():
    """
    Main function demonstrating different ways to load all configs.
    """
    print("Hydra All Configs Example")
    print("=" * 50)
    
    try:
        # Load all configurations
        all_configs = load_all_configs()
        print(all_configs)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()