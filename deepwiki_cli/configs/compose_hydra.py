#!/usr/bin/env python3
"""
Simple example of loading all YAML configurations from the configs folder using Hydra.
This example works with the existing YAML files without requiring dataclass definitions.
"""

import os
from pathlib import Path
from hydra import compose, initialize_config_dir, initialize
from omegaconf import DictConfig, OmegaConf

#! Though the following imports are not directly used, they are stored in globals() and will be used implicitly. So DO NOT REMOVE THEM!!
from deepwiki_cli.clients.huggingface_embedder_client import (
    HuggingfaceClient,
    HuggingfaceEmbedder,
)
from deepwiki_cli.clients.dashscope_client import DashScopeClient, DashScopeEmbedder
from adalflow import GoogleGenAIClient


def load_default_config() -> DictConfig:
    """
    Load all YAML configurations programmatically using Hydra's compose API.
    """
    with initialize(config_path=".", version_base=None):
        return compose(config_name="main")

def load_all_configs(cfg: DictConfig = None) -> dict:
    if cfg is None:
        cfg = load_default_config()
    all_configs = OmegaConf.to_container(cfg, resolve=True)
    load_generator_config(all_configs)
    load_rag_config(all_configs)
    return all_configs


PROVIDER_NAME_TO_CLASS = {
    "google": GoogleGenAIClient,
    "dashscope": DashScopeClient,
}


def load_generator_config(configs: dict):
    # Add client classes to each provider
    assert (
        "provider" in configs["generator"]
    ), "generator config must contain 'provider'"
    assert "model" in configs["generator"], "generator config must contain 'model'"
    assert (
        "model_kwargs" in configs["generator"]
    ), "generator config must contain 'model_kwargs'"
    assert (
        "temperature" in configs["generator"]["model_kwargs"]
    ), "generator config must contain 'temperature'"
    assert (
        "top_p" in configs["generator"]["model_kwargs"]
    ), "generator config must contain 'top_p'"

    configs["generator"]["model_client"] = PROVIDER_NAME_TO_CLASS[
        configs["generator"]["provider"]
    ]


def load_rag_config(configs: dict):
    # Process client classes

    class_name = configs["rag"]["embedder"]["client_class"]
    assert (
        class_name in globals()
    ), f"load_rag_config: {class_name} not in globals()  {globals()}"
    configs["rag"]["embedder"]["model_client"] = globals()[class_name]
    code_understanding_config = configs["rag"]["code_understanding"]
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
