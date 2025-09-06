#!/usr/bin/env python3
"""
Simple example of loading all YAML configurations from the configs folder using Hydra.
This example works with the existing YAML files without requiring dataclass definitions.
"""

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

#! Though the following imports are not directly used, they are stored in globals() and will be used implicitly. So DO NOT REMOVE THEM!!
from ragalyze.clients import *
from adalflow import GoogleGenAIClient
from ragalyze.logger.logging_config import get_tqdm_compatible_logger

global_configs = None
# Setup logging
logger = get_tqdm_compatible_logger(__name__)


def load_default_config() -> DictConfig:
    """
    Load all YAML configurations programmatically using Hydra's compose API.
    """
    with initialize(config_path=".", version_base=None):
        return compose(config_name="main")


def set_global_configs(cfg: DictConfig = None):
    if cfg is None:
        cfg = load_default_config()
    all_configs = OmegaConf.to_container(cfg, resolve=True)
    load_generator_config(all_configs)
    load_rag_config(all_configs)
    global global_configs
    global_configs = all_configs

def set_global_config_value(key: str, value):
    """
    Set a specific configuration value in the global configs.
    Supports nested keys using dot notation (e.g., 'rag.retriever.top_k').
    """
    global global_configs
    if global_configs is None:
        raise ValueError(
            "Global configs not initialized. Call set_global_configs() first."
        )
    
    keys = key.split('.')
    current = global_configs
    
    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Set the final value
    current[keys[-1]] = value
    logger.debug(f"Set config {key} = {value}")


def configs():
    if global_configs is None:
        raise ValueError(
            "May use global_configs before loading all configs. Probably ragalyze has incorrect initialization."
        )
    return global_configs


PROVIDER_NAME_TO_CLASS = {
    "google": GoogleGenAIClient,
    "dashscope": DashScopeClient,
    "lingxi": LingxiClient,
    "local_server": LocalServerClient,
    "openai": OpenAIClient,
}

# Define provider to embedder class name mappings
EMBEDDER_PROVIDER_TO_CLASS_NAMES = {
    "dashscope": ("DashScopeEmbedder", "DashScopeBatchEmbedder"),
    "huggingface": ("HuggingfaceEmbedder", "HuggingfaceBatchEmbedder"),
    "lingxi": ("LingxiEmbedder", "LingxiBatchEmbedder"),
    "local_server": ("LocalServerEmbedder", "LocalServerBatchEmbedder"),
    "openai": ("OpenAIEmbedder", "OpenAIBatchEmbedder"),
}

# Define provider to code understanding client class name mappings
CODE_UNDERSTANDING_PROVIDER_TO_CLASS_NAME = {
    "dashscope": "DashScopeClient",
    "huggingface": "HuggingfaceClient",
    "lingxi": "LingxiClient",
    "local_server": "LocalServerClient",
    "openai": "OpenAIClient",
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
    # Process embedder client classes
    embedder_provider = configs["rag"]["embedder"]["provider"]
    if embedder_provider not in EMBEDDER_PROVIDER_TO_CLASS_NAMES:
        raise ValueError(f"Unknown embedder provider: {embedder_provider}")

    embedder_class_name, batch_embedder_class_name = EMBEDDER_PROVIDER_TO_CLASS_NAMES[
        embedder_provider
    ]

    assert (
        batch_embedder_class_name in globals() and embedder_class_name in globals()
    ), f"load_rag_config: {batch_embedder_class_name} or {embedder_class_name} not in globals()  {globals()}"
    configs["rag"]["embedder"]["model_client"] = globals()[embedder_class_name]
    configs["rag"]["embedder"]["batch_model_client"] = globals()[
        batch_embedder_class_name
    ]

    # Handle base_url for embedder if present in config
    if (
        "base_url" in configs["rag"]["embedder"]
        and configs["rag"]["embedder"]["base_url"]
    ):
        configs["rag"]["embedder"]["base_url"] = configs["rag"]["embedder"]["base_url"]

    # Process code understanding client class
    code_understanding_config = configs["rag"]["code_understanding"]
    code_understanding_provider = code_understanding_config["provider"]
    if code_understanding_provider not in CODE_UNDERSTANDING_PROVIDER_TO_CLASS_NAME:
        raise ValueError(
            f"Unknown code understanding provider: {code_understanding_provider}"
        )

    class_name = CODE_UNDERSTANDING_PROVIDER_TO_CLASS_NAME[code_understanding_provider]
    model_client = globals().get(class_name)
    if not model_client:
        raise ValueError(f"Unknown client class: {class_name}")

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
        all_configs = set_global_configs()
        print(all_configs)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
