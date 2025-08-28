from .compose_hydra import load_all_configs, load_default_config, configs

import adalflow as adal


def get_batch_embedder() -> adal.BatchEmbedder:
    embedder_config = configs()["rag"]["embedder"]
    model_client_class = embedder_config["model_client"]
    batch_model_client_class = embedder_config["batch_model_client"]
    model_kwargs = embedder_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "model" not in model_kwargs:
        assert "model" in embedder_config, "embedder_config must contain model"
        model_kwargs["model"] = embedder_config["model"]
    
    # Handle API key
    api_key = embedder_config.get("api_key", "")
    base_url = embedder_config.get("base_url", "")
    client_kwargs = {"model_kwargs": model_kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    
    model_client = model_client_class(**client_kwargs)
    batch_model_client = batch_model_client_class(embedder=model_client, batch_size=embedder_config["batch_size"])
    return batch_model_client


def get_embedder() -> adal.Embedder:
    embedder_config = configs()["rag"]["embedder"]
    model_client_class = embedder_config["model_client"]
    model_kwargs = embedder_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "model" not in model_kwargs:
        assert "model" in embedder_config, "embedder_config must contain model"
        model_kwargs["model"] = embedder_config["model"]
    
    # Handle API key for embedder
    api_key = embedder_config.get("api_key", "")
    base_url = embedder_config.get("base_url", "")
    client_kwargs = {"model_kwargs": model_kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    model_client = model_client_class(**client_kwargs)
    return model_client

#! WARNING: under-test and unused function, may contain bugs
def get_generator() -> adal.Generator:
    generator_config = configs()["generator"]
    model_client_class = generator_config["model_client"]
    model_kwargs = generator_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "model" not in model_kwargs:
        assert "model" in generator_config, "generator_config must contain model"
        model_kwargs["model"] = generator_config["model"]
    
    # Handle API key for generator
    api_key = generator_config.get("api_key", "")
    base_url = generator_config.get("base_url", "")
    client_kwargs = {"model_kwargs": model_kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    model_client = model_client_class(**client_kwargs)
    return adal.Generator(model_client=model_client, model_kwargs=model_kwargs)

def get_code_understanding_client():
    code_understanding_config = configs()["rag"]["code_understanding"]
    model_client_class = code_understanding_config["model_client"]
    model_kwargs = code_understanding_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "model" not in model_kwargs:
        assert "model" in code_understanding_config, "code_understanding_config must contain model"
        model_kwargs["model"] = code_understanding_config["model"]
    
    # Handle API key for code understanding
    api_key = code_understanding_config.get("api_key", "")
    base_url = code_understanding_config.get("base_url", "")
    client_kwargs = {"model_kwargs": model_kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    model_client = model_client_class(**client_kwargs)
    return model_client