from .compose_hydra import load_all_configs, load_default_config, configs

import adalflow as adal


def get_embedder() -> adal.Embedder:
    embedder_config = configs()["rag"]["embedder"]
    model_client_class = embedder_config["model_client"]
    model_kwargs = embedder_config[
        "model_kwargs"
    ].copy()  # Create a copy to avoid modifying original
    if "batch_size" in model_kwargs:
        del model_kwargs["batch_size"]
    if "model" not in model_kwargs:
        assert "model" in embedder_config, "embedder_config must contain model"
        model_kwargs["model"] = embedder_config["model"]
    model_client = model_client_class(model_kwargs=model_kwargs)
    return model_client
