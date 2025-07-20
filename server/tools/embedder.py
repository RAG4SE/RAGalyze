import adalflow as adal

from server.config import configs

def get_embedder(is_huggingface_embedder: bool = False) -> adal.Embedder:
    embedder_config = configs["embedder"]

    # --- Initialize Embedder ---
    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    if is_huggingface_embedder:
        from server.huggingface_embedder_client import HuggingfaceEmbedder
        embedder = HuggingfaceEmbedder(
            model_client=model_client,
            model_kwargs=embedder_config["model_kwargs"],
        )
        return embedder
    else:
        from server.dashscope_client import DashScopeEmbedder
        embedder = DashScopeEmbedder(
            model_client=model_client,
            model_kwargs=embedder_config["model_kwargs"],
        )
        return embedder