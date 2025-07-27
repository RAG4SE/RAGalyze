import adalflow as adal

from core.config import configs

SUPPORT_EMBEDDER = [
    "HuggingfaceEmbedder",
    "DashScopeEmbedder"
]

def get_embedder() -> adal.Embedder:
    embedder_config = configs["embedder"]
    model_client_class = embedder_config["model_client"]
    model_client = model_client_class(model_kwargs=embedder_config["model_kwargs"])
    return model_client