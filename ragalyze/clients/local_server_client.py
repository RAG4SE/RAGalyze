"""Local Server ModelClient integration, inheriting from OpenAIClient."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Literal,
)

from openai.types import (
    Completion,
)

from adalflow.core.component import DataComponent

from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from .openai_client import OpenAIClient, OpenAIEmbedder, OpenAIBatchEmbedder

log = get_tqdm_compatible_logger(__name__)


class LocalServerClient(OpenAIClient):
    """A component wrapper for local server API client, inheriting from OpenAIClient.

    This client is designed to work with local LLM servers that expose an OpenAI-compatible API.
    It inherits all functionality from OpenAIClient without any modifications.

    Args:
        api_key (Optional[str], optional): API key for the local server. Defaults to None.
        base_url (str): The API base URL for the local server.
                       Defaults to "http://localhost:8000/v1" (common default for local servers).
        env_api_key_name (str): Environment variable name for the API key. Defaults to "LOCAL_SERVER_API_KEY".
        env_base_url_name (str): Environment variable name for the base URL. Defaults to "LOCAL_SERVER_BASE_URL".
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "LOCAL_SERVER_BASE_URL",
        env_api_key_name: str = "LOCAL_SERVER_API_KEY",
        **kwargs,
    ):
        # Set default base URL for local server if not provided
        if base_url is None:
            base_url = os.getenv(env_base_url_name, "http://localhost:8000/v1")

        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_base_url_name=env_base_url_name,
            env_api_key_name=env_api_key_name,
            **kwargs,
        )


class LocalServerEmbedder(OpenAIEmbedder):
    """A user-facing component for local server embedder, inheriting from OpenAIEmbedder.

    This embedder is designed to work with local embedding models that expose an OpenAI-compatible API.
    It inherits all functionality from OpenAIEmbedder without any modifications.

    Args:
        model_client (ModelClient): The local server model client to use for the embedder.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}.
        output_processors (Optional[Component], optional): The output processors after model call. Defaults to None.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Dict[str, Any] = {},
        output_processors: Optional[DataComponent] = None,
    ) -> None:
        model_client = LocalServerClient(
            api_key=api_key,
            base_url=base_url,
        )
        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            output_processors=output_processors,
        )


class LocalServerBatchEmbedder(OpenAIBatchEmbedder):
    """Batch embedder specifically designed for local server API, inheriting from OpenAIBatchEmbedder.

    This batch embedder is designed to work with local embedding models that expose an OpenAI-compatible API.
    It inherits all functionality from OpenAIBatchEmbedder without any modifications.

    Args:
        embedder: The embedder to use for batch processing.
        batch_size (int): The batch size for processing. Defaults to 100.
    """

    def __init__(self, embedder, batch_size: int = 100) -> None:
        super().__init__(embedder=embedder, batch_size=batch_size)
