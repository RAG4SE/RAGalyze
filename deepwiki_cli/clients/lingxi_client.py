"""Lingxi (Ant Chat) ModelClient integration."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Literal,
)
# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import Stream
from openai.types import (
    Completion,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from adalflow.core.types import (
    ModelType,
)
from adalflow.core.component import DataComponent

from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from .openai_client import OpenAIClient, OpenAIEmbedder, OpenAIBatchEmbedder

log = get_tqdm_compatible_logger(__name__)

class LingxiClient(OpenAIClient):
    """A component wrapper for the Lingxi (Ant Chat) API client.

    Lingxi provides access to Ant Group's models through an OpenAI-compatible API.

    Args:
        api_key (Optional[str], optional): Lingxi API key. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://antchat.alipay.com/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "LINGXI_API_KEY".

    References:
        - Lingxi API Documentation: [TODO: Add link if available]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "LINGXI_BASE_URL",
        env_api_key_name: str = "LINGXI_API_KEY",
        **kwargs,
    ):
        # Set default base URL for Lingxi if not provided
        if base_url is None:
            base_url = os.getenv(env_base_url_name, "https://antchat.alipay.com/v1")
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_base_url_name=env_base_url_name,
            env_api_key_name=env_api_key_name,
            **kwargs,
        )


class LingxiEmbedder(OpenAIEmbedder):
    r"""
    A user-facing component that orchestrates an embedder model via the Lingxi model client and output processors.

    Args:
        model_client (ModelClient): The Lingxi model client to use for the embedder.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}.
        output_processors (Optional[Component], optional): The output processors after model call. Defaults to None.
    """

    model_type: ModelType = ModelType.EMBEDDER

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Dict[str, Any] = {},
        output_processors: Optional[DataComponent] = None,
    ) -> None:
        model_client = LingxiClient(
            api_key=api_key,
            base_url=base_url
        )
        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            output_processors=output_processors,
        )


class LingxiBatchEmbedder(OpenAIBatchEmbedder):
    """Batch embedder specifically designed for Lingxi API"""

    def __init__(self, embedder, batch_size: int = 100) -> None:
        super().__init__(embedder=embedder, batch_size=batch_size)