"""Kimi (月之暗面) ModelClient integration based on OpenAI client."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Union,
    Literal,
)

import adalflow as adal
from adalflow.core.types import (
    ModelType
)

from ragalyze.clients.openai_client import OpenAIClient, get_first_message_content
from ragalyze.logger.logging_config import get_tqdm_compatible_logger

log = get_tqdm_compatible_logger(__name__)


class KimiClient(OpenAIClient):
    """
    Kimi (月之暗面) API client that inherits from OpenAIClient.

    Kimi provides OpenAI-compatible API endpoints for text generation,
    so we can inherit most functionality while setting the appropriate base URL.

    Args:
        api_key (Optional[str], optional): Kimi API key. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://api.moonshot.cn/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "MOONSHOT_API_KEY".

    References:
        - Kimi API Documentation: https://platform.moonshot.cn/docs/api-reference
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Any], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "MOONSHOT_BASE_URL",
        env_api_key_name: str = "MOONSHOT_API_KEY",
        **kwargs,
    ):
        # Set default base URL for Kimi
        if base_url is None:
            base_url = "https://api.moonshot.cn/v1"

        # Call parent constructor with Kimi-specific settings
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_base_url_name=env_base_url_name,
            env_api_key_name=env_api_key_name,
            **kwargs,
        )

    def _prepare_client_config(self):
        """
        Prepare client configuration for Kimi.

        Returns:
            tuple: (api_key, base_url) for client initialization

        Raises:
            ValueError: If API key is not provided
        """
        # Use provided API key first, then environment variable
        api_key = self._api_key or os.getenv(self._env_api_key_name)

        if not api_key:
            raise ValueError(
                f"clients/kimi_client.py:Environment variable {self._env_api_key_name} must be set."
            )

        base_url = self.base_url
        return api_key, base_url

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert inputs to Kimi API kwargs.

        Kimi follows OpenAI API format for text generation.
        """
        # Only support LLM model type for Kimi (no embedding support)
        if model_type == ModelType.EMBEDDER:
            raise ValueError("KimiClient does not support embedding operations")

        # Use parent implementation for LLM operations
        api_kwargs = super().convert_inputs_to_api_kwargs(
            input=input, model_kwargs=model_kwargs, model_type=model_type
        )

        return api_kwargs