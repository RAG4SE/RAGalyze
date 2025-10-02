"""DeepSeek ModelClient integration based on OpenAI client."""

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


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek API client that inherits from OpenAIClient.

    DeepSeek provides OpenAI-compatible API endpoints for text generation,
    so we can inherit most functionality while setting the appropriate base URL.

    Args:
        api_key (Optional[str], optional): DeepSeek API key. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://api.deepseek.com".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "DEEPSEEK_API_KEY".

    References:
        - DeepSeek API Documentation: https://api-docs.deepseek.com/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Any], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "DEEPSEEK_BASE_URL",
        env_api_key_name: str = "DEEPSEEK_API_KEY",
        **kwargs,
    ):
        # Set default base URL for DeepSeek
        if base_url is None:
            base_url = os.getenv(env_base_url_name, "https://api.deepseek.com")

        # Call parent constructor with DeepSeek-specific settings
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
        Prepare client configuration for DeepSeek.

        Returns:
            tuple: (api_key, base_url) for client initialization

        Raises:
            ValueError: If API key is not provided
        """
        # Use provided API key first, then environment variable
        api_key = self._api_key or os.getenv(self._env_api_key_name)

        if not api_key:
            raise ValueError(
                f"clients/deepseek_client.py:Environment variable {self._env_api_key_name} must be set."
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
        Convert inputs to DeepSeek API kwargs.

        DeepSeek follows OpenAI API format for text generation.
        """
        # Only support LLM model type for DeepSeek (no embedding support)
        if model_type == ModelType.EMBEDDER:
            raise ValueError("DeepSeekClient does not support embedding operations")

        # Use parent implementation for LLM operations
        api_kwargs = super().convert_inputs_to_api_kwargs(
            input=input, model_kwargs=model_kwargs, model_type=model_type
        )

        # DeepSeek-specific modifications can be added here if needed
        # For now, the default OpenAI format works fine

        return api_kwargs