"""Dashscope (Alibaba Cloud) ModelClient integration."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Union,
    Literal,
    List,
)

import adalflow as adal
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    EmbedderOutputType,
    EmbedderInputType,
)
from adalflow.core.component import DataComponent
from adalflow.core.embedder import (
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
)
import adalflow.core.functional as F
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from .openai_client import OpenAIClient

log = get_tqdm_compatible_logger(__name__)


class DashScopeClient(OpenAIClient):
    """
    A component wrapper for the Dashscope (Alibaba Cloud) API client.

    Dashscope provides access to Alibaba Cloud's Qwen and other models through an OpenAI-compatible API.
    This client inherits from OpenAIClient and overrides specific methods to handle DashScope's unique requirements.

    Args:
        api_key (Optional[str], optional): Dashscope API key. Defaults to None.
        workspace_id (Optional[str], optional): Dashscope workspace ID. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://dashscope.aliyuncs.com/compatible-mode/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "DASHSCOPE_API_KEY".
        env_workspace_id_name (str): Environment variable name for the workspace ID. Defaults to "DASHSCOPE_WORKSPACE_ID".

    References:
        - Dashscope API Documentation: https://help.aliyun.com/zh/dashscope/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "DASHSCOPE_BASE_URL",
        env_api_key_name: str = "DASHSCOPE_API_KEY",
        env_workspace_id_name: str = "DASHSCOPE_WORKSPACE_ID",
        **kwargs,
    ):
        # Store DashScope-specific attributes before calling parent constructor
        self._workspace_id = workspace_id
        self._env_workspace_id_name = env_workspace_id_name

        # Set default base URL for DashScope if not provided
        if base_url is None:
            base_url = os.getenv(
                env_base_url_name, "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        # Call parent constructor with DashScope-specific defaults
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
        Override parent method to include workspace_id configuration.

        Returns:
            tuple: (api_key, workspace_id, base_url) for client initialization

        Raises:
            ValueError: If API key is not provided
        """
        # Get base configuration from parent
        api_key, base_url = super()._prepare_client_config()

        # Add DashScope-specific workspace_id
        workspace_id = self._workspace_id or os.getenv(self._env_workspace_id_name)

        if not workspace_id:
            log.warning(
                f"Environment variable {self._env_workspace_id_name} not set. Some features may not work properly."
            )

        return api_key, workspace_id, base_url

    def init_sync_client(self):
        """Override to handle workspace_id storage."""
        api_key, workspace_id, base_url = self._prepare_client_config()

        # Use OpenAI client with DashScope base URL
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Store workspace_id for later use in requests
        if workspace_id:
            client._workspace_id = workspace_id

        return client

    def init_async_client(self):
        """Override to handle workspace_id storage."""
        api_key, workspace_id, base_url = self._prepare_client_config()

        # Use AsyncOpenAI client with DashScope base URL
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Store workspace_id for later use in requests
        if workspace_id:
            client._workspace_id = workspace_id

        return client

    def _add_dashscope_headers(self, api_kwargs: Dict) -> Dict:
        """
        Helper method to add DashScope-specific headers to API kwargs.

        Args:
            api_kwargs: The API keyword arguments to modify

        Returns:
            Modified API kwargs with DashScope headers
        """
        # Add workspace ID to headers if available
        workspace_id = getattr(self.sync_client, "_workspace_id", None) or getattr(
            self.async_client, "_workspace_id", None
        )
        if workspace_id:
            if "extra_headers" not in api_kwargs:
                api_kwargs["extra_headers"] = {}
            api_kwargs["extra_headers"]["X-DashScope-WorkSpace"] = workspace_id

        return api_kwargs

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Override parent method to add DashScope-specific headers.

        Args:
            input: The input data
            model_kwargs: Model parameters
            model_type: Type of model (LLM or EMBEDDER)

        Returns:
            API kwargs with DashScope-specific modifications
        """
        # Get base API kwargs from parent
        api_kwargs = super().convert_inputs_to_api_kwargs(
            input, model_kwargs, model_type
        )

        # Add DashScope-specific headers
        api_kwargs = self._add_dashscope_headers(api_kwargs)

        return api_kwargs

    def chat(self, api_kwargs: Dict = {}):
        """
        Override parent method to add DashScope-specific parameters.

        Args:
            api_kwargs: API keyword arguments

        Returns:
            Chat completion response
        """
        # For non-streaming, enable_thinking must be false.
        # Pass it via extra_body to avoid TypeError from openai client validation.
        if not api_kwargs.get("stream", False):
            extra_body = api_kwargs.get("extra_body", {})
            extra_body["enable_thinking"] = False
            api_kwargs["extra_body"] = extra_body

        return super().chat(api_kwargs)

    async def achat(self, api_kwargs: Dict = {}):
        """
        Override parent method to add DashScope-specific parameters.

        Args:
            api_kwargs: API keyword arguments

        Returns:
            Async chat completion response
        """
        # For non-streaming, enable_thinking must be false.
        # Pass it via extra_body to avoid TypeError from openai client validation.
        if not api_kwargs.get("stream", False):
            extra_body = api_kwargs.get("extra_body", {})
            extra_body["enable_thinking"] = False
            api_kwargs["extra_body"] = extra_body

        return await super().achat(api_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Override parent method to include workspace_id.

        Returns:
            Dictionary representation including DashScope-specific fields
        """
        result = super().to_dict()
        result["workspace_id"] = self._workspace_id
        return result

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """
        Override parent method to provide DashScope-specific error messaging.

        Args:
            response: The embedding response from DashScope API

        Returns:
            Parsed embedding output
        """
        try:
            return super().parse_embedding_response(response)
        except Exception as e:
            log.error(f"🔍 Error parsing DashScope embedding response: {e}")
            log.error(f"🔍 Raw response details: {repr(response)}")
            raise


class DashScopeEmbedder(adal.Embedder):
    """
    A user-facing component that orchestrates an embedder model via the DashScope model client and output processors.

    Args:
        model_client (ModelClient): The DashScope model client to use for the embedder.
        model_kwargs (Dict[str, Any], optional): The model kwargs to pass to the model client. Defaults to {}.
        output_processors (Optional[Component], optional): The output processors after model call. Defaults to None.
    """

    model_type: ModelType = ModelType.EMBEDDER

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Dict[str, Any] = {},
        model_client: Optional[DashScopeClient] = None,
        output_processors: Optional[DataComponent] = None,
    ) -> None:
        if model_client:
            super().__init__(
                model_client=model_client,
                model_kwargs=model_kwargs,
                output_processors=output_processors,
            )
        else:
            # Create model client with provided parameters
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if workspace_id:
                client_kwargs["workspace_id"] = workspace_id
            if base_url:
                client_kwargs["base_url"] = base_url

            super().__init__(
                model_client=DashScopeClient(**client_kwargs),
                model_kwargs=model_kwargs,
                output_processors=output_processors,
            )

        if not isinstance(model_kwargs, Dict):
            raise TypeError(
                f"clients/dashscope_client.py:{type(self).__name__} requires a dictionary for model_kwargs, not a string"
            )

    def call(
        self,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        """Call the DashScope embedder with input."""
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=self._compose_model_kwargs(**model_kwargs),
            model_type=self.model_type,
        )
        output = self.model_client.call(
            api_kwargs=api_kwargs, model_type=self.model_type
        )

        return output

    async def acall(
        self,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        """Async call to the DashScope embedder."""
        log.debug(f"Calling {self.__class__.__name__} with input: {input}")
        api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
            input=input,
            model_kwargs=self._compose_model_kwargs(**model_kwargs),
            model_type=self.model_type,
        )
        output: EmbedderOutputType = None
        try:
            response = await self.model_client.acall(
                api_kwargs=api_kwargs, model_type=self.model_type
            )
            output = self.model_client.parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error calling the DashScope model: {e}")
            output = EmbedderOutput(error=str(e))
            raise

        output.input = [input] if isinstance(input, str) else input
        log.debug(f"Output from {self.__class__.__name__}: {output}")
        return output

    def _compose_model_kwargs(self, **model_kwargs) -> Dict[str, object]:
        """Compose model kwargs with defaults."""
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)


# Batch Embedding Components for DashScope
class DashScopeBatchEmbedder(adal.BatchEmbedder):
    """
    Batch embedder specifically designed for DashScope API.

    DashScope has a smaller batch size limit compared to other providers.
    """

    def __init__(self, embedder, batch_size: int = 100) -> None:
        super().__init__(embedder=embedder, batch_size=batch_size)
        if self.batch_size > 10:
            log.warning(
                f"DashScope batch embedder initialization, batch size: {self.batch_size}, "
                f"note that DashScope batch embedding size cannot exceed 25, automatically set to 10"
            )
            self.batch_size = 10

    def _process_batch_with_retry(
        self, batch_input: List[str], model_kwargs: Optional[Dict] = {}
    ) -> EmbedderOutput:
        """
        Process a batch with recursive splitting if it exceeds token limits.

        Args:
            batch_input: List of input texts
            model_kwargs: Model parameters

        Returns:
            EmbedderOutput with results or error
        """
        try:
            # Try to process the batch normally
            return self.embedder(input=batch_input, model_kwargs=model_kwargs)
        except Exception as e:
            # Check if it's a token limit error
            error_str = str(e).lower()
            if "range of input length should be" in error_str:
                # If the batch has only one item, we can't split further
                if len(batch_input) <= 1:
                    log.error(
                        f"Single item exceeds token limit: {batch_input[0] if batch_input else 'empty'}"
                    )
                    raise e

                # Split the batch in half and recursively process each half
                mid = len(batch_input) // 2
                first_half = batch_input[:mid]
                second_half = batch_input[mid:]

                log.warning(
                    f"Batch of {len(batch_input)} items failed due to token limit. "
                    f"Splitting into two batches of {len(first_half)} and {len(second_half)} items."
                )

                # Process each half
                first_result = self._process_batch_with_retry(first_half, model_kwargs)
                second_result = self._process_batch_with_retry(
                    second_half, model_kwargs
                )

                # Combine results
                combined_data = []
                combined_error = None

                # Add data from first half
                if first_result.data:
                    combined_data.extend(first_result.data)
                if first_result.error:
                    combined_error = f"First half error: {first_result.error}"

                # Add data from second half
                if second_result.data:
                    combined_data.extend(second_result.data)
                if second_result.error:
                    combined_error = (
                        f"{combined_error}; Second half error: {second_result.error}"
                        if combined_error
                        else f"Second half error: {second_result.error}"
                    )

                # Create combined result
                combined_result = EmbedderOutput(
                    data=combined_data, error=combined_error, raw_response=None
                )
                return combined_result
            else:
                # Re-raise if it's not a token limit error
                raise e

    def call(
        self, input: BatchEmbedderInputType, model_kwargs: Optional[Dict] = {}
    ) -> BatchEmbedderOutputType:
        """
        Batch call to DashScope embedder.

        Args:
            input: List of input texts
            model_kwargs: Model parameters

        Returns:
            Batch embedding output
        """
        if isinstance(input, str):
            input = [input]

        n = len(input)
        embeddings: List[EmbedderOutput] = []

        log.info(
            f"Starting DashScope batch embedding processing, total {n} texts, batch size: {self.batch_size}"
        )

        for i in tqdm(
            range(0, n, self.batch_size),
            desc="DashScope batch embedding",
            disable=False,
        ):
            batch_input = input[i : min(i + self.batch_size, n)]

            try:
                # Use the retry mechanism with recursive splitting
                batch_output = self._process_batch_with_retry(batch_input, model_kwargs)
                embeddings.append(batch_output)

                # Validate batch output
                if batch_output.error:
                    log.error(
                        f"Batch {i//self.batch_size + 1} embedding failed: {batch_output.error}"
                    )
                elif batch_output.data:
                    log.debug(
                        f"Batch {i//self.batch_size + 1} successfully generated {len(batch_output.data)} embedding vectors"
                    )
                else:
                    log.warning(
                        f"Batch {i//self.batch_size + 1} returned no embedding data"
                    )

            except Exception as e:
                log.error(f"Batch {i//self.batch_size + 1} processing exception: {e}")
                # Create error embedding output
                error_output = EmbedderOutput(data=[], error=str(e), raw_response=None)
                embeddings.append(error_output)
                raise

        log.info(
            f"DashScope batch embedding completed, processed {len(embeddings)} batches"
        )

        return embeddings

    def __call__(
        self,
        input: BatchEmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> BatchEmbedderOutputType:
        """
        Call operator interface, delegates to call method.
        """
        return self.call(input=input, model_kwargs=model_kwargs)
