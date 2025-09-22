"""ModelScope ModelClient integration."""

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


class ModelScopeClient(OpenAIClient):
    """
    A component wrapper for the ModelScope API client.

    ModelScope provides access to various models through an OpenAI-compatible API.
    This client inherits from OpenAIClient and provides ModelScope-specific configurations.

    Args:
        api_key (Optional[str], optional): ModelScope API key. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://api-inference.modelscope.cn/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "MODELSCOPE_API_KEY".

    References:
        - ModelScope API Documentation: https://modelscope.cn/docs/api-inference/overview
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "MODELSCOPE_BASE_URL",
        env_api_key_name: str = "MODELSCOPE_API_KEY",
        **kwargs,
    ):
        # Set default base URL for ModelScope if not provided
        if base_url is None:
            base_url = os.getenv(
                env_base_url_name, "https://api-inference.modelscope.cn/v1"
            )

        # Call parent constructor with ModelScope-specific defaults
        super().__init__(
            api_key=api_key,
            chat_completion_parser=chat_completion_parser,
            input_type=input_type,
            base_url=base_url,
            env_base_url_name=env_base_url_name,
            env_api_key_name=env_api_key_name,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Override parent method to include ModelScope-specific fields.

        Returns:
            Dictionary representation including ModelScope-specific fields
        """
        result = super().to_dict()
        return result

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """
        Override parent method to provide ModelScope-specific error messaging.

        Args:
            response: The embedding response from ModelScope API

        Returns:
            Parsed embedding output
        """
        try:
            return super().parse_embedding_response(response)
        except Exception as e:
            log.error(f"🔍 Error parsing ModelScope embedding response: {e}")
            log.error(f"🔍 Raw response details: {repr(response)}")
            raise


class ModelScopeEmbedder(adal.Embedder):
    """
    A user-facing component that orchestrates an embedder model via the ModelScope model client and output processors.

    Args:
        model_client (ModelClient): The ModelScope model client to use for the embedder.
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
        model_client: Optional[ModelScopeClient] = None,
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
            if base_url:
                client_kwargs["base_url"] = base_url

            super().__init__(
                model_client=ModelScopeClient(**client_kwargs),
                model_kwargs=model_kwargs,
                output_processors=output_processors,
            )

        if not isinstance(model_kwargs, Dict):
            raise TypeError(
                f"clients/modelscope_client.py:{type(self).__name__} requires a dictionary for model_kwargs, not a string"
            )

    def call(
        self,
        input: EmbedderInputType,
        model_kwargs: Optional[Dict] = {},
    ) -> EmbedderOutputType:
        """Call the ModelScope embedder with input."""
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
        """Async call to the ModelScope embedder."""
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
            log.error(f"Error calling the ModelScope model: {e}")
            output = EmbedderOutput(error=str(e))
            raise

        output.input = [input] if isinstance(input, str) else input
        log.debug(f"Output from {self.__class__.__name__}: {output}")
        return output

    def _compose_model_kwargs(self, **model_kwargs) -> Dict[str, object]:
        """Compose model kwargs with defaults."""
        return F.compose_model_kwargs(self.model_kwargs, model_kwargs)


# Batch Embedding Components for ModelScope
class ModelScopeBatchEmbedder(adal.BatchEmbedder):
    """
    Batch embedder specifically designed for ModelScope API.
    """

    def __init__(self, embedder, batch_size: int = 100) -> None:
        super().__init__(embedder=embedder, batch_size=batch_size)
        # ModelScope may have specific batch size limits
        if self.batch_size > 100:
            log.warning(
                f"ModelScope batch embedder initialization, batch size: {self.batch_size}, "
                f"note that ModelScope batch embedding size might be limited, automatically set to 100"
            )
            self.batch_size = 100

    def call(
        self, input: BatchEmbedderInputType, model_kwargs: Optional[Dict] = {}
    ) -> BatchEmbedderOutputType:
        """
        Batch call to ModelScope embedder.

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
            f"Starting ModelScope batch embedding processing, total {n} texts, batch size: {self.batch_size}"
        )

        for i in tqdm(
            range(0, n, self.batch_size),
            desc="ModelScope batch embedding",
            disable=False,
        ):
            batch_input = input[i : min(i + self.batch_size, n)]

            try:
                batch_output = self.embedder(
                    input=batch_input, model_kwargs=model_kwargs
                )
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
            f"ModelScope batch embedding completed, processed {len(embeddings)} batches"
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