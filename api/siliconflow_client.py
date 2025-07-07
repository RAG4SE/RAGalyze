"""SiliconFlow ModelClient integration."""

import os
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Generator,
    Union,
    Literal,
)

import logging
import backoff

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    CompletionUsage,
    GeneratorOutput,
)
from adalflow.components.model_client.utils import parse_embedding_response

log = logging.getLogger(__name__)


def get_first_message_content(completion: ChatCompletion) -> str:
    """When we only need the content of the first message."""
    log.debug(f"raw completion: {completion}")
    return completion.choices[0].message.content


def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """Parse the response of the stream API."""
    return completion.choices[0].delta.content


def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """Handle the streaming response."""
    for completion in generator:
        log.debug(f"Raw chunk completion: {completion}")
        parsed_content = parse_stream_response(completion)
        yield parsed_content


class SiliconFlowClient(ModelClient):
    """A component wrapper for the SiliconFlow API client.

    SiliconFlow provides access to various AI models through an OpenAI-compatible API.
    
    Args:
        api_key (Optional[str], optional): SiliconFlow API key. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://api.siliconflow.cn/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "SILICONFLOW_API_KEY".

    References:
        - SiliconFlow API Documentation: https://docs.siliconflow.cn/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "SILICONFLOW_BASE_URL",
        env_api_key_name: str = "SILICONFLOW_API_KEY",
    ):
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._env_base_url_name = env_base_url_name
        self.base_url = base_url or os.getenv(self._env_base_url_name, "https://api.siliconflow.cn/v1")
        self.sync_client = self.init_sync_client()
        self.async_client = None
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )
        self._input_type = input_type
        self._api_kwargs = {}

    def init_sync_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """Parse the completion response to a GeneratorOutput."""
        try:
            if isinstance(completion, ChatCompletion):
                return GeneratorOutput(
                    data=self.chat_completion_parser(completion),
                    usage=CompletionUsage(
                        completion_tokens=completion.usage.completion_tokens,
                        prompt_tokens=completion.usage.prompt_tokens,
                        total_tokens=completion.usage.total_tokens,
                    ),
                    raw_response=str(completion),
                )
            else:
                # Handle streaming response
                def generator_with_usage():
                    content_parts = []
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            content_parts.append(chunk.choices[0].delta.content)
                            yield chunk.choices[0].delta.content
                    
                return GeneratorOutput(data=generator_with_usage(), raw_response="streaming")
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            raise

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """Track the completion usage."""
        if isinstance(completion, ChatCompletion):
            return CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        else:
            # For streaming, we can't track usage accurately
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        """Parse the embedding response to a EmbedderOutput."""
        return parse_embedding_response(response)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to API kwargs."""
        final_model_kwargs = model_kwargs.copy()
        
        if model_type == ModelType.LLM:
            messages = []
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, list):
                messages = input
            else:
                raise ValueError(f"Unsupported input type: {type(input)}")
            
            api_kwargs = {
                "messages": messages,
                **final_model_kwargs
            }
            return api_kwargs
            
        elif model_type == ModelType.EMBEDDER:
            api_kwargs = {
                "input": input,
                **final_model_kwargs
            }
            return api_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call the SiliconFlow API."""
        if model_type == ModelType.LLM:
            if api_kwargs.get("stream", False):
                completion = self.sync_client.chat.completions.create(**api_kwargs)
                return handle_streaming_response(completion)
            else:
                completion = self.sync_client.chat.completions.create(**api_kwargs)
                return self.parse_chat_completion(completion)
        elif model_type == ModelType.EMBEDDER:
            response = self.sync_client.embeddings.create(**api_kwargs)
            return self.parse_embedding_response(response)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """Async call to the SiliconFlow API."""
        if not self.async_client:
            self.async_client = self.init_async_client()

        if model_type == ModelType.LLM:
            if api_kwargs.get("stream", False):
                completion = await self.async_client.chat.completions.create(**api_kwargs)
                return handle_streaming_response(completion)
            else:
                completion = await self.async_client.chat.completions.create(**api_kwargs)
                return self.parse_chat_completion(completion)
        elif model_type == ModelType.EMBEDDER:
            response = await self.async_client.embeddings.create(**api_kwargs)
            return self.parse_embedding_response(response)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an instance from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_key": self._api_key,
            "base_url": self.base_url,
        } 