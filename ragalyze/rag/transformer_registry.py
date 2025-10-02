"""
Embedder Transformer Registry

Provides a registry pattern to dynamically select and create appropriate embedder transformers,
replacing the original if-else judgment logic.
"""

from typing import Dict, Type, Callable, Any, Optional
from abc import ABC, abstractmethod

import adalflow as adal
from adalflow.core.db import EntityMapping
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.rag.embedding_transformer import (
    ToEmbeddings,
    DashScopeToEmbeddings,
    HuggingfaceToEmbeddings,
    OpenAIToEmbeddings,
    DualVectorToEmbeddings,
)
from ragalyze.rag.code_understanding import CodeUnderstandingGenerator
from ragalyze.configs import get_batch_embedder, configs
from ragalyze.rag.dynamic_splitter_transformer import DynamicSplitterTransformer
from ragalyze.rag.splitter import MyTextSplitter
from ragalyze.rag.bm25_transformer import BM25Transformer

# Register MyTextSplitter with EntityMapping to enable proper deserialization
EntityMapping.register('MyTextSplitter', MyTextSplitter)
EntityMapping.register('BM25Transformer', BM25Transformer)

# Register all custom transformer classes for proper deserialization
from ragalyze.rag.dynamic_splitter_transformer import DynamicSplitterTransformer
from ragalyze.rag.embedding_transformer import (
    ToEmbeddings,
    DashScopeToEmbeddings,
    HuggingfaceToEmbeddings,
    OpenAIToEmbeddings,
    DualVectorToEmbeddings,
)

EntityMapping.register('DynamicSplitterTransformer', DynamicSplitterTransformer)
EntityMapping.register('ToEmbeddings', ToEmbeddings)
EntityMapping.register('DashScopeToEmbeddings', DashScopeToEmbeddings)
EntityMapping.register('HuggingfaceToEmbeddings', HuggingfaceToEmbeddings)
EntityMapping.register('OpenAIToEmbeddings', OpenAIToEmbeddings)
EntityMapping.register('DualVectorToEmbeddings', DualVectorToEmbeddings)

logger = get_tqdm_compatible_logger(__name__)


class EmbedderTransformerFactory(ABC):
    """Abstract base class for transformer factories"""

    @abstractmethod
    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> ToEmbeddings:
        """Create transformer instance"""
        pass

    @abstractmethod
    def can_handle(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> bool:
        """Determine if this factory can handle the given embedder type"""
        pass


class DualVectorEmbedderFactory(EmbedderTransformerFactory):
    """Factory for dual-vector transformers"""

    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> DualVectorToEmbeddings:
        code_understanding_generator = kwargs.get("code_understanding_generator")
        if not code_understanding_generator:
            raise ValueError(
                "DualVectorToEmbeddings requires code_understanding_generator"
            )

        return DualVectorToEmbeddings(
            embedder=embedder, generator=code_understanding_generator
        )

    def can_handle(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> bool:
        return kwargs.get("use_dual_vector", False)


class HuggingfaceEmbedderFactory(EmbedderTransformerFactory):
    """Factory for HuggingFace transformers"""

    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> HuggingfaceToEmbeddings:
        return HuggingfaceToEmbeddings(embedder=embedder)

    def can_handle(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> bool:
        if kwargs.get("use_dual_vector", False):
            return False
        embedder_class_name = embedder.__class__.__name__
        return embedder_class_name in [
            "HuggingfaceBatchEmbedder",
            "HuggingfaceEmbedder",
        ]


class DashScopeEmbedderFactory(EmbedderTransformerFactory):
    """Factory for DashScope transformers"""

    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> DashScopeToEmbeddings:
        return DashScopeToEmbeddings(embedder=embedder)

    def can_handle(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> bool:
        if kwargs.get("use_dual_vector", False):
            return False
        embedder_class_name = embedder.__class__.__name__
        return embedder_class_name in ["DashScopeBatchEmbedder", "DashScopeEmbedder"]


class OpenAIEmbedderFactory(EmbedderTransformerFactory):
    """Factory for OpenAI transformers"""

    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> OpenAIToEmbeddings:
        return OpenAIToEmbeddings(embedder=embedder)

    def can_handle(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> bool:
        if kwargs.get("use_dual_vector", False):
            return False
        embedder_class_name = embedder.__class__.__name__
        return embedder_class_name in ["OpenAIBatchEmbedder", "OpenAIEmbedder"]

class LocalServerEmbedderFactory(EmbedderTransformerFactory):
    """Factory for OpenAI transformers"""

    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> OpenAIToEmbeddings:
        return OpenAIToEmbeddings(embedder=embedder)

    def can_handle(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> bool:
        if kwargs.get("use_dual_vector", False):
            return False
        embedder_class_name = embedder.__class__.__name__
        return embedder_class_name in ["LocalServerBatchEmbedder", "LocalServerEmbedder"]


class EmbedderTransformerRegistry:
    """
    Transformer Registry

    Used to register and select appropriate embedder transformer factories
    """

    def __init__(self):
        self._factories: list[EmbedderTransformerFactory] = []
        self._register_default_factories()

    def _register_default_factories(self):
        """Register default factory classes"""
        # Note: DualVector factory has the highest priority as it may override other conditions
        self.register_factory(DualVectorEmbedderFactory())
        self.register_factory(HuggingfaceEmbedderFactory())
        self.register_factory(DashScopeEmbedderFactory())
        self.register_factory(OpenAIEmbedderFactory())
        self.register_factory(LocalServerEmbedderFactory())

    def register_factory(self, factory: EmbedderTransformerFactory):
        """Register a new factory"""
        if not isinstance(factory, EmbedderTransformerFactory):
            raise TypeError("Factory must be an instance of EmbedderTransformerFactory")
        self._factories.append(factory)
        logger.debug(f"Registered transformer factory: {factory.__class__.__name__}")

    def create_transformer(
        self, embedder: adal.Embedder | adal.BatchEmbedder, **kwargs
    ) -> ToEmbeddings:
        """
        Create an appropriate transformer

        Args:
            embedder: embedder instance
            **kwargs: additional parameters such as use_dual_vector, code_understanding_generator, etc.

        Returns:
            ToEmbeddings: appropriate transformer instance

        Raises:
            ValueError: if no suitable factory is found
        """
        for factory in self._factories:
            if factory.can_handle(embedder, **kwargs):
                transformer = factory.create_transformer(embedder, **kwargs)
                logger.info(f"Created transformer: {transformer.__class__.__name__}")
                return transformer

        # If no suitable factory is found, raise an exception
        embedder_type = embedder.__class__.__name__
        raise ValueError(f"Unknown embedder type: {embedder_type}")

    def list_registered_factories(self) -> list[str]:
        """List all registered factory class names"""
        return [factory.__class__.__name__ for factory in self._factories]


# Global registry instance
_global_registry = EmbedderTransformerRegistry()


def get_transformer_registry() -> EmbedderTransformerRegistry:
    """Get the global transformer registry"""
    return _global_registry


def create_embedder_transformer() -> ToEmbeddings:
    """
    Convenience function: create an appropriate embedder transformer

    Args:
        embedder: embedder instance
        use_dual_vector: whether to use dual vectors
        code_understanding_generator: code understanding generator (required for dual vector mode)

    Returns:
        ToEmbeddings: appropriate transformer instance
    """
    registry = get_transformer_registry()
    code_understanding_generator = None
    if configs()["rag"]["embedder"]["sketch_filling"]:
        code_understanding_generator = CodeUnderstandingGenerator(
            **configs()["rag"]["code_understanding"]
        )
    return registry.create_transformer(
        embedder=get_batch_embedder(),
        use_dual_vector=configs()["rag"]["embedder"]["sketch_filling"],
        code_understanding_generator=code_understanding_generator,
    )

def create_splitter_transformer():
    if configs()["rag"]["dynamic_splitter"]["enabled"]:
        # Use dynamic splitter that automatically selects appropriate splitter
        splitter = DynamicSplitterTransformer(
            batch_size=configs()["rag"]["dynamic_splitter"]["batch_size"],
            parallel=configs()["rag"]["dynamic_splitter"]["parallel"],
        )
    else:
        text_splitter_kwargs = configs()["rag"]["text_splitter"]
        if configs()["rag"]["adjacent_documents"]["enabled"]:
            text_splitter_kwargs["chunk_overlap"] = False
        splitter = MyTextSplitter(
            **text_splitter_kwargs,
        )
    return splitter


def create_bm25_transformer():
    return BM25Transformer()