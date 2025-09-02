"""
Embedder Transformer Registry

Provides a registry pattern to dynamically select and create appropriate embedder transformers,
replacing the original if-else judgment logic.
"""

from typing import Dict, Type, Callable, Any, Optional
from abc import ABC, abstractmethod
import inspect

import adalflow as adal
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.rag.embedding import (
    ToEmbeddings,
    DashScopeToEmbeddings,
    HuggingfaceToEmbeddings,
    OpenAIToEmbeddings,
    DualVectorToEmbeddings,
)
from ragalyze.rag.code_understanding import CodeUnderstandingGenerator

logger = get_tqdm_compatible_logger(__name__)


class TransformerFactory(ABC):
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


class DualVectorTransformerFactory(TransformerFactory):
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


class HuggingfaceTransformerFactory(TransformerFactory):
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


class DashScopeTransformerFactory(TransformerFactory):
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


class OpenAITransformerFactory(TransformerFactory):
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

class LocalServerTransformerFactory(TransformerFactory):
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


class TransformerRegistry:
    """
    Transformer Registry

    Used to register and select appropriate embedder transformer factories
    """

    def __init__(self):
        self._factories: list[TransformerFactory] = []
        self._register_default_factories()

    def _register_default_factories(self):
        """Register default factory classes"""
        # Note: DualVector factory has the highest priority as it may override other conditions
        self.register_factory(DualVectorTransformerFactory())
        self.register_factory(HuggingfaceTransformerFactory())
        self.register_factory(DashScopeTransformerFactory())
        self.register_factory(OpenAITransformerFactory())
        self.register_factory(LocalServerTransformerFactory())

    def register_factory(self, factory: TransformerFactory):
        """Register a new factory"""
        if not isinstance(factory, TransformerFactory):
            raise TypeError("Factory must be an instance of TransformerFactory")
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
_global_registry = TransformerRegistry()


def get_transformer_registry() -> TransformerRegistry:
    """Get the global transformer registry"""
    return _global_registry


def create_embedder_transformer(
    embedder: adal.Embedder | adal.BatchEmbedder,
    use_dual_vector: bool = False,
    code_understanding_generator: Optional[CodeUnderstandingGenerator] = None,
) -> ToEmbeddings:
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
    return registry.create_transformer(
        embedder=embedder,
        use_dual_vector=use_dual_vector,
        code_understanding_generator=code_understanding_generator,
    )
