"""Dynamic splitter transformer that selects appropriate splitter based on document type."""

import logging
from typing import List, Union, Any

from adalflow.core.types import Document
from adalflow.core.component import Component
from tqdm import tqdm

from deepwiki_cli.rag.splitter_factory import get_splitter_factory
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class DynamicSplitterTransformer(Component):
    """Transformer that dynamically selects appropriate splitter based on document type."""

    def __init__(self):
        """Initialize the dynamic splitter transformer."""
        super().__init__()
        self.splitter_factory = get_splitter_factory()
        logger.info("Initialized DynamicSplitterTransformer")

    def call(self, documents: List[Document]) -> List[Document]:
        """Process documents with appropriate splitters using batch optimization.

        Args:
            documents (List[Document]): Input documents

        Returns:
            List[Document]: Split documents
        """
        if not documents:
            return []

        # Step 1: Group documents by splitter type
        splitter_groups = (
            {}
        )  # key: splitter_key, value: {'splitter': splitter, 'docs': [docs]}

        for doc in tqdm(documents, desc="Grouping documents by splitter type"):
            try:
                # Get appropriate splitter for this document
                file_path = getattr(doc, "meta_data", {}).get("file_path", "")
                splitter = self.splitter_factory.get_splitter(
                    content=doc.text,
                    file_path=file_path,
                )

                # Create splitter key
                splitter_key = splitter.get_key()

                # Group documents by splitter
                if splitter_key not in splitter_groups:
                    splitter_groups[splitter_key] = {
                        "splitter": splitter,
                        "docs": [],
                    }

                splitter_groups[splitter_key]["docs"].append(doc)

            except Exception as e:
                logger.error(
                    f"Error analyzing document {getattr(doc, 'meta_data', {}).get('file_path', 'unknown')}: {e}"
                )
                raise

        # Step 2: Process each splitter group in batch
        result_documents = []
        logger.info(f"Found {len(splitter_groups)} unique splitter configurations")

        for splitter_key, group_info in splitter_groups.items():
            logger.info(f"Processing splitter group {splitter_key}")
            splitter = group_info["splitter"]
            docs = group_info["docs"]

            try:
                # Batch split all documents for this splitter
                split_docs = splitter.call(docs)
                result_documents.extend(split_docs)

            except Exception as e:
                # Skip this unsplittable document
                logger.warning(
                    f"Error processing splitter group {splitter_key}: {e}"
                )
                continue

        logger.info(
            f"Processed {len(documents)} documents into {len(result_documents)} chunks using {len(splitter_groups)} splitter groups"
        )

        return result_documents

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Make the transformer callable.

        Args:
            documents (List[Document]): Input documents

        Returns:
            List[Document]: Split documents
        """
        return self.call(documents)
