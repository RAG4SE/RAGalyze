"""Dynamic splitter transformer that selects appropriate splitter based on document type."""

import os
import threading
from typing import List, Union, Any

from adalflow.core.types import Document
from adalflow.core.component import Component
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ragalyze.rag.splitter_factory import get_splitter_factory
from ragalyze.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class DynamicSplitterTransformer(Component):
    """Transformer that dynamically selects appropriate splitter based on document type."""

    def __init__(self, batch_size: int = None, parallel: bool = True):
        """Initialize the dynamic splitter transformer."""
        super().__init__()
        self.splitter_factory = get_splitter_factory()
        self.batch_size = batch_size
        self.parallel = parallel
        if not self.batch_size:
            self.batch_size = os.cpu_count()
        logger.info("Initialized DynamicSplitterTransformer")

    def _process_batch(self, documents: List[Document]) -> List[Document]:
        """Process a batch of documents with appropriate splitters using threading."""
        result_documents = []

        # Create a stop event that can be shared across all workers
        stop_event = threading.Event()
        error_container = {"error": None}

        def safe_process_single_document(doc):
            """Wrapper that checks stop event before processing."""
            try:
                # Check if we should stop before starting
                if stop_event.is_set():
                    logger.debug("Skipping document processing due to stop event")
                    return []

                # Process the document
                result = self._process_single_document(doc)

                # Check again after processing (in case error occurred during processing)
                if stop_event.is_set():
                    logger.debug("Discarding result due to stop event")
                    return []

                return result

            except Exception as e:
                # Set stop event to signal other threads to stop
                if not stop_event.is_set():
                    stop_event.set()
                    error_container["error"] = e
                raise

        # Use ThreadPoolExecutor to parallelize document processing
        # (ThreadPoolExecutor shares memory space, avoiding config loading issues)
        with ThreadPoolExecutor(
            max_workers=min(len(documents), mp.cpu_count())
        ) as executor:
            # Submit all documents for processing with the wrapper function
            futures = [
                executor.submit(safe_process_single_document, doc) for doc in documents
            ]

            # Collect results as they complete, stop immediately on first error
            try:
                for future in as_completed(futures):
                    # Check if we should stop before getting result
                    if stop_event.is_set():
                        logger.debug("Stop event set, cancelling remaining futures")
                        break

                    result = future.result()
                    if result:  # Only extend if result is not empty
                        result_documents.extend(result)

            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                stop_event.set()

            finally:
                # Cancel all remaining futures
                cancelled_count = 0
                for remaining_future in futures:
                    if not remaining_future.done():
                        if remaining_future.cancel():
                            cancelled_count += 1

                if cancelled_count > 0:
                    logger.info(f"Cancelled {cancelled_count} remaining futures")

                # If there was an error, re-raise it
                if error_container["error"]:
                    raise error_container["error"]

        return result_documents

    def _process_single_document(self, doc: Document) -> List[Document]:
        """Process a single document with appropriate splitter."""
        splitter = self.splitter_factory.get_splitter(
            content=doc.text,
            file_path=getattr(doc, "meta_data", {}).get("file_path", ""),
        )
        try:
            splitted_docs = splitter.call([doc])
            del splitter
        except Exception as e:
            raise
        return splitted_docs

    def call(self, documents: List[Document]) -> List[Document]:
        """Process documents with appropriate splitters using batch optimization.

        Args:
            documents (List[Document]): Input documents

        Returns:
            List[Document]: Split documents
        """
        if not documents:
            return []

        result_documents = []
        if not self.parallel:
            for idx in tqdm(
                range(0, len(documents)),
                desc="Dynamcally Splitting Documents in Sequence",
            ):
                result_documents.extend(self._process_single_document(documents[idx]))
        else:
            for start_idx in tqdm(
                range(0, len(documents), self.batch_size),
                desc="Dynamcally Splitting Documents in Batches",
            ):
                end_idx = start_idx + self.batch_size
                batch_documents = documents[start_idx:end_idx]
                result_documents.extend(self._process_batch(batch_documents))

        return result_documents

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Make the transformer callable.

        Args:
            documents (List[Document]): Input documents

        Returns:
            List[Document]: Split documents
        """
        return self.call(documents)
