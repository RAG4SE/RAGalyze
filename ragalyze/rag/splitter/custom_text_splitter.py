from copy import deepcopy
import re

from adalflow.core.types import Document
from adalflow.components.data_process.text_splitter import (
    TextSplitter,
    DocumentSplitterInputType,
    DocumentSplitterOutputType,
)
from tqdm import tqdm

from ragalyze.logger.logging_config import get_tqdm_compatible_logger
logger = get_tqdm_compatible_logger(__name__)

# Import the fast chunk processor for parallel execution
try:
    from ragalyze.lib.fast_chunk_processor import process_document, set_debug_mode
    logger.info("Fast chunk processor loaded successfully")
except ImportError as e:
    logger.error(f"Fast chunk processor not available: {e}, falling back to Python implementation")
    raise
except Exception as e:
    logger.error(f"Error loading fast chunk processor: {e}, falling back to Python implementation")
    raise


class MyTextSplitter(TextSplitter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_debug_mode(0)

    def _call_c_impl(self, documents: DocumentSplitterInputType) -> DocumentSplitterOutputType:
        """C implementation using process_document with separator-based splitting."""
        split_docs = []

        for doc in tqdm(documents, desc="Splitting documents with C implementation"):
            if not isinstance(doc, Document):
                logger.error(f"Each item in documents should be an instance of Document, but got {type(doc).__name__}.")
                raise TypeError(f"Each item in documents should be an instance of Document, but got {type(doc).__name__}.")

            if doc.text is None:
                logger.error(f"Text should not be None. Doc id: {doc.id}")
                raise ValueError(f"Text should not be None. Doc id: {doc.id}")

            # Create document dictionary for C extension
            doc_dict = {
                'text': doc.text,
                'id': doc.id,
                'parent_doc_id': doc.parent_doc_id,
                'order': doc.order,
                'meta_data': doc.meta_data or {},
                'vector': doc.vector or [],
                'score': doc.score,
                'estimated_num_tokens': getattr(doc, 'estimated_num_tokens', 0)
            }

            # Add BM25 indexes if available
            if doc.meta_data and 'bm25_indexes' in doc.meta_data:
                doc_dict['meta_data']['bm25_indexes'] = doc.meta_data['bm25_indexes']

            # Call C function with separator-based splitting
            # Use the same split_by, chunk_size, and chunk_overlap settings as the parent class
            separators_dict = self.separators if hasattr(self, 'separators') else None

            chunk_results = process_document(
                doc_dict,
                self.chunk_size,
                self.chunk_overlap,
                "word",
                separators_dict
            )

            # Convert results to Documents and link them together
            prev_doc = None
            for i, chunk_dict in enumerate(chunk_results):
                chunk_doc = Document(
                    text=chunk_dict['text'],
                    meta_data=chunk_dict.get('meta_data', {}),
                    parent_doc_id=doc.id,  # Always use the original document ID as parent
                    order=chunk_dict.get('order', i),
                    vector=chunk_dict.get('vector', []),
                    id=chunk_dict.get('id')
                )
                chunk_doc.meta_data["end_line"] = chunk_doc.meta_data["start_line"] + len(chunk_dict['text'].splitlines()) - 1
                
                # Link documents together
                if prev_doc:
                    prev_doc.meta_data["next_doc_id"] = chunk_doc.id
                    chunk_doc.meta_data["prev_doc_id"] = prev_doc.id
                else:
                    chunk_doc.meta_data["prev_doc_id"] = None

                if i == len(chunk_results) - 1:
                    chunk_doc.meta_data["next_doc_id"] = None

                prev_doc = chunk_doc
                split_docs.append(chunk_doc)

        return split_docs

    def call(self, documents: DocumentSplitterInputType) -> DocumentSplitterOutputType:
        """
        Process the splitting task on a list of documents in batch.

        Uses fast C implementation when available, falls back to Python implementation.

        Args:
            documents (List[Document]): A list of Document objects to process.

        Returns:
            List[Document]: A list of new Document objects, each containing a chunk of text from the original documents.

        Raises:
            TypeError: If 'documents' is not a list or contains non-Document objects.
            ValueError: If any document's text is None.
        """
        
        if not isinstance(documents, list) or any(
            not isinstance(doc, Document) for doc in documents
        ):
            logger.error("Input should be a list of Documents.")
            raise
        
        # Use fast C implementation if available, otherwise fall back to Python
        try:
            split_docs = self._call_c_impl(documents)

            logger.info(
                f"Processed {len(documents)} documents into {len(split_docs)} split documents using fast C implementation."
            )
            return split_docs
        except Exception as e:
            logger.error(f"Fast C implementation failed: {e}, falling back to Python implementation")
            raise
