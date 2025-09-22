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


class MyTextSplitter(TextSplitter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, documents: DocumentSplitterInputType) -> DocumentSplitterOutputType:
        """
        Process the splitting task on a list of documents in batch.

        Batch processes a list of documents, splitting each document's text according to the configured
        split_by, chunk size, and chunk overlap.

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
            raise TypeError("Input should be a list of Documents.")

        split_docs = []
        # Using range and batch_size to create batches
        for start_idx in tqdm(
            range(0, len(documents), self.batch_size),
            desc="Splitting Documents in Batches",
        ):
            batch_docs = documents[start_idx : start_idx + self.batch_size]

            for doc in batch_docs:
                if not isinstance(doc, Document):
                    logger.error(
                        f"Each item in documents should be an instance of Document, but got {type(doc).__name__}."
                    )
                    raise TypeError(
                        f"Each item in documents should be an instance of Document, but got {type(doc).__name__}."
                    )

                if doc.text is None:
                    logger.error(f"Text should not be None. Doc id: {doc.id}")
                    raise ValueError(f"Text should not be None. Doc id: {doc.id}")

                # First split the text, then add line numbers to each chunk
                text_splits = self.split_text(doc.text)

                meta_data = deepcopy(doc.meta_data)
                bm25_indexes = meta_data.get("bm25_indexes", []) if meta_data else []
            
                # Sort BM25 indices by their positions (line, then column)
                if bm25_indexes:
                    bm25_indexes.sort(key=lambda x: (x.position[0], x.position[1]))

                prev_bm25_index_idx = 0

                # Add line numbers to each chunk
                line_numbered_splits = []
                for chunk in text_splits:
                    lines = chunk.splitlines()
                    if not lines:
                        continue

                    # Find the starting line number for this chunk in the original document
                    original_start_line = doc.text[:doc.text.find(chunk)].count('\n')
                    line_numbered_chunk = "\n".join(
                        f"{original_start_line + i}: {line}" for i, line in enumerate(lines)
                    )
                    line_numbered_splits.append(line_numbered_chunk)

                start_pos = 0
                prev_doc = None
                pre_chunk_extension_count = 0
                for i, chunk in enumerate(text_splits):
                    chunk = chunk[pre_chunk_extension_count:]
                    pattern = re.escape(chunk.lstrip())
                    match = re.search(pattern, doc.text[start_pos:])
                    if match:
                        chunk_start = start_pos + match.start()
                        start_pos = chunk_start
                    else:
                        chunk_start = start_pos  # fallback

                    # If bm25_indexes exists, check if chunk ends in the middle of an index
                    if bm25_indexes:
                        chunk_end = chunk_start + len(chunk)
                        extended_end = chunk_end

                        # Convert chunk position to line numbers (accounting for line number prefixes)
                        chunk_lines = doc.text[:chunk_end].split("\n")
                        current_line_0based = len(chunk_lines) - 1
                        current_col_0based = len(chunk_lines[-1]) - 1

                        # Find if current position falls within any bm25 index range
                        for bm25_index in bm25_indexes:
                            # Convert 1-based bm25 index to 0-based for comparison
                            # idx_string is like [FUNCDEF]function_name
                            idx_string = bm25_index.token
                            idx_line, idx_col = bm25_index.position
                            idx_text = idx_string.split("]")[1]
                            idx_line_0based = idx_line - 1
                            idx_start_col_0based = idx_col - 1
                            idx_end_col_0based = (
                                idx_start_col_0based + len(idx_text) - 1
                            )
                            # Find the position of this bm25 index in the text
                            if (
                                idx_line_0based == current_line_0based
                                and idx_start_col_0based
                                < current_col_0based
                                < idx_end_col_0based
                            ):
                                # Extend chunk to end of this bm25 index
                                extended_end = chunk_end + (
                                    idx_end_col_0based - current_col_0based
                                )
                                current_col_0based = idx_end_col_0based

                        # If extension is needed, update the chunk
                        if extended_end > chunk_end:
                            chunk = doc.text[chunk_start:extended_end]
                            pre_chunk_extension_count = extended_end - chunk_end

                    start_line = doc.text[:chunk_start].count("\n")
                    chunk_meta = deepcopy(meta_data) if meta_data else {}
                    chunk_meta["start_line"] = start_line

                    chunk_bm25_indexes = []

                    for i in range(prev_bm25_index_idx, len(bm25_indexes)):
                        if bm25_indexes[i].position[0] - 1 < current_line_0based:
                            chunk_bm25_indexes.append(bm25_indexes[i])
                        elif bm25_indexes[i].position[0] - 1 == current_line_0based and bm25_indexes[i].position[1] <= current_col_0based:
                            chunk_bm25_indexes.append(bm25_indexes[i])
                        else:
                            prev_bm25_index_idx = i
                            break
                    
                    chunk_meta["bm25_indexes"] = chunk_bm25_indexes
                    
                    lines = chunk.splitlines()

                    # Find the starting line number for this chunk in the original document
                    original_start_line = doc.text[:doc.text.find(chunk)].count('\n')
                    line_numbered_chunk = "\n".join(
                        f"{original_start_line + i}: {line}" for i, line in enumerate(lines)
                    )
                    
                    this_doc = Document(
                        text=line_numbered_chunk,
                        meta_data=chunk_meta,
                        parent_doc_id=f"{doc.id}",
                        order=i,
                        vector=[],
                    )
                    
                    if prev_doc:
                        prev_doc.meta_data["next_doc_id"] = this_doc.id
                        this_doc.meta_data["prev_doc_id"] = prev_doc.id
                        this_doc.meta_data["next_doc_id"] = None
                    else:
                        this_doc.meta_data["prev_doc_id"] = None
                        this_doc.meta_data["next_doc_id"] = None
                    this_doc.meta_data["original_text"] = chunk
                    split_docs.append(this_doc)
                    prev_doc = this_doc

        logger.info(
            f"Processed {len(documents)} documents into {len(split_docs)} split documents."
        )
        return split_docs
