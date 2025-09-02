from copy import deepcopy
import re

from adalflow.core.types import Document
from adalflow.components.data_process.text_splitter import (
    TextSplitter,
    DocumentSplitterInputType,
    DocumentSplitterOutputType
)
from tqdm import tqdm

from ragalyze.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

class MyTextSplitter(TextSplitter):

    def __init__(self, enable_line_number: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_line_number = enable_line_number

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

                # prefix each line of doc.text with the 0-based line number
                doc.text = "\n".join(f"{i}: {line}" for i, line in enumerate(doc.text.splitlines()))

                text_splits = self.split_text(doc.text)
                meta_data = deepcopy(doc.meta_data)
                
                if self.enable_line_number:
                    start_pos = 0
                    for i, chunk in enumerate(text_splits):
                        pattern = re.escape(chunk.lstrip())
                        match = re.search(pattern, doc.text[start_pos:])
                        if match:
                            chunk_start = start_pos + match.start()
                            start_pos = chunk_start
                        else:
                            chunk_start = start_pos  # fallback
                        start_line = doc.text[:chunk_start].count('\n')
                        chunk_meta = deepcopy(meta_data) if meta_data else {}
                        chunk_meta["start_line"] = start_line
                        split_docs.append(
                            Document(
                                text=chunk,
                                meta_data=chunk_meta,
                                parent_doc_id=f"{doc.id}",
                                order=i,
                                vector=[],
                            )
                        )
                else:
                    split_docs.extend(
                        [
                            Document(
                                text=txt,
                                meta_data=meta_data,
                                parent_doc_id=f"{doc.id}",
                                order=i,
                                vector=[],
                            )
                            for i, txt in enumerate(text_splits)
                        ]
                    )
        logger.info(
            f"Processed {len(documents)} documents into {len(split_docs)} split documents."
        )
        return split_docs
        