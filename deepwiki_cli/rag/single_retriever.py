from typing import List, Optional

from adalflow.core.types import Document, RetrieverOutput, RetrieverOutputType
from adalflow.components.retriever.faiss_retriever import (
    FAISSRetriever,
    FAISSRetrieverQueriesType,
)

from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class SingleVectorRetriever(FAISSRetriever):
    """A wrapper of FAISSRetriever with the additional feature of supporting docoments feature"""

    def __init__(self, documents: List[Document], *args, **kwargs):
        self.original_doc = documents
        super().__init__(documents=documents, *args, **kwargs)
        logger.info(
            f"SingleVectorRetriever initialized with {len(documents)} documents"
        )

    def call(
        self,
        input: FAISSRetrieverQueriesType,
    ) -> RetrieverOutputType:
        retriever_output = super().call(input, self.top_k)

        # Extract the first result from the list
        if not retriever_output:
            return []

        first_output = retriever_output[0]

        # Get the documents based on the indices
        retrieved_docs = [self.original_doc[i] for i in first_output.doc_indices]

        # Create a new RetrieverOutput with the documents
        return [
            RetrieverOutput(
                doc_indices=first_output.doc_indices,
                doc_scores=first_output.doc_scores,
                query=first_output.query,
                documents=retrieved_docs,
            )
        ]
