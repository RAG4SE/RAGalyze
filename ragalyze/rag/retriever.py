"""
This file implements various retrieval strategies including single vector, dual vector, 
hybrid BM25+FAISS, and query-driven retrievers for semantic document search in the RAG system.
"""

import re
from typing import List, Optional, Union, Callable
from rank_bm25 import BM25Okapi
from collections import defaultdict
from copy import deepcopy

from adalflow.core.types import RetrieverOutput, Document, RetrieverOutputType
from adalflow.components.retriever.faiss_retriever import (
    FAISSRetriever,
    FAISSRetrieverQueriesType,
)

from ragalyze.configs import get_embedder, configs
from ragalyze.core.types import DualVectorDocument
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.rag.db import DatabaseManager
from ragalyze.core.utils import minmax_norm, zscore_norm

logger = get_tqdm_compatible_logger(__name__)

class BM25Retriever:
    """Standalone BM25 retriever that can be used by other retrievers."""

    def __init__(self, documents: List[Union[Document, DualVectorDocument]], 
                 k1: float = 1.5, b: float = 0.75, top_k: int = 20):
        """
        Initialize the BM25 retriever.

        Args:
            documents: List of documents to index
            k1: BM25 k1 parameter
            b: BM25 b parameter
            top_k: Number of top documents to retrieve
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        assert top_k > 0, "Top k must be greater than 0"
        self.top_k = top_k
        self.bm25 = None
        self._initialize_bm25(documents)

    def _initialize_bm25(self, documents: List[Union[Document, DualVectorDocument]]):
        """Initialize BM25 index with document texts."""
        try:
            # Extract text content from documents for BM25 indexing
            corpus = []
            for doc in documents:
                if isinstance(doc, DualVectorDocument):
                    # For dual vector documents, combine code and understanding text
                    text = doc.original_doc.text + "\n" + doc.understanding_text
                else:
                    text = doc.text

                # Tokenize text for BM25 (simple whitespace + punctuation splitting)
                tokens = self._tokenize_text(text)
                corpus.append(tokens)

            # Initialize BM25 with custom parameters
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
            logger.info(f"BM25 index created with {len(corpus)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.bm25 = None

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25 indexing."""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Split on whitespace and common punctuation, keep alphanumeric tokens
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents given a query."""
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(query_tokens)
            return scores
        except Exception as e:
            logger.error(f"Failed to get BM25 scores: {e}")
            return []

    def filter_and_score(self, query: str) -> tuple[List[int], List[float]]:
        """Filter documents using BM25 and return document indices and normalized scores."""
        try:
            # Get BM25 scores for all documents
            scores = self.get_scores(query)
            
            # Get top-k document indices based on BM25 scores
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:self.top_k]

            filtered_scores = [scores[i] for i in doc_indices]

            logger.info(f"BM25 filtered {len(self.documents)} documents to {len(doc_indices)} candidates")
            return doc_indices, filtered_scores

        except Exception as e:
            logger.error(f"BM25 filtering failed: {e}")
            raise

    def call(self, query: str, top_k: Optional[int] = None) -> RetrieverOutputType:
        """Retrieve documents using BM25."""
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []

        if top_k is None:
            top_k = self.top_k

        assert top_k > 0, "Top k must be greater than 0"

        try:
            # Get BM25 scores for all documents
            scores = self.get_scores(query)
            
            # Get top-k document indices based on BM25 scores
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]

            # Get the actual documents and scores
            retrieved_docs = [self.documents[i] for i in doc_indices]
            doc_scores = [scores[i] for i in doc_indices]

            logger.info(f"BM25 retrieved {len(retrieved_docs)} documents")

            return [
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=doc_scores,
                    query=query,
                    documents=retrieved_docs,
                )
            ]

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []
    

class SingleVectorRetriever(FAISSRetriever):
    """A wrapper of FAISSRetriever with the additional feature of supporting documents feature"""

    def __init__(self, documents: List[Document], *args, **kwargs):
        self.original_doc = documents
        super().__init__(documents=documents, *args, **kwargs)
        logger.info(
            f"SingleVectorRetriever initialized with {len(documents)} documents"
        )

    def call(
        self,
        input: FAISSRetrieverQueriesType,
        top_k: int = None
    ) -> RetrieverOutputType:
        if top_k is None:
            top_k = self.top_k
        assert top_k > 0, "Top k must be greater than 0"
        retriever_output = super().call(input, top_k)

        # Extract the first result from the list
        if not retriever_output:
            return []

        first_output = retriever_output[0]

        # Get the documents based on the indices
        retrieved_docs = [self.original_doc[i] for i in first_output.doc_indices]
        doc_indices = first_output.doc_indices
        doc_scores = first_output.doc_scores

        # Create a new RetrieverOutput with the documents
        return [
            RetrieverOutput(
                doc_indices=doc_indices,
                doc_scores=doc_scores,
                query=first_output.query,
                documents=retrieved_docs,
            )
        ]


class DualVectorRetriever:
    """Dual vector retriever: supports dual retrieval from code and summary vectors."""

    def __init__(self, dual_docs: List[DualVectorDocument], embedder, top_k: int = 20):
        """
        Initializes the dual vector retriever.

        Args:
            dual_docs: A list of dual vector documents.
            embedder: The embedder instance for embedding queries.
            top_k: The number of most relevant documents to return.
        """
        self.dual_docs = dual_docs
        self.embedder = embedder
        assert top_k > 0, "Top k must be greater than 0"
        self.top_k = top_k
        self.doc_map = {doc.original_doc.id: doc for doc in dual_docs}

        logger.info(
            f"Dual vector retriever initialization completed, containing {len(dual_docs)} documents"
        )

    def _build_indices(self):
        """Builds the code index and the summary index."""
        if not self.dual_docs:
            logger.warning("No documents available for building indices")
            self.code_retriever = None
            self.understanding_retriever = None
            return

        # 1. Build the code index
        code_docs = []
        for dual_doc in self.dual_docs:
            # Create a document object for FAISS
            faiss_doc = Document(
                text=dual_doc.original_doc.text,
                meta_data=dual_doc.original_doc.meta_data,
                id=f"{dual_doc.original_doc.id}_code",
                vector=dual_doc.code_embedding,
            )
            code_docs.append(faiss_doc)

        self.code_retriever = FAISSRetriever(
            top_k=self.top_k,
            embedder=self.embedder,
            documents=code_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Code FAISS index built successfully.")

        # 2. Build the summary index
        understanding_docs = []
        for dual_doc in self.dual_docs:
            faiss_doc = Document(
                text=dual_doc.understanding_text,
                meta_data=dual_doc.original_doc.meta_data,
                id=f"{dual_doc.original_doc.id}_understanding",
                vector=dual_doc.understanding_embedding,
            )
            understanding_docs.append(faiss_doc)

        self.understanding_retriever = FAISSRetriever(
            top_k=self.top_k,
            embedder=self.embedder,
            documents=understanding_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Understanding FAISS index built successfully.")

    def call(self, query_str: str) -> RetrieverOutputType:
        """
        Performs dual retrieval.

        Args:
            query_str: The query string.

        Returns:
            A RetrieverOutput object containing the retrieved documents and scores.
        """
        assert isinstance(
            query_str, str
        ), f"Query must be a string, got {type(query_str)}"

        self._build_indices()

        if not self.dual_docs:
            return RetrieverOutput(
                doc_indices=[], doc_scores=[], query=query_str, documents=[]
            )

        # 1. Retrieve from the code index
        code_results = self.code_retriever.call(query_str, top_k=self.top_k)[0]
        # 2. Retrieve from the summary index
        understanding_results = self.understanding_retriever.call(
            query_str, top_k=self.top_k
        )[0]

        # 3. Merge and re-rank the results
        combined_scores = {}

        # Process code results - extract original chunk_id from FAISS document ID
        for i, score in zip(code_results.doc_indices, code_results.doc_scores):
            # Get the document from code retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_code", "")
            combined_scores[original_chunk_id] = score

        # Process understanding results - extract original chunk_id from FAISS document ID
        for i, score in zip(
            understanding_results.doc_indices, understanding_results.doc_scores
        ):
            # Get the document from understanding retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_understanding", "")
            if original_chunk_id not in combined_scores:
                combined_scores[original_chunk_id] = score
            else:
                combined_scores[original_chunk_id] = max(
                    combined_scores[original_chunk_id], score
                )

        # 4. Sort and get the top-k results
        # Sort by the combined score in descending order
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda chunk_id: combined_scores[chunk_id],
            reverse=True,
        )

        # 5. Retrieve the full documents for the top-k chunk_ids and create indices mapping
        top_k_docs = []
        doc_indices = []
        doc_scores = []
        for idx, chunk_id in enumerate(
            sorted_chunk_ids[: min(self.top_k, len(sorted_chunk_ids))]
        ):
            if chunk_id in self.doc_map:
                dual_doc = self.doc_map[chunk_id]
                top_k_docs.append(dual_doc)
                doc_indices.append(idx)
                doc_scores.append(combined_scores[chunk_id])

        logger.info(
            f"Retrieved {len(top_k_docs)} documents after merging code and understanding search results."
        )

        return [
            RetrieverOutput(
                doc_indices=doc_indices,
                doc_scores=doc_scores,
                query=query_str,
                documents=top_k_docs,
            )
        ]


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 keyword filtering with FAISS semantic search.

    The retrieval process:
    1. BM25 first filters documents by exact keyword matches, reducing the search space
    2. FAISS then performs semantic similarity search on this subset
    3. Results are merged and re-ranked for optimal relevance
    """

    def __init__(self, documents: List[Union[Document, DualVectorDocument]], **kwargs):
        """
        Initialize the hybrid retriever.

        Args:
            documents: List of transformed documents to index
        """
        self.documents = documents
        self.embedder = get_embedder()

        rag_config = configs()["rag"]
        self.use_dual_vector = rag_config["embedder"]["sketch_filling"]
        retriever_config = rag_config["retriever"]
        self.top_k = retriever_config["top_k"]
        assert self.top_k > 0, "Top k must be greater than 0"
        bm25_config = retriever_config["bm25"]
        self.bm25_k1 = bm25_config["k1"]
        self.bm25_b = bm25_config["b"]
        self.bm25_weight = bm25_config["weight"]
        assert 0 <= self.bm25_weight <= 1, "BM25 weight must be between 0 and 1."
        self.fusion = configs()["rag"]["retriever"]["fusion"]
        assert self.fusion in ["rrf", "normal_add"], f"Invalid fusion method: {self.fusion}"

        logger.info(
            f"Hybrid retriever initialized with dual_vector={'enabled' if self.use_dual_vector else 'disabled'}"
            f"BM25 parameters: k1={self.bm25_k1}, b={self.bm25_b}, weight={self.bm25_weight}"
            f"Other parameters: top_k={self.top_k}"
        )

    def _initialize_faiss_retriever(self, documents: List[Document | DualVectorDocument], top_k: int):
        """Initialize FAISS retriever based on vector type."""
        if self.use_dual_vector:
            faiss_retriever = DualVectorRetriever(
                dual_docs=documents,
                embedder=self.embedder,
                top_k=top_k,
            )
        else:
            faiss_retriever = SingleVectorRetriever(
                documents=documents,
                embedder=self.embedder,
                top_k=top_k,
                document_map_func=lambda doc: doc.vector,
            )
        logger.info(f"FAISS retriever initialized successfully")
        return faiss_retriever

    def _initialize_bm25_retriever(self, documents: List[Document | DualVectorDocument], top_k: int) -> BM25Retriever:
        """Initialize BM25 retriever."""
        bm25_retriever = BM25Retriever(
            documents=documents,
            k1=self.bm25_k1,
            b=self.bm25_b,
            top_k=top_k
        )
        logger.info(f"BM25 retriever initialized successfully")
        return bm25_retriever

    def _rrf(self, doc_indices_list: List[List[int]], weights: List[int]) -> dict:
        """
        Given several ranks, return the final rank based on Reciprocal Rank Fusion
        """
        k = configs()["rag"]["retriever"]["rrf"]["k"]
        # Initialize a dictionary to hold the cumulative scores for each document
        doc_scores = defaultdict(float)
        for doc_indices, weight in zip(doc_indices_list, weights):
            for rank, doc_id in enumerate(doc_indices):
                # Apply the Reciprocal Rank Fusion formula
                doc_scores[doc_id] += weight * (1.0 / float(rank + 1 + k))
        # Return the final ranked list of document IDs
        return dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))


    def _mix_bm25_score_faiss_score(self, documents: List[Document | DualVectorDocument], bm25_indices: List[int], bm25_scores: List[float], faiss_results: RetrieverOutputType) -> List[Document]:
        """Mix BM25 scores with FAISS results."""
        faiss_indices = faiss_results[0].doc_indices
        faiss_scores = faiss_results[0].doc_scores
        if isinstance(documents[0], DualVectorDocument):
            documents = [doc.original_doc for doc in documents]
        if self.fusion == "normal_add":
            minmax_bm25_scores = minmax_norm(bm25_scores)
            zscore_bm25_scores = zscore_norm(minmax_bm25_scores)
            self.doc_id_to_bm25_scores = {documents[doc_id].id: (original_score, minmax_score, zscore_score) for doc_id, original_score, minmax_score, zscore_score in zip(bm25_indices, bm25_scores, minmax_bm25_scores, zscore_bm25_scores)}
            minmax_faiss_scores = minmax_norm(faiss_scores)
            zscore_faiss_scores = zscore_norm(minmax_faiss_scores)
            self.doc_id_to_faiss_scores = {documents[doc_id].id: (original_score, minmax_score, zscore_score) for doc_id, original_score, minmax_score, zscore_score in zip(faiss_indices, faiss_scores, minmax_faiss_scores, zscore_faiss_scores)}
            scores = [0] * len(bm25_indices)
            assert len(faiss_indices) == len(bm25_indices), f"Mismatch in FAISS results: {len(faiss_indices)} vs {len(bm25_indices)}"
            assert all([faiss_indices[i] < len(faiss_indices) for i in range(len(faiss_indices))]), f"Invalid FAISS index found"
            assert all([bm25_indices[i] < len(bm25_indices) for i in range(len(bm25_indices))]), f"Invalid BM25 index found"
            for i, doc in enumerate(bm25_indices):
                scores[bm25_indices[i]] = zscore_bm25_scores[i] * self.bm25_weight
            for i, doc in enumerate(faiss_indices):
                scores[faiss_indices[i]] += zscore_faiss_scores[i] * (1 - self.bm25_weight)
            self.doc_id_to_bm25faiss_scores = {documents[id].id: scores[id] for id in bm25_indices}
        else:
            id_to_scores = self._rrf([bm25_indices, faiss_indices], [self.bm25_weight, 1 - self.bm25_weight])
            scores = list(id_to_scores.values())
            self.doc_id_to_rrf_scores = {documents[id].id: score for (id, score) in id_to_scores.items()}
        return scores

    def call(self, query: str) -> List[RetrieverOutput]:
        """Perform hybrid retrieval combining BM25 and FAISS."""

        try:
            bm25_retriever = self._initialize_bm25_retriever(self.documents, top_k=len(self.documents))
            bm25_indices, bm25_scores = bm25_retriever.filter_and_score(query)
            faiss_retriever = self._initialize_faiss_retriever(self.documents, top_k=len(self.documents))
            faiss_results = faiss_retriever.call(query)
            assert len(faiss_results[0].documents) == len(bm25_indices), f"Mismatch in number of documents between BM25 and FAISS results: {len(bm25_indices)} vs {len(faiss_results[0]['documents'])}"
            scores = self._mix_bm25_score_faiss_score(self.documents, bm25_indices, bm25_scores, faiss_results)
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:self.top_k]
            return [
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=[scores[i] for i in doc_indices],
                    query=query,
                    documents=[self.documents[i] for i in doc_indices],
                )
            ]

        except Exception as e:
            logger.error(
                f"Hybrid retrieval failed: {e}, falling back to pure FAISS search"
            )
            raise


class QueryDrivenRetriever(HybridRetriever):
    """Query-driven retriever that uses BM25 index and on-demand embedding with FAISS."""

    def __init__(
        self,
        documents: List[Union[Document, DualVectorDocument]],
        update_database: Callable = None,
    ):
        """
        Initialize the query-driven retriever.

        Args:
            documents: List of splitted documents to index
            update_database: Function to update the database with new embedded documents
        """
        self.update_database = update_database
        self.query_driven_top_k = configs()["rag"]["query_driven"]['top_k']
        logger.info(f"Query-driven retriever initialized with query_driven_top_k={self.query_driven_top_k}")
        super().__init__(documents)

    def call(self, query: str) -> List[RetrieverOutput]:
        """
        Retrieve documents using BM25 index and on-demand embedding with FAISS.

        Args:
            query: Query string

        Returns:
            List of RetrieverOutput objects
        """

        # Step 1: BM25 filtering to get candidates
        logger.info("Step 1: BM25 filtering")
        bm25_retriever = self._initialize_bm25_retriever(self.documents, top_k=self.query_driven_top_k)
        query_related_doc_indices, query_related_doc_scores = bm25_retriever.filter_and_score(query)

        filtered_docs = [self.documents[i] for i in query_related_doc_indices]
        doc_id_to_score = {doc.id if isinstance(doc, Document) else doc.original_doc.id: score for doc, score in zip(filtered_docs, query_related_doc_scores)}

        self.bm25_documents = deepcopy(filtered_docs)
        # Step 2: Use database manager to embed and cache documents
        logger.info("Step 2: Embedding and caching documents using DatabaseManager")
        if self.update_database:
            embedded_docs = self.update_database(filtered_docs)
        else:
            embedded_docs = filtered_docs
        logger.info(f"Embedded and cached {len(embedded_docs)} documents")

        # embedded_docs may reorder the filtered_docs, so update the query_related_doc_indices nad query_related_doc_scores correspondingly
        query_related_doc_indices = list(range(len(embedded_docs)))
        query_related_doc_scores = [doc_id_to_score[doc.original_doc.id] if isinstance(doc, DualVectorDocument) else doc_id_to_score[doc.id] for doc in embedded_docs]

        try:
            faiss_retriever = self._initialize_faiss_retriever(embedded_docs, top_k=len(embedded_docs))
            faiss_results = faiss_retriever.call(query)
            assert len(faiss_results[0].documents) == len(query_related_doc_indices), f"Mismatch in number of documents between BM25 and FAISS results: {len(query_related_doc_indices)} vs {len(faiss_results[0]['documents'])}"
            scores = self._mix_bm25_score_faiss_score(embedded_docs, query_related_doc_indices, query_related_doc_scores, faiss_results)
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:self.top_k]
            return [
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=[scores[i] for i in doc_indices],
                    query=query,
                    documents=[embedded_docs[i] for i in doc_indices],
                )
            ]

        except Exception as e:
            logger.error(
                f"Query-based retrieval failed: {e}, falling back to pure FAISS search"
            )
            raise