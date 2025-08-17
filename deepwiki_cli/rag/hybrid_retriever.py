import re
from typing import List, Optional, Union, Dict, Any
from rank_bm25 import BM25Okapi

from adalflow.core.types import RetrieverOutput, Document

from deepwiki_cli.configs import get_embedder, configs
from deepwiki_cli.rag.dual_vector import DualVectorDocument
from deepwiki_cli.rag.dual_vector_pipeline import DualVectorRetriever
from deepwiki_cli.rag.single_retriever import SingleVectorRetriever
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


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
            documents: List of documents to index
            embedder: Embedding model for FAISS
            use_bm25: Whether to enable BM25 filtering
            bm25_top_k: Number of documents to retrieve from BM25 before FAISS
            bm25_k1: BM25 k1 parameter (term frequency saturation)
            bm25_b: BM25 b parameter (field length normalization)
            top_k: Final number of documents to return
            use_dual_vector: Whether to use dual vector retrieval
            **kwargs: Additional arguments for FAISS retriever
        """
        self.documents = documents
        self.embedder = get_embedder()

        rag_config = configs()["rag"]
        assert "embedder" in rag_config, "rag_config must contain embedder section"
        self.use_dual_vector = rag_config["embedder"]["sketch_filling"]
        assert "hybrid" in rag_config, "rag_config must contain hybrid section"
        assert (
            "enabled" in rag_config["hybrid"]
        ), "hybrid_config must contain enabled section"
        self.use_bm25 = rag_config["hybrid"]["enabled"]
        assert "bm25" in rag_config["hybrid"], "hybrid_config must contain bm25 section"
        bm25_config = rag_config["hybrid"]["bm25"]
        assert (bm25_config is None) or (
            "top_k" in bm25_config and "k1" in bm25_config and "b" in bm25_config
        ), "bm25_config must contain top_k, k1, and b parameters"
        self.bm25_top_k = bm25_config["top_k"]
        self.bm25_k1 = bm25_config["k1"]
        self.bm25_b = bm25_config["b"]
        assert (
            "retriever" in rag_config
        ), "embedder_config must contain retriever section"
        retriever_config = rag_config["retriever"]
        assert (
            "top_k" in retriever_config
        ), "retriever_config must contain top_k section"
        self.top_k = retriever_config["top_k"]

        # Initialize BM25 if enabled
        if self.use_bm25:
            self._initialize_bm25()
            # FAISS retriever will be initialized later with filtered documents
            self.faiss_retriever = None
        else:
            # Initialize FAISS retriever with all documents when BM25 is disabled
            self._initialize_faiss_retriever(**kwargs)

        logger.info(
            f"Hybrid retriever initialized with BM25={'enabled' if self.use_bm25 else 'disabled'}, "
            f"dual_vector={'enabled' if self.use_dual_vector else 'disabled'}"
        )

    def _initialize_bm25(self):
        """Initialize BM25 index with document texts."""
        try:
            # Extract text content from documents for BM25 indexing
            corpus = []
            for doc in self.documents:
                if isinstance(doc, DualVectorDocument):
                    # For dual vector documents, combine code and understanding text
                    text = doc.original_doc.text + "\n" + doc.understanding_text
                else:
                    text = doc.text

                # Tokenize text for BM25 (simple whitespace + punctuation splitting)
                tokens = self._tokenize_text(text)
                corpus.append(tokens)

            # Initialize BM25 with custom parameters
            self.bm25 = BM25Okapi(corpus, k1=self.bm25_k1, b=self.bm25_b)
            logger.info(f"BM25 index created with {len(corpus)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.use_bm25 = False
            self.bm25 = None

    def _initialize_faiss_retriever(self, **kwargs):
        """Initialize FAISS retriever based on vector type."""
        if self.use_dual_vector:
            self.faiss_retriever = DualVectorRetriever(
                dual_docs=self.documents,
                embedder=self.embedder,
                top_k=self.top_k,
                **kwargs,
            )
        else:
            self.faiss_retriever = SingleVectorRetriever(
                documents=self.documents,
                embedder=self.embedder,
                top_k=self.top_k,
                document_map_func=lambda doc: doc.vector,
                **kwargs,
            )
        logger.info(f"FAISS retriever initialized successfully")

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25 indexing."""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Split on whitespace and common punctuation, keep alphanumeric tokens
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _bm25_filter(self, query: str) -> List[int]:
        """Filter documents using BM25 and return document indices."""
        if not self.use_bm25 or not self.bm25:
            # If BM25 is disabled, return all document indices
            return list(range(len(self.documents)))

        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)

            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k document indices based on BM25 scores
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )

            # Limit to bm25_top_k documents
            filtered_indices = doc_indices[: self.bm25_top_k]

            logger.info(
                f"BM25 filtered {len(self.documents)} documents to {len(filtered_indices)} candidates"
            )
            return filtered_indices

        except Exception as e:
            logger.error(f"BM25 filtering failed: {e}, falling back to all documents")
            raise

    def call(self, query: str, top_k: Optional[int] = None) -> List[RetrieverOutput]:
        """Perform hybrid retrieval combining BM25 and FAISS."""
        if top_k is None:
            top_k = self.top_k

        try:
            if self.use_bm25:
                # Step 1: BM25 filtering
                bm25_indices = self._bm25_filter(query)
                # Step 2: Filter documents
                self.documents = [self.documents[i] for i in bm25_indices]
                # Step 3: Initialize FAISS retriever with filtered documents
                self._initialize_faiss_retriever()

                if not bm25_indices:
                    logger.warning(
                        "BM25 returned no results, falling back to full FAISS search"
                    )
                    # Initialize FAISS with all documents as fallback
                    return self.faiss_retriever.call(query)

                faiss_results = self.faiss_retriever.call(query)

                return faiss_results

            else:
                # BM25 disabled, use pure FAISS search
                logger.info("BM25 disabled, using pure FAISS search")
                return self.faiss_retriever.call(query)

        except Exception as e:
            logger.error(
                f"Hybrid retrieval failed: {e}, falling back to pure FAISS search"
            )
            raise
            # Fall back to pure FAISS search - ensure retriever is initialized
            if self.faiss_retriever is None:
                self._initialize_faiss_retriever()
            return self.faiss_retriever.call(query)
