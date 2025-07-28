from logger.logging_config import get_tqdm_compatible_logger
import re
from typing import List, Optional, Union, Dict, Any
from rank_bm25 import BM25Okapi
from adalflow.core.types import RetrieverOutput, Document
from core.dual_vector import DualVectorDocument
from core.dual_vector_pipeline import DualVectorRetriever
from core.single_retriever import SingleVectorRetriever

logger = get_tqdm_compatible_logger(__name__)

class HybridRetriever:
    """
    Hybrid retriever that combines BM25 keyword filtering with FAISS semantic search.
    
    The retrieval process:
    1. BM25 first filters documents by exact keyword matches, reducing the search space
    2. FAISS then performs semantic similarity search on this subset
    3. Results are merged and re-ranked for optimal relevance
    """
    
    def __init__(self, 
                 documents: List[Union[Document, DualVectorDocument]], 
                 embedder,
                 use_bm25: bool = True,
                 bm25_top_k: int = 100,
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.75,
                 top_k: int = 20,
                 use_dual_vector: bool = False,
                 **kwargs):
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
        self.embedder = embedder
        self.use_bm25 = use_bm25
        self.bm25_top_k = bm25_top_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.top_k = top_k
        self.use_dual_vector = use_dual_vector

        # Initialize BM25 if enabled
        if self.use_bm25:
            self._initialize_bm25(bm25_k1, bm25_b)
            # FAISS retriever will be initialized later with filtered documents
            self.faiss_retriever = None
        else:
            # Initialize FAISS retriever with all documents when BM25 is disabled
            self._initialize_faiss_retriever(**kwargs)
        
        logger.info(f"Hybrid retriever initialized with BM25={'enabled' if use_bm25 else 'disabled'}, "
                   f"dual_vector={'enabled' if use_dual_vector else 'disabled'}")
    
    def _initialize_bm25(self, k1: float, b: float):
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
            self.bm25 = BM25Okapi(corpus, k1=k1, b=b)
            logger.info(f"BM25 index created with {len(corpus)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.use_bm25 = False
            self.bm25 = None
    
    def _initialize_faiss_retriever(self, **kwargs):
        """Initialize FAISS retriever based on vector type."""
        try:
            if self.use_dual_vector:
                self.faiss_retriever = DualVectorRetriever(
                    dual_docs=self.documents,
                    embedder=self.embedder,
                    top_k=self.top_k,
                    **kwargs
                )
            else:
                self.faiss_retriever = SingleVectorRetriever(
                    documents=self.documents,
                    embedder=self.embedder,
                    top_k=self.top_k,
                    document_map_func=lambda doc: doc.vector,
                    **kwargs
                )
            logger.info(f"FAISS retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS retriever: {e}")
            raise
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25 indexing."""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Split on whitespace and common punctuation, keep alphanumeric tokens
        tokens = re.findall(r'\b\w+\b', text)
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
            doc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # Limit to bm25_top_k documents
            filtered_indices = doc_indices[:self.bm25_top_k]
            
            logger.info(f"BM25 filtered {len(self.documents)} documents to {len(filtered_indices)} candidates")
            return filtered_indices
            
        except Exception as e:
            logger.error(f"BM25 filtering failed: {e}, falling back to all documents")
            return list(range(len(self.documents)))
    
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
                    logger.warning("BM25 returned no results, falling back to full FAISS search")
                    # Initialize FAISS with all documents as fallback
                    return self.faiss_retriever.call(query)

                faiss_results = self.faiss_retriever.call(query)
                
                return faiss_results
            
            else:
                # BM25 disabled, use pure FAISS search
                logger.info("BM25 disabled, using pure FAISS search")
                return self.faiss_retriever.call(query)
                
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}, falling back to pure FAISS search")
            # Fall back to pure FAISS search - ensure retriever is initialized
            if self.faiss_retriever is None:
                self._initialize_faiss_retriever()
            return self.faiss_retriever.call(query)