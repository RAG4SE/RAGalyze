import os
import logging
from typing import List, Optional, Union
from server.dual_vector import DualVectorDocument

from adalflow.core.types import Document, ModelType, RetrieverOutput, RetrieverOutputType
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from server.dashscope_client import DashscopeClient
from server.config import configs, get_code_understanding_config
from server.data_pipeline import get_embedder
from adalflow.core.component import DataComponent

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
# System prompt designed specifically for the code understanding task
CODE_UNDERSTANDING_SYSTEM_PROMPT = """
You are an expert programmer and a master of code analysis.
Your task is to provide a concise, high-level summary of the given code snippet.
Focus on the following aspects:
1.  **Purpose**: What is the main goal or functionality of the code?
2.  **Inputs**: What are the key inputs, arguments, or parameters?
3.  **Outputs**: What does the code return or produce?
4.  **Key Logic**: Briefly describe the core logic or algorithm.

Keep the summary in plain language and easy to understand for someone with technical background but not necessarily familiar with this specific code.
Do not get lost in implementation details. Provide a "bird's-eye view" of the code.
The summary should be in English and as concise as possible.
"""


class CodeUnderstandingGenerator:
    """
    Uses the Dashscope model to generate natural language summaries for code.
    """
    def __init__(self, provider=None, model=None):
        """
        Initializes the code understanding generator.
        
        Args:
            provider (str): Provider name. If None, uses default from config.
            model (str): Model name. If None, uses default for the provider.
        """
        # Get configuration using the centralized function
        config = get_code_understanding_config(provider, model)
        
        # Currently only support dashscope
        assert config["provider"] == "dashscope", f"Currently only 'dashscope' provider is supported, got '{config['provider']}'"
        
        # Extract configuration
        self.provider = config["provider"]
        self.model = config["model"]
        model_config = config["model_config"]
        
        # Set model parameters
        self.temperature = model_config.get("temperature", 0.7)
        self.top_p = model_config.get("top_p", 0.8)
        self.max_tokens = model_config.get("max_tokens", 2048)
        
        # Get API configuration from environment
        api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        
        # workspace_id = os.environ.get('DASHSCOPE_WORKSPACE_ID')
        
        # Initialize client
        model_client_class = config["model_client"]
        if model_client_class == DashscopeClient:
            self.client = model_client_class(
                api_key=api_key,
                # workspace_id=workspace_id
            )
        else:
            raise ValueError(f"Unsupported client class: {model_client_class}")
        
        logger.info(f"CodeUnderstandingGenerator initialized with provider: {self.provider}, model: {self.model}")

    def generate_code_understanding(self, code: str, file_path: Optional[str] = None) -> Union[str, None]:
        """
        Generates a summary for the given code snippet.

        Args:
            code: The code string to be summarized.
            file_path: The file path where the code is located (optional).

        Returns:
            The generated code summary string.
        """
        try:
            prompt = f"File Path: `{file_path}`\n\n```\n{code}\n```"
            
            result = self.client.call(
                api_kwargs={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": CODE_UNDERSTANDING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens
                },
                model_type=ModelType.LLM
            )
            
            # Extract content from GeneratorOutput data field
            summary = result.data if hasattr(result, 'data') else str(result)
            logger.debug(f"Successfully generated understanding for {file_path}")
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate code understanding for {file_path}: {e}")
            # Return an empty or default summary on error
            return None


class DualVectorToEmbeddings(DataComponent):
    """
    A data component that transforms documents into dual-vector embeddings,
    including both code and understanding vectors.
    """
    def __init__(self, embedder, force_recreate_db: bool = False, embedding_cache_file_name: str = "default"):
        super().__init__()
        self.embedder = embedder
        self.code_generator = CodeUnderstandingGenerator()
        self.force_recreate_db = force_recreate_db
        self.cache_path = f'./embedding_cache/{embedding_cache_file_name}_dual_vector_embeddings.pkl'

    def __call__(self, documents: List[Document]) -> List[DualVectorDocument]:
        """
        Processes a list of documents to generate and cache dual-vector embeddings.
        """
        import pickle
        if not self.force_recreate_db and os.path.exists(self.cache_path):
            logger.info("Loading cached dual-vector embeddings from %s", self.cache_path)
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        logger.info("Generating dual-vector embeddings for %s documents", len(documents))
        
        dual_docs = []

        from tqdm import tqdm
        for doc in tqdm(documents, desc="Generating dual-vector embeddings"):
            code_embedding_result = self.embedder.call(doc.text)
            code_vector = code_embedding_result.data[0].embedding if not code_embedding_result.error else []
            assert 'is_code' in doc.meta_data, f'No `is_code` key in meta_data: {doc.meta_data}'
            if not doc.meta_data.get('is_code'):
                understanding_text = ''
                # The summary vector is all zero when the understanding text is empty
                # Otherwise, FAISSRetriever will raise an error because summary_vectors are of different lengths.
                summary_vector = [0.0] * len(code_vector)
            else:
                understanding_text = self.code_generator.generate_code_understanding(
                    doc.text, doc.meta_data.get('file_path')
                )
                summary_embedding_result = self.embedder.call(understanding_text)
                summary_vector = summary_embedding_result.data[0].embedding if not summary_embedding_result.error else []

            dual_docs.append(
                DualVectorDocument(
                    original_doc=doc,
                    code_embedding=code_vector,
                    understanding_embedding=summary_vector,
                    understanding_text=understanding_text,
                )
            )

        # Cache the results
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(dual_docs, f)
        
        logger.info("Successfully generated and cached %s dual-vector documents.", len(dual_docs))
        return dual_docs

class DualVectorRetriever:
    """Dual vector retriever: supports hybrid retrieval from code and summary vectors."""
    
    def __init__(self, dual_docs: List[DualVectorDocument], embedder, top_k: int = 20):
        """
        Initializes the dual vector retriever.
        
        Args:
            dual_docs: A list of dual vector documents.
            embedder: The embedder instance.
            top_k: The number of most relevant documents to return.
        """
        self.dual_docs = dual_docs
        self.embedder = embedder
        self.top_k = top_k
        self.doc_map = {doc.original_doc.id: doc for doc in dual_docs}
        
        # Build the two FAISS indexes
        self._build_indices()
        logger.info(f"Dual vector retriever initialization completed, containing {len(dual_docs)} documents")
    
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
                vector=dual_doc.code_embedding
            )
            code_docs.append(faiss_doc)

        self.code_retriever = FAISSRetriever(
            **configs["retriever"],
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
                vector=dual_doc.understanding_embedding
            )
            understanding_docs.append(faiss_doc)

        self.understanding_retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=self.embedder,
            documents=understanding_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Understanding FAISS index built successfully.")

    def call(self, query_str: str) -> RetrieverOutputType:
        """
        Performs hybrid retrieval.
        
        Args:
            query_str: The query string.
            
        Returns:
            A RetrieverOutput object containing the retrieved documents and scores.
        """
        assert isinstance(query_str, str), f"Query must be a string, got {type(query_str)}"

        if not self.dual_docs:
            return RetrieverOutput(
                doc_indices=[],
                doc_scores=[],
                query=query_str,
                documents=[]
            )

        # 1. Retrieve from the code index
        code_results = self.code_retriever.call(query_str, top_k=self.top_k)[0]
        # 2. Retrieve from the summary index
        understanding_results = self.understanding_retriever.call(query_str, top_k=self.top_k)[0]

        # 3. Merge and re-rank the results
        combined_scores = {}

        # Process code results - extract original chunk_id from FAISS document ID
        for i, score in zip(code_results.doc_indices, code_results.doc_scores):
            # Get the document from code retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_code", "")
            combined_scores[original_chunk_id] = score

        # Process understanding results - extract original chunk_id from FAISS document ID
        for i, score in zip(understanding_results.doc_indices, understanding_results.doc_scores):
            # Get the document from understanding retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_understanding", "")
            if original_chunk_id not in combined_scores:
                combined_scores[original_chunk_id] = score
            else:
                combined_scores[original_chunk_id] = max(combined_scores[original_chunk_id], score)

        # 4. Sort and get the top-k results
        # Sort by the combined score in descending order
        sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda chunk_id: combined_scores[chunk_id], reverse=True)

        # 5. Retrieve the full documents for the top-k chunk_ids and create indices mapping
        top_k_docs = []
        doc_indices = []
        doc_scores = []
        for idx, chunk_id in enumerate(sorted_chunk_ids[:min(self.top_k, len(sorted_chunk_ids))]):
            if chunk_id in self.doc_map:
                dual_doc = self.doc_map[chunk_id]
                top_k_docs.append(dual_doc.original_doc)
                doc_indices.append(idx)
                doc_scores.append(combined_scores[chunk_id])

        logger.info(f"Retrieved {len(top_k_docs)} documents after merging code and understanding search results.")
        
        return [
            RetrieverOutput(
                doc_indices=doc_indices,
                doc_scores=doc_scores,
                query=query_str,
                documents=top_k_docs
            )
        ]
