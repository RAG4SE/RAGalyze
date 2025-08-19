import os
import logging
from typing import List, Optional, Union
from openai.types.chat import ChatCompletion

import adalflow as adal
from adalflow.core.types import (
    Document,
    ModelType,
    RetrieverOutput,
    RetrieverOutputType,
)
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.component import DataComponent

from deepwiki_cli.clients.dashscope_client import DashScopeClient
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from deepwiki_cli.configs import configs
from deepwiki_cli.rag.dual_vector import DualVectorDocument

logger = get_tqdm_compatible_logger(__name__)
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

CODE_UNDERSTANDING_SYSTEM_PROMPT = """
You are an expert programmer and a master of code analysis.
Your task is to provide a concise, high-level summary of the given code snippet.

Keep the summary in plain language and easy to understand for someone with technical background but not necessarily familiar with this specific code.
Do not get lost in implementation details. Provide a "bird's-eye view" of the code.
The summary should be in English and as concise as possible.
"""


class CodeUnderstandingGenerator:
    """
    Uses the Dashscope model to generate natural language summaries for code.
    """

    def __init__(self, **kwargs):
        """
        Initializes the code understanding generator.

        """
        code_understanding_generator_config = configs()["rag"]["code_understanding"]
        assert (
            "model" in code_understanding_generator_config
        ), f"rag/dual_vector_pipeline.py:model not found in code_understanding_generator_config"
        self.model = code_understanding_generator_config["model"]
        assert (
            "model_client" in code_understanding_generator_config
        ), f"rag/dual_vector_pipeline.py:model_client not found in code_understanding_generator_config"
        model_client = code_understanding_generator_config["model_client"]
        # Initialize client

        # Get API configuration from environment
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        if model_client == DashScopeClient:
            self.client = model_client(
                api_key=api_key,
                # workspace_id=workspace_id
            )
        else:
            raise ValueError(
                f"rag/dual_vector_pipeline.py:Unsupported client class: {model_client.__class__.name}"
            )

        # Extract configuration
        if "model_kwargs" in code_understanding_generator_config:
            self.model_kwargs = code_understanding_generator_config["model_kwargs"]
        else:
            self.model_kwargs = {}

        # Get API configuration from environment
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "rag/dual_vector_pipeline.py:DASHSCOPE_API_KEY environment variable not set"
            )

    def generate_code_understanding(
        self, code: str, file_path: Optional[str] = None
    ) -> Union[str, None]:
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
                        {"role": "user", "content": prompt},
                    ],
                    **self.model_kwargs,
                },
                model_type=ModelType.LLM,
            )

            # Extract content from GeneratorOutput data field
            assert isinstance(result, ChatCompletion), f"result is not a ChatCompletion: {type(result)}"
            summary = result.choices[0].message.content

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

    def __init__(self, embedder: adal.Embedder, generator: CodeUnderstandingGenerator):
        """
        Initialize the DualVectorToEmbeddings component.

        Args:
            embedder: the embedder instance
            generator: the code understanding generator instance
        """
        super().__init__()
        self.embedder = embedder
        self.code_generator = generator

    def __call__(self, documents: List[Document]) -> List[DualVectorDocument]:
        """
        Processes a list of documents to generate and cache dual-vector embeddings.
        """
        logger.info(
            "Generating dual-vector embeddings for %s documents", len(documents)
        )

        dual_docs = []

        from tqdm import tqdm

        for doc in tqdm(documents, desc="Generating dual-vector embeddings"):
            code_embedding_result = self.embedder.call(doc.text)
            code_vector = (
                code_embedding_result.data[0].embedding
                if not code_embedding_result.error
                else []
            )
            assert (
                "is_code" in doc.meta_data
            ), f"rag/dual_vector_pipeline.py:No `is_code` key in meta_data: {doc.meta_data}"
            if not doc.meta_data.get("is_code"):
                understanding_text = ""
                # The summary vector is all zero when the understanding text is empty
                # Otherwise, FAISSRetriever will raise an error because summary_vectors are of different lengths.
                summary_vector = [0.0] * len(code_vector)
            else:
                understanding_text = self.code_generator.generate_code_understanding(
                    doc.text, doc.meta_data.get("file_path")
                )
                summary_embedding_result = self.embedder.call(understanding_text)
                summary_vector = (
                    summary_embedding_result.data[0].embedding
                    if not summary_embedding_result.error
                    else []
                )

            dual_docs.append(
                DualVectorDocument(
                    original_doc=doc,
                    code_embedding=code_vector,
                    understanding_embedding=summary_vector,
                    understanding_text=understanding_text,
                )
            )

        logger.info("Successfully generated %s dual-vector documents.", len(dual_docs))
        return dual_docs


class DualVectorRetriever:
    """Dual vector retriever: supports dual retrieval from code and summary vectors."""

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
