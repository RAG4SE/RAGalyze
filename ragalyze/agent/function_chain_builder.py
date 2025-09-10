"""
Function Call Chain Builder Module

This module provides functionality to build bidirectional function call chains:
1. Forward chains: Starting from a function f, find all functions that f calls
2. Backward chains: Starting from a function f, find all functions that call f

The implementation leverages BM25 tokenization with [CALL] and [FUNC] prefixes
and uses the existing RAG components for retrieval and analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from adalflow.core.types import Document
from ragalyze.rag.rag import RAG
from ragalyze.prompts import FIND_FUNCTION_CALL_TEMPLATE
from ragalyze.core.types import DualVectorDocument
import re


@dataclass
class FunctionCallChain:
    """Represents a function call chain with caller and callee relationships."""
    target_function: str
    callers: List[str] = field(default_factory=list)
    callees: List[str] = field(default_factory=list)
    forward_chain: List[str] = field(default_factory=list)
    backward_chain: List[str] = field(default_factory=list)


class FunctionCallChainBuilder:
    """Builds bidirectional function call chains using RAG and BM25 tokenization."""
    
    def __init__(self, repo_path: str):
        """Initialize the chain builder with a repository path.
        
        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = repo_path
        self.rag = RAG()
        self.rag.prepare_retriever(repo_path)
        
    def build_forward_chain(self, function_name: str) -> List[str]:
        """Build forward call chain: find all functions that the given function calls.
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            List of function names that are called by the given function
        """
        # Use BM25 to find documents containing the function
        # The BM25 tokenizer prefixes [FUNC] to function definitions
        bm25_query = f"[FUNC]{function_name}"
        
        # Get documents that contain this function
        retrieved_docs = self.rag.call(bm25_keywords=bm25_query, faiss_query=function_name)
        
        callees = []
        if retrieved_docs:
            # Extract [CALL] prefixed tokens from the documents
            for doc_output in retrieved_docs:
                docs = doc_output.documents
                for doc in docs:
                    # For DualVectorDocument, use the original document
                    if isinstance(doc, DualVectorDocument):
                        text = doc.original_doc.text
                    else:
                        text = doc.text
                    
                    # Find all [CALL] prefixed function names in the text
                    call_matches = re.findall(r'\[CALL\](\w+)', text)
                    callees.extend(call_matches)
        
        return list(set(callees))  # Remove duplicates
    
    def build_backward_chain(self, function_name: str) -> List[str]:
        """Build backward call chain: find all functions that call the given function.
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            List of function names that call the given function
        """
        # Use the FIND_FUNCTION_CALL_TEMPLATE to find callers
        prompt = FIND_FUNCTION_CALL_TEMPLATE.call(function_name=function_name)
        
        # Query the RAG system with this prompt
        result = self.rag.call(bm25_keywords=function_name, faiss_query=prompt)
        
        callers = []
        if result:
            # Extract function names from the results
            for doc_output in result:
                docs = doc_output.documents
                for doc in docs:
                    # For DualVectorDocument, use the original document
                    if isinstance(doc, DualVectorDocument):
                        text = doc.original_doc.text
                    else:
                        text = doc.text
                    
                    # Extract function names from the document text
                    # Look for function definitions in the returned code snippets
                    lines = text.split('\n')
                    for line in lines:
                        # Match function definitions (def or function keyword)
                        func_match = re.search(r'(?:def|function)\s+(\w+)', line)
                        if func_match:
                            callers.append(func_match.group(1))
        
        return list(set(callers))  # Remove duplicates
    
    def build_bidirectional_chain(self, function_name: str) -> FunctionCallChain:
        """Build bidirectional call chain for a function.
        
        Args:
            function_name: Name of the function to analyze
            
        Returns:
            FunctionCallChain object with forward and backward chains
        """
        forward_chain = self.build_forward_chain(function_name)
        backward_chain = self.build_backward_chain(function_name)
        
        return FunctionCallChain(
            target_function=function_name,
            callers=backward_chain,
            callees=forward_chain,
            forward_chain=forward_chain,
            backward_chain=backward_chain
        )
    
    def get_function_context(self, function_name: str) -> List[Document]:
        """Get the context documents for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            List of documents containing the function
        """
        bm25_query = f"[FUNC]{function_name}"
        result = self.rag.call(bm25_keywords=bm25_query, faiss_query=function_name)
        
        documents = []
        if result:
            for doc_output in result:
                documents.extend(doc_output.documents)
                
        return documents