#!/usr/bin/env python3
"""
RAGalyze Query Module
"""

import os
import sys
from logger.logging_config import get_tqdm_compatible_logger
from typing import Dict, List, Optional, Any
# Add the parent directory to Python path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback

from rag.rag import RAG

# Setup logging
logger = get_tqdm_compatible_logger(__name__)


def analyze_repository(repo_path: str) -> RAG:
    """
    Analyze a code repository and return a RAG instance.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        RAG: Initialized RAG instance
        
    Raises:
        ValueError: If repository path doesn't exist
        Exception: If analysis fails
    """
    repo_path = os.path.abspath(repo_path)
    
    # Check if path exists
    if not os.path.exists(repo_path):
        raise ValueError(f"core/query.py:Repository path does not exist: {repo_path}")
        

    logger.info(f"🚀 Starting new analysis for: {repo_path}")
    
    # Initialize RAG with dual vector flag
    rag = RAG()

    # Prepare retriever
    rag.prepare_retriever(repo_path)
    
    logger.info(f"✅ Analysis complete for: {repo_path}")

    return rag

def query_repository(repo_path: str, 
                    question: str) -> Dict[str, Any]:
    """
    Query a repository about a question and return structured results.
    
    Args:
        repo_path: Path to the repository to analyze
        question: Question to ask about the repository
        
    Returns:
        Dict containing:
            - response: The assistant's response text
            - retrieved_documents: List of retrieved documents with metadata
            - error_msg: Error message if any (empty string if no error)
    """
    # Analyze repository
    rag = analyze_repository(repo_path=repo_path)
    
    logger.info(f"🔍 Processing question: {question}")
    
    # Call RAG system to retrieve documents
    result = rag.call(question)
    
    if result and len(result) > 0:
        # result[0] is RetrieverOutput
        retriever_output = result[0]
        retrieved_docs = retriever_output.documents if hasattr(retriever_output, 'documents') else []
        
        # Generate answer
        logger.info("🤖 Generating answer...")
        try:
            generator_result = rag.generator(
                prompt_kwargs={
                    "input_str": question,
                    "contexts": retrieved_docs
                }
            )
        except Exception as e:
            logger.error(f"Error calling generator: {e}")
            raise 
        
        # Only use the content of the first choice
        try:
            rag_answer = generator_result.data.strip()
        except Exception as e:
            logger.error(f"Error catching generator result: {e}")
            raise
            
        assert rag_answer, "Generator result is empty"

        logger.info(f"✅ Final answer length: {len(rag_answer)} characters")
            
        return {
            "response": rag_answer,
            "retrieved_documents": retrieved_docs,
            "error_msg": ""
        }
        
    else:
        logger.error("❌ No rag output retrieved")
        raise

def clear_cache():
    """Clear the RAG cache"""
    global rag_cache
    rag_cache.clear()
    logger.info("🧹 RAG cache cleared")

def get_cached_repositories() -> List[str]:
    """Get list of cached repository paths"""
    return list(rag_cache.keys())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query a repository using RAGalyze")
    parser.add_argument("--repo_path", "-r", required=True, help="Path to the repository")
    parser.add_argument("--question", "-q", required=True, help="Question to ask about the repository")
    # Configuration parameters are now loaded from config files
    
    args = parser.parse_args()
    
    try:
        result = query_repository(
            repo_path=args.repo_path,
            question=args.question
        )
    except Exception as e:
        print(traceback.format_exc())
        raise
    
    if result["error_msg"]:
        print(f"Error: {result['error_msg']}")
        raise
    else:
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        # Convert escaped newlines to actual line breaks for better readability
        formatted_response = result["response"].replace('\\n', '\n')
        print(formatted_response)
        
        if result["retrieved_documents"]:
            print("\n" + "="*50)
            print("RETRIEVED DOCUMENTS:")
            print("="*50)
            for i, doc in enumerate(result["retrieved_documents"], 1):
                print(f"\n{i}. File: {getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')}")
                print(f"   Preview: {(doc.text[:200] + "...") if len(doc.text) > 200 else doc.text}")
                # print(f"{doc.text}")