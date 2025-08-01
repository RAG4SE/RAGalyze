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
        
        generator_result = rag.generator(
            prompt_kwargs={
                "input_str": question,
                "contexts": retrieved_docs
            }
        )
        
        if generator_result and hasattr(generator_result, 'data'):
            rag_answer = generator_result.data
            
            # If data is None or empty, try raw_response as fallback
            if rag_answer is None and hasattr(generator_result, 'raw_response'):
                logger.info("🔄 Generator data is None, trying raw_response")
                rag_answer = generator_result.raw_response
                
            # Format relevant document information
            relevant_docs = []
            for doc in retrieved_docs[:5]:  # Only return top 5 most relevant documents
                doc_info = {
                    "file_path": getattr(doc, 'meta_data', {}).get('file_path', 'Unknown'),
                    "content_preview": (doc.text[:200] + "...") if len(doc.text) > 200 else doc.text
                }
                relevant_docs.append(doc_info)
                
            # Extract answer text from various possible formats
            answer_text = ""
            
            # First check for actual ChatCompletion objects
            if hasattr(rag_answer, 'choices') and len(rag_answer.choices) > 0 and hasattr(rag_answer.choices[0], 'message'):
                # ChatCompletion object - extract the actual message content
                try:
                    answer_text = rag_answer.choices[0].message.content
                    logger.info(f"✅ Generator returned ChatCompletion object, extracted message content: {answer_text[:100]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to extract content from ChatCompletion: {e}")
                    answer_text = str(rag_answer)
                    raise
            elif isinstance(rag_answer, str) and 'ChatCompletion' in rag_answer and 'content=' in rag_answer:
                # String representation of ChatCompletion - extract content using regex
                import re
                try:
                    # Extract content from string representation using regex
                    content_match = re.search(r'content=["\']([^"\']*)["\'\)]', rag_answer)
                    if content_match:
                        answer_text = content_match.group(1)
                    else:
                        # Try alternative pattern for multiline content
                        content_match = re.search(r'content="([^"]*(?:\\.[^"]*)*)"|content=\'([^\']*(?:\\.[^\']*)*)\'', rag_answer)
                        if content_match:
                            answer_text = content_match.group(1) or content_match.group(2)
                        else:
                            # Last resort: try to find content between quotes after 'content='
                            content_start = rag_answer.find('content="')
                            if content_start != -1:
                                content_start += len('content="')
                                # Find the matching closing quote, handling escaped quotes
                                quote_count = 0
                                content_end = content_start
                                while content_end < len(rag_answer):
                                    if rag_answer[content_end] == '"' and (content_end == 0 or rag_answer[content_end-1] != '\\'):
                                        break
                                    content_end += 1
                                if content_end < len(rag_answer):
                                    answer_text = rag_answer[content_start:content_end]
                                    # Unescape common escape sequences
                                    answer_text = answer_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\'', '\'').replace('\\\\', '\\')
                    
                    if answer_text:
                        logger.info(f"✅ Generator returned ChatCompletion string, extracted content: {answer_text[:100]}...")
                    else:
                        answer_text = rag_answer
                        logger.warning(f"⚠️ Could not extract content from ChatCompletion string, using full string")
                except Exception as e:
                    logger.error(f"❌ Failed to extract content from ChatCompletion string: {e}")
                    answer_text = rag_answer
                    raise
            elif isinstance(rag_answer, str):
                # Raw string response (ideal case)
                answer_text = rag_answer.strip()
                logger.info(f"✅ Generator returned string answer: {answer_text[:100]}...")
            elif hasattr(rag_answer, 'answer') and hasattr(rag_answer, 'rationale'):
                # Structured RAGAnswer object (legacy case)
                answer_text = str(rag_answer.answer) if rag_answer.answer else ""
                logger.info(f"✅ Generator returned structured RAGAnswer object: {answer_text[:100]}...")
            elif hasattr(rag_answer, 'content'):
                # Some response objects have a content field
                answer_text = str(rag_answer.content)
                logger.info(f"✅ Generator returned object with content field: {answer_text[:100]}...")
            elif hasattr(rag_answer, 'text'):
                # Some response objects have a text field
                answer_text = str(rag_answer.text)
                logger.info(f"✅ Generator returned object with text field: {answer_text[:100]}...")
            else:
                # Fallback: convert whatever we got to string
                answer_text = str(rag_answer)
                logger.warning(f"🔧 Generator returned unexpected format: {type(rag_answer)}, converted to string: {answer_text[:100]}...")
                

                
            # Clean and validate the answer text
            answer_text = answer_text.strip()
            if not answer_text:
                logger.error("❌ Empty answer after processing")
                return {
                    "response": "",
                    "retrieved_documents": relevant_docs,
                    "error_msg": "Empty answer generated"
                }
                
            logger.info(f"✅ Final answer length: {len(answer_text)} characters")
            
            return {
                "response": answer_text,
                "retrieved_documents": relevant_docs,
                "error_msg": ""
            }
        else:
            return {
                "response": "",
                "retrieved_documents": [],
                "error_msg": "Unable to generate answer"
            }
    else:
        return {
            "response": "",
            "retrieved_documents": [],
            "error_msg": "Unable to find relevant documents"
        }

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
        sys.exit(1)
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
                print(f"\n{i}. File: {doc['file_path']}")
                print(f"   Preview: {doc['content_preview']}")