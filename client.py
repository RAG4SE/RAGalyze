#!/usr/bin/env python3
"""
RAGalyze Client - Simple API for code repository analysis
"""

import requests
import logging
from typing import Dict, Any, List, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default server configuration
DEFAULT_SERVER_URL = "http://localhost:8000"

class RAGalyzeError(Exception):
    """Custom exception for RAGalyze errors"""
    pass

def _make_request(method: str, endpoint: str, data: dict = None, server_url: str = DEFAULT_SERVER_URL) -> Dict[str, Any]:
    """
    Make HTTP request to RAGalyze server
    
    Args:
        method: HTTP method (GET, POST, DELETE)
        endpoint: API endpoint
        data: Request data for POST requests
        server_url: Server URL
        
    Returns:
        Response data
        
    Raises:
        RAGalyzeError: If request fails
    """
    try:
        url = f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url)
        else:
            raise RAGalyzeError(f"Unsupported HTTP method: {method}")
            
        response.raise_for_status()
        return response.json()
        
    except requests.RequestException as e:
        if "Connection refused" in str(e):
            raise RAGalyzeError(
                f"Cannot connect to RAGalyze server at {server_url}. "
                f"Please make sure the server is running: python server.py"
            )
        raise RAGalyzeError(f"Request failed: {str(e)}")
    except Exception as e:
        raise RAGalyzeError(f"Unexpected error: {str(e)}")

def analyze_repository(repo_path: str, 
                      force_recreate: bool = False,
                      excluded_dirs: Optional[List[str]] = None,
                      excluded_files: Optional[List[str]] = None,
                      included_dirs: Optional[List[str]] = None,
                      included_files: Optional[List[str]] = None,
                      server_url: str = DEFAULT_SERVER_URL) -> Dict[str, Any]:
    """
    Analyze a code repository and build RAG system
    
    Args:
        repo_path: Local path to the repository
        force_recreate: Whether to force recreate the database (ignore cache)
        excluded_dirs: List of directory patterns to exclude
        excluded_files: List of file patterns to exclude  
        included_dirs: List of directory patterns to include
        included_files: List of file patterns to include
        server_url: RAGalyze server URL
        
    Returns:
        Analysis result with success status, document count, etc.
        
    Raises:
        RAGalyzeError: If analysis fails
        
    Example:
        >>> result = analyze_repository("/path/to/my/repo")
        >>> print(f"Analyzed {result['document_count']} documents")
    """
    logger.info(f"Starting repository analysis: {repo_path}")
    
    request_data = {
        "repo_path": repo_path,
        "force_recreate": force_recreate
    }
    
    if excluded_dirs:
        request_data["excluded_dirs"] = excluded_dirs
    if excluded_files:
        request_data["excluded_files"] = excluded_files
    if included_dirs:
        request_data["included_dirs"] = included_dirs
    if included_files:
        request_data["included_files"] = included_files
    
    try:
        result = _make_request("POST", "/analyze", request_data, server_url)
        
        if not result.get("success"):
            raise RAGalyzeError(f"Analysis failed: {result.get('message', 'Unknown error')}")
            
        logger.info(f"Analysis completed successfully: {result['document_count']} documents processed")
        return result
        
    except RAGalyzeError:
        raise
    except Exception as e:
        raise RAGalyzeError(f"Analysis failed: {str(e)}")

def ask_question(repo_path: str, 
                question: str,
                server_url: str = DEFAULT_SERVER_URL) -> Dict[str, Any]:
    """
    Ask a question about an analyzed repository
    
    Args:
        repo_path: Local path to the repository (must be analyzed first)
        question: Question to ask about the repository
        server_url: RAGalyze server URL
        
    Returns:
        Answer with relevant documents and reasoning
        
    Raises:
        RAGalyzeError: If query fails or repository not analyzed
        
    Example:
        >>> answer = ask_question("/path/to/my/repo", "What does this project do?")
        >>> print(answer['answer'])
        >>> for doc in answer['relevant_documents']:
        ...     print(f"- {doc['file_path']}")
    """
    logger.info(f"Asking question about {repo_path}: {question}")
    
    request_data = {
        "repo_path": repo_path,
        "question": question
    }
    
    try:
        result = _make_request("POST", "/chat", request_data, server_url)
        
        if not result.get("success"):
            error_msg = result.get("error_message", "Unknown error")
            if "not analyzed yet" in error_msg:
                raise RAGalyzeError(
                    f"Repository not analyzed yet. Please call analyze_repository('{repo_path}') first."
                )
            raise RAGalyzeError(f"Query failed: {error_msg}")
            
        logger.info("Question answered successfully")
        return result
        
    except RAGalyzeError:
        raise
    except Exception as e:
        raise RAGalyzeError(f"Query failed: {str(e)}")

def get_server_status(server_url: str = DEFAULT_SERVER_URL) -> Dict[str, Any]:
    """
    Get RAGalyze server status and information
    
    Args:
        server_url: RAGalyze server URL
        
    Returns:
        Server status information
        
    Raises:
        RAGalyzeError: If server is not accessible
    """
    try:
        return _make_request("GET", "/status", server_url=server_url)
    except RAGalyzeError:
        raise
    except Exception as e:
        raise RAGalyzeError(f"Failed to get server status: {str(e)}")

def clear_cache(repo_path: Optional[str] = None, 
               server_url: str = DEFAULT_SERVER_URL) -> Dict[str, Any]:
    """
    Clear repository cache
    
    Args:
        repo_path: Specific repository path to clear (if None, clears all cache)
        server_url: RAGalyze server URL
        
    Returns:
        Cache clearing result
        
    Raises:
        RAGalyzeError: If cache clearing fails
    """
    try:
        if repo_path:
            endpoint = f"/cache/{repo_path}"
        else:
            endpoint = "/cache"
            
        return _make_request("DELETE", endpoint, server_url=server_url)
        
    except RAGalyzeError:
        raise
    except Exception as e:
        raise RAGalyzeError(f"Failed to clear cache: {str(e)}")

# Simple usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <repo_path> [question]")
        print("Example: python client.py /path/to/repo 'What does this project do?'")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the main purpose of this project?"
    
    try:
        # Check server status
        print("🔍 Checking server status...")
        status = get_server_status()
        print(f"✅ Server is running (cached repos: {status['server_info']['cache_count']})")
        
        # Analyze repository
        print(f"\n📊 Analyzing repository: {repo_path}")
        analysis_result = analyze_repository(repo_path)
        print(f"✅ Analysis complete: {analysis_result['document_count']} documents")
        
        # Ask question
        print(f"\n❓ Question: {question}")
        answer_result = ask_question(repo_path, question)
        
        # Format the answer - handle ChatCompletion objects
        answer_text = answer_result['answer']
        if 'ChatCompletion' in str(answer_text) and 'choices' in str(answer_text):
            # Try to extract content from ChatCompletion object string
            try:
                # Look for content pattern in the string representation
                content_match = re.search(r"content='([^']*)'", str(answer_text))
                if content_match:
                    answer_text = content_match.group(1)
                    # Unescape common escape sequences
                    answer_text = answer_text.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                else:
                    # Fallback: try to find content in a different format
                    content_match = re.search(r'content="([^"]*)"', str(answer_text))
                    if content_match:
                        answer_text = content_match.group(1)
                        answer_text = answer_text.replace('\\n', '\n').replace('\\"', '"')
            except Exception as e:
                print(f"⚠️ Warning: Could not extract content from response: {e}")
                # Keep original answer as fallback
        
        print(f"\n💡 Answer: {answer_text}")
        
        if answer_result.get('relevant_documents'):
            print(f"\n📚 Relevant files:")
            for doc in answer_result['relevant_documents'][:3]:
                print(f"  - {doc['file_path']}")
                
    except RAGalyzeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0) 