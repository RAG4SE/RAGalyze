#!/usr/bin/env python3
"""
RAGalyze Client - API for code repository analysis with WebSocket support
"""

import requests
import logging
import json
import asyncio
import websockets
import re
import sys
from typing import Dict, Any, List, Optional, Union
from urllib.parse import unquote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default server configuration
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_WS_URL = "ws://localhost:8000/chat"

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
                f"Please make sure the server is running: python -m server.main"
            )
        raise RAGalyzeError(f"Request failed: {str(e)}")
    except Exception as e:
        raise RAGalyzeError(f"Unexpected error: {str(e)}")

async def _websocket_chat(repo_path: str, 
                        messages: List[Dict[str, str]],
                        provider: str = "google",
                        model: Optional[str] = None,
                        excluded_dirs: Optional[List[str]] = None,
                        excluded_files: Optional[List[str]] = None,
                        included_dirs: Optional[List[str]] = None,
                        included_files: Optional[List[str]] = None,
                        file_path: Optional[str] = None,
                        ws_url: str = DEFAULT_WS_URL) -> Dict[str, str]:
    """
    Send a chat request via WebSocket and stream the response
    
    Args:
        repo_path: Local path to the repository
        messages: List of chat messages with role and content
        provider: Model provider (google, openai, etc.)
        model: Model name for the specified provider
        excluded_dirs: List of directory patterns to exclude
        excluded_files: List of file patterns to exclude
        included_dirs: List of directory patterns to include
        included_files: List of file patterns to include
        file_path: Optional path to a file in the repository to include
        ws_url: WebSocket server URL
        
    Returns:
        A dictionary containing:
            - response: The assistant's response text
            - retrieved_documents: The retrieved documents text
            - error_msg: The error message if any
        
    Raises:
        RAGalyzeError: If WebSocket connection fails
    """
    try:
        # Prepare request data
        request_data = {
            "repo_path": repo_path,
            "messages": messages,
            "provider": provider,
            "model": model,
        }
        
        # Add optional parameters if provided
        if excluded_dirs:
            request_data["excluded_dirs"] = "\n".join(excluded_dirs)
        if excluded_files:
            request_data["excluded_files"] = "\n".join(excluded_files)
        if included_dirs:
            request_data["included_dirs"] = "\n".join(included_dirs)
        if included_files:
            request_data["included_files"] = "\n".join(included_files)
        if file_path:
            request_data["filePath"] = file_path
            
        # Connect to WebSocket
        logger.info(f"Connecting to WebSocket at {ws_url}")
        response_text = ""
        
        async with websockets.connect(ws_url) as websocket:
            # Send request
            await websocket.send(json.dumps(request_data))
            logger.info("Request sent, waiting for response...")
            
            # Receive and process streaming response
            # Add a timeout to prevent hanging indefinitely
            in_retrieved_docs = False
            in_response = False
            
            all_msg = ""
            response_msg = ""
            retrieved_docs = ""
            
            timeout = 10.0

            while True:
                try:
                    # Set a timeout for receiving messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    all_msg += message

                    # Track if we're in the retrieved documents section
                    if "## Retrieved Documents:" in message and not in_retrieved_docs:
                        in_retrieved_docs = True
                        in_response = False
                    elif "## Response:" in message and not in_response:
                        in_response = True
                    
                    # Store message in appropriate variable
                    if in_response:
                        response_msg += message
                    elif in_retrieved_docs:
                        retrieved_docs += message
                    else:
                        raise RAGalyzeError(f"Unexpected message format: {message}")
                    
                except asyncio.TimeoutError:
                    # If no message received for 10 seconds, assume the response is complete
                    logger.info(f"No message received for {timeout} seconds, assuming response is complete")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {str(e)}")
                    break
        
        import re

        def strip_blank_lines(s):
            # Remove leading and trailing blank lines (including lines with only whitespace)
            return re.sub(r'^\s*\n', '', re.sub(r'\n\s*$', '', s, flags=re.MULTILINE), flags=re.MULTILINE)

        if in_retrieved_docs:
            # If in_retrieved_docs, response and documents are captured
            return {
                "response": strip_blank_lines(response_msg.split('## Response:')[1].split('[Response complete]')[0]),
                "retrieved_documents": strip_blank_lines(retrieved_docs.split("## Retrieved Documents:")[1].split('[End of Retrieved Documents]')[0]),
                "error_msg": ""
            }
        else:
            # Othereise, the response is error message
            return {
                "response": "",
                "retrieved_documents": "",
                "error_msg": all_msg
            }
        
    except websockets.exceptions.ConnectionClosedError as e:
        raise RAGalyzeError(f"WebSocket connection closed: {str(e)}")
    except websockets.exceptions.WebSocketException as e:
        raise RAGalyzeError(f"WebSocket error: {str(e)}")
    except Exception as e:
        raise RAGalyzeError(f"Unexpected error in WebSocket communication: {str(e)}")

async def direct_websocket_chat(messages: List[Dict[str, str]], repo_path: str = None, ws_url: str = DEFAULT_WS_URL) -> Dict[str, str]:
    """
    Direct WebSocket chat without repository analysis
    
    Args:
        messages: List of chat messages with role and content
        repo_path: Optional repository path to use (if None, uses a placeholder)
        ws_url: WebSocket server URL
        
    Returns:
        A dictionary containing:
            - response: The assistant's response text
            - retrieved_documents: The retrieved documents text
            - full_text: The complete response including both response and retrieved documents
        
    Raises:
        RAGalyzeError: If WebSocket connection fails
    """
    try:
        # Connect to WebSocket
        logger.info(f"Connecting directly to WebSocket at {ws_url}")
        response_text = ""
        
        # If no repo_path is provided, use a placeholder
        # The server expects a repo_path, even for direct chat
        if not repo_path:
            repo_path = "/tmp/placeholder"
        
        async with websockets.connect(ws_url) as websocket:
            # Send request with messages and required repo_path
            # Use openai as provider since it doesn't require an API key in the server configuration
            await websocket.send(json.dumps({
                "messages": messages,
                "repo_path": repo_path,
                "provider": "openai"
            }))
            logger.info("Request sent, waiting for response...")
            
            # Receive and process streaming response
            # Add a timeout to prevent hanging indefinitely
            in_retrieved_docs = False
            retrieved_docs = ""
            assistant_response = ""
            
            while True:
                try:
                    # Set a timeout for receiving messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    
                    # Track if we're in the retrieved documents section
                    if "### File:" in message and not in_retrieved_docs:
                        in_retrieved_docs = True
                        print("\nRetrieved Documents:")
                    
                    # Store message in appropriate variable
                    if in_retrieved_docs:
                        retrieved_docs += message
                        print(message, end="", flush=True)
                    else:
                        # Check if this is an error message
                        if message.startswith("Error:") and not assistant_response:
                            print("\nResponse: ")
                        assistant_response += message
                        print(message, end="", flush=True)
                    
                    # Check for end of retrieved documents
                    if "[End of Retrieved Documents]" in message:
                        in_retrieved_docs = False
                    
                    # Store complete message
                    response_text += message
                    
                    # If we see the end marker, break the loop
                    if "[End of Retrieved Documents]" in message or "[Response complete]" in message:
                        # Wait a short time for any final messages
                        await asyncio.sleep(1.0)
                        break
                except asyncio.TimeoutError:
                    # If no message received for 10 seconds, assume the response is complete
                    logger.info("No message received for 10 seconds, assuming response is complete")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {str(e)}")
                    break
                
        print()  # Add newline after response
        return {
            "response": assistant_response,
            "retrieved_documents": retrieved_docs,
            "full_text": response_text
        }
        
    except websockets.exceptions.ConnectionClosedError as e:
        raise RAGalyzeError(f"WebSocket connection closed: {str(e)}")
    except websockets.exceptions.WebSocketException as e:
        raise RAGalyzeError(f"WebSocket error: {str(e)}")
    except Exception as e:
        raise RAGalyzeError(f"Unexpected error in WebSocket communication: {str(e)}")

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

async def ask_question_websocket(repo_path: str, 
                              question: str,
                              provider: str = "google",
                              model: Optional[str] = None,
                              excluded_dirs: Optional[List[str]] = None,
                              excluded_files: Optional[List[str]] = None,
                              included_dirs: Optional[List[str]] = None,
                              included_files: Optional[List[str]] = None,
                              ws_url: str = DEFAULT_WS_URL) -> Dict[str, str]:
    """
    Ask a question about an analyzed repository using WebSocket connection
    
    Args:
        repo_path: Local path to the repository (must be analyzed first)
        question: Question to ask about the repository
        provider: Model provider (google, openai, etc.)
        model: Model name for the specified provider
        excluded_dirs: List of directory patterns to exclude
        excluded_files: List of file patterns to exclude
        included_dirs: List of directory patterns to include
        included_files: List of file patterns to include
        ws_url: WebSocket server URL
        
    Returns:
        A dictionary containing:
            - response: The assistant's response text
            - retrieved_documents: The retrieved documents text
            - error_msg: The error message if any
        
    Raises:
        RAGalyzeError: If WebSocket connection fails
        
    Example:
        >>> answer = await ask_question_websocket("/path/to/my/repo", "What does this project do?")
        >>> print(answer)
    """
    logger.info(f"Asking question about {repo_path} via WebSocket: {question}")
    
    # Create messages list with the question
    messages = [{"role": "user", "content": question}]
    
    return await _websocket_chat(
        repo_path=repo_path,
        messages=messages,
        provider=provider,
        model=model,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files,
        ws_url=ws_url
    )

def ask_question(repo_path: str, 
                question: str,
                server_url: str = DEFAULT_SERVER_URL) -> Dict[str, Any]:
    """
    Ask a question about an analyzed repository (HTTP API)
    
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

# Simple usage example with WebSocket
async def main_websocket(repo_path: str, question: str):
    
    try:
        # Check server status
        print("🔍 Checking server status...")
        status = get_server_status()
        print(f"✅ Server is running (cached repos: {status['server_info']['cache_count']})")
        
        # Skip analysis and directly ask question using WebSocket
        print(f"\n❓ Question: {question}")
        print("💡 Answer:")
        
        # If history is enabled, we'll use a list of messages
        response = await ask_question_websocket(repo_path, question)
            # The response is already printed by the websocket functions
        return response
                
    except RAGalyzeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)

# Entry point
if __name__ == "__main__":
    import argparse
    
    # Create argument parser with add_help=False to handle -h manually
    parser = argparse.ArgumentParser(description="RAGalyze Client - API for code repository analysis", add_help=False)
    
    # Add arguments
    parser.add_argument("--repo", "-r", help="Local repository path to analyze")
    parser.add_argument("--question", "-q", help="Question to ask about the repository")
    parser.add_argument("--use_history", "-h", action="store_true", help="Load query-answer history")
    parser.add_argument("--help", action="help", help="Show this help message and exit")

    # Parse arguments
    args, unknown = parser.parse_known_args()
    
    #TODO: support use_history
    if args.use_history:
        print("Warning: --use_history is not supported yet")
        args.use_history = False
    
    # Handle positional arguments for backward compatibility
    if not args.repo and len(unknown) > 0:
        args.repo = unknown[0]
        if len(unknown) > 1 and not args.question:
            args.question = unknown[1]
    
    # Check if we have a repository path
    if not args.repo:
        print("Error: Repository path is required")
        parser.print_help()
        sys.exit(1)
    
    # Set default question if not provided
    if not args.question:
        args.question = "What is the main purpose of this project?"
    
    response = asyncio.run(main_websocket(args.repo, args.question))
    
    # Print response
    print('=====response=====')
    print(response['response'])
    print('=====retrieved_documents=====')
    print(response['retrieved_documents'])
    print('=====error_msg=====')
    print(response['error_msg'])