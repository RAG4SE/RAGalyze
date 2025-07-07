#!/usr/bin/env python3
"""
RAGalyze Server - Web API for analyzing code repositories
Based on analyze_solidity_repo.py functionality, providing Web API interface
"""

import sys
import os
import logging
import asyncio
import signal
import subprocess
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import json
from datetime import datetime

# Add api directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from api.data_pipeline import read_all_documents, prepare_data_pipeline, transform_documents_and_save_to_db
from api.rag import RAG
from api.config import configs
from adalflow.core.db import LocalDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG instance cache
rag_cache: Dict[str, RAG] = {}

# Global storage for query results (for web interface)
query_results_cache: List[Dict[str, Any]] = []

class ServerManager:
    """Manages the uvicorn server process with proper signal handling"""
    
    def __init__(self):
        self.server_process = None
        self.shutdown_event = threading.Event()
        self.is_subprocess_mode = False
        
    def signal_handler(self, sig, frame):
        """Handle termination signals"""
        print(f"\n🛑 Received signal {sig}. Shutting down server...")
        self.shutdown_event.set()
        
        if self.server_process and self.is_subprocess_mode:
            try:
                # Send SIGTERM to the server process
                print("📋 Terminating server process...")
                self.server_process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.server_process.wait(timeout=5)
                    print("✅ Server terminated gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️ Server didn't terminate gracefully, forcing kill...")
                    self.server_process.kill()
                    self.server_process.wait()
                    print("✅ Server killed")
                    
            except Exception as e:
                print(f"❌ Error during shutdown: {e}")
        else:
            # For direct uvicorn mode, just exit
            print("🛑 Shutting down server...")
            os._exit(0)
        
        print("🛑 Server manager exiting")
        sys.exit(0)
    
    def run_server_subprocess(self, host="localhost", port=8000, reload=False, log_level="info"):
        """Run the server using subprocess for better process management"""
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)  # Termination
        
        self.is_subprocess_mode = True
        
        print("🚀 Starting RAGalyze Server with advanced process management")
        print(f"📍 Server will be available at: http://{host}:{port}")
        print(f"📚 API documentation: http://{host}:{port}/docs")
        print(f"🌐 Web interface: http://{host}:{port}/web")
        print(f"💡 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Build uvicorn command
        cmd = [
            sys.executable, "-m", "uvicorn",
            "server:app",
            "--host", host,
            "--port", str(port),
            "--log-level", log_level,
            "--access-log"
        ]
        
        if reload:
            cmd.append("--reload")
            print("🔄 Auto-reload enabled for development")
        
        try:
            # Start server process
            print(f"🚀 Starting server: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"📋 Server process started with PID: {self.server_process.pid}")
            
            # Stream server output in a separate thread
            def stream_output():
                try:
                    for line in iter(self.server_process.stdout.readline, ''):
                        if line.strip():
                            print(f"[SERVER] {line.strip()}")
                        if self.shutdown_event.is_set():
                            break
                except Exception as e:
                    if not self.shutdown_event.is_set():
                        print(f"❌ Error streaming output: {e}")
            
            output_thread = threading.Thread(target=stream_output, daemon=True)
            output_thread.start()
            
            # Wait for process to complete or shutdown signal
            while not self.shutdown_event.is_set():
                try:
                    # Check if process is still running
                    exit_code = self.server_process.poll()
                    if exit_code is not None:
                        print(f"🛑 Server process exited with code: {exit_code}")
                        break
                    
                    # Sleep briefly to avoid busy waiting
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    # This should be caught by signal handler, but just in case
                    print("\n🛑 Keyboard interrupt received")
                    self.shutdown_event.set()
                    break
            
            # Final cleanup
            if self.server_process and self.server_process.poll() is None:
                print("🧹 Final cleanup...")
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                    self.server_process.wait()
            
            print("✅ Server shutdown complete")
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return 1
        
        return 0
    
    def run_server_direct(self, host="localhost", port=8000, reload=False, log_level="info"):
        """Run the server directly using uvicorn.Server for simpler mode"""
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)  # Termination
        
        self.is_subprocess_mode = False
        
        print("🚀 Starting RAGalyze Server (direct mode)")
        print(f"📍 Server will be available at: http://{host}:{port}")
        print(f"📚 API documentation: http://{host}:{port}/docs")
        print(f"🌐 Web interface: http://{host}:{port}/web")
        print(f"💡 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        try:
            # Create uvicorn server config
            config = uvicorn.Config(
                "server:app",
                host=host,
                port=port,
                reload=reload,
                log_level=log_level,
                access_log=True,
                use_colors=True,
            )
            server = uvicorn.Server(config)
            server.run()
            
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user (Ctrl+C)")
            return 0
        except Exception as e:
            print(f"\n❌ Server error: {e}")
            return 1
        
        return 0

# FastAPI app
app = FastAPI(
    title="RAGalyze Code Analysis Server",
    description="Code repository analysis service based on HuggingFace embedding models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Request/Response models
class AnalyzeRequest(BaseModel):
    repo_path: str = Field(..., description="Local code repository path")
    force_recreate: bool = Field(False, description="Whether to force recreate database")
    excluded_dirs: Optional[List[str]] = Field(None, description="List of directories to exclude")
    excluded_files: Optional[List[str]] = Field(None, description="List of file patterns to exclude")
    included_dirs: Optional[List[str]] = Field(None, description="List of directories to include")
    included_files: Optional[List[str]] = Field(None, description="List of file patterns to include")

class ChatRequest(BaseModel):
    repo_path: str = Field(..., description="Analyzed code repository path")
    question: str = Field(..., description="Question to ask")

class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    repo_path: str
    document_count: int = 0
    cache_key: str = ""

class ChatResponse(BaseModel):
    success: bool
    question: str
    answer: str = ""
    rationale: str = ""
    relevant_documents: List[Dict[str, Any]] = []
    error_message: str = ""

class StatusResponse(BaseModel):
    status: str
    cached_repos: List[str]
    server_info: Dict[str, Any]

def get_cache_key(repo_path: str) -> str:
    """Generate cache key"""
    return os.path.abspath(repo_path)

def analyze_repository_sync(repo_path: str, **kwargs) -> RAG:
    """
    Synchronously analyze code repository
    
    Args:
        repo_path: Local repository path
        **kwargs: Other parameters
    
    Returns:
        RAG: Initialized RAG system
    """
    
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    logger.info(f"🔍 Starting repository analysis: {repo_path}")
    
    try:
        logger.info("🔍 Building RAG system...")
        # 自动检测是否使用 HuggingFace 嵌入器
        from api.config import should_use_huggingface_embedder
        is_huggingface = should_use_huggingface_embedder()
        logger.info(f"Using {'HuggingFace' if is_huggingface else 'API-based'} embedder based on configuration")
        
        # Get default provider and model from configuration
        default_provider = configs.get("default_provider", "dashscope")
        default_model = None
        if "providers" in configs and default_provider in configs["providers"]:
            default_model = configs["providers"][default_provider].get("default_model")
        
        logger.info(f"Using provider: {default_provider}, model: {default_model}")
        rag = RAG(provider=default_provider, model=default_model, is_huggingface_embedder=is_huggingface)
        
        # Extract parameters
        excluded_dirs = kwargs.get('excluded_dirs', configs["file_filters"]["excluded_dirs"])
        excluded_files = kwargs.get('excluded_files', configs["file_filters"]["excluded_files"])
        included_files = kwargs.get('included_files')
        included_dirs = kwargs.get('included_dirs')
        force_recreate_db = kwargs.get('force_recreate', False)
        
        rag.prepare_retriever(
            repo_path, 
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_files=included_files,
            included_dirs=included_dirs,
            force_recreate_db=force_recreate_db,  # 使用请求中的force_recreate参数
            is_huggingface_embedder=is_huggingface
        )
        
        logger.info(f"✅ RAG system build completed, loaded {len(rag.transformed_docs)} documents")
        return rag
        
    except Exception as e:
        logger.error(f"❌ Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

def query_rag_sync(rag: RAG, question: str) -> Dict[str, Any]:
    """
    Synchronously query RAG system
    
    Args:
        rag: Initialized RAG system
        question: User question
    
    Returns:
        Dict: Query result
    """
    
    try:
        logger.info(f"🔍 Processing question: {question}")
        
        # Call RAG system
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
                
                # Format relevant document information
                relevant_docs = []
                for doc in retrieved_docs[:5]:  # Only return top 5 most relevant documents
                    doc_info = {
                        "file_path": getattr(doc, 'meta_data', {}).get('file_path', 'Unknown'),
                        "content_preview": (doc.text[:200] + "...") if len(doc.text) > 200 else doc.text
                    }
                    relevant_docs.append(doc_info)
                
                # Handle generator response (now returns raw string by design)
                if isinstance(rag_answer, str):
                    # Raw string response (normal case)
                    answer_text = rag_answer.strip()
                    rationale_text = ""
                    logger.info("✅ Generator returned markdown-formatted answer")
                elif hasattr(rag_answer, 'answer') and hasattr(rag_answer, 'rationale'):
                    # Structured RAGAnswer object (legacy case)
                    answer_text = rag_answer.answer
                    rationale_text = rag_answer.rationale
                    logger.info("✅ Generator returned structured RAGAnswer object")
                else:
                    # Unknown format, try to convert to string
                    answer_text = str(rag_answer)
                    rationale_text = ""
                    logger.warning(f"🔧 Generator returned unexpected format: {type(rag_answer)}")
                
                return {
                    "success": True,
                    "answer": answer_text,
                    "rationale": rationale_text,
                    "relevant_documents": relevant_docs
                }
            else:
                return {
                    "success": False,
                    "error_message": "Unable to generate answer"
                }
        else:
            return {
                "success": False,
                "error_message": "Unable to find relevant documents"
            }
            
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        return {
            "success": False,
            "error_message": str(e)
        }

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root path"""
    return {
        "message": "RAGalyze Code Analysis Server",
        "web_interface": "/web",
        "docs": "/docs",
        "status": "/status"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get server status"""
    return StatusResponse(
        status="running",
        cached_repos=list(rag_cache.keys()),
        server_info={
            "embedding_model": "intfloat/multilingual-e5-large-instruct",
            "cache_count": len(rag_cache),
            "python_version": sys.version,
        }
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Analyze code repository and build RAG system
    """
    try:
        repo_path = os.path.abspath(request.repo_path)
        cache_key = get_cache_key(repo_path)
        
        # Check if path exists
        if not os.path.exists(repo_path):
            raise HTTPException(status_code=400, detail=f"Repository path does not exist: {repo_path}")
        
        # Check cache
        if cache_key in rag_cache and not request.force_recreate:
            rag = rag_cache[cache_key]
            return AnalyzeResponse(
                success=True,
                message="Using cached RAG system",
                repo_path=repo_path,
                document_count=len(rag.transformed_docs),
                cache_key=cache_key
            )
        
        # Analyze repository
        logger.info(f"🚀 Starting analysis for: {repo_path}")
        
        # Run analysis in background (simplified as synchronous here, can be made async in production)
        rag = analyze_repository_sync(
            repo_path,
            force_recreate=request.force_recreate,
            excluded_dirs=request.excluded_dirs,
            excluded_files=request.excluded_files,
            included_dirs=request.included_dirs,
            included_files=request.included_files
        )
        
        # Cache result
        rag_cache[cache_key] = rag
        
        return AnalyzeResponse(
            success=True,
            message="Repository analysis completed successfully",
            repo_path=repo_path,
            document_count=len(rag.transformed_docs),
            cache_key=cache_key
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_repository(request: ChatRequest):
    """
    Chat with analyzed code repository
    """
    try:
        repo_path = os.path.abspath(request.repo_path)
        cache_key = get_cache_key(repo_path)
        
        # Check if RAG system exists in cache
        if cache_key not in rag_cache:
            raise HTTPException(
                status_code=400, 
                detail=f"Repository not analyzed yet. Please call /analyze first for: {repo_path}"
            )
        
        rag = rag_cache[cache_key]
        
        # Query RAG system
        result = query_rag_sync(rag, request.question)
        
        if result["success"]:
            # Store query result for web interface
            query_record = {
                "timestamp": datetime.now().isoformat(),
                "repo_path": repo_path,
                "question": request.question,
                "answer": result["answer"],
                "rationale": result.get("rationale", ""),
                "relevant_documents": result.get("relevant_documents", [])
            }
            
            # Keep only last 50 queries
            query_results_cache.append(query_record)
            if len(query_results_cache) > 50:
                query_results_cache.pop(0)
            
            return ChatResponse(
                success=True,
                question=request.question,
                answer=result["answer"],
                rationale=result.get("rationale", ""),
                relevant_documents=result.get("relevant_documents", [])
            )
        else:
            return ChatResponse(
                success=False,
                question=request.question,
                error_message=result.get("error_message", "Unknown error")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/web", response_class=HTMLResponse)
async def web_interface(request: Request):
    """
    Web interface for RAGalyze
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "RAGalyze Code Analysis"
    })

@app.get("/api/queries")
async def get_query_results():
    """
    Get all stored query results for web interface
    """
    return {
        "queries": list(reversed(query_results_cache)),  # Most recent first
        "total_count": len(query_results_cache)
    }

@app.get("/api/queries/latest")
async def get_latest_query():
    """
    Get the latest query result
    """
    if query_results_cache:
        return query_results_cache[-1]
    else:
        return {"message": "No queries yet"}

@app.delete("/cache/{repo_path:path}")
async def clear_cache(repo_path: str):
    """
    Clear cache for specified repository
    """
    cache_key = get_cache_key(repo_path)
    
    if cache_key in rag_cache:
        del rag_cache[cache_key]
        return {"message": f"Cache cleared for: {repo_path}"}
    else:
        raise HTTPException(status_code=404, detail=f"No cache found for: {repo_path}")

@app.delete("/cache")
async def clear_all_cache():
    """
    Clear all cache
    """
    count = len(rag_cache)
    rag_cache.clear()
    return {"message": f"Cleared {count} cached repositories"}

@app.on_event("startup")
async def startup_event():
    """Setup signal handlers after uvicorn starts"""
    logger.info("🚀 Server starting up...")
    
@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("🛑 Server shutting down...")
    rag_cache.clear()

def main():
    """Main entry point for the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGalyze Code Analysis Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--mode", choices=["subprocess", "direct"], default="subprocess", 
                       help="Server execution mode: subprocess (better process management) or direct (simpler)")
    
    args = parser.parse_args()
    
    # Create and run server manager
    server_manager = ServerManager()
    
    if args.mode == "subprocess":
        exit_code = server_manager.run_server_subprocess(
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    else:
        exit_code = server_manager.run_server_direct(
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 