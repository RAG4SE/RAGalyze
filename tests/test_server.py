#!/usr/bin/env python3
"""
RAGalyze Server Connection Test - Interactive client for testing server connection
This file tests the connection and functionality of the RAGalyze server
"""

import requests
import json
import sys
import time
from typing import Dict, Any

class RAGalyzeClient:
    """RAGalyze server client"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def check_status(self) -> Dict[str, Any]:
        """Check server status"""
        try:
            response = requests.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def analyze_repository(self, repo_path: str, force_recreate: bool = False, **kwargs) -> Dict[str, Any]:
        """Analyze code repository"""
        data = {
            "repo_path": repo_path,
            "force_recreate": force_recreate,
            **kwargs
        }
        
        try:
            print(f"🔍 Analyzing repository: {repo_path}")
            response = requests.post(f"{self.base_url}/analyze", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def chat(self, repo_path: str, question: str) -> Dict[str, Any]:
        """Chat with repository"""
        data = {
            "repo_path": repo_path,
            "question": question
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def clear_cache(self, repo_path: str = None) -> Dict[str, Any]:
        """Clear cache"""
        try:
            if repo_path:
                response = requests.delete(f"{self.base_url}/cache/{repo_path}")
            else:
                response = requests.delete(f"{self.base_url}/cache")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

def demo_usage():
    """Demo usage"""
    client = RAGalyzeClient()
    
    # Check server status
    print("🔍 Checking server status...")
    status = client.check_status()
    if "error" in status:
        print(f"❌ Server connection failed: {status['error']}")
        print("Please make sure the server is running: python -m server.main")
        return
    
    print(f"✅ Server status: {status['status']}")
    print(f"📚 Using embedding model: {status['server_info']['embedding_model']}")
    print(f"💾 Cached repositories count: {status['server_info']['cache_count']}")
    
    # Get repository path
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = input("\nPlease enter the repository path to analyze: ").strip()
    
    if not repo_path:
        print("❌ No repository path provided")
        return
    
    # Analyze repository
    print(f"\n🚀 Starting repository analysis: {repo_path}")
    result = client.analyze_repository(repo_path)
    
    if "error" in result:
        print(f"❌ Analysis failed: {result['error']}")
        return
    
    if result["success"]:
        print(f"✅ Analysis completed!")
        print(f"📄 Document count: {result['document_count']}")
        print(f"💾 Cache key: {result['cache_key']}")
        print(f"📝 Status: {result['message']}")
    else:
        print(f"❌ Analysis failed")
        return
    
    # Interactive Q&A
    print(f"\n🤖 Start chatting with {repo_path} (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Please enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🔍 Searching for relevant information...")
            chat_result = client.chat(repo_path, question)
            
            if "error" in chat_result:
                print(f"❌ Query failed: {chat_result['error']}")
                continue
            
            if chat_result["success"]:
                print(f"\n💡 Answer:")
                print(f"{chat_result['answer']}")
                
                if chat_result.get("rationale"):
                    print(f"\n📝 Reasoning process:")
                    print(f"{chat_result['rationale']}")
                
                relevant_docs = chat_result.get("relevant_documents", [])
                if relevant_docs:
                    print(f"\n📚 Relevant documents ({len(relevant_docs)} items):")
                    for i, doc in enumerate(relevant_docs, 1):
                        print(f"   {i}. {doc['file_path']}")
                        print(f"      Preview: {doc['content_preview']}")
            else:
                print(f"❌ Query failed: {chat_result.get('error_message', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

def interactive_mode():
    """Interactive mode"""
    client = RAGalyzeClient()
    
    print("🚀 RAGalyze Client - Interactive Mode")
    print("Available commands:")
    print("  status - Check server status")
    print("  analyze <path> - Analyze repository")
    print("  chat <path> <question> - Ask question")
    print("  clear [path] - Clear cache")
    print("  demo [path] - Run demo")
    print("  quit - Exit")
    print("-" * 50)
    
    while True:
        try:
            command = input("\n> ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif cmd == 'status':
                status = client.check_status()
                print(json.dumps(status, indent=2, ensure_ascii=False))
            elif cmd == 'analyze':
                if len(command) < 2:
                    print("❌ Usage: analyze <repo_path>")
                    continue
                result = client.analyze_repository(command[1])
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif cmd == 'chat':
                if len(command) < 3:
                    print("❌ Usage: chat <repo_path> <question>")
                    continue
                repo_path = command[1]
                question = ' '.join(command[2:])
                result = client.chat(repo_path, question)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif cmd == 'clear':
                repo_path = command[1] if len(command) > 1 else None
                result = client.clear_cache(repo_path)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif cmd == 'demo':
                repo_path = command[1] if len(command) > 1 else None
                if repo_path:
                    sys.argv = ['client.py', repo_path]
                demo_usage()
            else:
                print(f"❌ Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        demo_usage()