#!/usr/bin/env python3
"""
Script specifically for analyzing Solidity code repositories
Includes better error handling and debugging information
"""

import sys
import os
import logging

# Add api directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from server.rag import RAG
from server.config import configs
from adalflow.core.db import LocalDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_solidity_repository(repo_path: str, db_path: str = None):
    """
    Analyze Solidity code repository
    
    Args:
        repo_path: Local repository path
        db_path: Database save path (optional)
    """
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository path does not exist: {repo_path}")
        return None
    
    if db_path is None:
        # Use default path
        repo_name = os.path.basename(repo_path)
        db_path = f"~/.adalflow/databases/{repo_name}_solidity.db"
        db_path = os.path.expanduser(db_path)
    
    print(f"🔍 Starting Solidity repository analysis: {repo_path}")
    print(f"💾 Database path: {db_path}")
    
    try:
        print("\n🔍 Building RAG system...")
        rag = RAG(is_huggingface_embedder=True)

        
        rag.prepare_retriever(
            repo_path, 
            excluded_dirs=configs["file_filters"]["excluded_dirs"],
            excluded_files=configs["file_filters"]["excluded_files"],
            included_files=None,
            included_dirs=None,
            force_recreate_db=False,
            is_huggingface_embedder=True
        )
        
        print(f"✅ RAG system build completed, loaded {len(rag.transformed_docs)} documents")
        
        return rag
        
    except Exception as e:
        print(f"❌ Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_qa(rag: RAG):
    """
    Interactive Q&A
    
    Args:
        rag: Initialized RAG system
    """
    print("\n" + "="*50)
    print("🤖 Starting interactive Q&A (enter 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            query = input("\n❓ Please enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n🔍 Searching for relevant documents...")
            result = rag.call(query)
            
            if result and len(result) > 0:
                # result[0] is RetrieverOutput, not RAGAnswer
                retriever_output = result[0]
                retrieved_docs = retriever_output.documents if hasattr(retriever_output, 'documents') else []
                
                # Generate answer using the generator
                print("\n🤖 Generating answer...")
                try:
                    
                    # Generate answer using the RAG generator
                    generator_result = rag.generator(
                        prompt_kwargs={
                            "input_str": query,
                            "contexts": retrieved_docs
                        }
                    )
                    
                    if generator_result and hasattr(generator_result, 'data'):
                        rag_answer = generator_result.data
                        print(f"\n💡 Answer: {rag_answer.answer}")
                        if hasattr(rag_answer, 'rationale') and rag_answer.rationale:
                            print(f"📝 Reasoning process: {rag_answer.rationale}")
                    else:
                        print("❌ Unable to generate answer")
                        
                except Exception as e:
                    print(f"❌ Error generating answer: {e}")
                
                if retrieved_docs:
                    print(f"\n📚 Relevant documents ({len(retrieved_docs)} items):")
                    for i, doc in enumerate(retrieved_docs[:3]):  # Only show first 3
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')
                        print(f"   {i+1}. {file_path}")
            else:
                print("❌ Unable to find relevant documents")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error occurred while processing question: {e}")

def main():
    """Main function"""
    print("🚀 RAGalyze Solidity Repository Analysis Tool")
    print("Using HuggingFace multilingual-e5-large-instruct embedding model")
    print("-" * 50)
    
    # Get repository path
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = input("Please enter the Solidity repository path to analyze: ").strip()
    
    if not repo_path:
        print("❌ No repository path provided")
        return
    
    # Analyze repository
    rag = analyze_solidity_repository(repo_path)
    
    if rag:
        print(f"\n✅ Solidity repository analysis completed!")
        
        # Start interactive Q&A
        interactive_qa(rag)
    else:
        print("❌ Solidity repository analysis failed")

if __name__ == "__main__":
    main() 