#!/usr/bin/env python3
"""
Test the functionality of the dual-vector system.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from adalflow.core.types import Document
from api.dual_vector_pipeline import CodeUnderstandingGenerator
from api.rag import RAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_code_understanding_generator():
    """Test the Code Understanding Generator."""
    print("\n=== Testing Code Understanding Generator ===")
    
    try:
        generator = CodeUnderstandingGenerator()
        
        # Test code snippet
        test_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    print(fibonacci(10))

if __name__ == "__main__":
    main()
'''
        
        understanding = generator.generate_code_understanding(test_code, "test.py")
        if understanding == None:
            print('Failed to generate code understanding')
            return False
        print(f"Generated code understanding: {understanding}")
        return True
        
    except Exception as e:
        print(f"❌ Code Understanding Generator test failed: {e}")
        return False

def test_rag_with_dual_vector():
    """Test RAG system integration with dual-vector functionality."""
    print("\n=== Testing RAG Dual-Vector Integration ===")
    
    try:
        # Create a RAG system with dual-vector enabled
        rag = RAG(is_huggingface_embedder=True, use_dual_vector=True)
        
        # Use the current directory as the test repository
        current_dir = os.getcwd()
        print(f"Using directory: {current_dir}")
        
        # Prepare retriever (only including Python files)
        rag.prepare_retriever(
            repo_path=current_dir,
            included_dirs=["api"],
            included_files=None,  # No file type restrictions, let the system discover automatically
            excluded_dirs=[".git", "__pycache__", ".pytest_cache", "venv", "env", "cache", ".adalflow"],
            excluded_files=["*.pyc", "*.pyo", "*.egg-info", "*.dist-info"],
            force_recreate_db=True,
            is_huggingface_embedder=True,
            file_count_upperlimit=1
        )
        
        # Test query
        test_query = "What are these files about?"
        
        results = rag.call(test_query)
        print('>>>', results)
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'documents') and result.documents:
                print(f"✅ Successfully retrieved {len(result.documents)} documents")
                for i, doc in enumerate(result.documents[:3]):  # Display only the top 3
                    print(f"Document {i+1}: {doc.meta_data.get('file_path', 'unknown')}")
                    print(f"Document: {doc.text}")
                    #TODO: understanding_text is not in the meta_data, put it in the meta_data
                    if 'understanding_text' in doc.meta_data:
                        print(f"  Understanding text: {doc.meta_data['understanding_text'][:100]}...")
            else:
                print("❌ No documents retrieved")
                return False
        else:
            print("❌ RAG call returned empty result")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ RAG dual-vector integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting tests for the dual-vector system...")
    
    test_results = []
    
    # # Test 1: Code Understanding Generator
    # test_results.append(("Code Understanding Generator", test_code_understanding_generator()))
    
    # Test 2: RAG Dual-Vector Integration
    test_results.append(("RAG Dual-Vector Integration", test_rag_with_dual_vector()))
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dual-vector system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed, please check the error messages.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 