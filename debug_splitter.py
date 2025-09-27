#!/usr/bin/env python3

import sys
import traceback
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

from ragalyze.rag.splitter.custom_text_splitter import MyTextSplitter
from adalflow.core.types import Document

print("Import successful")

def test_simple():
    """Test with simple document"""
    print("Testing simple document...")

    try:
        splitter = MyTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc = Document(text="Hello world! This is a simple test.")

        print("About to call splitter...")
        result = splitter.call([doc])
        print(f"Success! Got {len(result)} chunks")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True

def test_with_bm25():
    """Test with BM25 indexes"""
    print("Testing with BM25 indexes...")

    try:
        splitter = MyTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Create a mock BM25 index
        from types import SimpleNamespace
        bm25_index = SimpleNamespace(
            token="test_token",
            position=(1, 5)
        )

        doc = Document(
            text="Hello world! This is a test with BM25 indexes.",
            meta_data={"bm25_indexes": [bm25_index]}
        )

        print("About to call splitter with BM25...")
        result = splitter.call([doc])
        print(f"Success! Got {len(result)} chunks")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True

def test_large_document():
    """Test with larger document"""
    print("Testing large document...")

    try:
        splitter = MyTextSplitter(chunk_size=500, chunk_overlap=100)

        # Create a larger text
        large_text = "This is a test sentence. " * 100
        doc = Document(text=large_text)

        print("About to call splitter with large document...")
        result = splitter.call([doc])
        print(f"Success! Got {len(result)} chunks")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    print("Testing C text splitter implementation...")

    # Test simple case first
    if not test_simple():
        sys.exit(1)

    # Test with BM25
    if not test_with_bm25():
        sys.exit(1)

    # Test large document
    if not test_large_document():
        sys.exit(1)

    print("All tests passed!")