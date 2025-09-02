"""Test script for MyTextSplitter.call method."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adalflow.core.types import Document
from ragalyze.rag.splitter import MyTextSplitter


def test_my_text_splitter_basic():
    """Test basic functionality of MyTextSplitter with start_line metadata."""
    print("Testing MyTextSplitter basic functionality...")
    
    # Create a test document with multiple lines
    test_text = """Line 0: This is the first line.
Line 1: This is the second line.
Line 2: This is the third line.
Line 3: This is the fourth line.
Line 4: This is the fifth line."""
    
    doc = Document(
        text=test_text,
        id="test_doc_1",
        meta_data={"file_path": "test.txt", "original_info": "test"}
    )
    
    # Create splitter with small chunk size to force multiple chunks
    splitter = MyTextSplitter(
        enable_line_number=True,
        split_by="word",
        chunk_size=8,  # Small chunk size to create multiple chunks
        chunk_overlap=2
    )
    
    # Process the document
    result_docs = splitter.call([doc])
    
    newline_count = test_text.count('\n')
    print(f"Original document has {newline_count + 1} lines")
    print(f"Split into {len(result_docs)} chunks")
    print()
    
    # Verify results
    for i, chunk_doc in enumerate(result_docs):
        print(f"Chunk {i}:")
        print(f"  Text: {repr(chunk_doc.text[:50])}{'...' if len(chunk_doc.text) > 50 else ''}")
        print(f"  Start line: {chunk_doc.meta_data.get('start_line', 'NOT_FOUND')}")
        print(f"  Parent doc ID: {chunk_doc.parent_doc_id}")
        print(f"  Order: {chunk_doc.order}")
        print(f"  Original meta_data preserved: {chunk_doc.meta_data.get('original_info', 'NOT_FOUND')}")
        print()
        
        # Verify start_line is present and is a number
        assert "start_line" in chunk_doc.meta_data, f"start_line missing in chunk {i}"
        assert isinstance(chunk_doc.meta_data["start_line"], int), f"start_line should be int in chunk {i}"
        assert chunk_doc.meta_data["start_line"] >= 0, f"start_line should be >= 0 in chunk {i}"
        
        # Verify original meta_data is preserved
        assert chunk_doc.meta_data.get("original_info") == "test", f"Original meta_data not preserved in chunk {i}"
        assert chunk_doc.meta_data.get("file_path") == "test.txt", f"Original meta_data not preserved in chunk {i}"


def test_my_text_splitter_edge_cases():
    """Test edge cases for MyTextSplitter."""
    print("Testing MyTextSplitter edge cases...")
    
    # Test single line document
    single_line_doc = Document(
        text="This is a single line document with many words to test splitting.",
        id="single_line_doc"
    )
    
    splitter = MyTextSplitter(enable_line_number=True, split_by="word", chunk_size=5, chunk_overlap=1)
    result = splitter.call([single_line_doc])
    
    print(f"Single line document split into {len(result)} chunks")
    for chunk in result:
        assert chunk.meta_data["start_line"] == 0, "Single line document should have start_line=0 for all chunks"
    print("✓ Single line test passed")
    
    # Test document with empty lines
    multi_line_doc = Document(
        text="First line\n\nThird line (second was empty)\nFourth line",
        id="multi_line_doc"
    )
    
    result = splitter.call([multi_line_doc])
    print(f"Multi-line document (with empty line) split into {len(result)} chunks")
    for i, chunk in enumerate(result):
        print(f"  Chunk {i}: start_line={chunk.meta_data['start_line']}, text={repr(chunk.text[:30])}")
    print("✓ Multi-line test passed")
    
    # Test empty document
    try:
        empty_doc = Document(text="", id="empty_doc")
        result = splitter.call([empty_doc])
        print(f"Empty document result: {len(result)} chunks")
        print("✓ Empty document test passed")
    except Exception as e:
        print(f"Empty document test failed: {e}")


def test_my_text_splitter_sentence_splitting():
    """Test MyTextSplitter with sentence splitting."""
    print("Testing MyTextSplitter with sentence splitting...")
    
    test_text = """First sentence on line 0.
Second sentence on line 1. Third sentence also on line 1.
Fourth sentence on line 2."""
    
    doc = Document(text=test_text, id="sentence_test")
    
    # Use sentence splitting
    splitter = MyTextSplitter(enable_line_number=True, split_by="sentence", chunk_size=2, chunk_overlap=0)
    result = splitter.call([doc])
    
    print(f"Sentence splitting result: {len(result)} chunks")
    for i, chunk in enumerate(result):
        print(f"  Chunk {i}: start_line={chunk.meta_data['start_line']}, text={repr(chunk.text)}")
    print("✓ Sentence splitting test passed")


def test_multiple_documents():
    """Test MyTextSplitter with multiple documents."""
    print("Testing MyTextSplitter with multiple documents...")
    
    docs = [
        Document(text="Doc1 line1\nDoc1 line2\nDoc1 line3", id="doc1"),
        Document(text="Doc2 line1\nDoc2 line2", id="doc2"),
        Document(text="Doc3 single line", id="doc3")
    ]
    
    splitter = MyTextSplitter(enable_line_number=True, split_by="word", chunk_size=4, chunk_overlap=1)
    result = splitter.call(docs)
    
    print(f"Multiple documents ({len(docs)}) split into {len(result)} chunks")
    
    # Group chunks by parent document
    doc_chunks = {}
    for chunk in result:
        parent_id = chunk.parent_doc_id
        if parent_id not in doc_chunks:
            doc_chunks[parent_id] = []
        doc_chunks[parent_id].append(chunk)
    
    for parent_id, chunks in doc_chunks.items():
        print(f"  {parent_id}: {len(chunks)} chunks")
        for chunk in chunks:
            print(f"{chunk.text}:   start_line={chunk.meta_data['start_line']}, order={chunk.order}")
    
    print("✓ Multiple documents test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MyTextSplitter.call method")
    print("=" * 60)
    
    try:
        test_my_text_splitter_basic()
        print("=" * 60)
        
        test_my_text_splitter_edge_cases()
        print("=" * 60)
        
        test_my_text_splitter_sentence_splitting()
        print("=" * 60)
        
        test_multiple_documents()
        print("=" * 60)
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
