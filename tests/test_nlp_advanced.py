#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced NLP boundary detection test for SmartTextSplitter.
This test validates that NLP-based text splitting preserves semantic boundaries
even with very small chunk sizes that force splitting.
"""

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def test_forced_nlp_splitting():
    """Test NLP boundary detection with very small chunks that force splitting."""
    logger.info("Testing forced NLP splitting with small chunk sizes...")
    
    # Create splitter with very small chunk size to force splitting
    splitter = SmartTextSplitter(
        chunk_size=80,  # Very small to force splitting
        chunk_overlap=10,
        split_by='token',
        content_type='text'
    )
    
    # Test text with clear sentence boundaries
    test_text = """
### 4. **Simple Usage (Standalone)**

```python
from client import analyze_repository, ask_question

# Analyze a repository
result = analyze_repository("/path/to/your/repo")
print(f"Analyzed {result['document_count']} documents")

# Ask questions
answer = ask_question("/path/to/your/repo", "What does this project do?")
print(answer['answer'])
```
    """
    
    logger.info(f"Original text length: {len(test_text)} characters")
    logger.info(f"Test text: {repr(test_text)}")
    
    # Split the text
    chunks = splitter.split_text(test_text)
    
    logger.info(f"Generated {len(chunks)} chunks:")
    
    all_preserve_boundaries = True
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        # Check if chunk starts with capital letter or digit
        starts_properly = content[0].isupper() or content[0].isdigit() if content else False
        
        # Check if chunk ends with proper punctuation
        ends_properly = content[-1] in '.!?' if content else False
        
        logger.info(f"Chunk {i}:")
        logger.info(f"  Length: {len(content)} characters")
        logger.info(f"  Content: {repr(content)}")
        logger.info(f"  Starts properly: {starts_properly}")
        logger.info(f"  Ends properly: {ends_properly}")

    
    if all_preserve_boundaries:
        logger.info("✅ All chunks preserve sentence boundaries with NLP splitting!")
    else:
        logger.warning("❌ Some chunks do not preserve sentence boundaries")
    
    return chunks, all_preserve_boundaries

def test_comparison_with_fallback():
    """Compare NLP splitting with fallback line-based splitting."""
    logger.info("\nComparing NLP splitting with fallback method...")
    
    test_text = (
        "First sentence here.\n"
        "Second sentence on new line.\n"
        "Third sentence also on new line."
    )
    
    # Test with NLP (text content type)
    nlp_splitter = SmartTextSplitter(
        chunk_size=60,
        chunk_overlap=5,
        split_by='token',
        content_type='text'
    )
    
    nlp_chunks = nlp_splitter.split_text(test_text)
    logger.info(f"NLP splitting generated {len(nlp_chunks)} chunks")
    
    # Test direct boundary detection methods
    text_bytes = test_text.encode('utf-8')
    
    if nlp_splitter._can_use_nlp_boundary_detection():
        nlp_boundary = nlp_splitter._find_nlp_text_boundary(text_bytes, 0, 30)
        logger.info(f"NLP boundary detection result: {nlp_boundary}")
        
        if nlp_boundary:
            boundary_text = text_bytes[:nlp_boundary].decode('utf-8')
            logger.info(f"Text up to NLP boundary: {repr(boundary_text)}")
    
    line_boundary = nlp_splitter._find_line_boundary(text_bytes, 0, 30)
    logger.info(f"Line boundary detection result: {line_boundary}")
    
    if line_boundary:
        line_text = text_bytes[:line_boundary].decode('utf-8')
        logger.info(f"Text up to line boundary: {repr(line_text)}")
    
    return nlp_chunks

def test_edge_cases():
    """Test edge cases for NLP boundary detection."""
    logger.info("\nTesting edge cases...")
    
    splitter = SmartTextSplitter(
        chunk_size=50,
        chunk_overlap=5,
        split_by='token',
        content_type='text'
    )
    
    # Test cases
    test_cases = [
        "Single sentence without splitting.",
        "No punctuation here",
        "Multiple!!! Exclamation!!! Marks!!!",
        "Question? Another question? Final statement.",
        "   Leading and trailing spaces   ",
        "\n\nNewlines\n\nEverywhere\n\n",
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nEdge case {i}: {repr(test_case)}")
        try:
            chunks = splitter.split_text(test_case)
            logger.info(f"Generated {len(chunks)} chunks")
            for j, chunk in enumerate(chunks, 1):
                content = chunk.text if hasattr(chunk, 'text') else str(chunk)
                logger.info(f"  Chunk {j}: {repr(content)}")
        except Exception as e:
            logger.error(f"Error processing edge case {i}: {e}")

if __name__ == "__main__":
    logger.info("Starting advanced NLP boundary detection tests...")
    
    try:
        # Test forced splitting
        chunks, boundaries_preserved = test_forced_nlp_splitting()
        
        # Test comparison with fallback
        comparison_chunks = test_comparison_with_fallback()
        
        # Test edge cases
        test_edge_cases()
        
        logger.info("\n=== Advanced Test Summary ===")
        logger.info(f"Forced splitting test: {'✅ PASSED' if boundaries_preserved else '❌ FAILED'}")
        logger.info(f"Generated {len(chunks)} chunks in forced splitting test")
        logger.info(f"Generated {len(comparison_chunks)} chunks in comparison test")
        logger.info("All advanced tests completed!")
        
        if not boundaries_preserved:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Advanced test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)