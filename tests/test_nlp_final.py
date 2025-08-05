#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final comprehensive test for NLP-based text boundary detection.
This test uses a longer text to ensure splitting occurs and validates
that sentence boundaries are preserved.
"""

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def test_comprehensive_nlp_splitting():
    """Test NLP boundary detection with a longer text that will definitely be split."""
    logger.info("Testing comprehensive NLP splitting...")
    
    # Create splitter with small chunk size
    splitter = SmartTextSplitter(
        chunk_size=100,  # Small chunk size
        chunk_overlap=20,
        split_by='token',
        content_type='text'
    )
    
    # Longer test text with multiple sentences
    test_text = (
        "Natural language processing is a fascinating field of artificial intelligence. "
        "It involves teaching computers to understand and generate human language. "
        "Machine learning algorithms are used to analyze patterns in text data. "
        "Deep learning models like transformers have revolutionized NLP tasks. "
        "Text preprocessing is crucial for good results in NLP applications. "
        "Tokenization breaks text into smaller units for processing. "
        "Named entity recognition identifies important entities in text. "
        "Sentiment analysis determines the emotional tone of text. "
        "Question answering systems can extract information from documents. "
        "Language models can generate coherent and contextually relevant text."
    )
    
    logger.info(f"Original text length: {len(test_text)} characters")
    logger.info(f"Original text: {test_text[:100]}...")
    
    # Split the text
    chunks = splitter.split_text(test_text)
    
    logger.info(f"Generated {len(chunks)} chunks:")
    
    all_preserve_boundaries = True
    total_length = 0
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        total_length += len(content)
        
        # Check if chunk starts with capital letter or digit
        starts_properly = content[0].isupper() or content[0].isdigit() if content else False
        
        # Check if chunk ends with proper punctuation
        ends_properly = content[-1] in '.!?' if content else False
        
        logger.info(f"\nChunk {i}:")
        logger.info(f"  Length: {len(content)} characters")
        logger.info(f"  Content: {repr(content)}")
        logger.info(f"  Starts properly: {starts_properly}")
        logger.info(f"  Ends properly: {ends_properly}")
        
        if not (starts_properly and ends_properly):
            all_preserve_boundaries = False
            logger.warning(f"  ⚠️ Chunk {i} does not preserve sentence boundaries!")
        else:
            logger.info(f"  ✅ Chunk {i} preserves sentence boundaries")
    
    logger.info(f"\nTotal reconstructed length: {total_length} characters")
    logger.info(f"Original length: {len(test_text)} characters")
    
    if all_preserve_boundaries:
        logger.info("✅ All chunks preserve sentence boundaries with NLP splitting!")
    else:
        logger.warning("❌ Some chunks do not preserve sentence boundaries")
    
    return chunks, all_preserve_boundaries

def test_direct_boundary_methods():
    """Test the boundary detection methods directly."""
    logger.info("\nTesting boundary detection methods directly...")
    
    splitter = SmartTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        split_by='token',
        content_type='text'
    )
    
    test_text = "First sentence here. Second sentence follows. Third sentence ends."
    text_bytes = test_text.encode('utf-8')
    
    logger.info(f"Test text: {repr(test_text)}")
    logger.info(f"Text bytes length: {len(text_bytes)}")
    
    # Test NLP boundary detection
    if splitter._can_use_nlp_boundary_detection():
        logger.info("\nTesting NLP boundary detection:")
        
        # Test different ranges
        test_ranges = [(0, 25), (0, 40), (20, 50), (25, 65)]
        
        for start, end in test_ranges:
            if end <= len(text_bytes):
                boundary = splitter._find_nlp_text_boundary(text_bytes, start, end)
                logger.info(f"  Range [{start}:{end}] -> boundary at {boundary}")
                
                if boundary is not None:
                    text_segment = text_bytes[start:boundary].decode('utf-8')
                    logger.info(f"    Text segment: {repr(text_segment)}")
    
    # Test line boundary detection
    logger.info("\nTesting line boundary detection:")
    for start, end in test_ranges:
        if end <= len(text_bytes):
            boundary = splitter._find_line_boundary(text_bytes, start, end)
            logger.info(f"  Range [{start}:{end}] -> boundary at {boundary}")
            
            if boundary is not None:
                text_segment = text_bytes[start:boundary].decode('utf-8')
                logger.info(f"    Text segment: {repr(text_segment)}")

def test_fallback_behavior():
    """Test fallback behavior when NLP is not available."""
    logger.info("\nTesting fallback behavior...")
    
    # Create splitter with code content type (should not use NLP)
    code_splitter = SmartTextSplitter(
        chunk_size=80,
        chunk_overlap=10,
        split_by='token',
        content_type='code'  # This should not use NLP
    )
    
    test_text = "First sentence. Second sentence. Third sentence."
    
    logger.info(f"Can use NLP boundary detection: {code_splitter._can_use_nlp_boundary_detection()}")
    
    chunks = code_splitter.split_text(test_text)
    logger.info(f"Code splitter generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        logger.info(f"  Chunk {i}: {repr(content)}")

if __name__ == "__main__":
    logger.info("Starting final comprehensive NLP tests...")
    
    try:
        # Test comprehensive splitting
        chunks, boundaries_preserved = test_comprehensive_nlp_splitting()
        
        # Test direct boundary methods
        test_direct_boundary_methods()
        
        # Test fallback behavior
        test_fallback_behavior()
        
        logger.info("\n=== Final Test Summary ===")
        logger.info(f"Comprehensive splitting test: {'✅ PASSED' if boundaries_preserved else '❌ FAILED'}")
        logger.info(f"Generated {len(chunks)} chunks in comprehensive test")
        logger.info("All final tests completed successfully!")
        
        if not boundaries_preserved:
            logger.error("NLP boundary preservation test failed!")
            sys.exit(1)
        else:
            logger.info("🎉 All NLP boundary detection tests passed!")
            
    except Exception as e:
        logger.error(f"Final test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)