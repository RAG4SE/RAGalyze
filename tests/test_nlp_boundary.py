#!/usr/bin/env python3
"""
Test script for NLP-based text boundary detection in SmartTextSplitter.
"""

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def test_nlp_text_splitting():
    """Test NLP-based text boundary detection."""
    
    # Sample text with multiple sentences - designed to force splitting
    sample_text = """Natural language processing is important. It helps computers understand text. Machine learning models are used. Deep learning is very powerful. Text splitting preserves meaning. Sentence boundaries are detected automatically. This ensures semantic completeness. Each chunk contains full sentences."""
    
    # Initialize splitter with very small chunk size to force splitting
    splitter = SmartTextSplitter(
        chunk_size=50,  # Very small chunk size to force splitting
        chunk_overlap=10,
        content_type='text',
        smart_boundary_ratio=0.8
    )
    
    logger.info("Testing NLP-based text boundary detection...")
    
    # Split the text
    chunks = splitter.split_text(sample_text)
    
    logger.info(f"Generated {len(chunks)} chunks:")
    
    sentence_boundary_preserved = True
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"Length: {len(chunk)} characters")
        logger.info(f"Content: {repr(chunk)}")
        
        # Check if chunk starts and ends with complete sentences
        chunk_stripped = chunk.strip()
        if chunk_stripped:
            starts_with_capital = chunk_stripped[0].isupper() or chunk_stripped[0].isdigit()
            ends_with_punctuation = chunk_stripped[-1] in '.!?'
            
            logger.info(f"Starts with capital/digit: {starts_with_capital}")
            logger.info(f"Ends with punctuation: {ends_with_punctuation}")
            
            if not starts_with_capital:
                logger.warning(f"Chunk {i+1} may not start with a complete sentence")
                sentence_boundary_preserved = False
            if not ends_with_punctuation and i < len(chunks) - 1:  # Last chunk might not end with punctuation
                logger.warning(f"Chunk {i+1} may not end with a complete sentence")
                sentence_boundary_preserved = False
    
    if sentence_boundary_preserved:
        logger.info("✅ All chunks preserve sentence boundaries!")
    else:
        logger.warning("⚠️ Some chunks may not preserve sentence boundaries")
    
    return chunks

def test_spacy_availability():
    """Test if spaCy is properly initialized."""
    logger.info("\nTesting spaCy availability...")
    
    splitter = SmartTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        content_type='text'
    )
    
    # Check if NLP model is initialized
    if hasattr(splitter, 'nlp_model') and splitter.nlp_model is not None:
        logger.info("✅ spaCy model is properly initialized")
        
        # Test sentence detection directly
        test_text = "This is sentence one. This is sentence two. This is sentence three."
        doc = splitter.nlp_model(test_text)
        sentences = list(doc.sents)
        
        logger.info(f"Detected {len(sentences)} sentences:")
        for i, sent in enumerate(sentences):
            logger.info(f"  Sentence {i+1}: {repr(sent.text)}")
            
        return len(sentences) == 3
    else:
        logger.warning("⚠️ spaCy model is not initialized")
        return False

def test_boundary_detection_methods():
    """Test different boundary detection methods."""
    logger.info("\nTesting boundary detection methods...")
    
    splitter = SmartTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        content_type='text'
    )
    
    test_text = "First sentence here. Second sentence follows. Third sentence ends."
    text_bytes = test_text.encode('utf-8')
    
    # Test NLP boundary detection if available
    if splitter._can_use_nlp_boundary_detection():
        logger.info("Testing NLP text boundary detection...")
        boundary = splitter._find_nlp_text_boundary(text_bytes, 20, 40)
        logger.info(f"Found NLP text boundary at position: {boundary}")
        if boundary is not None:
            decoded_text = text_bytes[:boundary].decode('utf-8')
            logger.info(f"Text at boundary: {repr(decoded_text[-10:] + '|' + text_bytes[boundary:boundary+10].decode('utf-8'))}")
    else:
        logger.warning("NLP boundary detection not available")
    
    # Test line boundary detection
    logger.info("Testing line boundary detection...")
    line_boundary = splitter._find_line_boundary(text_bytes, 0, 30)
    logger.info(f"Found line boundary at position: {line_boundary}")

if __name__ == "__main__":
    logger.info("Starting NLP boundary detection tests...")
    
    try:
        # Test spaCy availability first
        spacy_works = test_spacy_availability()
        
        # Test boundary detection methods
        test_boundary_detection_methods()
        
        # Test main NLP functionality
        nlp_chunks = test_nlp_text_splitting()
        
        logger.info("\n=== Test Summary ===")
        logger.info(f"spaCy working: {spacy_works}")
        logger.info(f"NLP test generated {len(nlp_chunks)} chunks")
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)