#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def debug_nlp_boundary_detection():
    """Debug NLP boundary detection to understand why it's not working."""
    
    splitter = SmartTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        split_by='token',
        content_type='text'
    )
    
    # Test text that should be split
    test_text = (
        "Natural language processing is a fascinating field of artificial intelligence. "
        "It involves teaching computers to understand and generate human language. "
        "Machine learning algorithms are used to analyze patterns in text data. "
        "Deep learning models like transformers have revolutionized NLP tasks."
    )
    
    print(f"Test text: {repr(test_text)}")
    print(f"Text length: {len(test_text)} characters")
    print()
    
    # Test NLP model availability
    print(f"Can use NLP boundary detection: {splitter._can_use_nlp_boundary_detection()}")
    print(f"NLP model available: {splitter.nlp_model is not None}")
    print()
    
    # Test spaCy sentence detection directly
    if splitter.nlp_model is not None:
        doc = splitter.nlp_model(test_text)
        sentences = list(doc.sents)
        print(f"spaCy detected {len(sentences)} sentences:")
        for i, sent in enumerate(sentences, 1):
            print(f"  Sentence {i}: {repr(str(sent))} (start: {sent.start_char}, end: {sent.end_char})")
        print()
    
    # Test boundary detection at different positions
    text_bytes = test_text.encode('utf-8')
    test_ranges = [
        (0, 100),   # Should find boundary around first sentence
        (0, 150),   # Should find boundary around second sentence
        (0, 200),   # Should find boundary around third sentence
        (100, 250), # Should find boundary in middle portion
    ]
    
    for start, end in test_ranges:
        if end <= len(text_bytes):
            print(f"Testing range [{start}:{end}]:")
            print(f"  Text segment: {repr(text_bytes[start:end].decode('utf-8'))}")
            
            # Test NLP boundary detection
            nlp_boundary = splitter._find_nlp_text_boundary(text_bytes, start, end)
            print(f"  NLP boundary: {nlp_boundary}")
            
            if nlp_boundary is not None:
                boundary_text = text_bytes[start:nlp_boundary].decode('utf-8')
                print(f"  Text up to boundary: {repr(boundary_text)}")
                print(f"  Boundary preserves sentence: {boundary_text.strip().endswith(('.', '!', '?'))}")
            
            # Test line boundary detection for comparison
            line_boundary = splitter._find_line_boundary(text_bytes, start, end)
            print(f"  Line boundary: {line_boundary}")
            print()

if __name__ == "__main__":
    debug_nlp_boundary_detection()