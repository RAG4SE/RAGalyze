#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def test_overlap_detection():
    """Test the new NLP overlap detection functionality."""
    
    splitter = SmartTextSplitter(
        chunk_size=50,
        chunk_overlap=15,
        split_by='token',
        content_type='text'
    )
    
    # Test text with clear sentence boundaries
    test_text = (
        "First sentence here. Second sentence follows. Third sentence ends. "
        "Fourth sentence continues. Fifth sentence completes the test. "
        "Sixth sentence adds more content. Seventh sentence extends further. "
        "Eighth sentence provides additional text. Ninth sentence continues on. "
        "Tenth sentence reaches the end."
    )
    
    print(f"Test text: {repr(test_text)}")
    print(f"Text length: {len(test_text)} characters")
    print()
    
    # Test the overlap detection directly
    text_bytes = test_text.encode('utf-8')
    
    # Test different desired start positions
    test_positions = [50, 100, 150, 200]
    
    for pos in test_positions:
        if pos < len(text_bytes):
            print(f"Testing overlap detection at position {pos}:")
            print(f"  Context: {repr(test_text[pos:])}")
            
            overlap_start = splitter._find_nlp_overlap_start(text_bytes, pos)
            print(f"  Detected overlap start: {overlap_start}")
            
            if overlap_start < len(test_text):
                print(f"  Text from overlap start: {repr(test_text[overlap_start:])}")
                
                # Check if it starts properly
                if test_text[overlap_start].isupper() or test_text[overlap_start].isdigit():
                    print(f"  ✅ Overlap starts with proper sentence beginning")
                else:
                    print(f"  ❌ Overlap does not start with proper sentence beginning")
            print()
    
    # Now test full splitting
    print("=== Full Splitting Test ===")
    chunks = splitter.split_text(test_text)
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        print(f"\nChunk {i}:")
        print(f"  Length: {len(content)} characters")
        print(f"  Content: {repr(content)}")
        
        # Check boundary preservation
        starts_properly = content[0].isupper() or content[0].isdigit() if content else False
        ends_properly = content[-1] in '.!?' if content else False
        
        print(f"  Starts properly: {starts_properly}")
        print(f"  Ends properly: {ends_properly}")
        
        if starts_properly and ends_properly:
            print(f"  ✅ Chunk {i} preserves sentence boundaries")
        else:
            print(f"  ❌ Chunk {i} does not preserve sentence boundaries")

if __name__ == "__main__":
    test_overlap_detection()