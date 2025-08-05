#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def test_boundary_detection():
    """Test the smart boundary detection directly."""
    
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
    
    # Test the boundary detection directly
    text_bytes = test_text.encode('utf-8')
    
    # Test different ranges to see what boundaries are found
    test_ranges = [
        (0, 100),   # Should find boundary around sentence end
        (50, 150),  # Should find boundary around sentence end
        (100, 200), # Should find boundary around sentence end
        (150, 250), # Should find boundary around sentence end
    ]
    
    for start_pos, max_pos in test_ranges:
        if max_pos <= len(text_bytes):
            print(f"Testing boundary detection in range [{start_pos}:{max_pos}]:")
            print(f"  Text in range: {repr(test_text[start_pos:max_pos])}")
            
            # Test with 80% ratio (like in the actual code)
            search_start = start_pos + int((max_pos - start_pos) * 0.8)
            print(f"  Search start (80%): {search_start}")
            print(f"  Search text: {repr(test_text[search_start:max_pos])}")
            
            boundary = splitter._find_smart_boundary(text_bytes, search_start, max_pos, 'text')
            print(f"  Found boundary at: {boundary}")
            
            if boundary < len(test_text):
                print(f"  Character at boundary: {repr(test_text[boundary])}")
                print(f"  Text ending at boundary: {repr(test_text[start_pos:boundary])}")
                
                # Check if it ends properly
                if boundary > start_pos:
                    ending_char = test_text[boundary-1]
                    if ending_char in '.!?':
                        print(f"  ✅ Boundary ends with proper punctuation: '{ending_char}'")
                    else:
                        print(f"  ❌ Boundary does not end with proper punctuation: '{ending_char}'")
            print()
    
    # Now test the actual splitting to see what happens
    print("=== Actual Splitting Test ===")
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
    test_boundary_detection()