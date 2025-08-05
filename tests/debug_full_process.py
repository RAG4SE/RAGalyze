#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def debug_full_splitting_process():
    """Debug the full splitting process to understand where NLP boundary detection fails."""
    
    # Create a custom splitter with debug logging
    class DebugSmartTextSplitter(SmartTextSplitter):
        def _find_smart_boundary(self, text: bytes, start_pos: int, max_pos: int, content_type: str) -> int:
            print(f"\n_find_smart_boundary called:")
            print(f"  start_pos: {start_pos}, max_pos: {max_pos}, content_type: {content_type}")
            print(f"  text segment: {repr(text[start_pos:max_pos].decode('utf-8'))}")
            
            result = super()._find_smart_boundary(text, start_pos, max_pos, content_type)
            print(f"  boundary result: {result}")
            
            if result < max_pos:
                boundary_text = text[start_pos:result].decode('utf-8')
                print(f"  text up to boundary: {repr(boundary_text)}")
            
            return result
        
        def _find_text_boundary(self, text: bytes, start_pos: int, max_pos: int) -> int:
            print(f"\n_find_text_boundary called:")
            print(f"  start_pos: {start_pos}, max_pos: {max_pos}")
            print(f"  can_use_nlp: {self._can_use_nlp_boundary_detection()}")
            
            result = super()._find_text_boundary(text, start_pos, max_pos)
            print(f"  text boundary result: {result}")
            
            return result
        
        def _find_nlp_text_boundary(self, text: bytes, start_pos: int, max_pos: int):
            print(f"\n_find_nlp_text_boundary called:")
            print(f"  start_pos: {start_pos}, max_pos: {max_pos}")
            
            result = super()._find_nlp_text_boundary(text, start_pos, max_pos)
            print(f"  nlp boundary result: {result}")
            
            return result
    
    splitter = DebugSmartTextSplitter(
        chunk_size=50,  # Smaller chunk size to force splitting
        chunk_overlap=10,
        split_by='token',
        content_type='text'
    )
    
    # Longer test text to trigger boundary detection
    test_text = (
        "First sentence here. Second sentence follows. Third sentence ends. "
        "Fourth sentence continues. Fifth sentence completes the test. "
        "Sixth sentence adds more content. Seventh sentence extends further. "
        "Eighth sentence provides additional text. Ninth sentence continues on. "
        "Tenth sentence reaches the end. Eleventh sentence finishes up. "
        "Twelfth sentence concludes everything nicely."
    )
    
    print(f"Test text: {repr(test_text)}")
    print(f"Text length: {len(test_text)} characters")
    print(f"Smart boundary threshold: {splitter.smart_boundary_threshold}")
    print(f"Smart boundary ratio: {splitter.smart_boundary_ratio}")
    print()
    
    chunks = splitter.split_text(test_text)
    
    print(f"\n=== Final Results ===")
    print(f"Generated {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        print(f"\nChunk {i}:")
        print(f"  Length: {len(content)} characters")
        print(f"  Content: {repr(content)}")
        print(f"  Starts properly: {content[0].isupper() or content[0].isdigit() if content else False}")
        print(f"  Ends properly: {content[-1] in '.!?' if content else False}")

if __name__ == "__main__":
    debug_full_splitting_process()