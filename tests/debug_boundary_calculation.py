#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def debug_boundary_calculation():
    """Debug the boundary calculation in detail."""
    
    # Create a custom splitter with detailed debug logging
    class DebugSmartTextSplitter(SmartTextSplitter):
        def _find_spacy_boundary(self, search_text: str, start_pos: int, max_pos: int):
            print(f"\n=== _find_spacy_boundary Debug ===")
            print(f"search_text: {repr(search_text)}")
            print(f"start_pos: {start_pos}, max_pos: {max_pos}")
            
            doc = self.nlp_model(search_text)
            sentences = list(doc.sents)
            
            print(f"Found {len(sentences)} sentences:")
            for i, sent in enumerate(sentences):
                print(f"  Sentence {i}: {repr(sent.text)} (start: {sent.start_char}, end: {sent.end_char})")
                absolute_end = start_pos + sent.end_char
                print(f"    Absolute end position: {absolute_end}")
                if absolute_end <= max_pos:
                    print(f"    ✓ Fits within max_pos ({max_pos})")
                else:
                    print(f"    ✗ Exceeds max_pos ({max_pos})")
            
            # Find the last complete sentence that fits within our range
            best_boundary = None
            for i, sent in enumerate(sentences[:-1]):  # Exclude the last sentence
                sent_end = start_pos + sent.end_char
                if sent_end <= max_pos:
                    best_boundary = sent_end
                    print(f"  Setting best_boundary to {best_boundary} (sentence {i})")
            
            print(f"Final best_boundary: {best_boundary}")
            
            if best_boundary is not None:
                # Show what text would be at this boundary
                full_text = self.current_debug_text  # We'll set this in the parent
                boundary_char = full_text[best_boundary] if best_boundary < len(full_text) else 'EOF'
                print(f"Character at boundary position {best_boundary}: {repr(boundary_char)}")
                print(f"Text before boundary: {repr(full_text[best_boundary-10:best_boundary])}")
                print(f"Text after boundary: {repr(full_text[best_boundary:best_boundary+10])}")
            
            return best_boundary
    
    splitter = DebugSmartTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        split_by='token',
        content_type='text'
    )
    
    # Test text
    test_text = (
        "First sentence here. Second sentence follows. Third sentence ends. "
        "Fourth sentence continues. Fifth sentence completes the test. "
        "Sixth sentence adds more content. Seventh sentence extends further. "
        "Eighth sentence provides additional text. Ninth sentence continues on. "
        "Tenth sentence reaches the end. Eleventh sentence finishes up. "
        "Twelfth sentence concludes everything nicely."
    )
    
    # Store the text for debugging
    splitter.current_debug_text = test_text
    
    print(f"Full test text: {repr(test_text)}")
    print(f"Text length: {len(test_text)} characters")
    print()
    
    # Test the boundary detection directly
    text_bytes = test_text.encode('utf-8')
    start_pos = 238  # From the previous debug output
    max_pos = 298
    
    print(f"Testing boundary detection with start_pos={start_pos}, max_pos={max_pos}")
    print(f"Text segment: {repr(test_text[start_pos:max_pos])}")
    
    boundary = splitter._find_nlp_text_boundary(text_bytes, start_pos, max_pos)
    print(f"\nReturned boundary: {boundary}")
    
    if boundary is not None:
        print(f"Character at boundary: {repr(test_text[boundary])}")
        print(f"Text up to boundary: {repr(test_text[:boundary])}")
        print(f"Text from boundary: {repr(test_text[boundary:])}")

if __name__ == "__main__":
    debug_boundary_calculation()