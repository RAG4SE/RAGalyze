#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def debug_token_boundary_conversion():
    """Debug the token boundary conversion process."""
    
    # Create a custom splitter with detailed debug logging
    class DebugSmartTextSplitter(SmartTextSplitter):
        def _merge_units_to_chunks(self, splits: list, chunk_size: int, chunk_overlap: int, separator: str) -> list:
            print(f"\n=== _merge_units_to_chunks Debug ===")
            print(f"Total tokens: {len(splits)}")
            print(f"Chunk size: {chunk_size}")
            print(f"Smart boundary threshold: {self.smart_boundary_threshold}")
            
            # Get content type like the original method
            full_text = self.tokenizer.decode(splits)
            content_type = self._detect_content_type(full_text)
            print(f"Content type: {content_type}")
            
            chunks = []
            idx = 0
            # chunk_overlap is already a parameter
            
            while idx < len(splits):
                print(f"\n--- Processing chunk starting at token {idx} ---")
                
                chunk_end = min(idx + chunk_size, len(splits))
                chunk_tokens = splits[idx:chunk_end]
                chunk_text_bytes = self.tokenizer.decode(chunk_tokens).encode('utf-8')
                
                print(f"Initial chunk_end: {chunk_end}")
                print(f"Chunk tokens: {len(chunk_tokens)}")
                print(f"Chunk text length: {len(chunk_text_bytes)} bytes")
                print(f"Chunk text: {repr(chunk_text_bytes.decode('utf-8'))}")
                
                # If this is not the last chunk and we have room for smart boundary detection
                if chunk_end < len(splits) and chunk_end - idx >= self.smart_boundary_threshold:
                    print(f"\nApplying smart boundary detection...")
                    
                    # Find smart boundary in the last portion of the chunk
                    search_start_pos = int(len(chunk_text_bytes) * self.smart_boundary_ratio)
                    print(f"Search start position: {search_start_pos}")
                    
                    smart_boundary_pos = self._find_smart_boundary(
                        chunk_text_bytes, search_start_pos, len(chunk_text_bytes), content_type
                    )
                    
                    print(f"Smart boundary position: {smart_boundary_pos}")
                    print(f"Chunk text bytes length: {len(chunk_text_bytes)}")
                    
                    # If we found a good boundary, adjust the chunk end
                    if smart_boundary_pos < len(chunk_text_bytes):
                        print(f"\nAdjusting chunk boundary...")
                        
                        # Re-encode to find the token boundary
                        boundary_text = chunk_text_bytes[:smart_boundary_pos].decode('utf-8')
                        print(f"Boundary text: {repr(boundary_text)}")
                        print(f"Boundary text length: {len(boundary_text)} chars")
                        
                        boundary_tokens = self.tokenizer.encode(boundary_text)
                        print(f"Boundary tokens: {len(boundary_tokens)}")
                        
                        old_chunk_end = chunk_end
                        chunk_end = idx + len(boundary_tokens)
                        print(f"Adjusted chunk_end: {old_chunk_end} -> {chunk_end}")
                        
                        # Update chunk tokens and text
                        chunk_tokens = splits[idx:chunk_end]
                        chunk_text_bytes = self.tokenizer.decode(chunk_tokens).encode('utf-8')
                        
                        print(f"Final chunk text: {repr(chunk_text_bytes.decode('utf-8'))}")
                        print(f"Final chunk length: {len(chunk_text_bytes.decode('utf-8'))} chars")
                
                if chunk_text_bytes.strip():  # Only add non-empty chunks
                    final_text = chunk_text_bytes.decode('utf-8')
                    chunks.append(final_text)
                    print(f"\nAdded chunk {len(chunks)}: {repr(final_text)}")
                
                # Smart overlap calculation
                num_non_overlap_tokens = len(chunk_tokens) - chunk_overlap
                idx += max(1, num_non_overlap_tokens)
                print(f"Next idx: {idx} (advanced by {max(1, num_non_overlap_tokens)})")
            
            print(f"\n=== Final Results ===")
            print(f"Generated {len(chunks)} chunks")
            return chunks
    
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
    
    print(f"Test text: {repr(test_text)}")
    print(f"Text length: {len(test_text)} characters")
    
    chunks = splitter.split_text(test_text)
    
    print(f"\n=== Summary ===")
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        print(f"Chunk {i}: {repr(content)}")
        print(f"  Starts properly: {content[0].isupper() or content[0].isdigit() if content else False}")
        print(f"  Ends properly: {content[-1] in '.!?' if content else False}")

if __name__ == "__main__":
    debug_token_boundary_conversion()