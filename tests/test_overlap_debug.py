#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def test_overlap_debug():
    """Debug the overlap detection in detail."""
    
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
    
    # Simulate what happens in the merging logic
    splits = splitter.tokenizer.encode(test_text)
    print(f"Total tokens: {len(splits)}")
    
    # Simulate first chunk processing
    chunk_size = 50
    chunk_overlap = 15
    idx = 0
    
    print("\n=== First Chunk Processing ===")
    chunk_end = min(idx + chunk_size, len(splits))
    chunk_tokens = splits[idx:chunk_end]
    chunk_text_bytes = splitter.tokenizer.decode(chunk_tokens).encode('utf-8')
    
    print(f"Initial chunk tokens: {len(chunk_tokens)}")
    print(f"Initial chunk text: {repr(chunk_text_bytes.decode('utf-8'))}")
    
    # Apply boundary detection
    if chunk_end < len(splits) and chunk_end - idx >= splitter.smart_boundary_threshold:
        search_start_pos = int(len(chunk_text_bytes) * splitter.smart_boundary_ratio)
        print(f"Search start position: {search_start_pos}")
        
        smart_boundary_pos = splitter._find_smart_boundary(
            chunk_text_bytes, search_start_pos, len(chunk_text_bytes), 'text'
        )
        print(f"Smart boundary position: {smart_boundary_pos}")
        
        if smart_boundary_pos < len(chunk_text_bytes):
            boundary_text = chunk_text_bytes[:smart_boundary_pos].decode('utf-8')
            boundary_tokens = splitter.tokenizer.encode(boundary_text)
            chunk_end = idx + len(boundary_tokens)
            chunk_text_bytes = boundary_text.encode('utf-8')
            
            print(f"Adjusted chunk text: {repr(boundary_text)}")
            print(f"Adjusted chunk tokens: {len(boundary_tokens)}")
    
    first_chunk_text = chunk_text_bytes.decode('utf-8')
    print(f"Final first chunk: {repr(first_chunk_text)}")
    
    # Now simulate overlap calculation for second chunk
    print("\n=== Second Chunk Overlap Calculation ===")
    actual_chunk_tokens = splitter.tokenizer.encode(chunk_text_bytes.decode('utf-8'))
    num_non_overlap_tokens = len(actual_chunk_tokens) - chunk_overlap
    
    print(f"Actual chunk tokens: {len(actual_chunk_tokens)}")
    print(f"Non-overlap tokens: {num_non_overlap_tokens}")
    
    if num_non_overlap_tokens > 0 and chunk_end < len(splits):
        bytetext_before_overlap = splitter.tokenizer.decode(actual_chunk_tokens[:num_non_overlap_tokens]).encode('utf-8')
        desired_start_byte = len(bytetext_before_overlap)
        
        print(f"Text before overlap: {repr(bytetext_before_overlap.decode('utf-8'))}")
        print(f"Desired start byte: {desired_start_byte}")
        print(f"Character at desired start: {repr(first_chunk_text[desired_start_byte])}")
        
        best_start_byte = splitter._find_smart_overlap_start(chunk_text_bytes, desired_start_byte, 'text')
        print(f"Best start byte: {best_start_byte}")
        
        if best_start_byte < len(chunk_text_bytes):
            new_overlap_text = chunk_text_bytes[best_start_byte:].decode('utf-8')
            print(f"New overlap text: {repr(new_overlap_text)}")
            print(f"Character at overlap start: {repr(new_overlap_text[0] if new_overlap_text else 'EMPTY')}")
            
            new_overlap_tokens_count = len(splitter.tokenizer.encode(new_overlap_text))
            adjusted_overlap = new_overlap_tokens_count
            print(f"Adjusted overlap tokens: {adjusted_overlap}")
            
            # New logic: find the token that starts with the overlap text
            overlap_start_text = chunk_text_bytes[best_start_byte:].decode('utf-8')
            next_idx = chunk_end - adjusted_overlap  # fallback
            print(f"Looking for overlap text: {repr(overlap_start_text[:50])}")
            
            # Search through tokens to find the one that matches the overlap start
            for token_idx in range(idx, min(len(splits), chunk_end + 5)):
                # Decode from this token to the end to see if it matches our overlap start
                remaining_tokens = splits[token_idx:]
                remaining_text = splitter.tokenizer.decode(remaining_tokens)
                
                # Check if this token position gives us the overlap text we want
                # Strip leading whitespace for comparison since tokenizer may add spaces
                if remaining_text.lstrip().startswith(overlap_start_text.lstrip()[:min(50, len(overlap_start_text.lstrip()))]):
                    next_idx = token_idx
                    print(f"Found matching token at position {token_idx}: {repr(splitter.tokenizer.decode([splits[token_idx]]))}")
                    break
            else:
                print("No matching token found, using fallback calculation")
        else:
            adjusted_overlap = 0
            next_idx = chunk_end - adjusted_overlap
            print("No overlap adjustment")
    else:
        adjusted_overlap = 0
        next_idx = chunk_end - adjusted_overlap
        print("No overlap calculation needed")
    
    print(f"Next chunk starts at token: {next_idx}")
    
    # Debug: show what's around token 34
    print(f"\n=== Token Analysis ===")
    for i in range(max(0, next_idx-2), min(len(splits), next_idx+3)):
        token_text = splitter.tokenizer.decode([splits[i]])
        marker = " <-- START" if i == next_idx else ""
        print(f"Token {i}: {repr(token_text)}{marker}")
    
    # Show what the second chunk would look like
    if next_idx < len(splits):
        second_chunk_end = min(next_idx + chunk_size, len(splits))
        second_chunk_tokens = splits[next_idx:second_chunk_end]
        second_chunk_text = splitter.tokenizer.decode(second_chunk_tokens)
        print(f"\nSecond chunk text: {repr(second_chunk_text)}")
        print(f"Second chunk starts with: {repr(second_chunk_text[0] if second_chunk_text else 'EMPTY')}")

if __name__ == "__main__":
    test_overlap_debug()