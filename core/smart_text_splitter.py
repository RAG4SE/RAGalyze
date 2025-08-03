"""Smart text splitter that finds appropriate stopping points near chunk boundaries."""

import re
from typing import List, Optional, Dict, Literal
from adalflow.components.data_process import TextSplitter
from adalflow.core.types import Document
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class SmartTextSplitter(TextSplitter):
    """Enhanced TextSplitter that finds intelligent stopping points near chunk boundaries.
    
    This splitter improves upon the base TextSplitter by:
    1. Looking for appropriate stopping points when approaching chunk_size
    2. Using different stopping criteria for code vs text content
    3. Avoiding splits in the middle of function calls, statements, etc.
    """
    
    # Stopping patterns for different content types
    CODE_STOP_PATTERNS = [
        r'\n\s*\n',  # Double newlines (blank lines)
        r';\s*\n',   # Semicolon followed by newline
        r'}\s*\n',   # Closing brace followed by newline
        r'\n\s*#',   # Comment lines
        r'\n\s*//|\n\s*/\*',  # Comment lines (C-style)
        r'\n\s*def\s+|\n\s*class\s+|\n\s*function\s+',  # Function/class definitions
        r'\n\s*if\s+|\n\s*for\s+|\n\s*while\s+',  # Control structures
        r'\n\s*import\s+|\n\s*from\s+',  # Import statements
    ]
    
    TEXT_STOP_PATTERNS = [
        r'\n\s*\n',  # Double newlines (paragraph breaks)
        r'\.\s*\n',  # Sentence endings
        r'\n\s*[-*+]\s+',  # List items
        r'\n\s*\d+\.\s+',  # Numbered lists
        r'\n\s*#{1,6}\s+',  # Markdown headers
        r'\n\s*```',  # Code blocks
    ]
    
    def __init__(self, 
                 split_by: Literal["word", "sentence", "page", "passage", "token"] = "token",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 64,
                 batch_size: int = 1000,
                 separators: Optional[dict] = None,
                 smart_boundary_ratio: float = 0.8,
                 content_type: str = "auto"):
        """
        Initialize SmartTextSplitter.
        
        Args:
            split_by: Same as TextSplitter
            chunk_size: Same as TextSplitter
            chunk_overlap: Same as TextSplitter
            batch_size: Same as TextSplitter
            separators: Same as TextSplitter
            smart_boundary_ratio: When to start looking for smart boundaries (0.8 = 80% of chunk_size)
            content_type: 'code', 'text', or 'auto' for automatic detection
        """
        # Set default separators if None - TextSplitter expects a dict
        if separators is None:
            separators = {
                "word": [" ", "\n"],
                "sentence": [".", "!", "?"],
                "page": ["\n\n"],
                "passage": ["\n\n"],
                "token": []
            }
        
        super().__init__(split_by, chunk_size, chunk_overlap, batch_size, separators)
        self.smart_boundary_ratio = smart_boundary_ratio
        self.content_type = content_type
        self.smart_boundary_threshold = int(chunk_size * smart_boundary_ratio)
        
        logger.info(f"Initialized SmartTextSplitter with smart_boundary_ratio={smart_boundary_ratio}, content_type={content_type}")
    
    def _detect_content_type(self, text: str) -> str:
        """Detect if content is code or text."""
        if self.content_type != "auto":
            return self.content_type
            
        # Simple heuristics for content detection
        code_indicators = ['def ', 'function ', 'class ', 'import ', 'from ', '{', '}', ';\n', '->', '=>']
        text_indicators = ['# ', '## ', '- ', '* ', 'http://', 'https://', '[', '](']
        
        code_score = sum(1 for indicator in code_indicators if indicator in text)
        text_score = sum(1 for indicator in text_indicators if indicator in text)
        
        return 'code' if code_score > text_score else 'text'
    
    def _find_smart_boundary(self, text: str, start_pos: int, max_pos: int, content_type: str) -> int:
        """Find a smart boundary position within the given range.
        
        Args:
            text: Full text to search in
            start_pos: Start position to search from
            max_pos: Maximum position (hard boundary)
            content_type: 'code' or 'text'
            
        Returns:
            int: Best boundary position found, or max_pos if none found
        """
        if start_pos >= max_pos:
            return max_pos
            
        # Choose patterns based on content type
        patterns = self.CODE_STOP_PATTERNS if content_type == 'code' else self.TEXT_STOP_PATTERNS
        
        # Search for patterns in the boundary region
        search_text = text[start_pos:max_pos]
        best_pos = max_pos
        best_priority = -1
        
        for priority, pattern in enumerate(patterns):
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Find the last match (closest to max_pos)
                last_match = matches[-1]
                match_pos = start_pos + last_match.end()
                
                # Prefer higher priority patterns, but also consider position
                if priority > best_priority or (priority == best_priority and match_pos > best_pos):
                    best_pos = match_pos
                    best_priority = priority
        
        return min(best_pos, max_pos)
    
    def _merge_units_to_chunks(self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str) -> List[str]:
        """Enhanced merge method with smart boundary detection."""
        if self.split_by != "token":
            # For non-token splitting, use the original method
            return super()._merge_units_to_chunks(splits, chunk_size, chunk_overlap, separator)
        
        # For token-based splitting, we need to work with the decoded text
        full_text = self.tokenizer.decode(splits)
        content_type = self._detect_content_type(full_text)
        
        chunks = []
        step = chunk_size - chunk_overlap
        idx = 0
        
        logger.debug(f"Merging {len(splits)} tokens with smart boundaries (content_type: {content_type})")
        
        while idx < len(splits):
            # Calculate the end position for this chunk
            chunk_end = min(idx + chunk_size, len(splits))
            
            # If this is not the last chunk and we have room for smart boundary detection
            if chunk_end < len(splits) and chunk_end - idx >= self.smart_boundary_threshold:
                # Decode the current chunk to find smart boundaries
                current_tokens = splits[idx:chunk_end]
                current_text = self.tokenizer.decode(current_tokens)
                
                # Find smart boundary in the last portion of the chunk
                search_start_pos = int(len(current_text) * self.smart_boundary_ratio)
                smart_boundary_pos = self._find_smart_boundary(
                    current_text, search_start_pos, len(current_text), content_type
                )
                
                # If we found a good boundary, adjust the chunk end
                if smart_boundary_pos < len(current_text):
                    # Re-encode to find the token boundary
                    boundary_text = current_text[:smart_boundary_pos]
                    boundary_tokens = self.tokenizer.encode(boundary_text)
                    chunk_end = idx + len(boundary_tokens)
                    
                    logger.debug(f"Found smart boundary at position {smart_boundary_pos} (token {chunk_end})")
            
            # Extract the chunk
            chunk_tokens = splits[idx:chunk_end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Move to next chunk
            if chunk_end >= len(splits):
                break
            
            # Calculate next starting position with overlap
            idx = max(idx + step, chunk_end - chunk_overlap)
        
        logger.info(f"Smart splitting created {len(chunks)} chunks from {len(splits)} tokens")
        return chunks
    
    def _extra_repr(self) -> str:
        base_repr = super()._extra_repr()
        return f"{base_repr}, smart_boundary_ratio={self.smart_boundary_ratio}, content_type={self.content_type}"