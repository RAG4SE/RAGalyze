"""Custom code tokenizer optimized for programming languages."""

import tiktoken
from typing import List
from adalflow.core.tokenizer import Tokenizer


class CodeTokenizer(Tokenizer):
    """Custom tokenizer optimized for code analysis.
    
    This tokenizer extends the base Tokenizer with code-specific preprocessing
    and tokenization strategies that better handle programming constructs.
    
    Args:
        name (str, optional): The name of the tokenizer. Defaults to "cl100k_base".
        preserve_code_structure (bool, optional): Whether to preserve code structure
            like indentation and line breaks. Defaults to True.
        remove_stop_words (bool, optional): Whether to remove common stop words.
            Defaults to False for code to preserve all tokens.
    """
    
    def __init__(self, name: str = "cl100k_base", preserve_code_structure: bool = True, 
                 remove_stop_words: bool = False):
        super().__init__(name=name, remove_stop_words=remove_stop_words)
        self.preserve_code_structure = preserve_code_structure
        
        # Code-specific tokens that should be preserved
        self.code_keywords = {
            'python': {'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 
                      'try', 'except', 'finally', 'with', 'as', 'return', 'yield', 'lambda',
                      'async', 'await', 'global', 'nonlocal', 'pass', 'break', 'continue'},
            'javascript': {'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while',
                          'do', 'switch', 'case', 'default', 'try', 'catch', 'finally',
                          'return', 'throw', 'new', 'this', 'class', 'extends', 'import', 'export'},
            'java': {'public', 'private', 'protected', 'static', 'final', 'abstract', 'class',
                    'interface', 'extends', 'implements', 'import', 'package', 'if', 'else',
                    'for', 'while', 'do', 'switch', 'case', 'default', 'try', 'catch', 'finally'},
            'cpp': {'#include', '#define', 'namespace', 'using', 'class', 'struct', 'public',
                   'private', 'protected', 'virtual', 'static', 'const', 'if', 'else', 'for',
                   'while', 'do', 'switch', 'case', 'default', 'try', 'catch', 'throw'},
            'go': {'package', 'import', 'func', 'var', 'const', 'type', 'struct', 'interface',
                  'if', 'else', 'for', 'range', 'switch', 'case', 'default', 'select',
                  'go', 'defer', 'return', 'break', 'continue', 'fallthrough'},
            'rust': {'fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl', 'trait',
                    'use', 'mod', 'pub', 'if', 'else', 'match', 'for', 'while', 'loop',
                    'break', 'continue', 'return', 'async', 'await'}
        }
    
    def preprocess(self, text: str) -> List[str]:
        """Preprocess text with code-specific handling.
        
        Args:
            text (str): Input code text
            
        Returns:
            List[str]: Preprocessed tokens
        """
        if not self.preserve_code_structure:
            return super().preprocess(text)
        
        # For code, we want to preserve structure, so we split more carefully
        lines = text.split('\n')
        words = []
        
        for line in lines:
            # Preserve indentation information
            stripped = line.lstrip()
            if stripped:
                indent_level = len(line) - len(stripped)
                if indent_level > 0:
                    words.append(f"<INDENT_{indent_level}>")
                
                # Split on whitespace but preserve some structure
                line_words = stripped.split()
                words.extend(line_words)
            
            # Add line break marker
            words.append('<NEWLINE>')
        
        return words
    
    def encode(self, text: str) -> List[int]:
        """Encode text with code-specific preprocessing.
        
        Args:
            text (str): Input code text
            
        Returns:
            List[int]: Token IDs
        """
        if self.preserve_code_structure:
            # For code, we might want to add special handling
            # but for now, use the standard encoding
            pass
        
        return self.tokenizer.encode(text)
    
    def get_code_tokens(self, text: str) -> List[str]:
        """Get string tokens with code-specific handling.
        
        Args:
            text (str): Input code text
            
        Returns:
            List[str]: String tokens optimized for code
        """
        token_ids = self.encode(text)
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]
        
        # Post-process tokens for better code representation
        processed_tokens = []
        for token in tokens:
            # Clean up tokens that might be split awkwardly
            if token.strip():
                processed_tokens.append(token)
        
        return processed_tokens
    
    def detect_language(self, text: str) -> str:
        """Detect the programming language of the code.
        
        Args:
            text (str): Input code text
            
        Returns:
            str: Detected language or 'unknown'
        """
        text_lower = text.lower()
        
        # Simple heuristic-based language detection
        if 'def ' in text and 'import ' in text:
            return 'python'
        elif 'function ' in text or 'const ' in text or 'let ' in text:
            return 'javascript'
        elif 'public class ' in text or 'import java' in text:
            return 'java'
        elif '#include' in text or 'namespace ' in text:
            return 'cpp'
        elif 'package ' in text and 'func ' in text:
            return 'go'
        elif 'fn ' in text and 'let ' in text:
            return 'rust'
        else:
            return 'unknown'
    
    def count_code_tokens(self, text: str) -> int:
        """Count tokens with code-specific considerations.
        
        Args:
            text (str): Input code text
            
        Returns:
            int: Number of tokens
        """
        return len(self.encode(text))