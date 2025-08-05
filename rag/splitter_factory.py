"""Factory for creating appropriate text splitters based on document type."""

from multiprocessing import Value
import os
from typing import Union, Dict, Any
from adalflow.components.data_process import TextSplitter
from adalflow.core.tokenizer import Tokenizer
from rag.smart_text_splitter import SmartTextSplitter
from configs import configs
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class SplitterFactory:
    """Factory class for creating appropriate splitters based on document type."""
    
    # File extensions that are considered code files
    CODE_EXTENSIONS = configs['repo']['file_extensions']['code_extensions']
    
    # File extensions that are considered text/documentation files
    TEXT_EXTENSIONS = configs['repo']['file_extensions']['doc_extensions']
    
    def __init__(self):
        """Initialize the splitter factory."""
        self._text_splitter = None
        self._code_splitter = None
    
    def _get_text_splitter(self) -> SmartTextSplitter:
        """Get or create text splitter instance.
        
        Returns:
            SmartTextSplitter: Configured smart text splitter
        """
        if self._text_splitter is None:
            text_splitter_config = configs["knowledge"]['text_splitter'].copy()
            # Add smart splitting parameters for text content
            text_splitter_config['content_type'] = 'text'
            text_splitter_config['smart_boundary_ratio'] = 0.8
            self._text_splitter = SmartTextSplitter(**text_splitter_config)
            logger.info(f"Created smart text splitter with config: {text_splitter_config}")
        return self._text_splitter
    
    def _get_code_splitter(self, extension: str) -> SmartTextSplitter:
        """Get or create code splitter instance.
        
        Returns:
            SmartTextSplitter: Configured smart code splitter with custom tokenizer
        """
        if self._code_splitter is None:
            code_splitter_config = configs["knowledge"]['code_splitter'].copy()
            
            # Add smart splitting parameters for code content
            code_splitter_config['content_type'] = 'code'
            code_splitter_config['smart_boundary_ratio'] = 0.75  # Slightly more aggressive for code
            code_splitter_config['file_extension'] = extension
            # Create the smart code splitter
            self._code_splitter = SmartTextSplitter(**code_splitter_config)
            
            logger.info(f"Created smart code splitter with config: {code_splitter_config}")
        return self._code_splitter
    
    def detect_document_type(self, file_path: str) -> tuple[str, str]:
        """Detect document type and extension based on file path.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            tuple: (document_type, extension)
        """
        if not file_path:
            # return 'unknown', ''
            raise ValueError("file_path is required for document type detection")
        
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in self.CODE_EXTENSIONS:
            return 'code', ext
        elif ext in self.TEXT_EXTENSIONS:
            return 'text', ext
        else:
            # For unknown extensions, try to detect based on content patterns
            raise ValueError(f"Unknown file extension: {ext}")
    def detect_content_type(self, content: str, file_path: str = "") -> str:
        """Detect content type based on file path and content analysis.
        
        Args:
            content (str): File content
            file_path (str): File path (optional)
            
        Returns:
            str: 'code', 'text', or 'unknown'
        """
        # First try file extension
        if file_path:
            doc_type, _ = self.detect_document_type(file_path)
            if doc_type != 'unknown':
                return doc_type
        
        else:
            #TODO: currently, file_path is required. The case where file_path == "" will be supported later.
            raise ValueError("file_path is required for content type detection")
        
        # Fallback to content analysis
        if not content.strip():
            return 'unknown'
        
        # Simple heuristics for content detection
        content_lower = content.lower()
        
        # Code indicators
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'from ', '#include',
            'namespace ', 'package ', 'public class', 'private ', 'protected ',
            'const ', 'let ', 'var ', 'fn ', 'func ', 'impl ', 'trait ',
            '#!/', '<?php', '<%', '{', '}', ';\n', '->', '=>', '::'
        ]
        
        # Text/documentation indicators
        text_indicators = [
            '# ', '## ', '### ', '- ', '* ', '1. ', '2. ', '3. ',
            'http://', 'https://', '[', '](', '**', '__', '*', '_'
        ]
        
        code_score = sum(1 for indicator in code_indicators if indicator in content_lower)
        text_score = sum(1 for indicator in text_indicators if indicator in content_lower)
        
        # Calculate ratios
        total_chars = len(content)
        if total_chars > 0:
            brace_ratio = (content.count('{') + content.count('}')) / total_chars
            semicolon_ratio = content.count(';') / total_chars
            
            # Code files typically have more braces and semicolons
            if brace_ratio > 0.01 or semicolon_ratio > 0.02:
                code_score += 2
        
        if code_score > text_score:
            return 'code'
        elif text_score > code_score:
            return 'text'
        else:
            return 'unknown'
    
    def get_splitter(self, content: str = "", file_path: str = "", 
                    force_type: str = None) -> SmartTextSplitter:
        """Get appropriate splitter based on content and file path.
        
        Args:
            content (str): File content (optional)
            file_path (str): File path (optional)
            force_type (str): Force specific splitter type ('code' or 'text')
            
        Returns:
            SmartTextSplitter: Appropriate splitter instance
        """
        if force_type:
            doc_type = force_type
            _, ext = self.detect_document_type(file_path)
        else:
            doc_type, ext = self.detect_document_type(file_path)
            if doc_type == 'unknown':
                doc_type = self.detect_content_type(content, file_path)

        splitter = self._get_code_splitter(ext) if doc_type == 'code' else self._get_text_splitter()
        
        return splitter


# Global factory instance
_splitter_factory = None


def get_splitter_factory() -> SplitterFactory:
    """Get global splitter factory instance.
    
    Returns:
        SplitterFactory: Global factory instance
    """
    global _splitter_factory
    if _splitter_factory is None:
        _splitter_factory = SplitterFactory()
    return _splitter_factory