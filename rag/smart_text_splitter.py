"""Smart text splitter that finds appropriate stopping points near chunk boundaries."""

import re
from typing import List, Optional, Dict, Literal, Any
from adalflow.components.data_process import TextSplitter
from adalflow.core.types import Document
from logger.logging_config import get_tqdm_compatible_logger

# NLP libraries for intelligent text boundary detection
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from tree_sitter import Language, Parser
    # Import language parsers
    try:
        import tree_sitter_python as tspython
        import tree_sitter_javascript as tsjavascript
        import tree_sitter_java as tsjava
        import tree_sitter_cpp as tscpp
        import tree_sitter_go as tsgo
        import tree_sitter_rust as tsrust
        TREE_SITTER_AVAILABLE = True
    except ImportError:
        TREE_SITTER_AVAILABLE = False
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = get_tqdm_compatible_logger(__name__)


class SmartTextSplitter(TextSplitter):
    """Enhanced TextSplitter that finds intelligent stopping points near chunk boundaries.
    
    This splitter improves upon the base TextSplitter by:
    1. For text content: stops at the end of lines
    2. For code content: uses tree-sitter to find appropriate syntax boundaries
    3. Avoiding splits in the middle of function calls, statements, etc.
    """
    
    # Language mappings for tree-sitter
    SUFFIX_TO_LANG = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust'
    }

    LANGUAGE_PARSERS = {
        'python': 'tspython',
        'javascript': 'tsjavascript',
        'typescript': 'tsjavascript',
        'java': 'tsjava',
        'cpp': 'tscpp',
        'c': 'tscpp',
        'go': 'tsgo',
        'rust': 'tsrust'
    }
    
    def __init__(self, 
                 split_by: Literal["word", "sentence", "page", "passage", "token"] = "token",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 64,
                 batch_size: int = 1000,
                 separators: Optional[dict] = None,
                 smart_boundary_ratio: float = 0.8,
                 content_type: str = "auto",
                 file_extension: str = None):
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
            file_extension: The file extension to use for parser selection
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

        assert content_type in ['code', 'text', 'auto'], "content_type must be 'code', 'text', or 'auto'"

        self.file_extension = file_extension
        self.smart_boundary_threshold = int(chunk_size * smart_boundary_ratio)
        
        # Initialize tree-sitter parsers
        self.parsers = {}
        if TREE_SITTER_AVAILABLE and content_type != 'text':
            self._init_parsers()
        else:
            if content_type == 'code':
                logger.warning("Tree-sitter not available. Code splitting will use fallback method.")
        
        # Initialize NLP models for text boundary detection
        self.nlp_model = None
        if SPACY_AVAILABLE and content_type in ['text', 'auto']:
            self._init_nlp_model()
        
        logger.info(f"Initialized SmartTextSplitter with smart_boundary_ratio={smart_boundary_ratio}, content_type={content_type}, file_extension={file_extension}")

    def __getstate__(self):
        """Exclude non-serializable attributes from pickling."""
        state = self.__dict__.copy()
        # Remove the non-serializable 'parsers' attribute
        if 'parsers' in state:
            del state['parsers']
        return state

    def __setstate__(self, state):
        """Re-initialize non-serializable attributes after unpickling."""
        self.__dict__.update(state)
        # Re-initialize the 'parsers' attribute
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()
        # Re-initialize NLP model
        self.nlp_model = None
        if SPACY_AVAILABLE and self.content_type in ['text', 'auto']:
            self._init_nlp_model()
    
    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available. Code splitting will use fallback method.")
            return

        if not self.file_extension:
            logger.warning("No file extension provided. Code splitting will use fallback method.")
            return

        lang_name = self.SUFFIX_TO_LANG.get(self.file_extension)
        if not lang_name or lang_name not in self.LANGUAGE_PARSERS:
            logger.warning(f"Unsupported file extension {self.file_extension} for tree-sitter parsing, using fallback method for code split.")
            return

        module_name = self.LANGUAGE_PARSERS[lang_name]
        try:
            if module_name == 'tspython':
                language = Language(tspython.language())
            elif module_name == 'tsjavascript':
                language = Language(tsjavascript.language())
            elif module_name == 'tsjava':
                language = Language(tsjava.language())
            elif module_name == 'tscpp':
                language = Language(tscpp.language())
            elif module_name == 'tsgo':
                language = Language(tsgo.language())
            elif module_name == 'tsrust':
                language = Language(tsrust.language())
            else:
                return

            parser = Parser(language)
            self.parsers[lang_name] = parser
            logger.info(f"Initialized tree-sitter parser for {lang_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize parser for {lang_name}: {e}")
    
    def _init_nlp_model(self):
        """Initialize NLP model for text boundary detection."""
        try:
            # Try to load a small English model first
            self.nlp_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model for text boundary detection")
        except OSError:
            try:
                # Fallback to blank English model with sentencizer
                self.nlp_model = English()
                self.nlp_model.add_pipe('sentencizer')
                logger.info("Loaded spaCy blank English model with sentencizer")
            except Exception as e:
                logger.error(f"Failed to initialize spaCy model: {e}")
                self.nlp_model = None
                raise
    
    def _detect_content_type(self, text: str) -> str:
        """Detect if content is code or text."""
        if self.content_type != "auto":
            return self.content_type
        
        #TODO: add more sophisticated content detection
        # Simple heuristics for content detection
        code_indicators = ['def ', 'function ', 'class ', 'import ', 'from ', '{', '}', ';}', '->', '=>', 'public ', 'private ']
        text_indicators = ['# ', '## ', '- ', '* ', 'http://', 'https://', '[', '](', '. ', ', ']
        
        code_score = sum(1 for indicator in code_indicators if indicator in text)
        text_score = sum(1 for indicator in text_indicators if indicator in text)
        
        return 'code' if code_score > text_score else 'text'

    
    def _find_text_boundary(self, text: bytes, start_pos: int, max_pos: int) -> int:
        """Find semantic boundary for text content using NLP or fallback to line endings."""
        if start_pos >= max_pos:
            return max_pos
        
        # Try NLP-based boundary detection first
        if self._can_use_nlp_boundary_detection():
            nlp_boundary = self._find_nlp_text_boundary(text, start_pos, max_pos)
            if nlp_boundary is not None:
                return nlp_boundary
        
        logger.warning('nlp models unavailable')
        # Fallback to line-based boundary detection
        return self._find_line_boundary(text, start_pos, max_pos)
    
    def _can_use_nlp_boundary_detection(self) -> bool:
        """Check if NLP boundary detection is available and suitable."""
        return ((self.nlp_model is not None or NLTK_AVAILABLE) and 
                self.content_type in ['text', 'auto'])
    
    def _find_nlp_text_boundary(self, text: bytes, start_pos: int, max_pos: int) -> Optional[int]:
        """Find sentence boundary using NLP tools."""
        try:
            # Convert bytes to string for NLP processing
            text_str = text.decode('utf-8', errors='ignore')
            search_text = text_str[start_pos:max_pos]
            
            if not search_text.strip():
                return None
            
            # Try spaCy first if available
            if self.nlp_model is not None:
                return self._find_spacy_boundary(search_text, start_pos, max_pos)
            
            # Fallback to NLTK if spaCy is not available
            elif NLTK_AVAILABLE:
                return self._find_nltk_boundary(search_text, start_pos, max_pos)
            
            return None
            
        except Exception as e:
            logger.error(f"NLP boundary detection failed: {e}")
            raise

    def _find_spacy_boundary(self, search_text: str, start_pos: int, max_pos: int) -> Optional[int]:
        """Find sentence boundary using spaCy."""
        doc = self.nlp_model(search_text)
        sentences = list(doc.sents)
        
        if len(sentences) <= 1:
            return None
        
        # Find the last complete sentence that fits within our range
        best_boundary = None
        for i, sent in enumerate(sentences[:-1]):  # Exclude the last sentence
            sent_end = start_pos + sent.end_char
            if sent_end <= max_pos:
                best_boundary = sent_end
        
        return best_boundary
    
    def _find_nltk_boundary(self, search_text: str, start_pos: int, max_pos: int) -> Optional[int]:
        """Find sentence boundary using NLTK."""
        try:
            sentences = sent_tokenize(search_text)
            
            if len(sentences) <= 1:
                return None
            
            # Find the last complete sentence that fits within our range
            current_pos = 0
            best_boundary = None
            
            for i, sent in enumerate(sentences[:-1]):  # Exclude the last sentence
                sent_end_in_search = current_pos + len(sent)
                sent_end = start_pos + sent_end_in_search
                
                if sent_end <= max_pos:
                    best_boundary = sent_end
                
                # Move to next sentence (account for spaces/punctuation)
                current_pos = search_text.find(sent, current_pos) + len(sent)
                # Skip whitespace to next sentence
                while current_pos < len(search_text) and search_text[current_pos].isspace():
                    current_pos += 1
            
            return best_boundary
            
        except Exception as e:
            logger.error(f"NLTK boundary detection failed: {e}")
            raise
    
    def _find_nlp_overlap_start(self, text: bytes, desired_start_byte: int) -> int:
        """Find appropriate sentence start position for overlap using NLP tools."""
        try:
            # Convert bytes to string for NLP processing
            text_str = text.decode('utf-8', errors='ignore')
            
            # Search in a reasonable range around the desired position
            search_start = desired_start_byte
            search_end = len(text)
            search_text = text_str[search_start:search_end]
            
            if not search_text.strip():
                return desired_start_byte
            
            # Try spaCy first if available
            if self.nlp_model is not None:
                return self._find_spacy_overlap_start(search_text, search_start, desired_start_byte)
            
            # Fallback to NLTK if spaCy is not available
            elif NLTK_AVAILABLE:
                return self._find_nltk_overlap_start(search_text, search_start, desired_start_byte)
            
            # If no NLP tools available, return desired position
            return desired_start_byte
            
        except Exception as e:
            logger.debug(f"NLP overlap start detection failed: {e}")
            return desired_start_byte
    
    def _find_spacy_overlap_start(self, search_text: str, search_start: int, desired_start_byte: int) -> int:
        """Find sentence start position using spaCy for overlap."""
        doc = self.nlp_model(search_text)
        sentences = list(doc.sents)
        
        if len(sentences) <= 1:
            return desired_start_byte
        
        # Find the best sentence start position at or after desired position
        relative_desired = desired_start_byte - search_start
        best_start = desired_start_byte
        
        for sent in sentences:
            sent_start = search_start + sent.start_char
            # Look for sentence starts at or after the desired position
            if sent_start >= desired_start_byte:
                # Check if this sentence start begins with capital letter (proper sentence start)
                if sent.text.strip() and (sent.text[0].isupper() or sent.text[0].isdigit()):
                    best_start = sent_start
                    break
        
        return best_start
    
    def _find_nltk_overlap_start(self, search_text: str, search_start: int, desired_start_byte: int) -> int:
        """Find sentence start position using NLTK for overlap."""
        try:
            sentences = sent_tokenize(search_text)
            
            if len(sentences) <= 1:
                return desired_start_byte
            
            # Find sentence start positions
            relative_desired = desired_start_byte - search_start
            current_pos = 0
            best_start = desired_start_byte
            
            for sent in sentences:
                sent_start_in_search = search_text.find(sent, current_pos)
                if sent_start_in_search != -1:
                    sent_start = search_start + sent_start_in_search
                    
                    # Look for sentence starts at or after the desired position
                    if sent_start >= desired_start_byte:
                        # Check if this sentence starts properly
                        if sent.strip() and (sent[0].isupper() or sent[0].isdigit()):
                            best_start = sent_start
                            break
                    
                    current_pos = sent_start_in_search + len(sent)
            
            return best_start
            
        except Exception as e:
            logger.debug(f"NLTK overlap start detection failed: {e}")
            return desired_start_byte
    
    def _find_line_boundary(self, text: bytes, start_pos: int, max_pos: int) -> int:
        """Find line ending boundary for text content (fallback method)."""
        # Search backwards from max_pos to find the last line ending
        search_text = text[start_pos:max_pos]
        
        # Find all line endings in the search region
        line_endings = []
        for i, char in enumerate(search_text):
            if char == ord('\n'):
                line_endings.append(start_pos + i + 1)  # +1 to include the newline
        
        if line_endings:
            # Return the last line ending position
            return line_endings[-1]
        
        # If no line endings found, return max_pos
        return max_pos
    
    def _is_statement_node(self, node, language: str) -> bool:
        """
        Check if a tree-sitter node represents a complete statement.
        This ensures we don't split at partial code elements like single parentheses.
        """
        # Define statement-level node types for different languages
        statement_types = {
            'python': {
                'expression_statement', 'function_definition', 'class_definition',
                'if_statement', 'for_statement', 'while_statement', 'try_statement',
                'with_statement', 'import_statement', 'import_from_statement',
                'return_statement', 'break_statement', 'continue_statement',
                'pass_statement', 'del_statement', 'raise_statement', 'assert_statement',
                'global_statement', 'nonlocal_statement', 'decorated_definition',
                'match_statement'
            },
            'javascript': {
                'expression_statement', 'function_declaration', 'class_declaration',
                'if_statement', 'for_statement', 'while_statement', 'try_statement',
                'return_statement', 'break_statement', 'continue_statement',
                'throw_statement', 'switch_statement', 'variable_declaration',
                'import_statement', 'export_statement'
            },
            'java': {
                'expression_statement', 'method_declaration', 'class_declaration',
                'if_statement', 'for_statement', 'while_statement', 'try_statement',
                'return_statement', 'break_statement', 'continue_statement',
                'throw_statement', 'switch_statement', 'local_variable_declaration',
                'import_declaration', 'package_declaration'
            },
            'cpp': {
                'expression_statement', 'function_definition', 'class_specifier',
                'if_statement', 'for_statement', 'while_statement', 'try_statement',
                'return_statement', 'break_statement', 'continue_statement',
                'throw_statement', 'switch_statement', 'declaration'
            },
            'go': {
                'expression_statement', 'function_declaration', 'type_declaration',
                'if_statement', 'for_statement', 'switch_statement',
                'return_statement', 'break_statement', 'continue_statement',
                'go_statement', 'defer_statement', 'var_declaration',
                'const_declaration', 'import_declaration', 'package_clause'
            },
            'rust': {
                'expression_statement', 'function_item', 'struct_item',
                'if_expression', 'loop_expression', 'while_expression',
                'for_expression', 'match_expression', 'return_expression',
                'break_expression', 'continue_expression', 'let_declaration',
                'use_declaration', 'mod_item', 'impl_item'
            }
        }
        
        return node.type in statement_types.get(language, set())

    def _find_code_boundary_with_treesitter(self, text: bytes, start_pos: int, max_pos: int, language: str) -> int:
        """
        Given a utf-8 encoded text, find the best code boundary using tree-sitter.
        The "best" boundary is the one that is closest to max_pos, but not less than start_pos.
        Only considers statement-level nodes to ensure complete code semantics.
        """
        assert isinstance(text, bytes), "text must be utf-8 encoded bytes"
        try:
            parser = self.parsers[language]
            tree = parser.parse(text)
            
            # Find nodes that end near our target position
            best_boundary = start_pos
            
            def find_good_boundary(node):
                nonlocal best_boundary
                
                # Only consider statement-level nodes for boundaries
                if self._is_statement_node(node, language):
                    node_end = node.end_byte
                    # Check if this node ends in our target range
                    if start_pos <= node_end <= max_pos:
                        best_boundary = max(best_boundary, node_end)
                
                # Recursively check children
                for child in node.children:
                    find_good_boundary(child)
            
            find_good_boundary(tree.root_node)
            return min(best_boundary, max_pos)
            
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {language}: {e}")
            raise
    
    def _find_code_boundary_fallback(self, text: bytes, start_pos: int, max_pos: int) -> int:
        raise NotImplementedError("Fallback method for finding code boundaries without tree-sitter is not implemented.")
    
    def _find_smart_boundary(self, text: bytes, start_pos: int, max_pos: int, content_type: str) -> int:
        """Find a smart boundary position within the given range.
        
        Args:
            text: Full utf-8 encoded text to search in
            start_pos: Start position to search from
            max_pos: Maximum position (hard boundary)
            content_type: 'code' or 'text'
            
        Returns:
            int: Best boundary position found, or max_pos if none found
        """
        
        if start_pos >= max_pos:
            return max_pos
        if content_type == 'text':
            return self._find_text_boundary(text, start_pos, max_pos)
        else:  # code
            # Detect language and use tree-sitter if available
            language = self.SUFFIX_TO_LANG[self.file_extension]
            if language and TREE_SITTER_AVAILABLE:
                return self._find_code_boundary_with_treesitter(text, start_pos, max_pos, language)
            else:
                logger.warning(f"Tree-sitter not available for language {language}. Falling back to simple boundary detection. This detection may not be as accurate and may split at suboptimal points.")
                return self._find_code_boundary_fallback(text, start_pos, max_pos)
    
    def _find_code_boundary_with_treesitter_v2(self, text: bytes, desired_start_char: int, language: str):
        """
        Given a utf-8 encoded text, find the best code boundary using tree-sitter.
        The "best" boundary is the one that is closest to desired_start_char, but not less than desired_start_char.
        The range of the boundary is [desired_start_char, len(text)].
        """
        assert isinstance(text, bytes), "text must be utf-8 encoded bytes"
        try:
            parser = self.parsers[language]
            tree = parser.parse(text)
            
            best_start = len(text) # default to the end of the text, meaning zero overlap

            #haoyang
            best_start_node_text = ""

            def find_best_start_recursive(node):
                nonlocal best_start
                nonlocal best_start_node_text
                # Check if the current node is a candidate and is a statement
                if node.start_byte >= desired_start_char and self._is_statement_node(node, language):
                    #haoyang
                    if best_start > node.start_byte:
                        best_start_node_text = node.text.decode('utf-8') if node.text else ""
                    best_start = min(best_start, node.start_byte)
                
                # Recursively check children only if they can potentially contain a better candidate.
                if node.start_byte < best_start:
                    for child in node.children:
                        find_best_start_recursive(child)
            
            find_best_start_recursive(tree.root_node)
            return best_start

        except Exception as e:
            logger.error(f"Tree-sitter parsing for overlap failed for {language}: {e}")
            raise

    def _find_smart_overlap_start(self, text: bytes, desired_start_byte: int, content_type: str) -> int:
        """
        Finds the best starting position for an overlap in a text chunk.
        It looks for a syntax node starting at or after the desired position.
        """

        if content_type == 'text':
            # Use NLP-based boundary detection for text overlap
            return self._find_nlp_overlap_start(text, desired_start_byte)
        
        language = self.SUFFIX_TO_LANG[self.file_extension]
        
        if not self.parsers.get(language):
            raise ValueError(f"Tree-sitter parser not available for language {language}")

        if content_type == 'code':
            return self._find_code_boundary_with_treesitter_v2(text, desired_start_byte, language)
        

    def _merge_units_to_chunks(self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str) -> List[str]:
        """Enhanced merge method with smart boundary detection."""
        if self.split_by != "token":
            # For non-token splitting, use the original method
            return super()._merge_units_to_chunks(splits, chunk_size, chunk_overlap, separator)
        
        # For token-based splitting, we need to work with the decoded text
        full_text = self.tokenizer.decode(splits)
        content_type = self._detect_content_type(full_text)
        
        chunks = []
        idx = 0
        logger.debug(f"Merging {len(splits)} tokens with smart boundaries (content_type: {content_type})")
        while idx < len(splits):
            # Calculate the end position for this chunk
            chunk_end = min(idx + chunk_size, len(splits))
            chunk_tokens = splits[idx:chunk_end]
            chunk_text_bytes = self.tokenizer.decode(chunk_tokens).encode('utf-8')
            # If this is not the last chunk and we have room for smart boundary detection
            if chunk_end < len(splits) and chunk_end - idx >= self.smart_boundary_threshold:
                # Find smart boundary in the last portion of the chunk
                search_start_pos = int(len(chunk_text_bytes) * self.smart_boundary_ratio)
                smart_boundary_pos = self._find_smart_boundary(
                    chunk_text_bytes, search_start_pos, len(chunk_text_bytes), content_type
                )
                
                # If we found a good boundary, adjust the chunk end
                if smart_boundary_pos < len(chunk_text_bytes):
                    # Re-encode to find the token boundary
                    boundary_text = chunk_text_bytes[:smart_boundary_pos].decode('utf-8')
                    boundary_tokens = self.tokenizer.encode(boundary_text)
                    chunk_end = idx + len(boundary_tokens)
                    # Update chunk_text_bytes to reflect the boundary adjustment
                    chunk_text_bytes = boundary_text.encode('utf-8')
                    
                    logger.debug(f"Found smart boundary at position {smart_boundary_pos} (token {chunk_end})")
            
            if chunk_text_bytes.strip():  # Only add non-empty chunks
                chunk_text = chunk_text_bytes.decode('utf-8')
                # Strip leading whitespace for continuation chunks (not the first chunk)
                if len(chunks) > 0 and chunk_text.startswith(' '):
                    chunk_text = chunk_text.lstrip()
                chunks.append(chunk_text)
            
            # Smart overlap calculation
            # Use the actual chunk tokens that were used (may be adjusted by boundary detection)
            actual_chunk_tokens = self.tokenizer.encode(chunk_text_bytes.decode('utf-8'))
            num_non_overlap_tokens = len(actual_chunk_tokens) - chunk_overlap
            
            if num_non_overlap_tokens > 0 and chunk_end < len(splits):
                bytetext_before_overlap = self.tokenizer.decode(actual_chunk_tokens[:num_non_overlap_tokens]).encode('utf-8')
                desired_start_byte = len(bytetext_before_overlap)
                
                best_start_byte = self._find_smart_overlap_start(chunk_text_bytes, desired_start_byte, content_type)
                if best_start_byte < len(chunk_text_bytes):
                    new_overlap_text = chunk_text_bytes[best_start_byte:].decode('utf-8')
                    new_overlap_tokens_count = len(self.tokenizer.encode(new_overlap_text))
                    adjusted_overlap = new_overlap_tokens_count
                    
                    # Calculate next_idx based on the smart overlap position
                    # Find the token position in the original splits that corresponds to the overlap start
                    overlap_start_text = chunk_text_bytes[best_start_byte:].decode('utf-8')
                    
                    # Find the token that starts with the overlap text
                    next_idx = chunk_end - adjusted_overlap  # fallback
                    
                    # Search through tokens to find the one that matches the overlap start
                    for token_idx in range(idx, min(len(splits), chunk_end + 5)):
                        # Decode from this token to the end to see if it matches our overlap start
                        remaining_tokens = splits[token_idx:]
                        remaining_text = self.tokenizer.decode(remaining_tokens)
                        
                        # Check if this token position gives us the overlap text we want
                        # Strip leading whitespace for comparison since tokenizer may add spaces
                        if remaining_text.lstrip().startswith(overlap_start_text.lstrip()[:min(50, len(overlap_start_text.lstrip()))]):
                            next_idx = token_idx
                            break
                else:
                    adjusted_overlap = 0
                    next_idx = chunk_end - adjusted_overlap
            else:
                adjusted_overlap = 0
                next_idx = chunk_end - adjusted_overlap
            assert next_idx > idx, f"next_idx ({next_idx}) must be greater than idx ({idx})"
            idx = next_idx
        
        logger.info(f"Smart splitting created {len(chunks)} chunks from {len(splits)} tokens")
        return chunks
    
    def _extra_repr(self) -> str:
        base_repr = super()._extra_repr()
        return f"{base_repr}, smart_boundary_ratio={self.smart_boundary_ratio}, content_type={self.content_type}"