"""Smart text splitter that finds appropriate stopping points near chunk boundaries."""

import re
from typing import List, Optional, Dict, Literal, Any
from adalflow.components.data_process import TextSplitter
from adalflow.core.types import Document
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

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


logger = get_tqdm_compatible_logger(__name__)


class TxtTextSplitter(TextSplitter):

    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage", "token"] = "token",
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
        batch_size: int = 1000,
        separators: Optional[dict] = None,
        smart_boundary_ratio: float = 0.8,
        file_extension: str = None,
    ):
        """
        Initialize TxtTextSplitter.

        Args:
            split_by: Same as TextSplitter
            chunk_size: Same as TextSplitter
            chunk_overlap: Same as TextSplitter
            batch_size: Same as TextSplitter
            separators: Same as TextSplitter
            smart_boundary_ratio: When to start looking for smart boundaries (0.8 = 80% of chunk_size)
            file_extension: The file extension to use for parser selection
        """

        assert split_by == "token", "TxtTextSplitter only supports split_by='token'"

        # Set default separators if None - TextSplitter expects a dict
        if separators is None:
            separators = {
                "word": [" ", "\n"],
                "sentence": [".", "!", "?"],
                "page": ["\n\n"],
                "passage": ["\n\n"],
                "token": [],
            }

        super().__init__(split_by, chunk_size, chunk_overlap, batch_size, separators)
        self.smart_boundary_ratio = smart_boundary_ratio

        self.file_extension = file_extension
        self.smart_boundary_threshold = int(chunk_size * smart_boundary_ratio)

        # Initialize NLP models for text boundary detection
        self.nlp_model = None
        if SPACY_AVAILABLE:
            self._init_nlp_model()

        logger.info(
            f"Initialized TxtTextSplitter with smart_boundary_ratio={smart_boundary_ratio}, file_extension={file_extension}"
        )

    def __getstate__(self):
        """Exclude non-serializable attributes from pickling."""
        state = self.__dict__.copy()
        # Remove the non-serializable 'nlp_model' attribute
        if "nlp_model" in state:
            del state["nlp_model"]
        return state

    def __setstate__(self, state):
        """Re-initialize non-serializable attributes after unpickling."""
        self.__dict__.update(state)
        # Re-initialize NLP model
        self.nlp_model = None
        if SPACY_AVAILABLE:
            self._init_nlp_model()

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
                self.nlp_model.add_pipe("sentencizer")
                logger.info("Loaded spaCy blank English model with sentencizer")
            except Exception as e:
                logger.error(f"Failed to initialize spaCy model: {e}")
                self.nlp_model = None
                raise

    def _find_text_boundary(self, text: bytes, start_pos: int, max_pos: int) -> int:
        """Find semantic boundary for text content using NLP or fallback to line endings."""
        if start_pos >= max_pos:
            return max_pos

        # Try NLP-based boundary detection first
        if self._can_use_nlp_boundary_detection():
            nlp_boundary = self._find_nlp_text_boundary(text, start_pos, max_pos)
            if nlp_boundary is not None:
                return nlp_boundary

        logger.warning(
            "Cannot find semantic boundary, fallback to line-based boundary detection"
        )
        # Fallback to line-based boundary detection
        return self._find_line_boundary(text, start_pos, max_pos)

    def _can_use_nlp_boundary_detection(self) -> bool:
        """Check if NLP boundary detection is available and suitable."""
        return self.nlp_model is not None or NLTK_AVAILABLE

    def _find_nlp_text_boundary(
        self, text: bytes, start_pos: int, max_pos: int
    ) -> Optional[int]:
        """Find sentence boundary using NLP tools."""
        try:
            # Convert bytes to string for NLP processing
            text_str = text.decode("utf-8", errors="ignore")
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

    def _find_spacy_boundary(
        self, search_text: str, start_pos: int, max_pos: int
    ) -> Optional[int]:
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

    def _find_nltk_boundary(
        self, search_text: str, start_pos: int, max_pos: int
    ) -> Optional[int]:
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
                while (
                    current_pos < len(search_text)
                    and search_text[current_pos].isspace()
                ):
                    current_pos += 1

            return best_boundary

        except Exception as e:
            logger.error(f"NLTK boundary detection failed: {e}")
            raise

    def _find_nlp_overlap_start(self, text: bytes, desired_start_byte: int) -> int:
        """Find appropriate sentence start position for overlap using NLP tools."""
        try:
            # Convert bytes to string for NLP processing
            text_str = text.decode("utf-8", errors="ignore")

            # Search in a reasonable range around the desired position
            search_start = desired_start_byte
            search_end = len(text)
            search_text = text_str[search_start:search_end]

            if not search_text.strip():
                return desired_start_byte

            # Try spaCy first if available
            if self.nlp_model is not None:
                return self._find_spacy_overlap_start(
                    search_text, search_start, desired_start_byte
                )

            # Fallback to NLTK if spaCy is not available
            elif NLTK_AVAILABLE:
                return self._find_nltk_overlap_start(
                    search_text, search_start, desired_start_byte
                )

            # If no NLP tools available, return desired position
            return desired_start_byte

        except Exception as e:
            logger.debug(f"NLP overlap start detection failed: {e}")
            return desired_start_byte

    def _find_spacy_overlap_start(
        self, search_text: str, search_start: int, desired_start_byte: int
    ) -> int:
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
                if sent.text.strip() and (
                    sent.text[0].isupper() or sent.text[0].isdigit()
                ):
                    best_start = sent_start
                    break

        return best_start

    def _find_nltk_overlap_start(
        self, search_text: str, search_start: int, desired_start_byte: int
    ) -> int:
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
            if char == ord("\n"):
                line_endings.append(start_pos + i + 1)  # +1 to include the newline

        if line_endings:
            # Return the last line ending position
            return line_endings[-1]

        # If no line endings found, return max_pos
        return max_pos

    def _find_boundary(self, text: bytes, start_pos: int, max_pos: int) -> int:
        """Find a smart boundary position within the given range.

        Args:
            text: Full utf-8 encoded text to search in
            start_pos: Start position to search from
            max_pos: Maximum position (hard boundary)

        Returns:
            int: Best boundary position found, or max_pos if none found
        """

        if start_pos >= max_pos:
            return max_pos
        return self._find_text_boundary(text, start_pos, max_pos)

    def _find_overlap_start(self, text: bytes, desired_start_byte: int) -> int:
        """
        Finds the best starting position for an overlap in a text chunk.
        It looks for a syntax node starting at or after the desired position.
        """

        # Use NLP-based boundary detection for text overlap
        return self._find_nlp_overlap_start(text, desired_start_byte)

    def _merge_units_to_chunks(
        self, splits: List[str], chunk_size: int, chunk_overlap: int, separator: str
    ) -> List[str]:
        """Enhanced merge method with smart boundary detection."""
        if self.split_by != "token":
            # For non-token splitting, use the original method
            return super()._merge_units_to_chunks(
                splits, chunk_size, chunk_overlap, separator
            )

        chunks = []
        idx = 0
        while idx < len(splits):
            # Calculate the end position for this chunk
            chunk_end = min(idx + chunk_size, len(splits))
            chunk_tokens = splits[idx:chunk_end]
            chunk_text_bytes = self.tokenizer.decode(chunk_tokens).encode("utf-8")
            # If this is not the last chunk and we have room for smart boundary detection
            if (
                chunk_end < len(splits)
                and chunk_end - idx >= self.smart_boundary_threshold
            ):
                # Find smart boundary in the last portion of the chunk
                search_start_pos = int(
                    len(chunk_text_bytes) * self.smart_boundary_ratio
                )
                smart_boundary_pos = self._find_boundary(
                    chunk_text_bytes, search_start_pos, len(chunk_text_bytes)
                )

                # If we found a good boundary, adjust the chunk end
                if smart_boundary_pos < len(chunk_text_bytes):
                    # Re-encode to find the token boundary
                    boundary_text = chunk_text_bytes[:smart_boundary_pos].decode(
                        "utf-8"
                    )
                    boundary_tokens = self.tokenizer.encode(boundary_text)
                    chunk_end = idx + len(boundary_tokens)
                    # Update chunk_text_bytes to reflect the boundary adjustment
                    chunk_text_bytes = boundary_text.encode("utf-8")

                    logger.debug(
                        f"Found smart boundary at position {smart_boundary_pos} (token {chunk_end})"
                    )

            if chunk_text_bytes.strip():  # Only add non-empty chunks
                chunk_text = chunk_text_bytes.decode("utf-8")
                # Strip leading whitespace for continuation chunks (not the first chunk)
                if len(chunks) > 0 and chunk_text.startswith(" "):
                    chunk_text = chunk_text.lstrip()
                chunks.append(chunk_text)

            # Smart overlap calculation
            # Use the actual chunk tokens that were used (may be adjusted by boundary detection)
            actual_chunk_tokens = self.tokenizer.encode(
                chunk_text_bytes.decode("utf-8")
            )
            num_non_overlap_tokens = len(actual_chunk_tokens) - chunk_overlap

            if num_non_overlap_tokens > 0 and chunk_end < len(splits):
                bytetext_before_overlap = self.tokenizer.decode(
                    actual_chunk_tokens[:num_non_overlap_tokens]
                ).encode("utf-8")
                desired_start_byte = len(bytetext_before_overlap)

                best_start_byte = self._find_overlap_start(
                    chunk_text_bytes, desired_start_byte
                )
                if best_start_byte < len(chunk_text_bytes):
                    new_overlap_text = chunk_text_bytes[best_start_byte:].decode(
                        "utf-8"
                    )
                    new_overlap_tokens_count = len(
                        self.tokenizer.encode(new_overlap_text)
                    )
                    adjusted_overlap = new_overlap_tokens_count

                    # Calculate next_idx based on the smart overlap position
                    # Find the token position in the original splits that corresponds to the overlap start
                    overlap_start_text = chunk_text_bytes[best_start_byte:].decode(
                        "utf-8"
                    )

                    # Find the token that starts with the overlap text
                    next_idx = chunk_end - adjusted_overlap  # fallback

                    # Search through tokens to find the one that matches the overlap start
                    for token_idx in range(idx, min(len(splits), chunk_end + 5)):
                        # Decode from this token to the end to see if it matches our overlap start
                        remaining_tokens = splits[token_idx:]
                        remaining_text = self.tokenizer.decode(remaining_tokens)

                        # Check if this token position gives us the overlap text we want
                        # Strip leading whitespace for comparison since tokenizer may add spaces
                        if remaining_text.lstrip().startswith(
                            overlap_start_text.lstrip()[
                                : min(50, len(overlap_start_text.lstrip()))
                            ]
                        ):
                            next_idx = token_idx
                            break
                else:
                    adjusted_overlap = 0
                    next_idx = chunk_end - adjusted_overlap
            else:
                adjusted_overlap = 0
                next_idx = chunk_end - adjusted_overlap
            assert (
                next_idx > idx
            ), f"next_idx ({next_idx}) must be greater than idx ({idx})"
            idx = next_idx

        logger.info(
            f"Smart splitting created {len(chunks)} chunks from {len(splits)} tokens"
        )
        return chunks

    def _extra_repr(self) -> str:
        base_repr = super()._extra_repr()
        return f"{base_repr}, smart_boundary_ratio={self.smart_boundary_ratio}"
