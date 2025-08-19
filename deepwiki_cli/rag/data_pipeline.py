import os
from typing import List, Optional, Tuple, Union
import glob
import re
import fnmatch

import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from adalflow.components.data_process import TextSplitter

from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger
from deepwiki_cli.clients.huggingface_embedder_client import (
    HuggingfaceClientToEmbeddings,
)
from deepwiki_cli.clients.dashscope_client import DashScopeToEmbeddings
from deepwiki_cli.rag.dual_vector_pipeline import DualVectorDocument
from deepwiki_cli.rag.dual_vector_pipeline import (
    DualVectorToEmbeddings,
    CodeUnderstandingGenerator,
)
from deepwiki_cli.rag.dynamic_splitter_transformer import DynamicSplitterTransformer
from deepwiki_cli.configs import get_embedder, configs

# The setting is from the observation that the maximum length of Solidity compiler's files is 919974
MAX_EMBEDDING_LENGTH = 1000000

# Configure logging
logger = get_tqdm_compatible_logger(__name__)


def count_tokens(
    text: str,
    model_provider: str,
    model_name: Optional[str] = None,
    hf_tokenizer=None,
) -> int:
    """
    Count the number of tokens in a text string using the correct tokenizer
    for the model provider: OpenAI (tiktoken), HuggingFace (transformers), DashScope (approx).

    Args:
        text (str): The text to count tokens for.
        model_provider (str): "openai", "huggingface", or "dashscope".
        model_name (str, optional): Model name (for auto-determine tokenizer).
        hf_tokenizer (PreTrainedTokenizer, optional): HuggingFace tokenizer instance.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        if model_provider.lower() == "openai":
            import tiktoken

            # You can switch model name if needed
            encoding = tiktoken.encoding_for_model(
                model_name or "text-embedding-3-small"
            )
            return len(encoding.encode(text))

        elif model_provider.lower() == "huggingface":
            # Needs transformers and a loaded tokenizer object
            if hf_tokenizer is None:
                from transformers import AutoTokenizer

                assert model_name, "model_name is required for HuggingFace tokenizer"
                hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            return len(hf_tokenizer.encode(text))

        elif model_provider.lower() == "dashscope":
            # DashScope doesn't officially publish a tokenizer; use character or word heuristic
            # You may customize this rule per DashScope's documentation if available.
            # OpenAI-style estimate: ~4 chars/token for English
            return max(1, len(text) // 4)
        elif model_provider.lower() == "google":
            # Following https://ai.google.dev/gemini-api/docs/tokens?lang=python, ~4 chars/token for English
            return max(1, len(text) // 4)
        else:
            logger.warning(
                f"Unknown model provider '{model_provider}'; using rough 4-char-per-token estimate."
            )
            return max(1, len(text) // 4)

    except Exception as e:
        logger.warning(f"Error counting tokens ({model_provider=}, {model_name=}): {e}")
        return max(1, len(text) // 4)


def safe_read_file(file_path: str) -> Tuple[Optional[str], str]:
    """
    Safely read a file with multiple encoding attempts and binary file detection.

    Args:
        file_path: Path to the file to read

    Returns:
        Tuple of (content, status_message):
        - content: File content as string if successful, None if failed
        - status_message: Description of what happened
    """

    # Check if file is likely binary by examining first few bytes
    try:
        with open(file_path, "rb") as f:
            # Read first 8192 bytes to check for binary content
            chunk = f.read(8192)
            if not chunk:
                return None, "empty_file"

            # Check for common binary file signatures
            binary_signatures = [
                b"\x00",  # NULL byte (common in binary files)
                b"\xFF\xD8\xFF",  # JPEG
                b"\x89PNG",  # PNG
                b"GIF8",  # GIF
                b"%PDF",  # PDF
                b"\x50\x4B",  # ZIP/JAR/etc
                b"\x7FELF",  # ELF executable
                b"MZ",  # DOS/Windows executable
            ]

            # If file contains NULL bytes or binary signatures, skip it
            if b"\x00" in chunk:
                return None, "binary_file_null_bytes"

            for sig in binary_signatures:
                if chunk.startswith(sig):
                    return None, f"binary_file_signature_{sig.hex()}"

            # Check if chunk has too many non-printable characters
            printable_chars = sum(
                1 for byte in chunk if 32 <= byte <= 126 or byte in [9, 10, 13]
            )
            if len(chunk) > 0 and printable_chars / len(chunk) < 0.7:
                return None, "binary_file_non_printable"

    except (IOError, OSError) as e:
        logger.error(f"File access error for {file_path}: {e}")
        raise

    # Try different encodings in order of preference
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",  # UTF-8 with BOM
        "latin1",  # Common fallback
        "cp1252",  # Windows encoding
        "iso-8859-1",  # Another common encoding
        "ascii",  # Basic ASCII
    ]

    # Try to detect encoding if chardet is available
    try:
        import chardet

        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                if detected["encoding"] and detected["confidence"] > 0.7:
                    detected_encoding = detected["encoding"].lower()
                    # Add detected encoding to the front of the list if not already there
                    if detected_encoding not in [
                        enc.lower() for enc in encodings_to_try
                    ]:
                        encodings_to_try.insert(0, detected["encoding"])
        except Exception:
            pass  # Continue with default encodings if detection fails
    except ImportError:
        pass  # chardet not available, use default encodings

    # Try each encoding
    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as f:
                content = f.read()

                # Additional validation for the content
                if not content.strip():
                    return None, "empty_content"

                # Check if content seems reasonable (not too many weird characters)
                if encoding != "utf-8":
                    logger.debug(
                        f"Successfully read {file_path} with {encoding} encoding"
                    )

                return content, f"success_{encoding}"

        except UnicodeDecodeError as e:
            logger.debug(f"Failed to read {file_path} with {encoding}: {e}")
            continue
        except (IOError, OSError) as e:
            logger.error(f"File error for {file_path}: {e}")
            return None, f"file_error: {e}"

    # If all encodings failed, try one more time with errors='replace'
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            if content.strip():
                logger.warning(
                    f"Read {file_path} with character replacement (some characters may be corrupted)"
                )
                return content, "success_utf8_with_replacement"
    except Exception as e:
        logger.error(f"Final fallback failed for {file_path}: {e}")

    return None, "all_encodings_failed"


def read_all_documents(path: str):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.

    Returns:
        list: A list of Document objects with metadata.
    """
    documents = []
    included_dirs = configs()["repo"]["file_filters"]["included_dirs"]
    included_files = configs()["repo"]["file_filters"]["included_files"]
    excluded_dirs = configs()["repo"]["file_filters"]["excluded_dirs"]
    excluded_files = configs()["repo"]["file_filters"]["excluded_files"]
    code_extensions = configs()["repo"]["file_extensions"]["code_extensions"]
    doc_extensions = configs()["repo"]["file_extensions"]["doc_extensions"]

    # Determine filtering mode: inclusion or exclusion
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (
        included_files is not None and len(included_files) > 0
    )

    if use_inclusion_mode:
        # Inclusion mode: only process specified directories and files
        final_included_dirs = set(included_dirs) if included_dirs else set()
        final_included_files = set(included_files) if included_files else set()

        logger.info(f"Using inclusion mode")
        logger.info(f"Included directories: {list(final_included_dirs)}")
        logger.info(f"Included files: {list(final_included_files)}")

        # Convert to lists for processing
        included_dirs = list(final_included_dirs)
        included_files = list(final_included_files)
        excluded_dirs = []
        excluded_files = []
    else:

        included_dirs = []
        included_files = []

    logger.info(f"Reading documents from {path}, use_inclusion_mode: {use_inclusion_mode}")

    def should_process_file(
        file_path: str,
        use_inclusion: bool,
        included_dirs: List[str],
        included_files: List[str],
        excluded_dirs: List[str],
        excluded_files: List[str],
    ) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.
        Supports glob patterns and regular expressions for matching.

        Args:
            file_path (str): The file path to check
            use_inclusion (bool): Whether to use inclusion mode
            included_dirs (List[str]): List of directories to include (supports glob patterns)
            included_files (List[str]): List of files to include (supports glob patterns)
            excluded_dirs (List[str]): List of directories to exclude (supports glob patterns)
            excluded_files (List[str]): List of files to exclude (supports glob patterns)

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        # Normalize the file path for consistent matching
        normalized_path = os.path.normpath(file_path).replace("\\", "/")
        file_name = os.path.basename(file_path)
        
        def matches_pattern(text: str, pattern: str) -> bool:
            """Check if text matches a glob pattern."""
            if fnmatch.fnmatch(text, pattern):
                return True

            return False

        if use_inclusion:
            # Inclusion mode: file must be in included directories or match included files
            is_included = False

            # Check if file is in an included directory
            if included_dirs:
                for included in included_dirs:
                    if matches_pattern(normalized_path, included):
                        is_included = True
                        break

            # Check if file matches included file patterns
            if not is_included and included_files:
                for included_file in included_files:
                    if matches_pattern(file_name, included_file) or matches_pattern(normalized_path, included_file):
                        is_included = True
                        break

            # If no inclusion rules are specified for a category, allow all files from that category
            if not included_dirs and not included_files:
                is_included = True
            elif not included_dirs and included_files:
                # Only file patterns specified, allow all directories
                pass  # is_included is already set based on file patterns
            elif included_dirs and not included_files:
                # Only directory patterns specified, allow all files in included directories
                pass  # is_included is already set based on directory patterns

            return is_included
        else:
            # Exclusion mode: file must not be in excluded directories or match excluded files
            is_excluded = False

            # Check if file is in an excluded directory
            for excluded in excluded_dirs:
                if matches_pattern(normalized_path, excluded):
                    is_excluded = True
                    break

            # Check if file matches excluded file patterns
            if not is_excluded:
                for excluded_file in excluded_files:
                    if matches_pattern(file_name, excluded_file) or matches_pattern(normalized_path, excluded_file):
                        is_excluded = True
                        break
            
            return not is_excluded

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(
                file_path,
                use_inclusion_mode,
                included_dirs,
                included_files,
                excluded_dirs,
                excluded_files,
            ):
                continue

            # Safely read file with encoding detection
            content, status = safe_read_file(file_path)

            if content is None:
                # Log specific reasons for skipping files
                relative_path = os.path.relpath(file_path, path)
                if status.startswith("binary_file"):
                    logger.debug(f"Skipping binary file {relative_path}: {status}")
                elif status == "empty_file" or status == "empty_content":
                    logger.debug(f"Skipping empty file {relative_path}")
                elif status == "all_encodings_failed":
                    logger.warning(
                        f"Skipping file {relative_path}: Unable to decode with any encoding"
                    )
                else:
                    logger.warning(f"Skipping file {relative_path}: {status}")
                continue

            relative_path = os.path.relpath(file_path, path)

            # Determine if this is an implementation file
            is_implementation = (
                not relative_path.startswith("test_")
                and not relative_path.startswith("app_")
                and not relative_path.startswith("build")
                and "test" not in relative_path.lower()
            )

            # Check token count
            if len(content) > MAX_EMBEDDING_LENGTH:
                logger.warning(
                    f"Skipping large file {relative_path}: Token count ({len(content)}) exceeds limit"
                )
                continue

            doc = Document(
                text=content,
                meta_data={
                    "file_path": relative_path,
                    "type": ext[1:],
                    "is_code": True,
                    "is_implementation": is_implementation,
                    "title": relative_path,
                    "encoding_status": status,  # Track how file was read
                },
            )
            documents.append(doc)

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(
                file_path,
                use_inclusion_mode,
                included_dirs,
                included_files,
                excluded_dirs,
                excluded_files,
            ):
                continue

            # Safely read file with encoding detection
            content, status = safe_read_file(file_path)

            if content is None:
                # Log specific reasons for skipping files
                relative_path = os.path.relpath(file_path, path)
                if status.startswith("binary_file"):
                    logger.debug(f"Skipping binary file {relative_path}: {status}")
                elif status == "empty_file" or status == "empty_content":
                    logger.debug(f"Skipping empty file {relative_path}")
                elif status == "all_encodings_failed":
                    logger.warning(
                        f"Skipping file {relative_path}: Unable to decode with any encoding"
                    )
                else:
                    logger.warning(f"Skipping file {relative_path}: {status}")
                continue

            try:
                relative_path = os.path.relpath(file_path, path)

                # Check token count
                if len(content) > MAX_EMBEDDING_LENGTH:
                    logger.warning(
                        f"Skipping large file {relative_path}: Token count ({len(content)}) exceeds limit"
                    )
                    continue

                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:],
                        "is_code": False,
                        "is_implementation": False,
                        "title": relative_path,
                        "encoding_status": status,  # Track how file was read
                    },
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Found {len(documents)} files to be processed")

    return documents


def prepare_data_transformer() -> adal.Sequential:
    """
    Creates and returns the data transformation pipeline.
    Uses dynamic splitter that automatically selects appropriate splitter
    (code_splitter or natural_language_splitter) based on document type.

    Returns:
        adal.Sequential: The data transformation pipeline
    """
    use_dual_vector = configs()["rag"]["embedder"]["sketch_filling"]
    code_understanding_config = configs()["rag"]["code_understanding"]

    if configs()["rag"]["dynamic_splitter"]["enabled"]:
        # Use dynamic splitter that automatically selects appropriate splitter
        splitter = DynamicSplitterTransformer()
    else:
        splitter = TextSplitter(**configs()["rag"]["text_splitter"])

    embedder_model_config = configs()["rag"]["embedder"]["model_kwargs"]
    embedder = get_embedder()
    if use_dual_vector:
        code_understanding_generator = CodeUnderstandingGenerator(
            **code_understanding_config
        )
        embedder_transformer = DualVectorToEmbeddings(
            embedder=embedder, generator=code_understanding_generator
        )
        logger.info("Using DualVectorToEmbeddings transformer.")
    elif embedder.__class__.__name__ == "HuggingfaceEmbedder":
        # HuggingFace can use larger batch sizes
        batch_size = embedder_model_config["batch_size"]
        embedder_transformer = HuggingfaceClientToEmbeddings(
            embedder=embedder,
            batch_size=batch_size,
        )
        logger.info(f"Using HuggingFace embedder with batch size: {batch_size}")
    elif embedder.__class__.__name__ == "DashScopeEmbedder":
        # DashScope API limits batch size to maximum of 10
        batch_size = min(embedder_model_config["batch_size"], 10)
        embedder_transformer = DashScopeToEmbeddings(
            embedder=embedder, batch_size=batch_size
        )
        logger.info(
            f"Using DashScope specialized embedder with batch size: {batch_size} (API limit: ≤10)"
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder.__class__.__name__}")

    data_transformer = adal.Sequential(splitter, embedder_transformer)

    return data_transformer


def transform_documents_and_save_to_db(
    documents: List[Document],
    db_path: str,
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.

    Returns:
        LocalDB: The local database instance.
    """
    data_transformer = prepare_data_transformer()

    # Save the documents to a local database
    db = LocalDB()
    # Build a relation from the key to the data transformer, required by adalflow.core.db
    db.register_transformer(transformer=data_transformer, key=os.path.basename(db_path))
    db.load(documents)
    # Suppose the data transformer is HuggingfaceClientToEmbeddings, then
    # this function will call the HuggingfaceClientToEmbeddings.__call__ function
    db.transform(key=os.path.basename(db_path))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db


class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self, repo_path: str):
        self.db = None
        self.db_info = None
        self.data_transformer = None
        self.repo_path = repo_path

        assert "rag" in configs(), "configs() must contain rag section"
        rag_config = configs()["rag"]
        assert "embedder" in rag_config, "rag_config must contain embedder section"
        assert (
            "sketch_filling" in rag_config["embedder"]
        ), "rag_config must contain sketch_filling section"
        self.use_dual_vector = rag_config["embedder"]["sketch_filling"]

        assert "hybrid" in rag_config, "rag_config must contain hybrid section"
        assert (
            "enabled" in rag_config["hybrid"]
        ), "hybrid_config must contain enabled section"
        self.use_bm25 = rag_config["hybrid"]["enabled"]

    def _create_db_info(self) -> None:
        logger.info(f"Preparing repo storage for {self.repo_path}...")

        root_path = get_adalflow_default_root_path()

        os.makedirs(root_path, exist_ok=True)

        save_db_file = os.path.join(
            root_path, "databases", f"{self._prepare_embedding_cache_file_name()}.pkl"
        )

        os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

        self.db_info = {
            "repo_path": self.repo_path,
            "db_file_path": save_db_file,
        }
        logger.info(f"DB info: {self.db_info}")

    def _prepare_embedding_cache_file_name(self):
        # Extract repository name from path
        repo_name = os.path.abspath(self.repo_path.rstrip("/"))
        file_name = repo_name
        if self.use_dual_vector:
            file_name += "-dual-vector"
        if self.use_bm25:
            file_name += "-bm25"
        file_name = file_name.replace("/", "#")
        embedding_provider = configs()["rag"]["embedder"]["client_class"]
        embedding_model = configs()["rag"]["embedder"]["model"]
        file_name += f"-{embedding_provider}-{embedding_model}".replace("/", "#")
        return file_name

    def prepare_database(self) -> List[Union[Document, DualVectorDocument]]:
        """
        Create a new database from the repository.

        Returns:
            List[Document]: List of Document objects
        """
        self.data_transformer = prepare_data_transformer()
        self._create_db_info()
        return self.prepare_db_index()

    def prepare_db_index(self) -> List[Union[Document, DualVectorDocument]]:
        """
        Prepare the indexed database for the repository.

        Returns:
            List[Document]: List of Document objects
        """

        force_recreate = configs()["rag"]["embedder"]["force_embedding"]

        # check the database
        if (
            self.db_info
            and os.path.exists(self.db_info["db_file_path"])
            and not force_recreate
        ):
            logger.info("Loading existing database...")
            self.db = LocalDB.load_state(self.db_info["db_file_path"])
            documents = self.db.get_transformed_data(
                key=os.path.basename(self.db_info["db_file_path"])
            )
            if documents:
                logger.info(f"Loaded {len(documents)} documents from existing database")
                return documents
            else:
                logger.warning("No documents found in the existing database")
                return []

        # prepare the database
        logger.info("Creating new database...")
        documents = read_all_documents(
            self.db_info["repo_path"],
        )
        self.db = transform_documents_and_save_to_db(
            documents, self.db_info["db_file_path"]
        )
        documents = self.db.get_transformed_data(
            key=os.path.basename(self.db_info["db_file_path"])
        )
        return documents
