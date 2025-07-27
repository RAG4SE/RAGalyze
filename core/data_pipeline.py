import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter
import os
from typing import List, Optional, Tuple, Union
import logging
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from core.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from core.dual_vector_pipeline import DualVectorDocument
from core.config import get_embedder_config
from core.huggingface_embedder_client import HuggingfaceClientToEmbeddings
from core.dashscope_client import DashScopeToEmbeddings
from core.dual_vector_pipeline import DualVectorToEmbeddings
from core.tools.embedder import get_embedder

# The setting is from the observation that the maximum length of Solidity compiler's files is 919974
MAX_EMBEDDING_LENGTH = 1000000

# Configure logging
logger = logging.getLogger(__name__)


from typing import Optional
import logging

logger = logging.getLogger(__name__)

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
            encoding = tiktoken.encoding_for_model(model_name or "text-embedding-3-small")
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
            logger.warning(f"Unknown model provider '{model_provider}'; using rough 4-char-per-token estimate.")
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
        with open(file_path, 'rb') as f:
            # Read first 8192 bytes to check for binary content
            chunk = f.read(8192)
            if not chunk:
                return None, "empty_file"
            
            # Check for common binary file signatures
            binary_signatures = [
                b'\x00',  # NULL byte (common in binary files)
                b'\xFF\xD8\xFF',  # JPEG
                b'\x89PNG',  # PNG
                b'GIF8',  # GIF
                b'%PDF',  # PDF
                b'\x50\x4B',  # ZIP/JAR/etc
                b'\x7FELF',  # ELF executable
                b'MZ',  # DOS/Windows executable
            ]
            
            # If file contains NULL bytes or binary signatures, skip it
            if b'\x00' in chunk:
                return None, "binary_file_null_bytes"
            
            for sig in binary_signatures:
                if chunk.startswith(sig):
                    return None, f"binary_file_signature_{sig.hex()}"
            
            # Check if chunk has too many non-printable characters
            printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in [9, 10, 13])
            if len(chunk) > 0 and printable_chars / len(chunk) < 0.7:
                return None, "binary_file_non_printable"
    
    except (IOError, OSError) as e:
        return None, f"file_access_error: {e}"
    
    # Try different encodings in order of preference
    encodings_to_try = [
        'utf-8',
        'utf-8-sig',  # UTF-8 with BOM
        'latin1',     # Common fallback
        'cp1252',     # Windows encoding
        'iso-8859-1', # Another common encoding
        'ascii',      # Basic ASCII
    ]
    
    # Try to detect encoding if chardet is available
    try:
        import chardet
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                if detected['encoding'] and detected['confidence'] > 0.7:
                    detected_encoding = detected['encoding'].lower()
                    # Add detected encoding to the front of the list if not already there
                    if detected_encoding not in [enc.lower() for enc in encodings_to_try]:
                        encodings_to_try.insert(0, detected['encoding'])
        except Exception:
            pass  # Continue with default encodings if detection fails
    except ImportError:
        pass  # chardet not available, use default encodings
    
    # Try each encoding
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                content = f.read()
                
                # Additional validation for the content
                if not content.strip():
                    return None, "empty_content"
                
                # Check if content seems reasonable (not too many weird characters)
                if encoding != 'utf-8':
                    logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                
                return content, f"success_{encoding}"
                
        except UnicodeDecodeError as e:
            logger.debug(f"Failed to read {file_path} with {encoding}: {e}")
            continue
        except (IOError, OSError) as e:
            return None, f"file_error: {e}"
    
    # If all encodings failed, try one more time with errors='replace'
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if content.strip():
                logger.warning(f"Read {file_path} with character replacement (some characters may be corrupted)")
                return content, "success_utf8_with_replacement"
    except Exception as e:
        logger.debug(f"Final fallback failed for {file_path}: {e}")
    
    return None, "all_encodings_failed"

def read_all_documents(path: str, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.
        excluded_dirs (List[str], optional): List of directories to exclude from processing.
            Overrides the default configuration if provided.
        excluded_files (List[str], optional): List of file patterns to exclude from processing.
            Overrides the default configuration if provided.
        included_dirs (List[str], optional): List of directories to include exclusively.
            When provided, only files in these directories will be processed.
        included_files (List[str], optional): List of file patterns to include exclusively.
            When provided, only files matching these patterns will be processed.

    Returns:
        list: A list of Document objects with metadata.
    """
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [
        # Popular languages
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
        ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs", ".sol",
        # Functional languages
        ".ml", ".mli",  # OCaml
        ".hs", ".lhs",  # Haskell
        ".fs", ".fsi", ".fsx",  # F#
        ".erl", ".hrl",  # Erlang
        ".ex", ".exs",  # Elixir
        ".clj", ".cljs", ".cljc",  # Clojure
        ".scm", ".ss",  # Scheme
        ".lisp", ".lsp",  # Lisp
        ".elm",  # Elm
        # Other languages
        ".rb", ".pl", ".pm",  # Ruby, Perl
        ".scala", ".sc",  # Scala
        ".kt", ".kts",  # Kotlin
        ".dart",  # Dart
        ".lua",  # Lua
        ".r", ".R",  # R
        ".m",  # MATLAB/Objective-C
        ".jl",  # Julia
        ".nim",  # Nim
        ".cr",  # Crystal
        ".zig",  # Zig
        ".v", ".sv",  # Verilog/SystemVerilog
        ".vhd", ".vhdl",  # VHDL
        # Assembly and low-level
        ".s", ".S", ".asm",  # Assembly
        # Shell scripts
        ".sh", ".bash", ".zsh", ".fish",  # Shell scripts
        ".ps1", ".psm1",  # PowerShell
        ".bat", ".cmd",  # Batch files
        # Configuration and data
        ".toml", ".ini", ".cfg", ".conf",  # Config files
        ".xml", ".xaml",  # XML
        ".proto",  # Protocol Buffers
        ".graphql", ".gql",  # GraphQL
        ".sql",  # SQL
        # Build files
        ".mk", ".cmake",  # Make, CMake
        ".gradle", ".sbt",  # Gradle, SBT
        ".bazel", ".bzl",  # Bazel
    ]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".adoc", ".org", ".tex"]

    # Determine filtering mode: inclusion or exclusion
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

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
        # Exclusion mode: use default exclusions plus any additional ones
        final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files = set(DEFAULT_EXCLUDED_FILES)

        # Add any additional excluded directories from config
        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        # Add any additional excluded files from config
        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        # Add any explicitly provided excluded directories and files
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        # Convert back to lists for compatibility
        excluded_dirs = list(final_excluded_dirs)
        excluded_files = list(final_excluded_files)
        included_dirs = []
        included_files = []

        logger.info(f"Using exclusion mode")
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

    logger.info(f"Reading documents from {path}")

    def should_process_file(file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.

        Args:
            file_path (str): The file path to check
            use_inclusion (bool): Whether to use inclusion mode
            included_dirs (List[str]): List of directories to include
            included_files (List[str]): List of files to include
            excluded_dirs (List[str]): List of directories to exclude
            excluded_files (List[str]): List of files to exclude

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        file_path_parts = os.path.normpath(file_path).split(os.sep)
        file_name = os.path.basename(file_path)

        if use_inclusion:
            # Inclusion mode: file must be in included directories or match included files
            is_included = False

            # Check if file is in an included directory
            if included_dirs:
                for included in included_dirs:
                    clean_included = included.strip("./").rstrip("/")
                    if clean_included in file_path_parts:
                        is_included = True
                        break

            # Check if file matches included file patterns
            if not is_included and included_files:
                for included_file in included_files:
                    if file_name == included_file or file_name.endswith(included_file):
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
                clean_excluded = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            # Check if file matches excluded file patterns
            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
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
                    logger.warning(f"Skipping file {relative_path}: Unable to decode with any encoding")
                else:
                    logger.warning(f"Skipping file {relative_path}: {status}")
                continue
            
            try:
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
                    logger.warning(f"Skipping large file {relative_path}: Token count ({len(content)}) exceeds limit")
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
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
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
                    logger.warning(f"Skipping file {relative_path}: Unable to decode with any encoding")
                else:
                    logger.warning(f"Skipping file {relative_path}: {status}")
                continue
            
            try:
                relative_path = os.path.relpath(file_path, path)

                # Check token count
                if len(content) > MAX_EMBEDDING_LENGTH:
                    logger.warning(f"Skipping large file {relative_path}: Token count ({len(content)}) exceeds limit")
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

    logger.info(f"Found {len(documents)} documents")

    return documents

def prepare_data_pipeline(
    force_recreate_db: bool = False,
    embedding_cache_file_name: str = "default",
    use_dual_vector: bool = False
) -> Tuple[adal.Sequential, str]:
    """
    Creates and returns the data transformation pipeline.

    Returns:
        Tuple[adal.Sequential, str]: The data transformation pipeline and the cache file name as the key of the transformer.
    """
    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()
    embedder = get_embedder()
    if use_dual_vector:
        embedder_transformer = DualVectorToEmbeddings(
            embedder=embedder,
            force_recreate_db=force_recreate_db,
            embedding_cache_file_name=embedding_cache_file_name
        )
        logger.info("Using DualVectorToEmbeddings transformer.")
    elif embedder.__class__.__name__ == "HuggingfaceEmbedder":
        # HuggingFace can use larger batch sizes
        batch_size = embedder_config.get("batch_size", 100)
        embedder_transformer = HuggingfaceClientToEmbeddings(
            embedder=embedder, 
            force_recreate_db=force_recreate_db,
            batch_size=batch_size,
            embedding_cache_file_name=embedding_cache_file_name
        )
        logger.info(f"Using HuggingFace embedder with batch size: {batch_size}, embedding_cache_file_name: {embedding_cache_file_name}")
    elif embedder.__class__.__name__ == "DashScopeEmbedder":
        # DashScope API limits batch size to maximum of 10
        batch_size = min(embedder_config.get("batch_size", 10), 10)
        embedder_transformer = DashScopeToEmbeddings(
            embedder=embedder, 
            batch_size=batch_size,
            force_recreate_db=force_recreate_db,
            embedding_cache_file_name=embedding_cache_file_name
        )
        logger.info(f"Using DashScope specialized embedder with batch size: {batch_size}, embedding_cache_file_name: {embedding_cache_file_name} (API limit: ≤10)")
    else:
        raise ValueError(f"Unknown embedder type: {embedder.__class__.__name__}")

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )

    assert hasattr(embedder_transformer, 'cache_path'), f"embedder_transformer {embedder_transformer.__class__.__name__} does not have a cache_path attribute"

    return data_transformer, os.path.basename(embedder_transformer.cache_path).replace('.pkl', '')

def transform_documents_and_save_to_db(
    documents: List[Document],
    db_path: str,
    force_recreate: bool = False,
    embedding_cache_file_name: str = "default",
    use_dual_vector: bool = False,
    data_transformer: adal.Sequential = None,
    data_transformer_key: str = None,
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
        embedding_cache_file_name (str): Repository name for cache file naming.
    """
    if not data_transformer or not data_transformer_key:
        data_transformer, data_transformer_key = prepare_data_pipeline(
            force_recreate_db=force_recreate,
            embedding_cache_file_name=embedding_cache_file_name,
            use_dual_vector=use_dual_vector
        )

    # Save the documents to a local database
    db = LocalDB()
    # Build a relation from the key to the data transformer, required by adalflow.core.db
    db.register_transformer(transformer=data_transformer, key=data_transformer_key)
    db.load(documents)
    # Suppose the data transformer is HuggingfaceClientToEmbeddings, then
    # this function will call the HuggingfaceClientToEmbeddings.__call__ function
    db.transform(key=data_transformer_key)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self, repo_path: str, use_dual_vector: bool = False):
        self.db = None
        self.db_info = None
        self.data_transformer = None
        self.data_transformer_key = None
        self.embedding_cache_file_name = None

        self.repo_path = repo_path
        self.use_dual_vector = use_dual_vector

    def _create_db_info(self) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.adalflow/repos/{owner}_{repo_name}.pkl

        Args:
            repo_path (str): The local path of the repository
        """
        logger.info(f"Preparing repo storage for {self.repo_path}...")

        try:
            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)

            save_db_file = os.path.join(root_path, "databases", f"{self._prepare_embedding_cache_file_name()}.pkl")
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.db_info = {
                "save_repo_dir": self.repo_path,
                "save_db_file": save_db_file,
                "data_transformer_key": self.data_transformer_key,
            }
            logger.info(f"DB info: {self.db_info}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def _prepare_embedding_cache_file_name(self):
        if self.embedding_cache_file_name:
            return self.embedding_cache_file_name
        # Extract repository name from path
        repo_name = os.path.abspath(self.repo_path.rstrip('/'))
        file_name = repo_name
        if self.use_dual_vector:
            file_name += "_dual_vector"
        self.embedding_cache_file_name = file_name
        return file_name

    def prepare_database(self,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None, force_recreate: bool = False) -> List[Union[Document, DualVectorDocument]]:
        """
        Create a new database from the repository.

        Args:
            repo_path (str): The local path of the repository
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively
            force_recreate (bool, optional): Whether to force recreate the database
        Returns:
            List[Document]: List of Document objects
        """
        self.data_transformer, self.data_transformer_key = prepare_data_pipeline(
            force_recreate_db=force_recreate,
            embedding_cache_file_name=self._prepare_embedding_cache_file_name(),
            use_dual_vector=self.use_dual_vector
        )
        self._create_db_info()    
        return self.prepare_db_index(excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files, force_recreate=force_recreate)
    
    def prepare_db_index(self, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None, force_recreate: bool = False) -> List[Union[Document, DualVectorDocument]]:
        """
        Prepare the indexed database for the repository.

        Args:
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively
            force_recreate (bool, optional): Whether to force recreate the database
        Returns:
            List[Document]: List of Document objects
        """
        # check the database
        if self.db_info and os.path.exists(self.db_info["save_db_file"]) and not force_recreate:
            logger.info("Loading existing database...")
            try:
                self.db = LocalDB.load_state(self.db_info["save_db_file"])
                documents = self.db.get_transformed_data(key=self.db_info["data_transformer_key"])
                if documents:
                    logger.info(f"Loaded {len(documents)} documents from existing database")
                    return documents
                else:
                    logger.warning("No documents found in the existing database")
                    return []
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                # Continue to create a new database

        # prepare the database
        logger.info("Creating new database...")
        documents = read_all_documents(
            self.db_info["save_repo_dir"],
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )
        self.db = transform_documents_and_save_to_db(
            documents, self.db_info["save_db_file"],
            force_recreate=force_recreate, embedding_cache_file_name=self._prepare_embedding_cache_file_name(),
            use_dual_vector=self.use_dual_vector,
            data_transformer=self.data_transformer,
            data_transformer_key=self.data_transformer_key
        )
        documents = self.db.get_transformed_data(key=self.data_transformer_key)
        return documents

    def prepare_retriever(self, repo_path: str, type: str = "github", access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_path, type, access_token)
