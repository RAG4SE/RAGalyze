import os
from typing import List, Optional, Tuple, Union, Literal
import glob
import re
import fnmatch
import pickle
import asyncio
import aiofiles
from functools import partial

import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB

from ragalyze.rag.splitter import MyTextSplitter
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.rag.transformer_registry import create_embedder_transformer
from ragalyze.core.types import DualVectorDocument
from ragalyze.rag.code_understanding import CodeUnderstandingGenerator
from ragalyze.rag.dynamic_splitter_transformer import DynamicSplitterTransformer
from ragalyze.configs import get_batch_embedder, configs

# The setting is from the observation that the maximum length of Solidity compiler's files is 919974
MAX_EMBEDDING_LENGTH = 1000000

# Configure logging
logger = get_tqdm_compatible_logger(__name__)


def get_programming_language(file_extension: str) -> str:
    """
    Determine the programming language based on file extension.
    
    Args:
        file_extension: The file extension (e.g., '.py', '.js', '.md')
        
    Returns:
        str: The programming language name or 'documentation' for non-code files
    """
    extension_to_language = {
        # Python
        '.py': 'python',
        '.pyi': 'python',
        
        # JavaScript/TypeScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        
        # Java
        '.java': 'java',
        
        # C/C++
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        
        # Go
        '.go': 'go',
        
        # Rust
        '.rs': 'rust',
        
        # Web
        '.html': 'html',
        '.css': 'css',
        '.php': 'php',
        
        # Swift
        '.swift': 'swift',
        
        # C#
        '.cs': 'csharp',
        
        # Solidity
        '.sol': 'solidity',
        
        # OCaml
        '.ml': 'ocaml',
        '.mli': 'ocaml',
        
        # Haskell
        '.hs': 'haskell',
        '.lhs': 'haskell',
        
        # F#
        '.fs': 'fsharp',
        '.fsi': 'fsharp',
        '.fsx': 'fsharp',
        
        # Erlang
        '.erl': 'erlang',
        '.hrl': 'erlang',
        
        # Elixir
        '.ex': 'elixir',
        '.exs': 'elixir',
        
        # Clojure
        '.clj': 'clojure',
        '.cljs': 'clojure',
        '.cljc': 'clojure',
        
        # Scheme/Lisp
        '.scm': 'scheme',
        '.ss': 'scheme',
        '.lisp': 'lisp',
        '.lsp': 'lisp',
        
        # Elm
        '.elm': 'elm',
        
        # Ruby
        '.rb': 'ruby',
        
        # Perl
        '.pl': 'perl',
        '.pm': 'perl',
        
        # Scala
        '.scala': 'scala',
        '.sc': 'scala',
        
        # Kotlin
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        
        # Dart
        '.dart': 'dart',
        
        # Lua
        '.lua': 'lua',
        
        # R
        '.r': 'r',
        '.R': 'r',
        
        # Objective-C
        '.m': 'objectivec',
        
        # Julia
        '.jl': 'julia',
        
        # Nim
        '.nim': 'nim',
        
        # Crystal
        '.cr': 'crystal',
        
        # Zig
        '.zig': 'zig',
        
        # V
        '.v': 'vlang',
        
        # SystemVerilog/Verilog
        '.sv': 'systemverilog',
        '.vhd': 'vhdl',
        '.vhdl': 'vhdl',
        
        # Assembly
        '.s': 'assembly',
        '.S': 'assembly',
        '.asm': 'assembly',
        
        # Shell scripts
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.fish': 'shell',
        '.ps1': 'shell',
        '.psm1': 'shell',
        '.bat': 'shell',
        '.cmd': 'shell',
        
        # Configuration files
        '.toml': 'ini',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.xaml': 'ini',
        '.proto': 'ini',
        '.graphql': 'ini',
        '.gql': 'ini',
        '.sql': 'ini',
        
        # Build systems
        '.mk': 'makefile',
        '.cmake': 'makefile',
        '.gradle': 'makefile',
        '.sbt': 'makefile',
        '.bazel': 'makefile',
        '.bzl': 'makefile',
        
        # Documentation and data formats
        '.md': 'documentation',
        '.markdown': 'documentation',
        '.mdown': 'documentation',
        '.mkd': 'documentation',
        '.mkdn': 'documentation',
        '.rst': 'documentation',
        '.txt': 'documentation',
        '.json': 'documentation',
        '.jsonl': 'documentation',
        '.yaml': 'documentation',
        '.yml': 'documentation',
        # A special case for mybatis framework, xml is actually code
        '.xml': 'documentation',
    }
    
    return extension_to_language.get(file_extension.lower(), 'unknown')


async def safe_read_file_async(file_path: str) -> Tuple[Optional[str], str]:
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
        async with aiofiles.open(file_path, "rb") as f:
            # Read first 8192 bytes to check for binary content
            chunk = await f.read(8192)
            if not chunk:
                return None, "empty_file"

            # Check for common binary file signatures
            binary_signatures = [
                b"\x00",  # NULL byte (common in binary files)
                b"\xff\xd8\xff",  # JPEG
                b"\x89PNG",  # PNG
                b"GIF8",  # GIF
                b"%PDF",  # PDF
                b"\x50\x4B",  # ZIP/JAR/etc
                b"\x7fELF",  # ELF executable
                b"MZ",  # DOS/Windows executable
            ]

            # If file contains NULL bytes or binary signatures, skip it
            if b"\x00" in chunk:
                return None, "binary_file_null_bytes"

            for sig in binary_signatures:
                if chunk.startswith(sig):
                    return None, f"binary_file_signature_{sig.hex()}"

            # Check if chunk has too many non-printable characters
            # Allow byte >= 128 to support UTF-8 multi-byte chars, like Chinese
            printable_chars = sum(
                1
                for byte in chunk
                if 32 <= byte <= 126 or byte in [9, 10, 13] or byte >= 128
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
            async with aiofiles.open(file_path, "rb") as f:
                raw_data = await f.read()
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
            async with aiofiles.open(file_path, "r", encoding=encoding, errors="strict") as f:
                content = await f.read()

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
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = await f.read()
            if content.strip():
                logger.warning(
                    f"Read {file_path} with character replacement (some characters may be corrupted)"
                )
                return content, "success_utf8_with_replacement"
    except Exception as e:
        logger.error(f"Final fallback failed for {file_path}: {e}")

    return None, "all_encodings_failed"


def safe_read_file(file_path: str) -> Tuple[Optional[str], str]:
    """Synchronous wrapper for safe_read_file_async for backward compatibility."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an event loop, run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, safe_read_file_async(file_path))
                return future.result()
        else:
            # Otherwise, just run the async function
            return asyncio.run(safe_read_file_async(file_path))
    except Exception:
        # Fallback to sync version if asyncio fails
        return _safe_read_file_sync(file_path)


def _safe_read_file_sync(file_path: str) -> Tuple[Optional[str], str]:
    """Synchronous implementation for fallback."""
    # Check if file is likely binary by examining first few bytes
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            if not chunk:
                return None, "empty_file"

            binary_signatures = [
                b"\x00", b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"%PDF", 
                b"\x50\x4B", b"\x7fELF", b"MZ"
            ]

            if b"\x00" in chunk:
                return None, "binary_file_null_bytes"

            for sig in binary_signatures:
                if chunk.startswith(sig):
                    return None, f"binary_file_signature_{sig.hex()}"

            printable_chars = sum(
                1 for byte in chunk
                if 32 <= byte <= 126 or byte in [9, 10, 13] or byte >= 128
            )
            if len(chunk) > 0 and printable_chars / len(chunk) < 0.7:
                return None, "binary_file_non_printable"

    except (IOError, OSError) as e:
        logger.error(f"File access error for {file_path}: {e}")
        raise

    encodings_to_try = [
        "utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1", "ascii"
    ]

    try:
        import chardet
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                if detected["encoding"] and detected["confidence"] > 0.7:
                    detected_encoding = detected["encoding"].lower()
                    if detected_encoding not in [enc.lower() for enc in encodings_to_try]:
                        encodings_to_try.insert(0, detected["encoding"])
        except Exception:
            pass
    except ImportError:
        pass

    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as f:
                content = f.read()
                if not content.strip():
                    return None, "empty_content"
                if encoding != "utf-8":
                    logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content, f"success_{encoding}"
        except UnicodeDecodeError as e:
            logger.debug(f"Failed to read {file_path} with {encoding}: {e}")
            continue
        except (IOError, OSError) as e:
            logger.error(f"File error for {file_path}: {e}")
            return None, f"file_error: {e}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            if content.strip():
                logger.warning(f"Read {file_path} with character replacement")
                return content, "success_utf8_with_replacement"
    except Exception as e:
        logger.error(f"Final fallback failed for {file_path}: {e}")

    return None, "all_encodings_failed"


async def read_all_documents_async(path: str, max_concurrent: int = 50):
    """
    Recursively reads all documents in a directory and its subdirectories using multicoroutine for acceleration.

    Args:
        path (str): The root directory path.
        max_concurrent (int): Maximum number of concurrent coroutines for file processing.

    Returns:
        list: A list of Document objects with metadata.
    """

    excluded_patterns = list(
        set(
            configs()["repo"]["file_filters"]["excluded_patterns"]
            + configs()["repo"]["file_filters"]["extra_excluded_patterns"]
        )
    )
    code_extensions = configs()["repo"]["file_extensions"]["code_extensions"]
    doc_extensions = configs()["repo"]["file_extensions"]["doc_extensions"]

    logger.info(f"Reading documents from {path} with {max_concurrent} concurrent coroutines")

    def should_process_file(
        file_path: str,
        excluded_patterns: List[str],
    ) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.
        Supports glob patterns and regular expressions for matching.

        Args:
            file_path (str): The file path to check
            excluded_patterns (List[str]): List of patterns to exclude (supports glob patterns)

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        # Normalize the file path for consistent matching
        normalized_path = os.path.normpath(file_path).replace("\\", "/")

        def matches_pattern(text: str, pattern: str) -> bool:
            """Check if text matches a glob pattern."""
            if fnmatch.fnmatch(text, pattern):
                return True

            return False

        # Check if file is in an excluded directory
        for excluded_pattern in excluded_patterns:
            if matches_pattern(normalized_path, excluded_pattern):
                return False

        return True

    async def process_file_async(file_path: str, path: str, is_code: bool) -> Optional[Document]:
        """Process a single file asynchronously and return Document object or None if skipped."""
        # Check if file should be processed based on inclusion/exclusion rules
        if not should_process_file(file_path, excluded_patterns):
            return None

        # Safely read file with encoding detection
        content, status = await safe_read_file_async(file_path)

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
            return None

        relative_path = os.path.relpath(file_path, path)

        # Check token count
        if len(content) > MAX_EMBEDDING_LENGTH:
            logger.warning(
                f"Skipping large file {relative_path}: Token count ({len(content)}) exceeds limit"
            )
            return None

        try:
            ext = os.path.splitext(relative_path)[1]
            programming_language = get_programming_language(ext)
            
            # Determine if this is actually code based on the programming language
            is_actually_code = programming_language not in ['documentation', 'ini', 'makefile', 'shell', 'unknown']
            
            if is_code and is_actually_code:
                # Determine if this is an implementation file
                is_implementation = (
                    not relative_path.startswith("test_")
                    and not relative_path.startswith("app_")
                    and not relative_path.startswith("build")
                    and "test" not in relative_path.lower()
                )
                
                return Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:] if ext else "",
                        "is_code": True,
                        "is_implementation": is_implementation,
                        "programming_language": programming_language,
                        "title": relative_path,
                        "encoding_status": status,
                    },
                )
            else:
                # For documentation files or files that aren't actually code
                return Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:] if ext else "",
                        "is_code": is_actually_code,
                        "is_implementation": False,
                        "programming_language": programming_language,
                        "title": relative_path,
                        "encoding_status": status,
                    },
                )
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    # Collect all files to process
    all_files = []
    
    # Process code files
    for ext in set(code_extensions):
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            all_files.append((file_path, True))  # True indicates code file
    
    # Process documentation files
    for ext in set(doc_extensions):
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            all_files.append((file_path, False))  # False indicates documentation file

    logger.info(f"Found {len(all_files)} files to process")

    # Process files concurrently with asyncio
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_path: str, path: str, is_code: bool) -> Optional[Document]:
        async with semaphore:
            return await process_file_async(file_path, path, is_code)

    # Create tasks for all files
    tasks = [
        process_with_semaphore(file_path, path, is_code)
        for file_path, is_code in all_files
    ]
    
    # Wait for all tasks to complete and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    documents = []
    for i, result in enumerate(results):
        file_path, _ = all_files[i]
        if isinstance(result, Exception):
            logger.error(f"Error processing file {file_path}: {result}")
        elif result is not None:
            documents.append(result)

    logger.info(f"Successfully processed {len(documents)} files")
    return documents


def read_all_documents(path: str, max_concurrent: int = 50):
    """
    Synchronous wrapper for read_all_documents_async for backward compatibility.
    
    Args:
        path (str): The root directory path.
        max_concurrent (int): Maximum number of concurrent coroutines for file processing.
    
    Returns:
        list: A list of Document objects with metadata.
    """
    try:
        return asyncio.run(read_all_documents_async(path, max_concurrent))
    except Exception as e:
        logger.error(f"Async processing failed, falling back to sync: {e}")
        # Fallback to sync implementation if async fails
        return _read_all_documents_sync(path, max_concurrent)


def _read_all_documents_sync(path: str, max_workers: int = 8):
    """Fallback synchronous implementation."""
    # This is the original implementation for fallback
    excluded_patterns = list(
        set(
            configs()["repo"]["file_filters"]["excluded_patterns"]
            + configs()["repo"]["file_filters"]["extra_excluded_patterns"]
        )
    )
    code_extensions = configs()["repo"]["file_extensions"]["code_extensions"]
    doc_extensions = configs()["repo"]["file_extensions"]["doc_extensions"]

    logger.info(f"Reading documents from {path} (sync fallback with {max_workers} workers)")

    def should_process_file(file_path: str, excluded_patterns: List[str]) -> bool:
        normalized_path = os.path.normpath(file_path).replace("\\", "/")
        def matches_pattern(text: str, pattern: str) -> bool:
            return fnmatch.fnmatch(text, pattern)
        for excluded_pattern in excluded_patterns:
            if matches_pattern(normalized_path, excluded_pattern):
                return False
        return True

    documents = []
    
    # Process code files
    for ext in set(code_extensions):
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            if not should_process_file(file_path, excluded_patterns):
                continue
            content, status = safe_read_file(file_path)
            if content is None:
                relative_path = os.path.relpath(file_path, path)
                if status.startswith("binary_file"):
                    logger.debug(f"Skipping binary file {relative_path}: {status}")
                elif status in ["empty_file", "empty_content"]:
                    logger.debug(f"Skipping empty file {relative_path}")
                elif status == "all_encodings_failed":
                    logger.warning(f"Skipping file {relative_path}: Unable to decode")
                else:
                    logger.warning(f"Skipping file {relative_path}: {status}")
                continue
            relative_path = os.path.relpath(file_path, path)
            if len(content) > MAX_EMBEDDING_LENGTH:
                logger.warning(f"Skipping large file {relative_path}: exceeds limit")
                continue
            
            ext = os.path.splitext(relative_path)[1]
            programming_language = get_programming_language(ext)
            
            # Determine if this is actually code based on the programming language
            # xml is not actually code, but java can be used with xml in mybatis framework,
            # so we need to count xml as code
            is_actually_code = programming_language not in ['documentation', 'text', 'json', 'yaml', 'unknown']
            
            if is_actually_code:
                # Determine if this is an implementation file
                is_implementation = (
                    not relative_path.startswith("test_")
                    and not relative_path.startswith("app_")
                    and not relative_path.startswith("build")
                    and "test" not in relative_path.lower()
                )
                
                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:] if ext else "",
                        "is_code": True,
                        "is_implementation": is_implementation,
                        "programming_language": programming_language,
                        "title": relative_path,
                        "encoding_status": status,
                    },
                )
            else:
                # File extension is in code_extensions but it's not actually code (e.g., .md files)
                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:] if ext else "",
                        "is_code": False,
                        "is_implementation": False,
                        "programming_language": programming_language,
                        "title": relative_path,
                        "encoding_status": status,
                    },
                )
            documents.append(doc)

    # Process documentation files
    for ext in set(doc_extensions):
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            if not should_process_file(file_path, excluded_patterns):
                continue
            content, status = safe_read_file(file_path)
            if content is None:
                relative_path = os.path.relpath(file_path, path)
                if status.startswith("binary_file"):
                    logger.debug(f"Skipping binary file {relative_path}: {status}")
                elif status in ["empty_file", "empty_content"]:
                    logger.debug(f"Skipping empty file {relative_path}")
                elif status == "all_encodings_failed":
                    logger.warning(f"Skipping file {relative_path}: Unable to decode")
                else:
                    logger.warning(f"Skipping file {relative_path}: {status}")
                continue
            relative_path = os.path.relpath(file_path, path)
            if len(content) > MAX_EMBEDDING_LENGTH:
                logger.warning(f"Skipping large file {relative_path}: exceeds limit")
                continue
            ext = os.path.splitext(relative_path)[1]
            programming_language = get_programming_language(ext)
            
            doc = Document(
                text=content,
                meta_data={
                    "file_path": relative_path,
                    "type": ext[1:] if ext else "",
                    "is_code": False,
                    "is_implementation": False,
                    "programming_language": programming_language,
                    "title": relative_path,
                    "encoding_status": status,
                },
            )
            documents.append(doc)

    logger.info(f"Found {len(documents)} files to be processed")
    return documents


def prepare_data_transformer(
    mode: Literal[
        "only_splitter", "only_embedder", "splitter_and_embedder"
    ] = "splitter_and_embedder"
) -> adal.Sequential:
    """
    Creates and returns the data transformation pipeline.
    Uses dynamic splitter that automatically selects appropriate splitter
    (code_splitter or natural_language_splitter) based on document type.

    Returns:
        adal.Sequential: The data transformation pipeline
    """
    use_dual_vector = configs()["rag"]["embedder"]["sketch_filling"]
    code_understanding_config = configs()["rag"]["code_understanding"]

    if mode != "only_embedder":
        if configs()["rag"]["dynamic_splitter"]["enabled"]:
            # Use dynamic splitter that automatically selects appropriate splitter
            splitter = DynamicSplitterTransformer(
                batch_size=configs()["rag"]["dynamic_splitter"]["batch_size"],
                parallel=configs()["rag"]["dynamic_splitter"]["parallel"],
            )
        else:
            text_splitter_kwargs = configs()["rag"]["text_splitter"]
            if configs()["rag"]["adjacent_documents"]["enabled"]:
                text_splitter_kwargs["chunk_overlap"] = False
            splitter = MyTextSplitter(
                enable_line_number=configs()["generator"]["enable_line_number"],
                **text_splitter_kwargs,
            )

        if mode == "only_splitter":
            return adal.Sequential(splitter)

    embedder = get_batch_embedder()

    # Create code understanding generator (required for dual vector mode)
    code_understanding_generator = None
    if use_dual_vector:
        code_understanding_generator = CodeUnderstandingGenerator(
            **code_understanding_config
        )

    # Use registry pattern to create appropriate embedder transformer
    embedder_transformer = create_embedder_transformer(
        embedder=embedder,
        use_dual_vector=use_dual_vector,
        code_understanding_generator=code_understanding_generator,
    )

    if mode == "splitter_and_embedder":
        data_transformer = adal.Sequential(splitter, embedder_transformer)
    elif mode == "only_embedder":
        data_transformer = adal.Sequential(embedder_transformer)

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
    data_transformer = prepare_data_transformer(mode="splitter_and_embedder")

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
        self.repo_path = repo_path

        assert "rag" in configs(), "configs() must contain rag section"
        rag_config = configs()["rag"]
        assert "embedder" in rag_config, "rag_config must contain embedder section"
        assert (
            "sketch_filling" in rag_config["embedder"]
        ), "rag_config must contain sketch_filling section"
        self.use_dual_vector = rag_config["embedder"]["sketch_filling"]

        # Query-driven specific attributes
        self.query_driven = rag_config.get("query_driven", {}).get("enabled", False)

        self.dynamic_splitter = rag_config.get("dynamic_splitter", {}).get(
            "enabled", False
        )

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
        if self.query_driven:
            file_name += "-query-driven"
        if self.dynamic_splitter:
            file_name += "-dynamic-splitter"
        file_name = file_name.replace("/", "#")
        embedding_provider = configs()["rag"]["embedder"]["provider"]
        embedding_model = configs()["rag"]["embedder"]["model"]
        file_name += f"-{embedding_provider}-{embedding_model}".replace("/", "#")
        return file_name

    def prepare_database(self) -> List[Union[Document, DualVectorDocument]]:
        """
        Create a new database from the repository.

        Returns:
            List[Document]: List of Document objects
        """
        self._create_db_info()
        return self.prepare_db_index()

    def prepare_db_index(self) -> List[Union[Document, DualVectorDocument]]:
        """
        Prepare the indexed database for the repository.

        Returns:
            List[Document]: List of Document objects
        """

        force_recreate = configs()["rag"]["embedder"]["force_embedding"]

        if not self.query_driven:
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
                    logger.info(
                        f"Loaded {len(documents)} documents from existing database"
                    )
                    return documents
                else:
                    logger.warning("No documents found in the existing database")
                    return []
            
            cache_dir = os.path.expanduser(configs()["doc_cache_path"])
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file_path = os.path.join(
                cache_dir, self.db_info["repo_path"].replace("/", "#") + ".pkl"
            )
            if not configs()["rag"]["embedder"]["force_embedding"] and os.path.exists(
                self.cache_file_path
            ):
                logger.info(f"Loading documents from cache file {self.cache_file_path}")
                documents = pickle.load(open(self.cache_file_path, "rb"))
            else:
                documents = read_all_documents(
                    self.db_info["repo_path"],
                )
                pickle.dump(documents, open(self.cache_file_path, "wb"))

            self.db = transform_documents_and_save_to_db(
                documents, self.db_info["db_file_path"]
            )
            documents = self.db.get_transformed_data(
                key=os.path.basename(self.db_info["db_file_path"])
            )
            if isinstance(documents[0], Document):
                id2doc = {doc.id: doc for doc in documents}
            elif isinstance(documents[0], DualVectorDocument):
                id2doc = {doc.original_doc.id: doc.original_doc for doc in documents}
            else:
                raise ValueError("documents must be a list of Document or DualVectorDocument")
            pickle.dump(id2doc, open(self.cache_file_path + ".id2doc.pkl", "wb"))
            # If query-driven, return the original documents directly
            return documents

        else:
            cache_dir = os.path.expanduser(configs()["doc_cache_path"])
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file_path = os.path.join(
                cache_dir, self.db_info["repo_path"].replace("/", "#") + ".pkl"
            )
            if not configs()["rag"]["embedder"]["force_embedding"] and os.path.exists(
                self.cache_file_path
            ):
                logger.info(f"Loading documents from cache file {self.cache_file_path}")
                return pickle.load(open(self.cache_file_path, "rb"))

            documents = read_all_documents(
                self.db_info["repo_path"],
            )
            splitter = prepare_data_transformer(mode="only_splitter")
            splitted_documents = splitter(documents)
            id2doc = {doc.id: doc for doc in splitted_documents}
            pickle.dump(splitted_documents, open(self.cache_file_path, "wb"))
            pickle.dump(id2doc, open(self.cache_file_path + ".id2doc.pkl", "wb"))
            return splitted_documents

    def update_database_with_documents(
        self, documents: List[Document]
    ) -> List[Union[Document, DualVectorDocument]]:
        """
        Update the database with new documents by embedding them and saving to the database.

        Args:
            documents: List of Document objects to be embedded and added to the database

        Returns:
            List[Union[Document, DualVectorDocument]]: List of embedded documents
        """
        if not documents:
            return []

        key = (
            os.path.basename(self.db_info["db_file_path"])
            if self.db_info
            else "default"
        )

        # Ensure we have a database instance
        if self.db is None:
            # Try to load existing database first
            if (
                self.db_info
                and os.path.exists(self.db_info["db_file_path"])
                and not configs()["rag"]["embedder"]["force_embedding"]
            ):
                logger.info("Loading existing database...")
                self.db = LocalDB.load_state(self.db_info["db_file_path"])
            else:
                # Create new database if none exists
                logger.info("Creating new database...")
                self.db = LocalDB()
                self.db.transformed_items[key] = []

        # Add documents to database
        # Only add documents that are not already in the database
        existing_item_ids = set(
            item.id if isinstance(item, Document) else item.original_doc.id
            for item in self.db.items
        )

        new_documents = [
            doc
            for doc in documents
            if (hasattr(doc, "id") and doc.id not in existing_item_ids)
            or (
                hasattr(doc, "original_doc")
                and hasattr(doc.original_doc, "id")
                and doc.original_doc.id not in existing_item_ids
            )
        ]

        document_ids = set(
            doc.id if isinstance(doc, Document) else doc.original_doc.id
            for doc in documents
        )

        cached_embedded_documents = [
            doc
            for doc in self.db.transformed_items[key]
            if (hasattr(doc, "id") and doc.id in document_ids)
            or (
                hasattr(doc, "original_doc")
                and hasattr(doc.original_doc, "id")
                and doc.original_doc.id in document_ids
            )
        ]

        new_embedded_documents = []

        if len(new_documents) > 0:
            embedder = prepare_data_transformer(mode="only_embedder")
            new_embedded_documents = embedder(new_documents)
            self.db.transformed_items[key] = (
                self.db.transformed_items[key] + new_embedded_documents
            )
            self.db.items = self.db.items + new_documents

        logger.info(
            f"Updated database with {len(new_embedded_documents)} new documents and {len(cached_embedded_documents)} cached documents"
        )

        # Save the updated database
        if self.db_info and self.db_info.get("db_file_path"):
            self.db.save_state(filepath=self.db_info["db_file_path"])
            logger.info(f"Database updated and saved to {self.db_info['db_file_path']}")
        return new_embedded_documents + cached_embedded_documents

    def get_embedded_documents(self) -> List[Union[Document, DualVectorDocument]]:
        """
        Get all embedded documents from the database.

        Returns:
            List[Union[Document, DualVectorDocument]]: List of embedded documents
        """
        if self.db is None:
            return []

        key = (
            os.path.basename(self.db_info["db_file_path"])
            if self.db_info
            else "default"
        )
        return self.db.get_transformed_data(key=key)
