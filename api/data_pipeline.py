import adalflow as adal
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
from typing import List, Optional, Callable, Any, TypeVar
from adalflow.core.component import Component
import logging
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES

from api.tools.embedder import get_embedder

# The setting is from the observation that the maximum length of Solidity compiler's files is 919974
MAX_EMBEDDING_LENGTH = 1000000

# Configure logging
logger = logging.getLogger(__name__)

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
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs", ".sol"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

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

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
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
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
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
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def prepare_data_pipeline(is_huggingface_embedder: bool = False, force_recreate_db: bool = False):
    """
    Creates and returns the data transformation pipeline.

    Args:
        is_huggingface_embedder (bool, optional): Whether to use HuggingFace for embedding.
        force_recreate_db (bool, optional): Whether to force recreate the database.

    Returns:
        adal.Sequential: The data transformation pipeline
    """
    from api.config import get_embedder_config
    from api.huggingface_embedder_client import HuggingfaceClientToEmbeddings, HuggingfaceClientBatchEmbedder

    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()

    embedder = get_embedder(is_huggingface_embedder)

    if is_huggingface_embedder:
        # Use Huggingface document processor for single-document processing
        batch_size = embedder_config.get("batch_size", 500)
        embedder_transformer = HuggingfaceClientToEmbeddings(embedder=embedder, force_recreate_db=force_recreate_db)
    else:
        # Use batch processing for other embedders
        batch_size = embedder_config.get("batch_size", 500)
        embedder_transformer = ToEmbeddings(
            embedder=embedder, batch_size=batch_size
        )

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer

# T = TypeVar("T")  # Allow any type as items

# class CustomDB(LocalDB):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.data_transformer = None
    
#     def transform(
#         self,
#         *,
#         force_recreate: bool = False,
#         transformer: Optional[Component] = None,
#         key: Optional[str] = None,
#         map_fn: Optional[Callable[[T], Any]] = None,
#     ) -> str:
#         """The main method to apply the transformer to the data in two ways:
#         1. Apply the transformer by key to the data using ``transform(key="test")``.
#         2. Register and apply the transformer to the data using ``transform(transformer, key="test")``.

#         Args:
#             transformer (Optional[Component], optional): The transformer to use. Defaults to None.
#             key (Optional[str], optional): The key to use for the transformer. Defaults to None.
#             map_fn (Optional[Callable[[T], Any]], optional): The map function to use. Defaults to None.

#         Returns:
#             str: The key used for the transformation, from which the transformed data can be accessed.
#         """
#         key_to_use = key
#         if transformer:
#             key = self.register_transformer(transformer, key, map_fn)
#             key_to_use = key
#         if key_to_use is None:
#             raise ValueError("Key must be provided.")

#         if map_fn is not None:
#             items_to_use = [map_fn(item) for item in self.items]
#         else:
#             items_to_use = self.items.copy()

#         transformer_to_use = self.transformer_setups[key_to_use]
#         assert hasattr(transformer_to_use, '__call__'), "The transformer must have a __call__ method"
#         import inspect
#         sig = inspect.signature(transformer_to_use.__call__)
#         if 'force_recreate' in sig.parameters:
#             self.transformed_items[key_to_use] = transformer_to_use(items_to_use, force_recreate=force_recreate)
#         else:
#             self.transformed_items[key_to_use] = transformer_to_use(items_to_use)
#         return key_to_use

def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, is_huggingface_embedder: bool = False, force_recreate: bool = False
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
    """
    # Get the data transformer
    # E.g., if is_huggingface_embedder is True, the data transformer will be HuggingfaceClientToEmbeddings defined in api/huggingface_embedder_client.py
    data_transformer = prepare_data_pipeline(is_huggingface_embedder, force_recreate_db=force_recreate)

    # Save the documents to a local database
    db = LocalDB()
    # Build a relation from the key to the data transformer, required by adalflow.core.db
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    # Suppose the data transformer is HuggingfaceClientToEmbeddings, then
    # this function will call the HuggingfaceClientToEmbeddings.__call__ function
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_path: str,
                       excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                       included_dirs: List[str] = None, included_files: List[str] = None, force_recreate: bool = False,
                       is_huggingface_embedder: bool = False) -> List[Document]:
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
        self.reset_database()
        self._create_repo(repo_path)    
        return self.prepare_db_index(excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files, force_recreate=force_recreate,
                                   is_huggingface_embedder=is_huggingface_embedder)

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _create_repo(self, repo_path: str) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.adalflow/repos/{owner}_{repo_name} (for url, local path will be the same)
        ~/.adalflow/databases/{owner}_{repo_name}.pkl

        Args:
            repo_path (str): The local path of the repository
        """
        logger.info(f"Preparing repo storage for {repo_path}...")

        try:
            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)

            repo_name = os.path.basename(repo_path)

            save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": repo_path,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path = repo_path
            logger.info(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(self, excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None, force_recreate: bool = False,
                        is_huggingface_embedder: bool = False) -> List[Document]:
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
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]) and not force_recreate:
            logger.info("Loading existing database...")
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                documents = self.db.get_transformed_data(key="split_and_embed")
                if documents:
                    logger.info(f"Loaded {len(documents)} documents from existing database")
                    return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                # Continue to create a new database

        # prepare the database
        logger.info("Creating new database...")
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"],
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )

        self.db = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], is_huggingface_embedder=is_huggingface_embedder,
            force_recreate=force_recreate
        )
        logger.info(f"Total documents: {len(documents)}")
        transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        logger.info(f"Total transformed documents: {len(transformed_docs)}")
        return transformed_docs

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_url_or_path, type, access_token)
