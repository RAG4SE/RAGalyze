import time
import multiprocessing
from typing import List, Optional, Union, Tuple
from concurrent.futures import as_completed
from tqdm import tqdm
from dataclasses import dataclass

from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.rag.treesitter_parse_interface import tokenize_for_bm25, set_debug_mode
from ragalyze.core.types import DualVectorDocument
from adalflow.core.types import Document
from adalflow.core.component import Component

# Import DaemonThreadPoolExecutor from db.py or define it here
from ragalyze.core.utils import DaemonThreadPoolExecutor

logger = get_tqdm_compatible_logger(__name__)


@dataclass
class BM25Index:
    """Class to store BM25 tokens with their positions."""
    token: str
    position: Tuple[int, int]  # (line_number, column_number)

    def __init__(self, token: str, position: Tuple[int, int]):
        self.token = token
        self.position = position


class BM25Transformer(Component):
    """Transformer class for building BM25 indexes for documents."""

    def __init__(
        self, use_multithreading: bool = True, max_workers: Optional[int] = None
    ):
        super().__init__()
        self.use_multithreading = use_multithreading
        self.max_workers = max_workers
        if self.max_workers is None:
            self.max_workers = multiprocessing.cpu_count()

    def build_bm25_index_for_a_file(
        self, code: str, language: str, file_path: str
    ) -> List[BM25Index]:
        """Build BM25 index for a single file."""
        set_debug_mode(0)
        bm25_indexes = tokenize_for_bm25(code, language, file_path)
        tokens = [index[0] for index in bm25_indexes]
        positions = [(index[1], index[2]) for index in bm25_indexes]
        return [BM25Index(token=token, position=pos) for token, pos in zip(tokens, positions)]

    def call(
        self, documents: List[Union[Document, DualVectorDocument]]
    ) -> List[Union[Document, DualVectorDocument]]:
        """Initialize BM25 index with document texts."""
        try:
            # Build BM25 index for documents
            logger.info("Building BM25 index...")
            if self.use_multithreading and len(documents) > 1:
                # Determine number of worker threads
                self.max_workers = min(self.max_workers, len(documents))
                logger.info(f"Using {self.max_workers} threads for bm25 indexing")
                start_time = time.time()
                # Use ThreadPoolExecutor to parallelize tokenization
                with DaemonThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit tokenization tasks
                    try:
                        future_to_index = {}
                        for i, doc in enumerate(documents):
                            if isinstance(doc, DualVectorDocument):
                                # For dual vector documents, combine code and understanding text
                                if (
                                    "original_text" in doc.original_doc.meta_data
                                    and doc.original_doc.meta_data["original_text"]
                                ):
                                    text = doc.original_doc.meta_data["original_text"]
                                else:
                                    text = doc.original_doc.text
                            else:
                                if (
                                    "original_text" in doc.meta_data
                                    and doc.meta_data["original_text"]
                                ):
                                    text = doc.meta_data["original_text"]
                                else:
                                    text = doc.text
                            assert doc.meta_data[
                                "programming_language"
                            ], "Programming language is required for BM25 tokenization"
                            assert doc.meta_data[
                                "file_path"
                            ], "File path is required for BM25 tokenization"
                            future_to_index[
                                executor.submit(
                                    self.build_bm25_index_for_a_file,
                                    text,
                                    doc.meta_data["programming_language"],
                                    doc.meta_data["file_path"],
                                )
                            ] = i
                        logger.info("All tasks submitted, waiting for results...")
                        # Collect results in the correct order with progress bar
                        with tqdm(
                            total=len(documents), desc="BM25 multithreading indexing"
                        ) as pbar:
                            for future in as_completed(future_to_index):
                                index = future_to_index[future]
                                try:
                                    bm25_index = future.result()
                                    documents[index].meta_data["bm25_indexes"] = bm25_index
                                    pbar.update(1)
                                except Exception as e:
                                    logger.error(
                                        f"Error processing document {index}: {e}"
                                    )
                                    raise

                    except KeyboardInterrupt as e:
                        logger.error(f"Keyboard interrupt: {e}")
                        raise
                    finally:
                        logger.info("BM25 multithreading indexing completed")
                end_time = time.time()
                logger.info(
                    f"time taken for bm25 multithreading: {end_time - start_time}"
                )
            else:
                # Process documents sequentially (single-threaded)
                start_time = time.time()
                for doc in tqdm(documents, desc="BM25 single threading indexing"):
                    if isinstance(doc, DualVectorDocument):
                        # For dual vector documents, combine code and understanding text
                        if (
                            "original_text" in doc.original_doc.meta_data
                            and doc.original_doc.meta_data["original_text"]
                        ):
                            text = doc.original_doc.meta_data["original_text"]
                        else:
                            text = doc.original_doc.text
                    else:
                        if (
                            "original_text" in doc.meta_data
                            and doc.meta_data["original_text"]
                        ):
                            text = doc.meta_data["original_text"]
                        else:
                            text = doc.text

                    assert doc.meta_data[
                        "programming_language"
                    ], "Programming language is required for BM25 tokenization"
                    assert doc.meta_data[
                        "file_path"
                    ], "File path is required for BM25 tokenization"
                    bm25_index = self.build_bm25_index_for_a_file(
                        text,
                        doc.meta_data["programming_language"],
                        doc.meta_data["file_path"],
                    )
                    doc.meta_data["bm25_indexes"] = bm25_index
                end_time = time.time()
                logger.info(
                    f"time taken for bm25 single threading: {end_time - start_time}"
                )
            return documents

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            raise
