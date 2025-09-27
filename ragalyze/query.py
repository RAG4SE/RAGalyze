#!/usr/bin/env python3
"""
RAGalyze Query Module
"""

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import pickle
from omegaconf import DictConfig
from copy import deepcopy
import os
import sys
import hydra

from adalflow.core.types import Document

from ragalyze.rag.rag import RAG, GeneratorWrapper
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.configs import *
from ragalyze.core.types import DualVectorDocument

# Setup logging
logger = get_tqdm_compatible_logger(__name__)


def save_query_results(
    result: Dict[str, Any], bm25_keywords: str, faiss_query: str, question: str
) -> str:
    """
    Save query results to reply folder with timestamp

    Args:
        result: Query result dictionary
        bm25_keywords: BM25 keywords
        faiss_query: FAISS query
        question: Question that was asked

    Returns:
        Path to the created output directory
    """
    # Create timestamp-based folder name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
    output_dir = Path("reply") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save response
    if result.get("response"):
        response_file = output_dir / "response.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("QUERY INFORMATION\n")
            f.write("=" * 50 + "\n")
            f.write(f"Repository: {configs()['repo_path']}\n")
            f.write(f"BM25 keywords: {bm25_keywords}\n")
            f.write(f"FAISS query: {faiss_query}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=" * 50 + "\n")
            f.write("RESPONSE\n")
            f.write("=" * 50 + "\n")
            # Convert escaped newlines to actual line breaks
            formatted_response = (
                result["response"].replace("\\n", "\n").replace("\\t", "    ")
            )
            f.write(formatted_response + "\n")
            f.write(f"Estimated Token Count: {result['estimated_token_count']}\n")

    # Save retrieved documents if available
    if result.get("retrieved_documents"):
        docs_file = output_dir / "retrieved_documents.txt"
        with open(docs_file, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("RETRIEVED DOCUMENTS\n")
            f.write("=" * 50 + "\n")
            for i, doc in enumerate(result["retrieved_documents"], 1):
                if isinstance(doc, DualVectorDocument):
                    f.write(
                        f"\n{i}. File: {getattr(doc.original_doc, 'meta_data', {}).get('file_path', 'Unknown')}\n"
                    )
                    f.write(f"Document ID: {doc.original_doc.id}\n")
                    f.write("Full Content:\n")
                    f.write(doc.original_doc.text + "\n")
                    understanding_text = getattr(doc, "understanding_text", "")
                    if understanding_text:
                        f.write("Code Understanding:\n")
                        f.write("---\n")
                        f.write(understanding_text + "\n")
                        f.write("---\n\n")
                    f.write("=" * 80 + "\n")
                else:
                    f.write(
                        f"\n{i}. File: {getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')}\n"
                    )
                    f.write(f"Document ID: {doc.id}\n")
                    f.write("Full Content:\n")
                    f.write(doc.text + "\n")

    if result.get("context"):
        docs_file = output_dir / "context.txt"
        with open(docs_file, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("CONTEXTS\n")
            f.write("=" * 50 + "\n")
            for i, doc in enumerate(result["context"], 1):
                f.write(
                    f"\n{i}. File: {getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')}\n"
                )
                f.write(f"Document ID: {doc.id}\n")
                f.write("Full Content:\n")
                f.write(doc.text + "\n")

    # Save metadata
    metadata_file = output_dir / "metadata.txt"
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write("Query Metadata\n")
        f.write("=" * 30 + "\n")
        f.write(f"Repository: {configs()['repo_path']}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error Message: {result.get('error_msg', 'None')}\n")
        f.write(
            f"Retrieved Documents Count: {len(result.get('retrieved_documents', []))}\n"
        )

        # Count documents with understanding text
        docs_with_understanding = 0
        if result.get("retrieved_documents"):
            for doc in result["retrieved_documents"]:
                understanding_text = getattr(doc, "meta_data", {}).get(
                    "understanding_text", ""
                )
                if understanding_text:
                    docs_with_understanding += 1
        f.write(f"Documents with Code Understanding: {docs_with_understanding}\n")

    if result.get("bm25_docs"):
        bm25_doc_file = output_dir / "bm25_retrieved.txt"
        with open(bm25_doc_file, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("BM25-retrieved docs\n")
            f.write("=" * 50 + "\n")
            for i, doc in enumerate(result["bm25_docs"], 1):
                file_path = (
                    doc.meta_data.get('file_path', 'Unknown')
                    if isinstance(doc, Document)
                    else doc.original_doc.meta_data.get('file_path', 'Unknown')
                )
                f.write(f"\n{i}. File: {file_path}\n")
                f.write(
                    f"Document ID: {doc.id if isinstance(doc, Document) else doc.original_doc.id}\n"
                )
                f.write("Full Content:\n")
                f.write(
                    doc.text
                    if isinstance(doc, Document)
                    else doc.original_doc.text + "\n"
                )

    if result.get("bm25_scores"):
        bm25_score_file = output_dir / "bm25_scores.csv"
        with open(bm25_score_file, "w", encoding="utf-8") as f:
            # Sort by original score (score[0]) in descending order
            if isinstance(list(result["bm25_scores"].values())[0], tuple):
                f.write("doc_id, original_score, minmax_score, zscore_score\n")
                sorted_scores = sorted(
                    result["bm25_scores"].items(), key=lambda x: x[1][0], reverse=True
                )
            else:
                f.write("doc_id, original_score\n")
                sorted_scores = sorted(
                    result["bm25_scores"].items(), key=lambda x: x[1], reverse=True
                )
            for doc_id, score in sorted_scores:
                if isinstance(score, tuple):
                    f.write(f"{doc_id}, {score[0]}, {score[1]}, {score[2]}\n")
                else:
                    f.write(f"{doc_id}, {score}\n")

    if result.get("faiss_scores"):
        faiss_score_file = output_dir / "faiss_scores.csv"
        with open(faiss_score_file, "w", encoding="utf-8") as f:
            f.write("doc_id, original_score, minmax_score, zscore_score\n")
            # Sort by original score (score[0]) in descending order
            sorted_scores = sorted(
                result["faiss_scores"].items(), key=lambda x: x[1][0], reverse=True
            )
            for doc_id, score in sorted_scores:
                f.write(f"{doc_id}, {score[0]}, {score[1]}, {score[2]}\n")

    if result.get("bm25faiss_scores"):
        bm25faiss_score_file = output_dir / "bm25faiss_scores.csv"
        with open(bm25faiss_score_file, "w", encoding="utf-8") as f:
            f.write("doc_id, original_score, minmax_score, zscore_score\n")
            # Sort by original score (score[0]) in descending order
            sorted_scores = sorted(
                result["bm25faiss_scores"].items(), key=lambda x: x[1], reverse=True
            )
            for doc_id, score in sorted_scores:
                f.write(f"{doc_id}, {score}\n")

    if result.get("rrf_scores"):
        rrf_score_file = output_dir / "rrf_scores.csv"
        with open(rrf_score_file, "w", encoding="utf-8") as f:
            f.write("doc_id, rrf_score\n")
            # Sort by original score (score[0]) in descending order
            sorted_scores = sorted(
                result["rrf_scores"].items(), key=lambda x: x[1], reverse=True
            )
            for doc_id, score in sorted_scores:
                f.write(f"{doc_id}, {score}\n")

    if result.get("prompt"):
        prompt_file = output_dir / "prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(result["prompt"])

    return str(output_dir)


def build_context(
    retrieved_doc: Document | DualVectorDocument,
    id2doc: Dict[str, Document],
    direction: str = "both",
    count: int = 0,
) -> Document:
    """
    Build context strings from retrieved documents.

    Args:
        retrieved_documents: List of retrieved Document objects.

    Returns:
        List of context strings.
    """
    if isinstance(retrieved_doc, DualVectorDocument):
        doc = retrieved_doc.original_doc
    else:
        doc = retrieved_doc

    assert direction in ["both", "previous", "next"], f"Invalid direction: {direction}"

    def start_line_num(text):
        return text.split('\n')[0].split(':', 1)[0].strip()

    def end_line_num(text):
        return text.split('\n')[-1].split(':', 1)[0].strip()

    new_doc = deepcopy(doc)
    cnt = 0
    this_doc = doc
    cannot_extend_previous = False
    cannot_extend_next = False
    if direction == "both" or direction == "previous":
        while doc.meta_data["prev_doc_id"] is not None and cnt < count:
            prev_doc = id2doc[doc.meta_data["prev_doc_id"]]
            if end_line_num(prev_doc.text) == start_line_num(new_doc.text):
                new_doc.text = prev_doc.text + new_doc.text.split(':', 1)[1].strip()
            else:
                new_doc.text = prev_doc.text + new_doc.text
            new_doc.meta_data["original_text"] = prev_doc.meta_data["original_text"] + new_doc.meta_data["original_text"]
            doc = prev_doc
            cnt += 1
        if doc.meta_data["prev_doc_id"] is None:
            cannot_extend_previous = True
        doc = this_doc
        cnt = 0
    if direction == "both" or direction == "next":
        while doc.meta_data["next_doc_id"] is not None and cnt < count:
            next_doc = id2doc[doc.meta_data["next_doc_id"]]
            if end_line_num(new_doc.text) == start_line_num(next_doc.text):
                new_doc.text = new_doc.text + next_doc.text.split(':', 1)[1].strip()
            else:
                new_doc.text = new_doc.text + next_doc.text
            new_doc.meta_data["original_text"] = new_doc.meta_data["original_text"] + next_doc.meta_data["original_text"]
            doc = next_doc
            cnt += 1
        if doc.meta_data["next_doc_id"] is None:
            cannot_extend_next = True
        doc = this_doc
        cnt = 0
    if direction == "both":
        cannot_extend = cannot_extend_previous and cannot_extend_next
    elif direction == "previous":
        cannot_extend = cannot_extend_previous
    elif direction == "next":
        cannot_extend = cannot_extend_next
    return new_doc, cannot_extend


def query(question: str) -> str:
    generator = GeneratorWrapper()
    result = generator(input_str=question)
    return result.data.strip()


def query_repository(
    bm25_keywords: str, faiss_query: str, question: str
) -> Dict[str, Any]:
    """
    Query a repository about a question and return structured results.

    Args:
        bm25_keywords: BM25 keywords
        faiss_query: FAISS query
        question: Question to ask about the repository
    """
    rag = RAG()
    result = rag.retrieve(bm25_keywords=bm25_keywords, faiss_query=faiss_query)

    # result[0] is RetrieverOutput
    retriever_output = result[0] if isinstance(result, list) else result
    retrieved_docs = (
        retriever_output.documents if hasattr(retriever_output, "documents") else []
    )

    # Generate answer
    logger.info("🤖 Generating answer...")
    try:
        contexts = []
        for doc in retrieved_docs:
            contexts.append(build_context(doc, rag.id2doc)[0])
        return rag.query(input_str=question, contexts=contexts)
    except Exception as e:
        logger.error(f"Error calling generator: {e}")
        raise


def print_result(
    result: Dict[str, Any], print_retrieved_documents: bool = False
) -> None:
    if result.get("response"):
        print("\n" + "=" * 50)
        print("RESPONSE:")
        print("=" * 50)
        # Convert escaped newlines to actual line breaks for better readability
        formatted_response = (
            result["response"].replace("\\n", "\n").replace("\\t", "    ")
        )
        print(formatted_response)

    if result["retrieved_documents"] and print_retrieved_documents:
        print("\n" + "=" * 50)
        print("RETRIEVED DOCUMENTS:")
        print("=" * 50)
        for i, doc in enumerate(result["retrieved_documents"], 1):
            if isinstance(doc, DualVectorDocument):
                print(
                    f"\n{i}. File: {getattr(doc.original_doc, 'meta_data', {}).get('file_path', 'Unknown')}"
                )
                print(
                    f"   Preview: {(doc.original_doc.text[:200] + '...') if len(doc.original_doc.text) > 200 else doc.original_doc.text}"
                )
            else:
                print(
                    f"\n{i}. File: {getattr(doc, 'meta_data', {}).get('file_path', 'Unknown')}"
                )
                print(
                    f"   Preview: {(doc.text[:200] + '...') if len(doc.text) > 200 else doc.text}"
                )
    print("For more details, please check the reply folder")


def main(cfg: DictConfig) -> None:
    assert cfg.repo_path, "repo_path must be set"
    set_global_configs(cfg)
    try:
        result = query_repository(repo_path=cfg.repo_path, question=cfg.question)

        if result is None:
            logger.warning("❌ No question provided, only embedding the repository")
            return

        # Save results to reply folder
        output_dir = save_query_results(result, cfg.repo_path, cfg.question)

        print_result(result)

        # Inform user about saved files
        print(f"\n📁 Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def hydra_wrapped_query_repository(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_wrapped_query_repository()
