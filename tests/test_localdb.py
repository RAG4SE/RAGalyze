#!/usr/bin/env python3
"""
Test file for LocalDB functionality with data transformers.

This test demonstrates:
1. Creating sample documents
2. Building custom data transformers
3. Saving transformed data to LocalDB
4. Loading and verifying the saved data
"""

import os
import tempfile
import pickle
from typing import List
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.db import LocalDB


class SimpleTextTransformer(adal.Component):
    """Simple transformer that converts text to uppercase."""

    def __init__(self):
        super().__init__()

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Transform documents by converting text to uppercase."""
        transformed_docs = []
        for doc in documents:
            transformed_text = doc.text.upper()
            transformed_meta = doc.meta_data.copy()
            transformed_meta["transformed_by"] = "SimpleTextTransformer"
            transformed_meta["original_length"] = len(doc.text)

            transformed_doc = Document(
                text=transformed_text,
                meta_data=transformed_meta
            )
            transformed_docs.append(transformed_doc)
        return transformed_docs


class WordCountTransformer(adal.Component):
    """Transformer that adds word count and extracts first 3 words."""

    def __init__(self):
        super().__init__()

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Transform documents by adding word count and extracting first 3 words."""
        transformed_docs = []
        for doc in documents:
            words = doc.text.split()
            word_count = len(words)
            first_3_words = ' '.join(words[:3]) if words else ""

            transformed_meta = doc.meta_data.copy()
            transformed_meta["transformed_by"] = "WordCountTransformer"
            transformed_meta["word_count"] = word_count
            transformed_meta["first_3_words"] = first_3_words

            transformed_doc = Document(
                text=f"[{word_count} words] {first_3_words}...",
                meta_data=transformed_meta
            )
            transformed_docs.append(transformed_doc)
        return transformed_docs


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    sample_texts = [
        "This is the first document. It contains multiple sentences and should demonstrate the transformer capabilities.",
        "The second document is shorter but still meaningful for testing purposes.",
        "Third document with some technical terms like API, JSON, and database operations.",
        "Final document to ensure we have a good variety of content for our test suite."
    ]

    documents = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            text=text,
            meta_data={
                "doc_id": f"doc_{i+1}",
                "author": f"author_{i+1}",
                "category": "test",
                "created_date": "2024-01-01"
            }
        )
        documents.append(doc)

    return documents


def test_simple_transformer():
    """Test the SimpleTextTransformer."""
    print("=== Testing SimpleTextTransformer ===")

    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")

    # Create transformer
    transformer = SimpleTextTransformer()

    # Transform documents
    transformed_docs = transformer(documents)

    # Print results
    for i, doc in enumerate(transformed_docs):
        print(f"Document {i+1}:")
        print(f"  Original: {documents[i].text[:50]}...")
        print(f"  Transformed: {doc.text[:50]}...")
        print(f"  Metadata: {doc.meta_data}")
        print()

    return transformed_docs


def test_word_count_transformer():
    """Test the WordCountTransformer."""
    print("=== Testing WordCountTransformer ===")

    # Create sample documents
    documents = create_sample_documents()

    # Create transformer
    transformer = WordCountTransformer()

    # Transform documents
    transformed_docs = transformer(documents)

    # Print results
    for i, doc in enumerate(transformed_docs):
        print(f"Document {i+1}:")
        print(f"  Original: {documents[i].text}")
        print(f"  Transformed: {doc.text}")
        print(f"  Metadata: {doc.meta_data}")
        print()

    return transformed_docs


def test_localdb_with_single_transformer():
    """Test LocalDB with a single transformer."""
    print("=== Testing LocalDB with SimpleTextTransformer ===")

    # Create temporary database file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
        db_path = temp_file.name

    try:
        # Create sample documents
        documents = create_sample_documents()

        # Create transformer
        transformer = SimpleTextTransformer()

        # Create and setup LocalDB
        db = LocalDB()
        db.register_transformer(transformer=transformer, key="simple_transform")
        db.load(documents)
        db.transform(key="simple_transform")

        # Save database
        db.save_state(filepath=db_path)
        print(f"Database saved to: {db_path}")

        # Load database from file
        loaded_db = LocalDB.load_state(db_path)
        print("Database loaded successfully")

        # Get transformed data
        transformed_data = loaded_db.get_transformed_data(key="simple_transform")
        print(f"Retrieved {len(transformed_data)} transformed documents")

        # Verify data integrity
        for i, doc in enumerate(transformed_data):
            original_doc = documents[i]
            print(f"Document {i+1} verification:")
            print(f"  Original: {original_doc.text[:30]}...")
            print(f"  Transformed: {doc.text[:30]}...")
            print(f"  Transform metadata: {doc.meta_data.get('transformed_by')}")
            print(f"  Length preserved: {doc.meta_data.get('original_length') == len(original_doc.text)}")
            print()

        return transformed_data

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_localdb_with_sequential_transformer():
    """Test LocalDB with a sequential transformer (multiple steps)."""
    print("=== Testing LocalDB with Sequential Transformer ===")

    # Create temporary database file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
        db_path = temp_file.name

    try:
        # Create sample documents
        documents = create_sample_documents()

        # Create sequential transformer (first uppercase, then word count)
        sequential_transformer = adal.Sequential(
            SimpleTextTransformer(),
            WordCountTransformer()
        )

        # Create and setup LocalDB
        db = LocalDB()
        db.register_transformer(transformer=sequential_transformer, key="sequential_transform")
        db.load(documents)
        db.transform(key="sequential_transform")

        # Save database
        db.save_state(filepath=db_path)
        print(f"Sequential database saved to: {db_path}")

        # Load database from file
        loaded_db = LocalDB.load_state(db_path)
        print("Sequential database loaded successfully")

        # Get transformed data
        transformed_data = loaded_db.get_transformed_data(key="sequential_transform")
        print(f"Retrieved {len(transformed_data)} sequentially transformed documents")

        # Verify data integrity
        for i, doc in enumerate(transformed_data):
            original_doc = documents[i]
            print(f"Document {i+1} sequential verification:")
            print(f"  Original: {original_doc.text[:30]}...")
            print(f"  Final transformed: {doc.text}")
            print(f"  Transform chain: {doc.meta_data.get('transformed_by')}")
            print(f"  Word count: {doc.meta_data.get('word_count')}")
            print()

        return transformed_data

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_multiple_transformers_same_db():
    """Test LocalDB with multiple transformers registered to the same database."""
    print("=== Testing LocalDB with Multiple Transformers ===")

    # Create temporary database file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
        db_path = temp_file.name

    try:
        # Create sample documents
        documents = create_sample_documents()

        # Create transformers
        simple_transformer = SimpleTextTransformer()
        word_count_transformer = WordCountTransformer()

        # Create and setup LocalDB with multiple transformers
        db = LocalDB()
        db.register_transformer(transformer=simple_transformer, key="simple")
        db.register_transformer(transformer=word_count_transformer, key="wordcount")
        db.load(documents)

        # Transform with both transformers
        db.transform(key="simple")
        db.transform(key="wordcount")

        # Save database
        db.save_state(filepath=db_path)
        print(f"Multi-transformer database saved to: {db_path}")

        # Load database from file
        loaded_db = LocalDB.load_state(db_path)
        print("Multi-transformer database loaded successfully")

        # Get transformed data from both transformers
        simple_data = loaded_db.get_transformed_data(key="simple")
        wordcount_data = loaded_db.get_transformed_data(key="wordcount")

        print(f"Retrieved {len(simple_data)} simple-transformed documents")
        print(f"Retrieved {len(wordcount_data)} wordcount-transformed documents")

        # Compare results
        for i, (simple_doc, wordcount_doc) in enumerate(zip(simple_data, wordcount_data)):
            original_doc = documents[i]
            print(f"Document {i+1} comparison:")
            print(f"  Original: {original_doc.text[:30]}...")
            print(f"  Simple transform: {simple_doc.text[:30]}...")
            print(f"  Wordcount transform: {wordcount_doc.text}")
            print()

        return simple_data, wordcount_data

    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all tests."""
    # print("Starting LocalDB functionality tests...\n")

    # # Test individual transformers
    # print("1. Testing individual transformers:")
    # test_simple_transformer()
    # test_word_count_transformer()

    # print("\n2. Testing LocalDB integration:")
    # test_localdb_with_single_transformer()
    # test_localdb_with_sequential_transformer()
    test_multiple_transformers_same_db()

    print("All tests completed successfully!")


if __name__ == "__main__":
    main()