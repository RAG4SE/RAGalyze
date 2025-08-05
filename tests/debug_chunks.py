#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.smart_text_splitter import SmartTextSplitter
from logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)

def debug_chunk_boundaries():
    """Debug chunk boundaries to see why test is failing."""
    
    splitter = SmartTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        split_by='token',
        content_type='text'
    )
    
    test_text = (
        "Natural language processing is a fascinating field of artificial intelligence. "
        "It involves teaching computers to understand and generate human language. "
        "Machine learning algorithms are used to analyze patterns in text data. "
        "Deep learning models like transformers have revolutionized NLP tasks. "
        "Text preprocessing is crucial for good results in NLP applications. "
        "Tokenization breaks text into smaller units for processing. "
        "Named entity recognition identifies important entities in text. "
        "Sentiment analysis determines the emotional tone of text. "
        "Question answering systems can extract information from documents. "
        "Language models can generate coherent and contextually relevant text."
    )
    
    print(f"Original text ({len(test_text)} chars):")
    print(repr(test_text))
    print()
    
    chunks = splitter.split_text(test_text)
    
    print(f"Generated {len(chunks)} chunks:")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.text if hasattr(chunk, 'text') else str(chunk)
        
        print(f"=== Chunk {i} ===")
        print(f"Length: {len(content)} characters")
        print(f"Content: {repr(content)}")
        print(f"First char: {repr(content[0]) if content else 'N/A'}")
        print(f"Last char: {repr(content[-1]) if content else 'N/A'}")
        print(f"Starts with capital/digit: {content[0].isupper() or content[0].isdigit() if content else False}")
        print(f"Ends with punctuation: {content[-1] in '.!?' if content else False}")
        print()

if __name__ == "__main__":
    debug_chunk_boundaries()