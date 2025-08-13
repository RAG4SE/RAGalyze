"""Dynamic splitter transformer that selects appropriate splitter based on document type."""

from typing import List, Union, Any

from adalflow.core.types import Document
from adalflow.core.component import Component

from RAGalyze.rag.splitter_factory import get_splitter_factory
from RAGalyze.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class DynamicSplitterTransformer(Component):
    """Transformer that dynamically selects appropriate splitter based on document type."""
    
    def __init__(self):
        """Initialize the dynamic splitter transformer."""
        super().__init__()
        self.splitter_factory = get_splitter_factory()
        logger.info("Initialized DynamicSplitterTransformer")
    
    def call(self, documents: List[Document]) -> List[Document]:
        """Process documents with appropriate splitters.
        
        Args:
            documents (List[Document]): Input documents
            
        Returns:
            List[Document]: Split documents
        """
        if not documents:
            return []
        
        result_documents = []
        
        for doc in documents:
            try:
                # Get appropriate splitter for this document
                splitter = self.splitter_factory.get_splitter(
                    content=doc.text,
                    file_path=getattr(doc, 'meta_data', {}).get('file_path', '')
                )
                
                # Split the document
                split_docs = splitter.call([doc])
                result_documents.extend(split_docs)
                
                # Log splitting info
                file_path = getattr(doc, 'meta_data', {}).get('file_path', 'unknown')
                doc_type = self.splitter_factory.detect_content_type(
                    doc.text, file_path
                )
                logger.debug(
                    f"Split {file_path} (type: {doc_type}) into {len(split_docs)} chunks"
                )
                
            except Exception as e:
                logger.error(f"Error splitting document {getattr(doc, 'meta_data', {}).get('file_path', 'unknown')}: {e}")
                raise
        
        logger.info(f"Processed {len(documents)} documents into {len(result_documents)} chunks")
        return result_documents
    
    def __call__(self, documents: List[Document]) -> List[Document]:
        """Make the transformer callable.
        
        Args:
            documents (List[Document]): Input documents
            
        Returns:
            List[Document]: Split documents
        """
        return self.call(documents)