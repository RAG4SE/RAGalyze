import logging
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import torch
import json
from pathlib import Path
from dataclasses import dataclass

import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.db import LocalDB
from adalflow.components.retriever.faiss_retriever import FAISSRetriever

from api.tools.embedder import get_embedder
from api.config import configs

logger = logging.getLogger(__name__)

@dataclass
class DualVectorDocument:
    """Document containing both code and understanding vectors"""
    original_doc: Document
    code_embedding: List[float]
    understanding_embedding: List[float]
    understanding_text: str
    file_path: str
    chunk_id: str

class CodeUnderstandingGenerator:
    """Generate functional descriptions for code snippets using local model"""
    
    def __init__(self, model_name: str = "qwen2.5:7b", ollama_host: str = "http://localhost:11434"):
        """
        Initialize code understanding generator
        
        Args:
            model_name: Local model name to use, defaults to qwen2.5:7b
            ollama_host: Ollama service address
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self._setup_ollama_client()
    
    def _setup_ollama_client(self):
        """Set up Ollama client"""
        try:
            from adalflow import OllamaClient
            self.client = OllamaClient(host=self.ollama_host)
            logger.info(f"Connected to Ollama service: {self.ollama_host}, model: {self.model_name}")
        except Exception as e:
            logger.error(f"Unable to connect to Ollama service: {e}")
            raise
    
    def generate_code_understanding(self, code_content: str, file_path: str = "") -> str:
        """
        Generate functional description for code snippet
        
        Args:
            code_content: Code content
            file_path: File path (for context)
            
        Returns:
            Natural language description of code functionality
        """
        # Detect programming language
        language = self._detect_language(file_path, code_content)
        
        # Construct prompt
        prompt = self._create_understanding_prompt(code_content, language, file_path)
        
        try:
            # Call model to generate understanding
            response = self.client.call(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            if hasattr(response, 'message'):
                understanding = response.message.content.strip()
            elif hasattr(response, 'content'):
                understanding = response.content.strip()
            else:
                understanding = str(response).strip()
            
            # Clean and validate output
            understanding = self._clean_understanding_text(understanding)
            
            logger.debug(f"Generated code understanding for {file_path}: {understanding[:100]}...")
            return understanding
            
        except Exception as e:
            logger.error(f"Error generating code understanding: {e}")
            # Return default description
            return self._generate_fallback_understanding(code_content, language, file_path)
    
    def _detect_language(self, file_path: str, code_content: str) -> str:
        """Detect programming language"""
        if not file_path:
            return "unknown"
        
        ext = Path(file_path).suffix.lower()
        language_map = {
            # Popular languages
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.ts': 'TypeScript',
            '.jsx': 'React JSX',
            '.tsx': 'React TSX',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C Header',
            '.hpp': 'C++ Header',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.sol': 'Solidity',
            '.cs': 'C#',
            # Functional languages
            '.ml': 'OCaml',
            '.mli': 'OCaml Interface',
            '.hs': 'Haskell',
            '.lhs': 'Literate Haskell',
            '.fs': 'F#',
            '.fsi': 'F# Interface',
            '.fsx': 'F# Script',
            '.erl': 'Erlang',
            '.hrl': 'Erlang Header',
            '.ex': 'Elixir',
            '.exs': 'Elixir Script',
            '.clj': 'Clojure',
            '.cljs': 'ClojureScript',
            '.cljc': 'Clojure Common',
            '.scm': 'Scheme',
            '.ss': 'Scheme',
            '.lisp': 'Common Lisp',
            '.lsp': 'Lisp',
            '.elm': 'Elm',
            # Other languages
            '.scala': 'Scala',
            '.sc': 'Scala',
            '.kt': 'Kotlin',
            '.kts': 'Kotlin Script',
            '.dart': 'Dart',
            '.lua': 'Lua',
            '.r': 'R',
            '.R': 'R',
            '.m': 'MATLAB/Objective-C',
            '.jl': 'Julia',
            '.nim': 'Nim',
            '.cr': 'Crystal',
            '.zig': 'Zig',
            '.pl': 'Perl',
            '.pm': 'Perl Module',
            # Hardware description
            '.v': 'Verilog',
            '.sv': 'SystemVerilog',
            '.vhd': 'VHDL',
            '.vhdl': 'VHDL',
            # Assembly
            '.s': 'Assembly',
            '.S': 'Assembly',
            '.asm': 'Assembly',
            # Web and markup
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.less': 'Less',
            '.vue': 'Vue.js',
            '.xml': 'XML',
            '.xaml': 'XAML',
            # Shell scripts
            '.sh': 'Shell Script',
            '.bash': 'Bash Script',
            '.zsh': 'Zsh Script',
            '.fish': 'Fish Script',
            '.ps1': 'PowerShell',
            '.psm1': 'PowerShell Module',
            '.bat': 'Batch Script',
            '.cmd': 'Command Script',
            # Data and config
            '.sql': 'SQL',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Config',
            '.conf': 'Config',
            '.proto': 'Protocol Buffers',
            '.graphql': 'GraphQL',
            '.gql': 'GraphQL',
            # Build systems
            '.mk': 'Makefile',
            '.cmake': 'CMake',
            '.gradle': 'Gradle',
            '.sbt': 'SBT',
            '.bazel': 'Bazel',
            '.bzl': 'Bazel',
        }
        
        return language_map.get(ext, 'unknown')
    
    def _create_understanding_prompt(self, code_content: str, language: str, file_path: str) -> str:
        """Create prompt for generating code understanding"""
        prompt = f"""Please analyze the following {language} code snippet and provide a concise English description of its main functionality and purpose.

File path: {file_path}
Code content:
```{language.lower()}
{code_content}
```

Please provide a concise, accurate functional description (1-3 sentences), focusing on:
1. The main functionality of this code
2. Key logic or algorithms implemented
3. Its role in the project

Description requirements:
- Use English
- Be concise and clear, suitable for search matching
- Focus on functionality rather than implementation details
- Avoid overly technical terminology

Functional description:"""
        
        return prompt
    
    def _clean_understanding_text(self, text: str) -> str:
        """Clean and standardize understanding text"""
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove possible markdown formatting
        text = re.sub(r'```[\w]*', '', text)
        text = re.sub(r'```', '', text)
        
        # Remove quotes
        text = text.strip('"\'')
        
        # Limit length
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text
    
    def _generate_fallback_understanding(self, code_content: str, language: str, file_path: str) -> str:
        """Generate fallback code understanding (when model call fails)"""
        # Generate simple description based on code content and file path
        understanding_parts = []
        
        # Generate description based on filename
        if file_path:
            filename = Path(file_path).stem
            understanding_parts.append(f"{filename} module")
        
        # Based on language type
        if language != "unknown":
            understanding_parts.append(f"{language} code")
        
        # Based on code length
        lines = len(code_content.strip().split('\n'))
        if lines > 50:
            understanding_parts.append("complex functionality implementation")
        elif lines > 20:
            understanding_parts.append("moderate functionality implementation")
        else:
            understanding_parts.append("simple functionality implementation")
        
        # Based on code patterns
        if 'class ' in code_content:
            understanding_parts.append("with class definitions")
        if 'def ' in code_content or 'function' in code_content:
            understanding_parts.append("with function definitions")
        if 'import ' in code_content or '#include' in code_content:
            understanding_parts.append("with external dependencies")
        
        understanding = " ".join(understanding_parts)
        return understanding if understanding else f"Code file: {file_path}"

class DualVectorEmbedder:
    """Dual vector embedder: generates both code vectors and understanding vectors"""
    
    def __init__(self, is_huggingface_embedder: bool = True):
        """
        Initialize dual vector embedder
        
        Args:
            is_huggingface_embedder: Whether to use HuggingFace embedder
        """
        self.embedder = get_embedder(is_huggingface_embedder=is_huggingface_embedder)
        self.code_generator = CodeUnderstandingGenerator()
        logger.info("Dual vector embedder initialization completed")
    
    def embed_documents(self, documents: List[Document]) -> List[DualVectorDocument]:
        """
        Generate dual vector embeddings for document list
        
        Args:
            documents: Original document list
            
        Returns:
            Document list containing dual vectors
        """
        dual_docs = []
        
        logger.info(f"Starting dual vector embedding processing for {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            try:
                # Generate code understanding
                file_path = doc.meta_data.get('file_path', '')
                understanding = self.code_generator.generate_code_understanding(
                    doc.text, file_path
                )
                
                # Generate code vector
                code_result = self.embedder.call(doc.text)
                if code_result.error:
                    logger.error(f"Code embedding failed: {code_result.error}")
                    continue
                
                code_embedding = code_result.data[0].embedding if code_result.data else []
                
                # Generate understanding vector
                understanding_result = self.embedder.call(understanding)
                if understanding_result.error:
                    logger.error(f"Understanding embedding failed: {understanding_result.error}")
                    continue
                
                understanding_embedding = understanding_result.data[0].embedding if understanding_result.data else []
                
                # Create dual vector document
                dual_doc = DualVectorDocument(
                    original_doc=doc,
                    code_embedding=code_embedding,
                    understanding_embedding=understanding_embedding,
                    understanding_text=understanding,
                    file_path=file_path,
                    chunk_id=f"{file_path}_{i}"
                )
                
                dual_docs.append(dual_doc)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                continue
        
        logger.info(f"Dual vector embedding completed, successfully processed {len(dual_docs)} documents")
        return dual_docs

class DualVectorRetriever:
    """Dual vector retriever: supports retrieval in both code vectors and understanding vectors"""
    
    def __init__(self, dual_docs: List[DualVectorDocument], embedder, top_k: int = 20):
        """
        Initialize dual vector retriever
        
        Args:
            dual_docs: List of dual vector documents
            embedder: Embedder
            top_k: Number of most relevant documents to return
        """
        self.dual_docs = dual_docs
        self.embedder = embedder
        self.top_k = top_k
        
        # Build two FAISS indices
        self._build_indices()
        logger.info(f"Dual vector retriever initialization completed, containing {len(dual_docs)} documents")
    
    def _build_indices(self):
        """Build code index and understanding index"""
        if not self.dual_docs:
            logger.warning("No documents available for building indices")
            return
        
        # Build code index
        code_docs = []
        for dual_doc in self.dual_docs:
            # Create document for FAISS
            faiss_doc = Document(
                text=dual_doc.original_doc.text,
                meta_data=dual_doc.original_doc.meta_data,
                id=dual_doc.chunk_id,
                vector=dual_doc.code_embedding
            )
            code_docs.append(faiss_doc)
        
        def safe_document_map_func(doc):
            """Safely extract embedding vector from document, handling both Embedding objects and lists"""
            if hasattr(doc, 'vector') and doc.vector is not None:
                if hasattr(doc.vector, 'embedding'):
                    # If it's an Embedding object, extract the embedding attribute
                    return doc.vector.embedding
                else:
                    # If it's already a list or array, return as-is
                    return doc.vector
            return []
            
        self.code_retriever = FAISSRetriever(
            embedder=self.embedder,
            documents=code_docs,
            top_k=self.top_k,
            document_map_func=safe_document_map_func
        )
        
        # Build understanding index
        understanding_docs = []
        for dual_doc in self.dual_docs:
            # Create document for FAISS, using understanding text as content
            faiss_doc = Document(
                text=dual_doc.understanding_text,
                meta_data={
                    **dual_doc.original_doc.meta_data,
                    'original_text': dual_doc.original_doc.text,
                    'understanding_text': dual_doc.understanding_text
                },
                id=dual_doc.chunk_id,
                vector=dual_doc.understanding_embedding
            )
            understanding_docs.append(faiss_doc)
        
        self.understanding_retriever = FAISSRetriever(
            embedder=self.embedder,
            documents=understanding_docs,
            top_k=self.top_k,
            document_map_func=safe_document_map_func
        )
        
        logger.info("Code index and understanding index construction completed")
    
    def retrieve(self, query: str, mode: str = "hybrid", 
                code_weight: float = 0.4, understanding_weight: float = 0.6) -> List[Document]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query text
            mode: Retrieval mode ("code", "understanding", "hybrid")
            code_weight: Weight for code retrieval (only used in hybrid mode)
            understanding_weight: Weight for understanding retrieval (only used in hybrid mode)
            
        Returns:
            List of relevant documents
        """
        try:
            if mode == "code":
                # Use only code index for retrieval
                result = self.code_retriever.call(query)
                return result.documents if hasattr(result, 'documents') else []
            
            elif mode == "understanding":
                # Use only understanding index for retrieval
                result = self.understanding_retriever.call(query)
                return result.documents if hasattr(result, 'documents') else []
            
            elif mode == "hybrid":
                # Hybrid retrieval: combine code and understanding retrieval results
                code_result = self.code_retriever.call(query)
                understanding_result = self.understanding_retriever.call(query)
                
                code_docs = code_result.documents if hasattr(code_result, 'documents') else []
                understanding_docs = understanding_result.documents if hasattr(understanding_result, 'documents') else []
                
                # Merge and rerank results
                return self._merge_results(code_docs, understanding_docs, 
                                         code_weight, understanding_weight)
            
            else:
                raise ValueError(f"Unsupported retrieval mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _merge_results(self, code_docs: List[Document], understanding_docs: List[Document],
                      code_weight: float, understanding_weight: float) -> List[Document]:
        """
        Merge results from code retrieval and understanding retrieval
        
        Args:
            code_docs: Code retrieval results
            understanding_docs: Understanding retrieval results
            code_weight: Code retrieval weight
            understanding_weight: Understanding retrieval weight
            
        Returns:
            Merged document list
        """
        # Create scoring dictionary
        doc_scores = {}
        
        # Process code retrieval results
        for i, doc in enumerate(code_docs):
            doc_id = getattr(doc, 'id', f"code_{i}")
            # Calculate score based on ranking (higher rank = higher score)
            score = code_weight * (len(code_docs) - i) / len(code_docs)
            doc_scores[doc_id] = {
                'doc': doc,
                'score': score,
                'source': 'code'
            }
        
        # Process understanding retrieval results
        for i, doc in enumerate(understanding_docs):
            doc_id = getattr(doc, 'id', f"understanding_{i}")
            score = understanding_weight * (len(understanding_docs) - i) / len(understanding_docs)
            
            if doc_id in doc_scores:
                # If document already exists, accumulate score
                doc_scores[doc_id]['score'] += score
                doc_scores[doc_id]['source'] = 'hybrid'
            else:
                # New document, but need to use original text
                original_text = doc.meta_data.get('original_text', doc.text)
                merged_doc = Document(
                    text=original_text,
                    meta_data=doc.meta_data,
                    id=doc_id
                )
                doc_scores[doc_id] = {
                    'doc': merged_doc,
                    'score': score,
                    'source': 'understanding'
                }
        
        # Sort by score and return documents
        sorted_items = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        result_docs = [item['doc'] for item in sorted_items[:self.top_k]]
        
        logger.debug(f"Merged retrieval results: code {len(code_docs)}, understanding {len(understanding_docs)}, final {len(result_docs)}")
        return result_docs

class DualVectorPipeline:
    """Dual vector processing pipeline: integrates all functionality"""
    
    def __init__(self, db_path: str = None, is_huggingface_embedder: bool = True):
        """
        Initialize dual vector processing pipeline
        
        Args:
            db_path: Database path
            is_huggingface_embedder: Whether to use HuggingFace embedder
        """
        self.db_path = db_path or os.path.expanduser("~/.adalflow/databases/dual_vector.db")
        self.is_huggingface_embedder = is_huggingface_embedder
        self.embedder = get_embedder(is_huggingface_embedder=is_huggingface_embedder)
        self.dual_docs = []
        self.retriever = None
        
        logger.info(f"Dual vector processing pipeline initialization completed, database path: {self.db_path}")
    
    def process_documents(self, documents: List[Document]) -> DualVectorRetriever:
        """
        Process documents and create dual vector retriever
        
        Args:
            documents: Original document list
            
        Returns:
            Dual vector retriever
        """
        logger.info(f"Starting to process {len(documents)} documents")
        
        # Create dual vector embedder
        dual_embedder = DualVectorEmbedder(self.is_huggingface_embedder)
        
        # Generate dual vector documents
        self.dual_docs = dual_embedder.embed_documents(documents)
        
        # Create retriever
        top_k = configs.get("retriever", {}).get("top_k", 20)
        self.retriever = DualVectorRetriever(self.dual_docs, self.embedder, top_k)
        
        # Save to database
        self._save_to_database()
        
        logger.info("Dual vector processing pipeline completed")
        return self.retriever
    
    def _save_to_database(self):
        """Save dual vector data to database"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Prepare data to save
            save_data = {
                'dual_docs': [],
                'metadata': {
                    'total_docs': len(self.dual_docs),
                    'embedder_config': configs.get("embedder", {}),
                    'created_at': str(torch.datetime.datetime.now()) if hasattr(torch, 'datetime') else str(None)
                }
            }
            
            # Serialize dual vector documents
            for dual_doc in self.dual_docs:
                doc_data = {
                    'original_text': dual_doc.original_doc.text,
                    'meta_data': dual_doc.original_doc.meta_data,
                    'code_embedding': dual_doc.code_embedding,
                    'understanding_embedding': dual_doc.understanding_embedding,
                    'understanding_text': dual_doc.understanding_text,
                    'file_path': dual_doc.file_path,
                    'chunk_id': dual_doc.chunk_id
                }
                save_data['dual_docs'].append(doc_data)
            
            # Save to JSON file
            json_path = self.db_path.replace('.db', '_dual_vector.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Dual vector data saved to: {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving dual vector data: {e}")
    
    def load_from_database(self) -> Optional[DualVectorRetriever]:
        """Load dual vector data from database"""
        try:
            json_path = self.db_path.replace('.db', '_dual_vector.json')
            
            if not os.path.exists(json_path):
                logger.info("Dual vector data file not found")
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # Rebuild dual vector documents
            self.dual_docs = []
            for doc_data in save_data['dual_docs']:
                original_doc = Document(
                    text=doc_data['original_text'],
                    meta_data=doc_data['meta_data']
                )
                
                dual_doc = DualVectorDocument(
                    original_doc=original_doc,
                    code_embedding=doc_data['code_embedding'],
                    understanding_embedding=doc_data['understanding_embedding'],
                    understanding_text=doc_data['understanding_text'],
                    file_path=doc_data['file_path'],
                    chunk_id=doc_data['chunk_id']
                )
                self.dual_docs.append(dual_doc)
            
            # Create retriever
            top_k = configs.get("retriever", {}).get("top_k", 20)
            self.retriever = DualVectorRetriever(self.dual_docs, self.embedder, top_k)
            
            logger.info(f"Loaded {len(self.dual_docs)} dual vector documents from database")
            return self.retriever
            
        except Exception as e:
            logger.error(f"Error loading dual vector data from database: {e}")
            return None 