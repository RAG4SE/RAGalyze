"""
This file implements various retrieval strategies including single vector, dual vector, 
hybrid BM25+FAISS, and query-driven retrievers for semantic document search in the RAG system.
"""

import re
from typing import List, Optional, Union, Callable
from rank_bm25 import BM25Okapi
from collections import defaultdict
from copy import deepcopy
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np

from adalflow.core.types import RetrieverOutput, Document, RetrieverOutputType
from adalflow.components.retriever.faiss_retriever import (
    FAISSRetriever,
    FAISSRetrieverQueriesType,
)

from ragalyze.configs import get_embedder, configs
from ragalyze.core.types import DualVectorDocument
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.core.utils import minmax_norm, zscore_norm
from ragalyze.rag.treesitter_parse_interface import tokenize_for_bm25, set_debug_mode

logger = get_tqdm_compatible_logger(__name__)


class BM25Retriever:
    """Standalone BM25 retriever that can be used by other retrievers."""

    def __init__(self, documents: List[Union[Document, DualVectorDocument]], 
                 k1: float = 1.5, b: float = 0.75, top_k: int = 20, 
                 use_multithreading: bool = True, max_workers: Optional[int] = None):
        """
        Initialize the BM25 retriever.

        Args:
            documents: List of documents to index
            k1: BM25 k1 parameter
            b: BM25 b parameter
            top_k: Number of top documents to retrieve
            use_multithreading: Whether to use multithreading for tokenization
            max_workers: Maximum number of worker threads (None for CPU count)
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        assert top_k > 0, "Top k must be greater than 0"
        self.top_k = top_k
        self.use_multithreading = use_multithreading
        self.max_workers = max_workers
        self.bm25 = None
        self._initialize_bm25(documents)

    def _initialize_bm25(self, documents: List[Union[Document, DualVectorDocument]]):
        """Initialize BM25 index with document texts."""
        # for doc in documents:
        #     print(doc.meta_data)
        #     if not doc.meta_data["programming_language"]:
        #         import sys; sys.exit(1)
        try:
            # Extract text content from documents for BM25 indexing
            corpus = []
            import time
            if self.use_multithreading and len(documents) > 1:
                # Determine number of worker threads
                if self.max_workers is not None:
                    num_workers = self.max_workers
                else:
                    num_workers = min(multiprocessing.cpu_count(), len(documents))
                # start_time = time.time()
                # Use ThreadPoolExecutor to parallelize tokenization
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tokenization tasks
                    future_to_index = {}
                    for i, doc in enumerate(documents):
                        if isinstance(doc, DualVectorDocument):
                            # For dual vector documents, combine code and understanding text
                            text = doc.original_doc.text + "\n" + doc.understanding_text
                        else:
                            text = doc.text
                        assert doc.meta_data["programming_language"], "Programming language is required for BM25 tokenization"
                        future_to_index[executor.submit(self.build_bm25_index, text, doc.meta_data["programming_language"])] = i

                    # Collect results in the correct order
                    results = [None] * len(documents)
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            tokens = future.result()
                            results[index] = tokens
                        except Exception as e:
                            logger.error(f"Error processing document {index}: {e}")
                            raise
                    
                    corpus = results
                # end_time = time.time()
                # print('time taken for bm25 multithreading:', end_time - start_time)
            else:
                # Process documents sequentially (single-threaded)
                # start_time = time.time()
                for doc in documents:
                    if isinstance(doc, DualVectorDocument):
                        # For dual vector documents, combine code and understanding text
                        text = doc.original_doc.text + "\n" + doc.understanding_text
                    else:
                        text = doc.text

                    # Tokenize text for BM25 (simple whitespace + punctuation splitting)
                    assert doc.meta_data["programming_language"], "Programming language is required for BM25 tokenization"
                    tokens = self.build_bm25_index(text, doc.meta_data["programming_language"])
                    print('tokens:' ,tokens)
                    corpus.append(tokens)
                # end_time = time.time()
                # print('time taken for bm25 single threading:', end_time - start_time)            
            # Initialize BM25 with custom parameters
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            raise

    def build_bm25_index(self, code: str, language: str) -> List[str]:
        set_debug_mode(0)
        return tokenize_for_bm25(code, language)

    def _build_bm25_index_python(self, code: str) -> List[str]:
        """纯Python实现的BM25索引构建"""
        # 跨语言的函数定义模式
        function_definition_patterns = [
            r'def\s+(\w+)',           # Python: def function_name
            r'function\s+(\w+)',      # JavaScript: function name
            r'(?:public|private|protected)?\s*(?:static)?\s*(?:\w+(?:\s*\*)*\s+)?(\w+)\s*\([^)]*\)\s*{',  # C/C++/Java: return_type function_name(...)
            r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*;',   # C++/Java prototype
            r'(?:public|private|protected)?\s*(?:static)?\s*(?:\w+(?:\s*\*)*\s+)?(\w+)\s*\([^)]*\)\s*throws',  # Java with throws
            r'(?:\w+\s+)+(\w+)\s*\([^)]*\)(?:\s*const)?\s*{',  # C++: return_type function_name(...) const {
        ]
        
        # 类定义模式
        class_definition_patterns = [
            r'class\s+(\w+)',         # Python/Java/C++: class ClassName
        ]
        
        # 常见的关键字，不应被识别为函数调用
        # C/C++ keywords
        cpp_keywords = {
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 
            'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t', 'class', 'compl', 'concept', 
            'const', 'consteval', 'constexpr', 'constinit', 'const_cast', 'continue', 'co_await', 
            'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 
            'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 
            'if', 'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 
            'nullptr', 'operator', 'or', 'or_eq', 'private', 'protected', 'public', 'register', 
            'reinterpret_cast', 'requires', 'return', 'short', 'signed', 'sizeof', 'static', 
            'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 
            'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 
            'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
        }
        
        # Java keywords
        java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 
            'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 
            'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 
            'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 
            'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 
            'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', '_', 'true', 'false', 
            'null', 'var', 'record', 'sealed', 'non-sealed', 'permits', 'module', 'open', 'requires', 
            'exports', 'opens', 'uses', 'provides', 'to', 'with', 'transitive'
        }
        
        # Python keywords
        python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 
            'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 
            'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 
            'try', 'while', 'with', 'yield', 'match', 'case', 'type'
        }
        
        # JavaScript/TypeScript keywords
        js_ts_keywords = {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 
            'do', 'else', 'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 
            'in', 'instanceof', 'new', 'null', 'return', 'super', 'switch', 'this', 'throw', 'true', 
            'try', 'typeof', 'var', 'void', 'while', 'with', 'let', 'static', 'yield', 'await', 'enum', 
            'implements', 'interface', 'package', 'private', 'protected', 'public', 'arguments', 'as', 
            'async', 'eval', 'from', 'get', 'of', 'set', 'type', 'declare', 'namespace', 'module', 
            'abstract', 'any', 'boolean', 'constructor', 'symbol', 'readonly', 'keyof', 'infer', 'is', 
            'asserts', 'global', 'bigint', 'object', 'number', 'string', 'undefined', 'unknown', 
            'never', 'override', 'intrinsic'
        }
        
        # 合并所有关键字
        keywords = cpp_keywords | java_keywords | python_keywords | js_ts_keywords
        
        # 简单的分词方法：按空格和换行符分割，然后进一步处理
        tokens = []
        lines = code.split('\n')
        for line in lines:
            # 先提取完整的函数调用字符串（用于规则3）
            # 匹配 a.b.c->f() 或 a->f() 或 a.b.f() 等模式
            full_calls = []
            
            # 对象方法调用: a.f(
            obj_full_calls = re.findall(r'(\w+(?:\.\w+)*\.\w+)\s*\(', line)
            full_calls.extend(obj_full_calls)
            
            # 指针方法调用: a->f(
            ptr_full_calls = re.findall(r'(\w+(?:\.\w+)*(?:->\w+)+)\s*\(', line)
            full_calls.extend(ptr_full_calls)
            
            # 查找类定义并添加[CLASSDEF]前缀
            class_name = None
            for pattern in class_definition_patterns:
                class_def_match = re.search(pattern, line)
                if class_def_match:
                    class_name = class_def_match.group(1)
                    # 添加[CLASSDEF]标记作为单独的token
                    tokens.append('[CLASSDEF]{}'.format(class_name))
                    break
            
            # 查找函数定义并添加[FUNCDEF]前缀
            func_name = None
            for pattern in function_definition_patterns:
                func_def_match = re.search(pattern, line)
                if func_def_match:
                    potential_func_name = func_def_match.group(1)
                    # 排除关键字作为函数名（避免将for、if等控制结构误认为函数定义）
                    if potential_func_name not in keywords:
                        func_name = potential_func_name
                        # 添加[FUNCDEF]标记作为单独的token
                        tokens.append('[FUNCDEF]{}'.format(func_name))
                        break
            
            # 提取不同类型的函数调用模式
            # 1. 对象方法调用: a.f(
            obj_method_calls = re.findall(r'\.(\w+)\s*\(', line)
            for method in obj_method_calls:
                # 添加[OBJCALL]标记
                tokens.append('[OBJCALL]{}'.format(method))
            
            # 2. 指针方法调用: a->f(
            ptr_method_calls = re.findall(r'->(\w+)\s*\(', line)
            for method in ptr_method_calls:
                # 添加[PTRCALL]标记
                tokens.append('[PTRCALL]{}'.format(method))
                
            # Keep track of all processed identifiers to avoid duplicates
            processed_identifiers = set()
            
            # 3. 类方法调用: Class::method(
            # Handle complex cases like std::vector<int>::push_back and CodeTransform::operator()
            # This pattern matches namespace::class_name::method or class_name::method
            # It also handles templates like vector<int>
            class_method_calls = re.findall(r'((?:\w+::)*\w+(?:<[^>]*>)?)::(\w+)\s*\(', line)
            for full_class_name, method in class_method_calls:
                # 添加[CLASSCALL]标记给方法
                classcall_token = '[CLASSCALL]{}'.format(method)
                if classcall_token not in tokens:  # Avoid duplicates
                    tokens.append(classcall_token)
                processed_identifiers.add(method)
                
                # Extract base class/namespace names
                # Split by '::' but be careful with templates
                parts = full_class_name.split('::')
                for part in parts:
                    # For template classes, extract the base name
                    base_name = part.split('<')[0]
                    if base_name not in processed_identifiers:  # Avoid duplicates
                        tokens.append(base_name)
                        processed_identifiers.add(base_name)
            
            # 4. 普通函数调用: f( 但排除关键字和函数定义
            # First, find all potential function calls
            all_func_calls = re.findall(r'\b(\w+)\s*\(', line)
            for func in all_func_calls:
                # Skip if this function was already processed
                if func in processed_identifiers:
                    continue
                    
                # 排除关键字 - keywords should not be treated as function calls
                if func not in keywords:
                    # Check if this is actually a function definition rather than a call
                    is_func_def = False
                    
                    # Check for Python/JavaScript function definitions
                    func_def_pattern = r'(?:def|function)\s+' + re.escape(func) + r'\s*\('
                    if re.search(func_def_pattern, line):
                        is_func_def = True
                    
                    # Check for C++/Java function definitions by looking for the specific function we found
                    if not is_func_def:
                        # For each function definition pattern, check if it matches and captures this function name
                        for pattern in function_definition_patterns:
                            match = re.search(pattern, line)
                            if match and match.group(1) == func:
                                is_func_def = True
                                break
                    
                    # If it's not a function definition, treat it as a function call
                    if not is_func_def:
                        # But also check that it's not part of a method call (which we handle separately)
                        # Method calls like obj.method() are handled by the object method patterns
                        # Class method calls like Class::method() are handled by the class method patterns
                        # So we only add [CALL] for standalone function calls
                        # Check if it's preceded by '.' or '->' which would indicate it's a method call
                        method_call_pattern = r'(?:\.\s*|->\s*)' + re.escape(func) + r'\s*\('
                        class_method_call_pattern = r'(?:\w+(?:::\w+)*)::' + re.escape(func) + r'\s*\('
                        if not re.search(method_call_pattern, line) and not re.search(class_method_call_pattern, line):
                            # 添加[CALL]标记
                            call_token = '[CALL]{}'.format(func)
                            if call_token not in tokens:  # Avoid duplicates
                                tokens.append(call_token)
                            processed_identifiers.add(func)
                # Note: We do NOT add keywords to processed_identifiers here
                # because keywords should still be processed in the general identifier section
                # Keywords found in function call patterns are NOT added to processed_identifiers
            
            # 添加完整的函数调用字符串作为token（规则3）
            for full_call in full_calls:
                tokens.append(full_call)
            
            # 处理字符串字面量
            string_literals = re.findall(r'"([^"]*)"', line)  # 双引号字符串
            string_literals.extend(re.findall(r"'([^']*)'", line))  # 单引号字符串
            for literal in string_literals:
                # 清理字符串字面量，只保留字母数字和下划线
                clean_literal = re.sub(r'[^\w]', '', literal)
                if clean_literal:  # 只添加非空字符串
                    tokens.append(clean_literal)
            
            # 提取所有标识符，包括关键字
            identifiers = re.findall(r'\b(\w+)\b', line)
            for identifier in identifiers:
                # 排除函数名（避免重复）
                is_function_name = False
                if func_name and identifier == func_name:
                    is_function_name = True
                
                # 排除类名（避免重复）
                is_class_name = False
                if class_name and identifier == class_name:
                    is_class_name = True
                    
                # 排除已经被标记为函数调用的标识符
                # We need to check if the identifier part of the token matches exactly
                # For example, [CALL]print should not match identifier 'int' even though 'print' ends with 'int'
                is_function_call = False
                for token in tokens:
                    if token.startswith(('[CALL]', '[OBJCALL]', '[PTRCALL]', '[CLASSCALL]')):
                        # Extract the identifier part (after the last ])
                        if ']' in token:
                            token_identifier = token[token.rfind(']')+1:]
                        else:
                            token_identifier = token
                        if token_identifier == identifier:
                            is_function_call = True
                            break
                
                # 排除已在类方法处理中处理过的标识符
                already_processed = identifier in processed_identifiers
                
                # NOTE: We're NOT excluding keywords here as per requirements
                # Keywords like 'int', 'def', 'return' should be included
                if not is_function_name and not is_class_name and not is_function_call and not already_processed:
                    tokens.append(identifier)
            
            # 添加参数（如果是函数定义）
            func_def_match = None
            for pattern in function_definition_patterns:
                func_def_match = re.search(pattern, line)
                if func_def_match:
                    func_name = func_def_match.group(1)
                    break
            
            if func_def_match and func_name:
                # 获取参数部分
                param_match = re.search(r'{}\s*\(([^)]*)\)'.format(re.escape(func_name)), line)
                if param_match:
                    params = param_match.group(1)
                    # 分割参数，处理各种情况
                    for param in params.split(','):
                        # 去除类型声明(如 int a, String b)
                        param = param.strip().split()[-1] if param.strip() else ""
                        # 去除默认值(如 a=1)
                        param = param.split('=')[0]
                        # 去除指针符号等
                        param = re.sub(r'[&*]+', '', param)
                        # 清理参数名，去除标点符号
                        param = re.sub(r'[^\w]', '', param)
                        # 添加参数名作为token
                        if param and param != 'void':  # void不是参数名
                            tokens.append(param)
        # 不去除重复的token，保留所有token
        return tokens
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents given a query."""
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []

        try:
            # Tokenize query
            query_tokens = query.split()
            logger.info(f"query tokens: {query_tokens}")
            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(query_tokens)
            return scores
        except Exception as e:
            logger.error(f"Failed to get BM25 scores: {e}")
            return []

    def filter_and_score(self, query: str) -> tuple[List[int], List[float]]:
        """Filter documents using BM25 and return document indices and normalized scores."""
        try:
            # Get BM25 scores for all documents
            scores = self.get_scores(query)

            # Get top-k document indices based on BM25 scores
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:self.top_k]

            filtered_scores = [scores[i] for i in doc_indices]

            logger.info(f"BM25 filtered {len(doc_indices)} candidates from {len(self.documents)} documents")
            return doc_indices, filtered_scores

        except Exception as e:
            logger.error(f"BM25 filtering failed: {e}")
            raise

    def call(self, query: str, top_k: Optional[int] = None) -> RetrieverOutputType:
        """Retrieve documents using BM25."""
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []

        if top_k is None:
            top_k = self.top_k

        assert top_k > 0, "Top k must be greater than 0"

        try:
            # Get BM25 scores for all documents
            scores = self.get_scores(query)
            
            # Get top-k document indices based on BM25 scores
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]

            # Get the actual documents and scores
            retrieved_docs = [self.documents[i] for i in doc_indices]
            doc_scores = [scores[i] for i in doc_indices]

            logger.info(f"BM25 retrieved {len(retrieved_docs)} documents")

            return [
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=doc_scores,
                    query=query,
                    documents=retrieved_docs,
                )
            ]

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []
    

class SingleVectorRetriever(FAISSRetriever):
    """A wrapper of FAISSRetriever with the additional feature of supporting documents feature"""

    def __init__(self, documents: List[Document], *args, **kwargs):
        self.original_doc = documents
        super().__init__(documents=documents, *args, **kwargs)
        logger.info(
            f"SingleVectorRetriever initialized with {len(documents)} documents"
        )

    def call(
        self,
        input: FAISSRetrieverQueriesType,
        top_k: int = None
    ) -> RetrieverOutputType:
        if top_k is None:
            top_k = self.top_k
        assert top_k > 0, "Top k must be greater than 0"
        retriever_output = super().call(input, top_k)

        # Extract the first result from the list
        if not retriever_output:
            return []

        first_output = retriever_output[0]

        # Get the documents based on the indices
        retrieved_docs = [self.original_doc[i] for i in first_output.doc_indices]
        doc_indices = first_output.doc_indices
        doc_scores = first_output.doc_scores


        # Create a new RetrieverOutput with the documents
        return [
            RetrieverOutput(
                doc_indices=doc_indices,
                doc_scores=doc_scores,
                query=first_output.query,
                documents=retrieved_docs,
            )
        ]


class DualVectorRetriever:
    """Dual vector retriever: supports dual retrieval from code and summary vectors."""

    def __init__(self, dual_docs: List[DualVectorDocument], embedder, top_k: int = 20):
        """
        Initializes the dual vector retriever.

        Args:
            dual_docs: A list of dual vector documents.
            embedder: The embedder instance for embedding queries.
            top_k: The number of most relevant documents to return.
        """
        self.dual_docs = dual_docs
        self.embedder = embedder
        assert top_k > 0, f"Top k must be greater than 0, got {top_k}"
        self.top_k = top_k
        self.doc_map = {doc.original_doc.id: doc for doc in dual_docs}

        logger.info(
            f"Dual vector retriever initialization completed, containing {len(dual_docs)} documents"
        )

    def _build_indices(self):
        """Builds the code index and the summary index."""
        if not self.dual_docs:
            logger.warning("No documents available for building indices")
            self.code_retriever = None
            self.understanding_retriever = None
            return

        # 1. Build the code index
        code_docs = []
        for dual_doc in self.dual_docs:
            # Create a document object for FAISS
            faiss_doc = Document(
                text=dual_doc.original_doc.text,
                meta_data=dual_doc.original_doc.meta_data,
                id=f"{dual_doc.original_doc.id}_code",
                vector=dual_doc.code_embedding,
            )
            code_docs.append(faiss_doc)

        self.code_retriever = FAISSRetriever(
            top_k=self.top_k,
            embedder=self.embedder,
            documents=code_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Code FAISS index built successfully.")

        # 2. Build the summary index
        understanding_docs = []
        for dual_doc in self.dual_docs:
            faiss_doc = Document(
                text=dual_doc.understanding_text,
                meta_data=dual_doc.original_doc.meta_data,
                id=f"{dual_doc.original_doc.id}_understanding",
                vector=dual_doc.understanding_embedding,
            )
            understanding_docs.append(faiss_doc)

        self.understanding_retriever = FAISSRetriever(
            top_k=self.top_k,
            embedder=self.embedder,
            documents=understanding_docs,
            document_map_func=lambda doc: doc.vector,
        )
        logger.info("Understanding FAISS index built successfully.")

    def call(self, query_str: str) -> RetrieverOutputType:
        """
        Performs dual retrieval.

        Args:
            query_str: The query string.

        Returns:
            A RetrieverOutput object containing the retrieved documents and scores.
        """
        assert isinstance(
            query_str, str
        ), f"Query must be a string, got {type(query_str)}"

        self._build_indices()

        if not self.dual_docs:
            return RetrieverOutput(
                doc_indices=[], doc_scores=[], query=query_str, documents=[]
            )

        # 1. Retrieve from the code index
        code_results = self.code_retriever.call(query_str, top_k=self.top_k)[0]
        # 2. Retrieve from the summary index
        understanding_results = self.understanding_retriever.call(
            query_str, top_k=self.top_k
        )[0]

        # 3. Merge and re-rank the results
        combined_scores = {}

        # Process code results - extract original chunk_id from FAISS document ID
        for i, score in zip(code_results.doc_indices, code_results.doc_scores):
            # Get the document from code retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_code", "")
            combined_scores[original_chunk_id] = score

        # Process understanding results - extract original chunk_id from FAISS document ID
        for i, score in zip(
            understanding_results.doc_indices, understanding_results.doc_scores
        ):
            # Get the document from understanding retriever to extract original chunk_id
            doc_id = self.dual_docs[i].original_doc.id
            original_chunk_id = doc_id.replace("_understanding", "")
            if original_chunk_id not in combined_scores:
                combined_scores[original_chunk_id] = score
            else:
                combined_scores[original_chunk_id] = max(
                    combined_scores[original_chunk_id], score
                )

        # 4. Sort and get the top-k results
        # Sort by the combined score in descending order
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda chunk_id: combined_scores[chunk_id],
            reverse=True,
        )

        # 5. Retrieve the full documents for the top-k chunk_ids and create indices mapping
        top_k_docs = []
        doc_indices = []
        doc_scores = []
        for idx, chunk_id in enumerate(
            sorted_chunk_ids[: min(self.top_k, len(sorted_chunk_ids))]
        ):
            if chunk_id in self.doc_map:
                dual_doc = self.doc_map[chunk_id]
                top_k_docs.append(dual_doc)
                doc_indices.append(idx)
                doc_scores.append(combined_scores[chunk_id])

        logger.info(
            f"Retrieved {len(top_k_docs)} documents after merging code and understanding search results."
        )

        return [
            RetrieverOutput(
                doc_indices=doc_indices,
                doc_scores=doc_scores,
                query=query_str,
                documents=top_k_docs,
            )
        ]


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 keyword filtering with FAISS semantic search.

    The retrieval process:
    1. BM25 first filters documents by exact keyword matches, reducing the search space
    2. FAISS then performs semantic similarity search on this subset
    3. Results are merged and re-ranked for optimal relevance
    """

    def __init__(self, documents: List[Union[Document, DualVectorDocument]], **kwargs):
        """
        Initialize the hybrid retriever.

        Args:
            documents: List of transformed documents to index
        """
        self.documents = documents
        self.embedder = get_embedder()

        rag_config = configs()["rag"]
        self.use_dual_vector = rag_config["embedder"]["sketch_filling"]
        retriever_config = rag_config["retriever"]
        self.top_k = retriever_config["top_k"]
        assert self.top_k > 0, "Top k must be greater than 0"
        bm25_config = retriever_config["bm25"]
        self.bm25_k1 = bm25_config["k1"]
        self.bm25_b = bm25_config["b"]
        self.bm25_weight = bm25_config["weight"]
        assert 0 <= self.bm25_weight <= 1, "BM25 weight must be between 0 and 1."
        self.fusion = configs()["rag"]["retriever"]["fusion"]
        assert self.fusion in ["rrf", "normal_add"], f"Invalid fusion method: {self.fusion}"

        logger.info(
            f"Hybrid retriever initialized with dual_vector={'enabled' if self.use_dual_vector else 'disabled'}"
            f"BM25 parameters: k1={self.bm25_k1}, b={self.bm25_b}, weight={self.bm25_weight}"
            f"Other parameters: top_k={self.top_k}"
        )

    def _initialize_faiss_retriever(self, documents: List[Document | DualVectorDocument], top_k: int):
        """Initialize FAISS retriever based on vector type."""
        if self.use_dual_vector:
            faiss_retriever = DualVectorRetriever(
                dual_docs=documents,
                embedder=self.embedder,
                top_k=top_k,
            )
        else:
            faiss_retriever = SingleVectorRetriever(
                documents=documents,
                embedder=self.embedder,
                top_k=top_k,
                document_map_func=lambda doc: doc.vector,
            )
        logger.info(f"FAISS retriever initialized successfully")
        return faiss_retriever

    def _initialize_bm25_retriever(self, documents: List[Document | DualVectorDocument], top_k: int) -> BM25Retriever:
        """Initialize BM25 retriever."""
        bm25_retriever = BM25Retriever(
            documents=documents,
            k1=self.bm25_k1,
            b=self.bm25_b,
            top_k=top_k,
        )
        logger.info(f"BM25 retriever initialized successfully")
        return bm25_retriever

    def _rrf(self, doc_indices_list: List[List[int]], weights: List[int]) -> dict:
        """
        Given several ranks, return the final rank based on Reciprocal Rank Fusion
        """
        k = configs()["rag"]["retriever"]["rrf"]["k"]
        # Initialize a dictionary to hold the cumulative scores for each document
        doc_scores = defaultdict(float)
        for doc_indices, weight in zip(doc_indices_list, weights):
            for rank, doc_id in enumerate(doc_indices):
                # Apply the Reciprocal Rank Fusion formula
                doc_scores[doc_id] += weight * (1.0 / float(rank + 1 + k))
        # Return the final ranked list of document IDs
        return dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))


    def _mix_bm25_score_faiss_score(self, documents: List[Document | DualVectorDocument], bm25_indices: List[int], bm25_scores: List[float], faiss_results: RetrieverOutputType) -> List[Document]:
        """Mix BM25 scores with FAISS results."""
        faiss_indices = faiss_results[0].doc_indices
        faiss_scores = faiss_results[0].doc_scores
        if isinstance(documents[0], DualVectorDocument):
            documents = [doc.original_doc for doc in documents]
        if self.fusion == "normal_add":
            minmax_bm25_scores = minmax_norm(bm25_scores)
            zscore_bm25_scores = zscore_norm(minmax_bm25_scores)
            self.doc_id_to_bm25_scores = {documents[doc_id].id: (original_score, minmax_score, zscore_score) for doc_id, original_score, minmax_score, zscore_score in zip(bm25_indices, bm25_scores, minmax_bm25_scores, zscore_bm25_scores)}
            minmax_faiss_scores = minmax_norm(faiss_scores)
            zscore_faiss_scores = zscore_norm(minmax_faiss_scores)
            self.doc_id_to_faiss_scores = {documents[doc_id].id: (original_score, minmax_score, zscore_score) for doc_id, original_score, minmax_score, zscore_score in zip(faiss_indices, faiss_scores, minmax_faiss_scores, zscore_faiss_scores)}
            scores = [0] * len(bm25_indices)
            assert len(faiss_indices) == len(bm25_indices), f"Mismatch in FAISS results: {len(faiss_indices)} vs {len(bm25_indices)}"
            assert all([faiss_indices[i] < len(faiss_indices) for i in range(len(faiss_indices))]), f"Invalid FAISS index found"
            assert all([bm25_indices[i] < len(bm25_indices) for i in range(len(bm25_indices))]), f"Invalid BM25 index found"
            for i, doc in enumerate(bm25_indices):
                scores[bm25_indices[i]] = zscore_bm25_scores[i] * self.bm25_weight
            for i, doc in enumerate(faiss_indices):
                scores[faiss_indices[i]] += zscore_faiss_scores[i] * (1 - self.bm25_weight)
            self.doc_id_to_bm25faiss_scores = {documents[id].id: scores[id] for id in bm25_indices}
        else:
            id_to_scores = self._rrf([bm25_indices, faiss_indices], [self.bm25_weight, 1 - self.bm25_weight])
            scores = list(id_to_scores.values())
            self.doc_id_to_rrf_scores = {documents[id].id: score for (id, score) in id_to_scores.items()}
        return scores

    def call(self, bm25_keywords: str, faiss_query: str) -> List[RetrieverOutput]:
        """Perform hybrid retrieval combining BM25 and FAISS."""


        try:
            if bm25_keywords:
                bm25_retriever = self._initialize_bm25_retriever(self.documents, top_k=len(self.documents))
                bm25_indices, bm25_scores = bm25_retriever.filter_and_score(bm25_keywords)
            else:
                bm25_indices, bm25_scores = [], []
            faiss_retriever = self._initialize_faiss_retriever(self.documents, top_k=len(self.documents))
            faiss_results = faiss_retriever.call(faiss_query)
            if bm25_keywords:
                assert len(faiss_results[0].documents) == len(bm25_indices), f"Mismatch in number of documents between BM25 and FAISS results: {len(bm25_indices)} vs {len(faiss_results[0]['documents'])}"
                scores = self._mix_bm25_score_faiss_score(self.documents, bm25_indices, bm25_scores, faiss_results)
            else:
                scores = [0] * len(faiss_results[0].doc_indices)
                for i, doc in enumerate(faiss_results[0].doc_indices):
                    scores[doc] = faiss_results[0].doc_scores[i]
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:self.top_k]
            return [
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=[scores[i] for i in doc_indices],
                    query=f"BM25 keywords: {bm25_keywords}, FAISS query: {faiss_query}",
                    documents=[self.documents[i] for i in doc_indices],
                )
            ]

        except Exception as e:
            logger.error(
                f"Hybrid retrieval failed: {e}, falling back to pure FAISS search"
            )
            raise


class QueryDrivenRetriever(HybridRetriever):
    """Query-driven retriever that uses BM25 index and on-demand embedding with FAISS."""

    def __init__(
        self,
        documents: List[Union[Document, DualVectorDocument]],
        update_database: Callable = None,
        **kwargs
    ):
        """
        Initialize the query-driven retriever.

        Args:
            documents: List of splitted documents to index
            update_database: Function to update the database with new embedded documents
        """
        self.update_database = update_database
        self.query_driven_top_k = configs()["rag"]["query_driven"]['top_k']
        logger.info(f"Query-driven retriever initialized with query_driven_top_k={self.query_driven_top_k}")
        super().__init__(documents, **kwargs)

    def call(self, bm25_keywords: str, faiss_query: str) -> List[RetrieverOutput]:
        """
        Retrieve documents using BM25 index and on-demand embedding with FAISS.

        Args:
            bm25_keywords: BM25 keywords
            faiss_query: FAISS query
            bm25_keywords: BM25 keywords
            faiss_query: FAISS query

        Returns:
            List of RetrieverOutput objects
        """

        # Step 1: BM25 filtering to get candidates
        logger.info("Step 1: BM25 filtering")
        bm25_retriever = self._initialize_bm25_retriever(self.documents, top_k=self.query_driven_top_k)
        query_related_doc_indices, query_related_doc_scores = bm25_retriever.filter_and_score(bm25_keywords)

        filtered_docs = [self.documents[i] for i in query_related_doc_indices]
        doc_id_to_score = {doc.id if isinstance(doc, Document) else doc.original_doc.id: score for doc, score in zip(filtered_docs, query_related_doc_scores)}

        self.bm25_documents = deepcopy(filtered_docs)
        # Step 2: Use database manager to embed and cache documents
        logger.info("Step 2: Embedding and caching documents using DatabaseManager")

        if self.update_database:
            embedded_docs = self.update_database(filtered_docs)
        else:
            embedded_docs = filtered_docs
        logger.info(f"Embedded and cached {len(embedded_docs)} documents")

        # embedded_docs may reorder the filtered_docs, so update the query_related_doc_indices nad query_related_doc_scores correspondingly
        query_related_doc_indices = list(range(len(embedded_docs)))
        query_related_doc_scores = [doc_id_to_score[doc.original_doc.id] if isinstance(doc, DualVectorDocument) else doc_id_to_score[doc.id] for doc in embedded_docs]

        try:
            faiss_retriever = self._initialize_faiss_retriever(embedded_docs, top_k=len(embedded_docs))
            faiss_results = faiss_retriever.call(faiss_query)
            assert len(faiss_results[0].documents) == len(query_related_doc_indices), f"Mismatch in number of documents between BM25 and FAISS results: {len(query_related_doc_indices)} vs {len(faiss_results[0]['documents'])}"
            scores = self._mix_bm25_score_faiss_score(embedded_docs, query_related_doc_indices, query_related_doc_scores, faiss_results)
            doc_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:self.top_k]
            return [
                RetrieverOutput(
                    doc_indices=doc_indices,
                    doc_scores=[scores[i] for i in doc_indices],
                    query=f"BM25 keywords: {bm25_keywords}, FAISS query: {faiss_query}",
                    documents=[embedded_docs[i] for i in doc_indices],
                )
            ]

        except Exception as e:
            logger.error(
                f"Query-based retrieval failed: {e}, falling back to pure FAISS search"
            )
            raise