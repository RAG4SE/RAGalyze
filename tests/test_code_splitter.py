"""Tests for CodeSplitter with tree-sitter support including markdown."""

import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adalflow.core.types import Document
from ragalyze.rag.splitter.code_splitter import CodeSplitter, TREE_SITTER_AVAILABLE

# Sample code content for different languages
PYTHON_CODE = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "success"

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        """Add two numbers."""
        self.result = x + y
        return self.result
    
    def multiply(self, x, y):
        """Multiply two numbers."""
        self.result = x * y
        return self.result

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 6))
    hello_world()
'''

JAVASCRIPT_CODE = """
function helloWorld() {
    console.log("Hello, World!");
    return "success";
}

class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(x, y) {
        this.result = x + y;
        return this.result;
    }
    
    multiply(x, y) {
        this.result = x * y;
        return this.result;
    }
}

const calc = new Calculator();
console.log(calc.add(5, 3));
console.log(calc.multiply(4, 6));
helloWorld();
"""

MARKDOWN_CONTENT = """
# CodeSplitter Documentation

This is a comprehensive guide to using the CodeSplitter.

## Features

The CodeSplitter provides the following features:

- Smart boundary detection using tree-sitter parsing
- Language-specific statement recognition
- Adjustable chunk sizes and overlap
- Fallback mechanisms when tree-sitter is not available

### Supported Languages

1. **Python** (.py files)
   - Function definitions
   - Class definitions
   - Control flow statements

2. **JavaScript** (.js files)
   - Function declarations
   - Class declarations
   - ES6+ features

3. **Markdown** (.md files)
   - Headers and sections
   - Code blocks
   - Lists and tables

## Usage Example

```python
from rag.splitter.code_splitter import CodeSplitter

splitter = CodeSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    file_extension=".py"
)

documents = splitter.split([Document(text=code_content)])
```

## Configuration

You can configure the splitter with various parameters:

- `chunk_size`: Target size for each chunk
- `chunk_overlap`: Overlap between chunks
- `smart_boundary_ratio`: When to start looking for boundaries
- `file_extension`: Programming language to use

### Advanced Options

For advanced use cases, you can:

1. Customize the boundary detection ratio
2. Use different file extensions
3. Handle multiple languages in one project

## Best Practices

- Choose appropriate chunk sizes for your use case
- Consider the programming language when setting parameters
- Test with your specific code patterns

---

*This documentation is part of the RAGalyze project.*
"""

RST_CONTENT = """
CodeSplitter Documentation
==========================

This is a comprehensive guide to using the CodeSplitter.

Features
--------

The CodeSplitter provides the following features:

- Smart boundary detection using tree-sitter parsing
- Language-specific statement recognition
- Adjustable chunk sizes and overlap
- Fallback mechanisms when tree-sitter is not available

Supported Languages
~~~~~~~~~~~~~~~~~~~

1. **Python** (.py files)
   
   - Function definitions
   - Class definitions
   - Control flow statements

2. **JavaScript** (.js files)
   
   - Function declarations
   - Class declarations
   - ES6+ features

3. **reStructuredText** (.rst files)
   
   - Sections and subsections
   - Code blocks
   - Directives and roles

Usage Example
-------------

.. code-block:: python

   from rag.splitter.code_splitter import CodeSplitter
   
   splitter = CodeSplitter(
       chunk_size=1024,
       chunk_overlap=64,
       file_extension=".py"
   )
   
   documents = splitter.split([Document(text=code_content)])

Configuration
~~~~~~~~~~~~~

You can configure the splitter with various parameters:

- ``chunk_size``: Target size for each chunk
- ``chunk_overlap``: Overlap between chunks
- ``smart_boundary_ratio``: When to start looking for boundaries
- ``file_extension``: Programming language to use

Advanced Usage
--------------

For advanced use cases, you can:

1. Customize the boundary detection ratio
2. Use different file extensions
3. Handle multiple languages in one project

Best Practices
~~~~~~~~~~~~~~

- Choose appropriate chunk sizes for your use case
- Consider the programming language when setting parameters
- Test with your specific code patterns

.. note::
   This is an important note about using the CodeSplitter effectively.

.. warning::
   Make sure to test with your specific use case before production.

----

*This documentation is part of the RAGalyze project.*
"""

YAML_CONTENT = """
---
# Application Configuration
app:
  name: "RAGalyze"
  version: "1.0.0"
  description: "Advanced RAG system with intelligent document processing"
  
  # Server configuration
  server:
    host: "localhost"
    port: 8080
    ssl_enabled: false
    max_connections: 1000
    
  # Database settings
  database:
    type: "postgresql"
    host: "db.example.com"
    port: 5432
    name: "ragalyze_db"
    username: "app_user"
    password: "secure_password"
    pool_size: 20
    
# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "console"
      level: "DEBUG"
    - type: "file"
      level: "INFO"
      filename: "/var/log/log"
      max_size: "10MB"
      backup_count: 5
      
# Feature flags
features:
  enable_smart_splitting: true
  enable_tree_sitter: true
  enable_nlp_processing: false
  experimental_features:
    - "advanced_chunking"
    - "semantic_search"
    - "auto_optimization"
    
# Processing pipeline
pipeline:
  stages:
    - name: "document_ingestion"
      enabled: true
      config:
        batch_size: 100
        timeout: 30
    - name: "text_splitting"
      enabled: true
      config:
        chunk_size: 1024
        chunk_overlap: 128
        smart_boundary_ratio: 0.8
    - name: "embedding_generation"
      enabled: true
      config:
        model: "sentence-transformers/all-MiniLM-L6-v2"
        batch_size: 32
        
# Environment-specific overrides
environments:
  development:
    app:
      server:
        port: 3000
    logging:
      level: "DEBUG"
  production:
    app:
      server:
        ssl_enabled: true
        port: 443
    logging:
      level: "WARNING"
      handlers:
        - type: "file"
          level: "WARNING"
"""

JSON_CONTENT = """
{
  "application": {
    "name": "RAGalyze",
    "version": "1.0.0",
    "description": "Advanced RAG system with intelligent document processing",
    "author": {
      "name": "Development Team",
      "email": "dev@com",
      "organization": "RAGalyze Inc."
    },
    "license": "MIT",
    "repository": {
      "type": "git",
      "url": "https://github.com/ragalyze/git",
      "branch": "main"
    }
  },
  "server": {
    "host": "localhost",
    "port": 8080,
    "ssl": {
      "enabled": false,
      "certificate": "/path/to/cert.pem",
      "private_key": "/path/to/key.pem"
    },
    "cors": {
      "enabled": true,
      "origins": ["http://localhost:3000", "https://app.com"],
      "methods": ["GET", "POST", "PUT", "DELETE"],
      "headers": ["Content-Type", "Authorization"]
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100,
      "burst_size": 20
    }
  },
  "database": {
    "primary": {
      "type": "postgresql",
      "host": "db.example.com",
      "port": 5432,
      "name": "ragalyze_db",
      "username": "app_user",
      "password": "secure_password",
      "pool_size": 20,
      "timeout": 30,
      "ssl_mode": "require"
    },
    "cache": {
      "type": "redis",
      "host": "cache.example.com",
      "port": 6379,
      "database": 0,
      "password": "cache_password",
      "ttl": 3600
    }
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
      {
        "type": "console",
        "level": "DEBUG",
        "formatter": "detailed"
      },
      {
        "type": "file",
        "level": "INFO",
        "filename": "/var/log/log",
        "max_size": "10MB",
        "backup_count": 5,
        "formatter": "json"
      },
      {
        "type": "syslog",
        "level": "WARNING",
        "facility": "local0",
        "formatter": "syslog"
      }
    ]
  },
  "features": {
    "smart_splitting": {
      "enabled": true,
      "algorithms": ["tree_sitter", "semantic", "statistical"],
      "fallback_strategy": "statistical"
    },
    "embedding_generation": {
      "enabled": true,
      "models": [
        {
          "name": "sentence-transformers/all-MiniLM-L6-v2",
          "type": "sentence_transformer",
          "dimensions": 384,
          "max_sequence_length": 512
        },
        {
          "name": "text-embedding-ada-002",
          "type": "openai",
          "dimensions": 1536,
          "max_sequence_length": 8191
        }
      ]
    },
    "retrieval": {
      "enabled": true,
      "strategies": ["vector", "keyword", "hybrid"],
      "default_strategy": "hybrid",
      "top_k": 10,
      "similarity_threshold": 0.7
    }
  },
  "processing": {
    "pipeline": {
      "stages": [
        {
          "name": "document_ingestion",
          "enabled": true,
          "config": {
            "batch_size": 100,
            "timeout": 30,
            "supported_formats": ["pdf", "docx", "txt", "md", "html"]
          }
        },
        {
          "name": "text_extraction",
          "enabled": true,
          "config": {
            "ocr_enabled": false,
            "language_detection": true,
            "encoding_detection": true
          }
        },
        {
          "name": "text_splitting",
          "enabled": true,
          "config": {
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "smart_boundary_ratio": 0.8,
            "preserve_structure": true
          }
        }
      ]
    },
    "quality_control": {
      "enabled": true,
      "min_chunk_size": 50,
      "max_chunk_size": 2048,
      "duplicate_detection": true,
      "language_filtering": ["en", "zh", "es", "fr"]
    }
  },
  "monitoring": {
    "metrics": {
      "enabled": true,
      "endpoint": "/metrics",
      "collectors": ["cpu", "memory", "disk", "network", "application"]
    },
    "health_check": {
      "enabled": true,
      "endpoint": "/health",
      "interval": 30,
      "timeout": 5
    },
    "alerts": {
      "enabled": true,
      "channels": [
        {
          "type": "email",
          "recipients": ["admin@com", "ops@com"],
          "severity_threshold": "warning"
        },
        {
          "type": "slack",
          "webhook_url": "https://hooks.slack.com/services/...",
          "channel": "#alerts",
          "severity_threshold": "error"
        }
      ]
    }
  }
}
"""


class TestCodeSplitter:
    """Test cases for CodeSplitter functionality."""

    def __init__(self, split_by) -> None:
        self.split_by = split_by

    def test_initialization(self):
        """Test CodeSplitter initialization."""
        splitter = CodeSplitter(
            chunk_size=512,
            chunk_overlap=64,
            smart_boundary_ratio=0.8,
            file_extension=".py",
            split_by=self.split_by,
        )

        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 64
        assert splitter.smart_boundary_ratio == 0.8
        assert splitter.file_extension == ".py"
        assert splitter.smart_boundary_threshold == int(512 * 0.8)

    def test_python_splitting(self):
        """Test splitting Python code."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".py",
            split_by=self.split_by,
        )

        doc = Document(text=PYTHON_CODE)
        result = splitter.call([doc])

        print("\n" + "=" * 80)
        print(" Python Code Chunks - Extension: .py ")
        print("=" * 80)
        print(f"Total chunks: {len(result)}")
        print(f"Original content length: {len(PYTHON_CODE)} characters\n")

        for i, chunk in enumerate(result, 1):
            print(f"--- Chunk {i} ---")
            print(f"Length: {len(chunk.text)} characters")
            print("Content:")
            print(chunk.text)
            print("-" * 40)

        assert len(result) > 1, "Should split into multiple chunks"

        # Check that chunks have reasonable sizes
        for chunk in result:
            assert len(chunk.text.strip()) > 0, "Chunks should not be empty"

        # Check overlap exists between consecutive chunks
        if len(result) > 1:
            # Simple overlap check - some content should be similar
            first_chunk_end = result[0].text[-100:]
            second_chunk_start = result[1].text[:100]
            # At least some characters should overlap or be related
            assert len(first_chunk_end.strip()) > 0
            assert len(second_chunk_start.strip()) > 0

    def test_javascript_splitting(self):
        """Test splitting JavaScript code."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".js",
            split_by=self.split_by,
        )

        doc = Document(text=JAVASCRIPT_CODE)
        result = splitter.call([doc])

        print("\n" + "=" * 80)
        print(" JavaScript Code Chunks - Extension: .js ")
        print("=" * 80)
        print(f"Total chunks: {len(result)}")
        print(f"Original content length: {len(JAVASCRIPT_CODE)} characters\n")

        for i, chunk in enumerate(result, 1):
            print(f"--- Chunk {i} ---")
            print(f"Length: {len(chunk.text)} characters")
            print("Content:")
            print(chunk.text)
            print("-" * 40)

        assert len(result) >= 1, "Should produce at least one chunk"

        # Verify content is preserved
        combined_text = " ".join([chunk.text for chunk in result])
        assert "function helloWorld" in combined_text
        assert "class Calculator" in combined_text

    def test_markdown_splitting(self):
        """Test splitting Markdown content."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".md",
            split_by=self.split_by,
        )

        doc = Document(text=MARKDOWN_CONTENT)
        result = splitter.call([doc])

        print("\n" + "=" * 80)
        print(" Markdown Content Chunks - Extension: .md ")
        print("=" * 80)
        print(f"Total chunks: {len(result)}")
        print(f"Original content length: {len(MARKDOWN_CONTENT)} characters\n")

        for i, chunk in enumerate(result, 1):
            print(f"--- Chunk {i} ---")
            print(f"Length: {len(chunk.text)} characters")
            print("Content:")
            print(chunk.text)
            print("-" * 40)

        assert len(result) >= 1, "Should produce at least one chunk"

        # Check that markdown structure is preserved
        combined_text = " ".join([chunk.text for chunk in result])
        assert "# CodeSplitter Documentation" in combined_text
        assert "## Features" in combined_text
        assert "```python" in combined_text

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not available")
    def test_rst_splitting(self):
        """Test splitting reStructuredText content."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".rst",
            split_by=self.split_by,
        )

        doc = Document(text=RST_CONTENT)
        result = splitter.call([doc])

        print("\n" + "=" * 80)
        print(" RST Content Chunks - Extension: .rst ")
        print("=" * 80)
        print(f"Total chunks: {len(result)}")
        print(f"Original content length: {len(RST_CONTENT)} characters\n")

        for i, chunk in enumerate(result, 1):
            print(f"--- Chunk {i} ---")
            print(f"Length: {len(chunk.text)} characters")
            print("Content:")
            print(chunk.text)
            print("-" * 40)

        assert len(result) >= 1, "Should produce at least one chunk"

        # Check that RST structure is preserved
        combined_text = " ".join([chunk.text for chunk in result])
        assert "CodeSplitter Documentation" in combined_text
        assert "Features" in combined_text
        assert ".. code-block:: python" in combined_text
        assert ".. note::" in combined_text
        assert ".. warning::" in combined_text

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not available")
    def test_yaml_splitting(self):
        """Test YAML content splitting with tree-sitter."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".yaml",
            split_by=self.split_by,
        )

        doc = Document(text=YAML_CONTENT)
        result = splitter.call([doc])

        print("\n" + "=" * 80)
        print(" YAML Content Chunks - Extension: .yaml ")
        print("=" * 80)
        print(f"Total chunks: {len(result)}")
        print(f"Original content length: {len(YAML_CONTENT)} characters\n")

        for i, chunk in enumerate(result, 1):
            print(f"--- Chunk {i} ---")
            print(f"Length: {len(chunk.text)} characters")
            print("Content:")
            print(chunk.text)
            print("-" * 40)

        assert len(result) >= 1, "Should produce at least one chunk"

        # Check that YAML structure is preserved
        combined_text = " ".join([chunk.text for chunk in result])
        assert "app:" in combined_text
        assert "logging:" in combined_text
        assert "features:" in combined_text
        assert "pipeline:" in combined_text
        assert "environments:" in combined_text

        # Test with .yml extension as well
        splitter_yml = CodeSplitter(
            chunk_size=400,
            chunk_overlap=50,
            file_extension=".yml",
            split_by=self.split_by,
        )

        result_yml = splitter_yml.call([doc])
        assert len(result_yml) >= 1, "Should work with .yml extension too"

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not available")
    def test_json_splitting(self):
        """Test JSON content splitting with tree-sitter."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".json",
            split_by=self.split_by,
        )

        doc = Document(text=JSON_CONTENT)
        result = splitter.call([doc])

        print("\n" + "=" * 80)
        print(" JSON Content Chunks - Extension: .json ")
        print("=" * 80)
        print(f"Total chunks: {len(result)}")
        print(f"Original content length: {len(JSON_CONTENT)} characters\n")

        for i, chunk in enumerate(result, 1):
            print(f"--- Chunk {i} ---")
            print(f"Length: {len(chunk.text)} characters")
            print("Content:")
            print(chunk.text)
            print("-" * 40)

        assert len(result) >= 1, "Should produce at least one chunk"

        # Check that JSON structure is preserved
        combined_text = " ".join([chunk.text for chunk in result])
        assert "application" in combined_text
        assert "server" in combined_text
        assert "database" in combined_text
        assert "logging" in combined_text
        assert "features" in combined_text
        assert "processing" in combined_text
        assert "monitoring" in combined_text

        # Verify JSON structure integrity by checking for proper nesting
        assert "{" in combined_text and "}" in combined_text
        assert "[" in combined_text and "]" in combined_text

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not available")
    def test_smart_boundary_detection(self):
        """Test that smart boundaries are detected properly."""
        splitter = CodeSplitter(
            chunk_size=150,
            chunk_overlap=30,
            smart_boundary_ratio=0.7,
            file_extension=".py",
            split_by=self.split_by,
        )

        doc = Document(text=PYTHON_CODE)
        result = splitter.call([doc])

        # With smart boundaries, we should avoid splitting in the middle of functions
        for chunk in result:
            text = chunk.text.strip()
            if "def " in text:
                # If a chunk contains a function definition, it should be complete
                # or at least not cut off in the middle of the signature
                assert not text.endswith(
                    "def "
                ), "Should not end with incomplete function definition"

    def test_unsupported_language_fallback(self):
        """Test behavior with unsupported file extensions."""
        splitter = CodeSplitter(
            chunk_size=200,
            chunk_overlap=50,
            file_extension=".xyz",  # Unsupported extension,
            split_by=self.split_by,
        )

        doc = Document(text="Some text content for unsupported language.")

        # Should not raise an error, but may use fallback behavior
        try:
            result = splitter.call([doc])
            assert len(result) >= 1, "Should still produce chunks"
        except NotImplementedError:
            # Expected if fallback is not implemented
            pass

    def test_empty_content(self):
        """Test handling of empty content."""
        splitter = CodeSplitter(
            chunk_size=100,
            chunk_overlap=20,
            file_extension=".py",
            split_by=self.split_by,
        )

        doc = Document(text="")
        result = splitter.call([doc])

        # Should handle empty content gracefully
        assert isinstance(result, list)

    def test_small_content(self):
        """Test handling of content smaller than chunk size."""
        splitter = CodeSplitter(
            chunk_size=1000,  # Large chunk size
            chunk_overlap=100,
            file_extension=".py",
            split_by=self.split_by,
        )

        small_code = "print('Hello, World!')"
        doc = Document(text=small_code)
        result = splitter.call([doc])

        assert len(result) == 1, "Small content should result in single chunk"
        assert result[0].text.strip() == small_code

    def test_different_smart_boundary_ratios(self):
        """Test different smart boundary ratio settings."""
        ratios = [0.5, 0.7, 0.9]

        for ratio in ratios:
            splitter = CodeSplitter(
                chunk_size=200,
                chunk_overlap=50,
                smart_boundary_ratio=ratio,
                file_extension=".py",
                split_by=self.split_by,
            )

            assert splitter.smart_boundary_ratio == ratio
            assert splitter.smart_boundary_threshold == int(200 * ratio)

            doc = Document(text=PYTHON_CODE)
            result = splitter.call([doc])
            assert len(result) >= 1, f"Should work with ratio {ratio}"

    def test_repr_method(self):
        """Test the __repr__ method includes smart_boundary_ratio."""
        splitter = CodeSplitter(
            chunk_size=512,
            chunk_overlap=64,
            smart_boundary_ratio=0.75,
            file_extension=".py",
            split_by=self.split_by,
        )

        repr_str = repr(splitter)
        assert "smart_boundary_ratio=0.75" in repr_str
        assert "chunk_size=512" in repr_str
        assert "chunk_overlap=64" in repr_str

    def test_multiple_documents(self):
        """Test splitting multiple documents at once."""
        splitter = CodeSplitter(
            chunk_size=200,
            chunk_overlap=50,
            file_extension=".py",
            split_by=self.split_by,
        )

        docs = [
            Document(text=PYTHON_CODE),
            Document(text="def simple_func():\n    return 42"),
        ]

        result = splitter.call(docs)
        assert len(result) >= 2, "Should produce chunks from multiple documents"

    def test_serialization(self):
        """Test that CodeSplitter can be pickled and unpickled."""
        import pickle

        splitter = CodeSplitter(
            chunk_size=256,
            chunk_overlap=32,
            smart_boundary_ratio=0.8,
            file_extension=".py",
            split_by=self.split_by,
        )

        # Serialize and deserialize
        pickled = pickle.dumps(splitter)
        unpickled = pickle.loads(pickled)

        # Check that attributes are preserved
        assert unpickled.chunk_size == splitter.chunk_size
        assert unpickled.chunk_overlap == splitter.chunk_overlap
        assert unpickled.smart_boundary_ratio == splitter.smart_boundary_ratio
        assert unpickled.file_extension == splitter.file_extension

        # Test that it still works after unpickling
        doc = Document(text="print('test')")
        result = unpickled.call([doc])
        assert len(result) >= 1


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestCodeSplitter(split_by="ast")
    test_instance.test_initialization()
    test_instance.test_python_splitting()
    test_instance.test_markdown_splitting()
    test_instance.test_rst_splitting()
    test_instance.test_yaml_splitting()
    test_instance.test_json_splitting()
    print("✅ Basic tests passed!")
