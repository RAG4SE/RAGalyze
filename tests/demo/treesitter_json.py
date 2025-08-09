import sys
from tree_sitter import Parser, Node, Language
import tree_sitter_json as tsjson

def print_node(node: Node, source_code: bytes, indent: int = 0) -> None:
    """Recursively print all nodes in the AST with their details"""
    # Get node type and text content
    node_type = node.type
    node_text = source_code[node.start_byte:node.end_byte].decode("utf-8").strip()
    
    # Format output with indentation
    indent_str = "  " * indent
    print(f"{indent_str}[{node_type}]")
    # print(f"{indent_str}  Range: {node.start_byte}-{node.end_byte}")
    if node_type == 'document':
        print(f"{indent_str}  Text: {node_text}")  # Truncate long text
    # print(f"{indent_str}  ---")
    
    # Recurse into child nodes
    for child in node.children:
        print_node(child, source_code, indent + 1)

def parse_markdown(markdown_text: str) -> None:
    """Parse Markdown text and print the AST nodes"""
    # Initialize parser with Markdown grammar
    language = Language(tsjson.language())
    parser = Parser(language)
    
    # Convert text to bytes (required by Tree-sitter)
    source_bytes = markdown_text.encode("utf-8")
    
    # Parse into AST
    tree = parser.parse(source_bytes)
    root_node = tree.root_node
    
    print("Markdown AST Nodes:")
    print("====================")
    print_node(root_node, source_bytes)

if __name__ == "__main__":
    # Example Markdown content
    sample_json = """
{
  "application": {
    "name": "RAGalyze",
    "version": "1.0.0",
    "description": "Advanced RAG system with intelligent document processing",
    "author": {
      "name": "Development Team",
      "email": "dev@ragalyze.com",
      "organization": "RAGalyze Inc."
    },
    "license": "MIT",
    "repository": {
      "type": "git",
      "url": "https://github.com/ragalyze/ragalyze.git",
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
      "origins": ["http://localhost:3000", "https://app.ragalyze.com"],
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
        "filename": "/var/log/ragalyze.log",
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
          "recipients": ["admin@ragalyze.com", "ops@ragalyze.com"],
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
    
    # Parse and print nodes
    parse_markdown(sample_json)
