# RAGalyze 🚀

A powerful RAG (Retrieval-Augmented Generation) system with advanced text splitting capabilities.

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies

#### Install Pandoc (Required for RST document processing)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y pandoc
```

**macOS:**
```bash
brew install pandoc
```

**Windows:**
- Download and install from [Pandoc releases](https://github.com/jgm/pandoc/releases)
- Or use chocolatey: `choco install pandoc`

**Note:** Pandoc is required for the `UnstructuredRstTextSplitter` to work properly. It's used by the `unstructured` library to convert RST documents to HTML for parsing.

## Features

- **Advanced Text Splitting**: Support for multiple document formats (Markdown, RST, JSON, YAML, Code, etc.)
- **Intelligent RST Processing**: Uses `unstructured` library with Pandoc for semantic RST document splitting
- **Hybrid Retrieval**: Combines multiple retrieval strategies for better results
- **Configurable Pipeline**: Flexible configuration system with Hydra

## TODO
1. Support hybrid text splitting for markdown

## Acknowledgements

[deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open).