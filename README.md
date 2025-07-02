# RAGalyze 🚀

**Advanced Code Repository Analysis with RAG (Retrieval-Augmented Generation)**

RAGalyze is a powerful AI-powered tool that analyzes local code repositories and enables natural language querying about codebases. It uses advanced embedding techniques and multiple LLM providers to provide intelligent answers about your code.

## ✨ Features

### 🔍 **Intelligent Code Analysis**
- **Multi-language Support**: Analyzes Python, JavaScript, TypeScript, Java, C++, Go, Rust, Solidity, and more
- **Smart Document Processing**: Automatically chunks and processes code files with proper context preservation
- **Advanced Embedding**: Uses HuggingFace's `multilingual-e5-large-instruct` model for embedding, and faiss for high-quality semantic search

### 🤖 **Multiple LLM Provider Support**
- **Google Gemini**: gemini-2.5-flash, gemini-2.5-pro
- **OpenAI**: GPT-4o, GPT-4.1, o1, o3, o4-mini
- **DeepSeek**: deepseek-chat, deepseek-reasoner
- **Dashscope (Qwen)**: qwen-max, qwen-plus, qwen-turbo
- **SiliconFlow**: DeepSeek-V3, Llama-3.1-405B, Qwen2.5-72B

### 🔄 **Dual Vector Pipeline**
- **Code Embeddings**: Direct embedding of source code
- **Understanding Embeddings**: AI-generated code explanations for better semantic matching
- **Hybrid Retrieval**: Combines both approaches for optimal results

### 🌐 **Multiple Usage Modes**
- **Standalone Scripts**: Direct command-line analysis
- **Web API Server**: RESTful API with FastAPI
- **Web Interface**: Beautiful browser-based UI to view query results and retrieved code snippets
- **Simple Client Library**: Easy-to-use Python functions

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd RAGalyze-open

# Install dependencies
pip install -r requirements_server.txt
```

### 2. **LLM Provider Setup**

```bash
cd api/config
```

Modify `generator.json` to select your favorate LLM, and export the corresponding API

```bash
# Google Gemini
export GOOGLE_API_KEY="your_api_key"

# OpenAI
export OPENAI_API_KEY="your_api_key"

# DeepSeek
export DEEPSEEK_API_KEY="your_api_key"

# Dashscope (Qwen)
export DASHSCOPE_API_KEY="your_api_key"

# SiliconFlow
export SILICONFLOW_API_KEY="your_api_key"
```

### 3. **Simple Usage (Standalone)**

```python
from client import analyze_repository, ask_question

# Analyze a repository
result = analyze_repository("/path/to/your/repo")
print(f"Analyzed {result['document_count']} documents")

# Ask questions
answer = ask_question("/path/to/your/repo", "What does this project do?")
print(answer['answer'])
```

### 4. **Command Line Usage**

```bash
# Analyze a repository and ask a question
python client.py /path/to/repo "What is the main functionality?"

# Interactive Solidity analysis
python analyze_solidity_repo.py /path/to/solidity/repo
```

### 5. **Web Server Mode with Browser Interface**

```bash
# Start the server
python server.py --port 8000

# Use the web interface (recommended)
# Open browser: http://localhost:8000/web

# Ask questions via client (appears in web interface)
python client.py /path/to/repo "How does this work?"
```

Then visit http://localhost:8000/web to check the history and retrieval log.


## Acknowledgements

[deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open).