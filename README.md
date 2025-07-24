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
pip install -r requirements.txt
```

### 2. **API Keys Required**

RAGalyze requires API keys from LLM providers. Choose one or more providers and set up the corresponding API keys:

| Provider | Required Environment Variables | How to Get API Key |
|----------|-------------------------------|-------------------|
| **Google Gemini** | `GOOGLE_API_KEY` | [Get API Key](https://aistudio.google.com/app/apikey) |
| **OpenAI** | `OPENAI_API_KEY` | [Get API Key](https://platform.openai.com/api-keys) |
| **DeepSeek** | `DEEPSEEK_API_KEY` | [Get API Key](https://platform.deepseek.com/api_keys) |
| **Dashscope (Qwen)** | `DASHSCOPE_API_KEY`, `DASHSCOPE_WORKSPACE_ID` (optional) | [Get API Key and WorkSpace](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.6ebe48238qeoit&tab=api#/api)  |
| **SiliconFlow** | `SILICONFLOW_API_KEY` | [Get API Key](https://cloud.siliconflow.cn/i/api-keys) |

**Note**: 
- For **code understanding features**, you need at least one Dashscope API key since we only support dashscope for this task.
- For **general RAG queries**, you can use any supported provider
- Dashscope workspace ID is optional but required when you use Dashscope's embedding model, e.g., `text-embedding-v4` provided by Qwen Lab

### 3. **LLM Provider Setup**

```bash
cd api/config
```

Modify `generator.json` to select your favorite LLM, and export the corresponding API keys:

```bash
# Google Gemini
export GOOGLE_API_KEY="your_api_key"

# OpenAI
export OPENAI_API_KEY="your_api_key"

# DeepSeek
export DEEPSEEK_API_KEY="your_api_key"

# Dashscope (Qwen) - Required for code understanding
export DASHSCOPE_API_KEY="your_api_key"
export DASHSCOPE_WORKSPACE_ID="your_workspace_id"  # Optional

# SiliconFlow
export SILICONFLOW_API_KEY="your_api_key"
```

### 4. **Simple Usage (Standalone)**

```python
from client import analyze_repository, ask_question

# Analyze a repository
result = analyze_repository("/path/to/your/repo")
print(f"Analyzed {result['document_count']} documents")

# Ask questions
answer = ask_question("/path/to/your/repo", "What does this project do?")
print(answer['answer'])
```

### 5. **Command Line Usage**

```bash
# Analyze a repository and ask a question
python client.py /path/to/repo "What is the main functionality?"

# Interactive Solidity analysis
python analyze_solidity_repo.py /path/to/solidity/repo
```

### 6. **Web Server Mode with Browser Interface**

```bash
# Start the server (configuration is loaded from server_config.json)
python -m server.main

# Use the web interface (recommended)
# Open browser: http://localhost:8000/web (or the configured host:port)
```

### 7. **Server Configuration**

The server is configured using the `server_config.json` file in the project root directory:

```json
{
    "server": {
        "host": "localhost",
        "port": 8000,
        "reload": false,
        "log_level": "info",
        "mode": "subprocess"
    }
}
```

Modify this file to change the server settings. The available options are:

- `host`: The host to bind to (default: "localhost")
- `port`: The port to bind to (default: 8000)
- `reload`: Enable auto-reload for development (default: false)
- `log_level`: Log level (default: "info")
- `mode`: Server execution mode: "subprocess" (better process management) or "direct" (simpler) (default: "subprocess")

### 8. **Custom Configuration Path**

By default, the server looks for `server_config.json` in the project root directory. You can specify a custom path using the environment variable:

```bash
export RAGalyze_SERVER_CONFIG="/path/to/your/server_config.json"
python -m server.main
```

```bash
# Ask questions via client (appears in web interface)
python client.py /path/to/repo "How does this work?"
```

Then visit http://localhost:8000/web to check the history and retrieval log.

## Contributions

[CONTRIBUTION.md](CONTRIBUTION.md)

## Acknowledgements

[deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open).