# RAGalyze ([中文 README](README_zh.md))

Repository Analysis and Query Toolchain with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Repository Analysis**: Analyze code repositories and extract meaningful information
- **RAG-powered Q&A**: Ask questions about your codebase and get intelligent answers
- **Dual Usage**: Use as a Python library or command-line tool
- **Configurable**: Flexible configuration system using Hydra
- **Advanced Retrieval**: Supports dual-vector embedding (code + semantic interpretation) and hybrid search (BM25 + FAISS) for enhanced precision
- **Performance Optimized**: C extension for BM25 tokenization to bypass Python GIL limitations
- **Language-Aware Analysis**: Integrates a fork of [multilspy](https://github.com/RAG4SE/multilspy) to provide language-server powered reasoning across repositories

## Installation

well-tested under Python 3.11

### From Source

```bash
git clone git@github.com:RAG4SE/RAGalyze.git
cd RAGalyze
pip install -r requirements.txt
python python setup.py build_ext --inplace
pip install -e .
```

## Before Start

### Prerequisite 0: Install the multilspy Fork

RAGalyze depends on a forked version of `multilspy` for language-server powered analysis. Install it before running the advanced workflows:

```bash
git clone https://github.com/RAG4SE/multilspy.git
cd multilspy
pip install -r requirements.txt
pip install .
```

### Prerequisite 1: Set up API keys of LLM providers

RAGalyze requires API keys from LLM providers to function properly. The tool has been thoroughly tested with the following model configurations:

**Embedding Models:**
- Huggingface: `intfloat/multilingual-e5-large-instruct`
- Dashscope: `text-embedding-v4`

**Question Answering Models:**
- Dashscope: `qwen-plus`, `qwen-max`, `qwen-turbo`
- Google: `gemini-2.5-flash-preview-05-20`

**Code Understanding Model (for dual-vector embedding):**
- Dashscope: `qwen3-32b`

> **Note**: Support for additional mainstream models and providers is planned for future releases.

You can refer to the following coarse-grained table to get the corresponding API keys and store them into your env.

| Provider | Required Environment Variables | How to Get API Key |
|----------|-------------------------------|-------------------|
| **Google Gemini** | `GOOGLE_API_KEY` | [Get API Key](https://aistudio.google.com/app/apikey) |
| **OpenAI** | `OPENAI_API_KEY` | [Get API Key](https://platform.openai.com/api-keys) |
| **DeepSeek** | `DEEPSEEK_API_KEY` | [Get API Key](https://platform.deepseek.com/api_keys) |
| **Dashscope (Qwen)** | `DASHSCOPE_API_KEY`, `DASHSCOPE_WORKSPACE_ID` (optional) | [Get API Key and WorkSpace](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.6ebe48238qeoit&tab=api#/api)  |
| **SiliconFlow** | `SILICONFLOW_API_KEY` | [Get API Key](https://cloud.siliconflow.cn/i/api-keys) |

## Quick Start

### Analyze function call chains from the function entry with the agent

One of the main functions of `RAGalyze` is extracting function call chains.
Below is an example of how to extract all function call chains starting from the entry function.

```python
from ragalyze import *
from ragalyze.configs import *
from ragalyze.agent import *
from ragalyze.rag import retriever
from pathlib import Path
import os
import traceback

set_global_config_value("repo_path", "<The path to the repo you want to analyze>")
set_global_config_value("repo.file_filters.extra_excluded_patterns", ["*txt"]) # You don't care for text files in this repo
set_global_config_value("rag.recreate_db", True) # RAGalyze can cache, recreate_db means ignore the cache, rebuild the database and fill in the cache. This only happens when you want to debug or the repo has been updated.
set_global_config_value("generator.provider", "dashscope") # The LLM provider
set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")

# For other settings you can tune, refer to .yaml files in configs/ folder. 
# Tune the settings using set_global_config_value function with respect to hydra grammar.

r = FindAllFuncHeaderPipeline(debug=True)
all_func_infos = r() # Find all functions in the repo, this only happens when there is no entry function or every function can be entry function.
r = FunctionCallExtractorFromEntryFuncsPipeline(debug=True) # Treat each collected function as the entry, and collect call chains starting from it.
call_chain_forest = r(all_func_infos)
call_chain_forest.serialize("call_chain_forest.pkl") # cache the result
call_chain_forest.write2file_call_chains("call_chains.txt")
call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
call_chain_forest.print_call_chains()

r = FunctionCallExtractorFromEntryFuncWithLSPPipeline(debug=True) 
call_chain_tree = r('initWallet', 'parity_wallet_bug_1.sol', 222) # Different from the above exaple, this example shows how to extract function call chains starting from the specified function
call_chain_tree.print_nocode_call_chains()

```

Swap out the `repo_path` and generator settings for your project before running the script.

<!-- ### Query the repo like deepwiki

RAGalyze can perform like deepwiki: you use natural languages to query about the repo to catch its functions.

```python
from ragalyze.rag.rag import RAG
from ragalyze.configs import *
from ragalyze.query import build_context, save_query_results

import json

set_global_config_value("repo_path", "<The path to the repo you want to analyze>")
set_global_config_value("rag.recreate_db", True)

set_global_config_value("generator.provider", "dashscope")
set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")
set_global_config_value("generator.json_output", True)
rag = RAG(embed=True)
retrieved_docs = rag.retrieve(bm25_keywords="a", faiss_query="the definition of a")[
    0
].documents

contexts = []

for count in [5, 10, 15]:
    for doc in retrieved_docs:
        # build_context can combine adjacent code chunks to enrich the context
        context, _ = build_context(
            retrieved_doc=doc,
            id2doc=rag.id2doc,
            direction="both",
            count=count,
        )
        contexts.append(context)

    prompt = "Tell me the file structure of the repo"
    result = rag.query(prompt, contexts)
    save_query_results(
        result=result,
        bm25_keywords="<YOUR BM25 Keywords>", # If you want to query about 
        faiss_query="the definition of a",
        question=prompt,
    )
    data = json.loads(result["response"])
    if data.get("definition") is None:
        print("definition is none")
        continue
    if data.get("is_complete") == "No":
        print(f"is not complete")
        continue
    print(data["definition"])
    break
```

Customize the retrieval keywords, FAISS query, and prompt structure to match your task. Updating the generator or embedding provider only requires changing the configuration calls at the top of the script. -->


## Acknowledgements

[deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open).
