# RAGalyze

RAGalyze 是一个基于 RAG（检索增强生成）的代码仓库分析与问答工具链。

## 功能特性

- **仓库分析**：解析代码仓库并提取关键信息
- **RAG 问答**：针对代码库提出问题并获得智能回答
- **双重用法**：既可以作为 Python 库，也可以作为命令行工具
- **高度可配置**：通过 Hydra 提供灵活的配置体系
- **高级检索**：支持双向向量嵌入（代码 + 语义）和混合检索（BM25 + FAISS），精度更高
- **性能优化**：提供 BM25 词法分析的 C 扩展以绕过 Python GIL
- **语言感知分析**：集成来自 [multilspy](https://github.com/RAG4SE/multilspy) 的 fork，提供基于语言服务器的推理能力

## 安装

该项目在 Python 3.11 环境下经过充分测试。

### 从源码安装

```bash
git clone git@github.com:RAG4SE/RAGalyze.git
cd RAGalyze
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
```

## 使用前准备

### 前置条件 0：安装 multilspy 分支

RAGalyze 依赖一个 fork 版本的 `multilspy` 来获得语言服务器能力。请先完成如下操作：

```bash
git clone https://github.com/RAG4SE/multilspy.git
cd multilspy
pip install -r requirements.txt
pip install .
```

### 前置条件 1：配置大模型服务的 API Key

RAGalyze 需要配置多个大模型服务的 API Key。我们主要在以下模型配置上进行测试：

**向量嵌入模型：**
- Huggingface：`intfloat/multilingual-e5-large-instruct`
- Dashscope：`text-embedding-v4`

**问答模型：**
- Dashscope：`qwen-plus`、`qwen-max`、`qwen-turbo`
- Google：`gemini-2.5-flash-preview-05-20`

**代码理解模型（用于双向向量嵌入）：**
- Dashscope：`qwen3-32b`

> **说明**：未来会支持更多主流模型与服务商。

下表列出了各服务商所需的环境变量及申请方式：

| 服务商 | 所需环境变量 | 申请方式 |
|--------|--------------|----------|
| Google Gemini | `GOOGLE_API_KEY` | [申请链接](https://aistudio.google.com/app/apikey) |
| OpenAI | `OPENAI_API_KEY` | [申请链接](https://platform.openai.com/api-keys) |
| DeepSeek | `DEEPSEEK_API_KEY` | [申请链接](https://platform.deepseek.com/api_keys) |
| Dashscope (Qwen) | `DASHSCOPE_API_KEY`、`DASHSCOPE_WORKSPACE_ID`（可选） | [申请链接](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.6ebe48238qeoit&tab=api#/api) |
| SiliconFlow | `SILICONFLOW_API_KEY` | [申请链接](https://cloud.siliconflow.cn/i/api-keys) |

## 快速上手

### 使用 Agent 从入口函数分析调用链

RAGalyze 的核心功能之一是提取函数调用链。下面的示例展示了如何从入口函数开始，构建整个调用链。

```python
from ragalyze import *
from ragalyze.configs import *
from ragalyze.agent import *
from ragalyze.rag import retriever
from pathlib import Path
import os
import traceback

set_global_config_value("repo_path", "<The path to the repo you want to analyze>")
set_global_config_value("repo.file_filters.extra_excluded_patterns", ["*txt"]) # 不需要处理文本文件
set_global_config_value("rag.recreate_db", True) # 重新构建数据库和缓存，通常在调试或仓库变更时启用
set_global_config_value("generator.provider", "dashscope") # 指定 LLM 提供商
set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")

# 更多可调配置参见 configs/ 目录下的 .yaml 文件
# 使用 set_global_config_value 并遵循 hydra 语法修改配置

r = FindAllFuncHeaderPipeline(debug=True)
all_func_infos = r() # 遍历仓库中的所有函数，适用于没有明确入口的情况
r = FunctionCallExtractorFromEntryFuncsPipeline(debug=True) # 让每个函数都作为入口，提取对应的调用链
call_chain_forest = r(all_func_infos)
call_chain_forest.serialize("call_chain_forest.pkl") # 缓存执行结果
call_chain_forest.write2file_call_chains("call_chains.txt")
call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
call_chain_forest.print_call_chains()

r = FunctionCallExtractorFromEntryFuncWithLSPPipeline(debug=True) 
call_chain_tree = r('initWallet', 'parity_wallet_bug_1.sol', 222) # 指定入口函数，提取以该函数为起点的调用链
call_chain_tree.print_nocode_call_chains()
```

在运行脚本之前，请将示例中的 `repo_path` 和模型配置替换为你自己的项目设置。

<!-- ### 像 deepwiki 一样查询仓库

RAGalyze 也可以像 deepwiki 那样，通过自然语言查询仓库以了解其功能。

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
        # build_context 可以将相邻代码片段拼接，以丰富上下文
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
        bm25_keywords="<YOUR BM25 Keywords>",
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

你可以根据需求调整检索关键词、FAISS 查询语句以及提示词结构。只需修改顶部的配置即可切换不同的模型或嵌入器。 -->

## 鸣谢

[deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open).
