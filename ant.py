from ragalyze import *

repo_path = "/path/to/repository"
question = """
Your question here.
"""
dict_config = load_default_config()
dict_config.generator.provider = "lingxi"
dict_config.generator.model = "qwen3-coder-480b-a35b-instruct"
dict_config.rag.embedder.provider = "local_server"
dict_config.rag.embedder.model = "text-embedding-qwen3-embedding-8b"
dict_config.rag.code_understanding.provider = "lingxi"
dict_config.rag.code_understanding.model = "qwen3-8b"
# retriever的top_k越大，传入大模型的代码片段越多，思考依据更多，但同时消耗更多token
dict_config.rag.retriever.top_k = 20
# bm25 weight越大，rag在搜索中更倾向于稀疏检索而非稠密检索。稀疏检索更依赖于关键词匹配，而稠密检索则更依赖于语义理解。因此，如果问题关于查找而非分析，可以把weight调大，最大至1.
dict_config.rag.retriever.bm25.weight = 0.5
# 按需向量化的前提是用一个合适的检索方式搜索出大量代码片段，然后在此基础上进一步检索，query_driven.top_k指定了第一次检索的量
dict_config.rag.query_driven.top_k = 100
set_global_configs(dict_config)

result = query_repository(
    repo_path=repo_path,
    question=question,
)

print_result(result)
save_query_results(result, repo_path, question)
