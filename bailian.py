from ragalyze import *

repo_path = "/Users/mac/repo/solidity"
question = \
"""
Tell me the implementation of `EVMDialect::builtin`
"""

question = \
"""
List all the function bodies that contain function call(s) to `EVMDialect::builtin`
"""

dict_config = load_default_config()
# retriever的top_k越大，传入大模型的代码片段越多，思考依据更多，但同时消耗更多token
dict_config.rag.retriever.top_k = 30
# bm25 weight越大，rag在搜索中更倾向于稀疏检索而非稠密检索。稀疏检索更依赖于关键词匹配，而稠密检索则更依赖于语义理解。因此，如果问题关于查找而非分析，可以把weight调大，最大至1.
dict_config.rag.retriever.bm25.weight = 1.0
# 按需向量化的前提是用一个合适的检索方式搜索出大量代码片段，然后在此基础上进一步检索，query_driven.top_k指定了第一次检索的量
dict_config.rag.query_driven.top_k = 150
# 排序策略
dict_config.rag.retriever.fusion = "normal_add"
# dict_config.rag.embedder.force_embedding = True
load_all_configs(dict_config)

result = query_repository(
    repo_path=repo_path,
    question=question,
)

print_result(result)
save_query_results(result, repo_path, question)