from ragalyze import *

repo_path = "/Users/mac/repo/solidity"
repo_path = "/home/lyr/solidity"

question = """
EVMDialect::builtin
"""

dict_config = load_default_config()
# retriever的top_k越大，传入大模型的代码片段越多，思考依据更多，但同时消耗更多token
dict_config.rag.retriever.top_k = 10
# bm25 weight越大，rag在搜索中更倾向于稀疏检索而非稠密检索。稀疏检索更依赖于关键词匹配，而稠密检索则更依赖于语义理解。因此，如果问题关于查找而非分析，可以把weight调大，最大至1.
dict_config.rag.retriever.bm25.weight = 1.0
# 按需向量化的前提是用一个合适的检索方式搜索出大量代码片段，然后在此基础上进一步检索，query_driven.top_k指定了第一次检索的量
dict_config.rag.query_driven.top_k = 50
# 排序策略
dict_config.rag.retriever.fusion = "normal_add"

# dict_config.rag.embedder.force_embedding = True

set_global_configs(dict_config)
# result = query_repository(
#     repo_path=repo_path,
#     question=question,
# )


structured_question = {
    "what_to_find": "EVMDialect::builtin",
    "what_to_do": """
List every function **definition** (not declaration)  
where the **body** contains a **CallExpr** to  
`EVMDialect::builtin( … )`  
- exclude any occurrence inside  
  - return statements  
  - function signatures  
  - comments or string literals  
For each hit, output only the file name of this function + function signature + its body.  
If no such caller exists, reply “None”.
""",
}

result = query_repository_with_format_find_then_do(
    repo_path=repo_path, question=structured_question
)

print_result(result)
save_query_results(result, repo_path, question)
