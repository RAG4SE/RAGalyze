from ragalyze import *


repo_path = "/Users/mac/repo/RAGalyzeBench/solidity"
repo_path = "/Users/mac/repo/test-solidity/"

set_global_config_value("repo_path", repo_path)
# set_global_config_value("repo.file_filters.extra_excluded_patterns", ["*deps/*", "*test*/*", "workspace/*", "*/tags"])
set_global_config_value("rag.retriever.bm25.top_k", 25)
set_global_config_value("rag.retriever.query_driven", True)
set_global_config_value("rag.retriever.bm25.weight", 0.5)
set_global_config_value("rag.retriever.fusion", "normal_add")
set_global_config_value("rag.recreate_db", True)


bm25_keywords = "[CALL]builtin"
question = "list all functions that call builtin"
result = query_repository(
    bm25_keywords=bm25_keywords, faiss_query="", question=question
)
print_result(result)
save_query_results(result, bm25_keywords, "", question)
