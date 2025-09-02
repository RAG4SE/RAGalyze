from ragalyze import *

repo_path = "/path/to/repository"
question = \
"""
Your question here.
"""
dict_config = load_default_config()
dict_config.generator.provider = "lingxi"
dict_config.generator.model = "qwen3-coder-480b-a35b-instruct"
dict_config.rag.embedder.provider = "local_server"
dict_config.rag.embedder.model = "text-embedding-qwen3-embedding-8b"
dict_config.rag.code_understanding.provider = "lingxi"
dict_config.rag.code_understanding.model = "qwen3-8b"
load_all_configs(dict_config)

result = query_repository(
    repo_path=repo_path,
    question=question,
)

print_result(result)
save_query_results(result, repo_path, question)