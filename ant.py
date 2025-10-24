from ragalyze.configs import *
from ragalyze.agent import *

set_global_config_value("repo_path", "")
set_global_config_value("rag.recreate_db", True)

set_global_config_value("rag.embedder.provider", "local_server")
set_global_config_value("rag.embedder.model", "text-embedding-qwen3-embedding-8b")

set_global_config_value("rag.code_understanding.provider", "lingxi")
set_global_config_value("rag.code_understanding.model", "qwen3-8b")

set_global_config_value("generator.provider", "lingxi")
set_global_config_value("generator.model", "qwen3-coder-480b-a35b-instruct")
set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")

set_global_config_value("rag.retriever.bm25.top_k", 100)


r = FindAllFuncHeaderPipeline(debug=True)
all_func_infos = r()
print(all_func_infos)
r = FunctionCallExtractorFromEntryFuncsPipeline(debug=True)
call_chain_forest = r(all_func_infos)
call_chain_forest.serialize("call_chain_forest.pkl")
call_chain_forest.write2file_call_chains("call_chains.txt")
call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
call_chain_forest.print_call_chains()