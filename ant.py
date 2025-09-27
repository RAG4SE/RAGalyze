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

set_global_config_value("rag.retriever.bm25.top_k", 25)


agent = FetchCallerHeaderPipeline(debug=True)

print(
    agent(
        callee_name="builtin",
        # callee_body="""
        # BuiltinFunctionForEVM const& EVMDialect::builtin(BuiltinHandle const& _handle) const
        # """,
    )
)
