from ragalyze.rag.rag import RAG
from ragalyze.configs import *
from ragalyze.query import build_context

set_global_config_value("repo_path", "./bench/test_var_from_other_file_python")
# set_global_config_value("rag.recreate_db", True)

set_global_config_value("generator.provider", "dashscope")
set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")
set_global_config_value("generator.json_output", True)
rag = RAG(embed=True)
retrieved_docs = rag.retrieve(
    bm25_keywords="UserManager", faiss_query="the definition of UserManager"
)[0].documents

contexts = []

for doc in retrieved_docs:
    context, _ = build_context(
        retrieved_doc=doc,
        id2doc=rag.id2doc,
        direction="both",
        count=0,
    )
    contexts.append(context)

prompt = """
Return JSON:
{
    "definition": "The full definition or declaration of UserManager"
}
If you cannot find the definition statement, return
{
    "definition": None
}
"""

result = rag.query(prompt, contexts)
print(result["response"])
