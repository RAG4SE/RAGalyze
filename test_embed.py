from ragalyze.rag.rag import RAG
from ragalyze.configs import *
from ragalyze.query import build_context, save_query_results

import json

set_global_config_value("repo_path", "./bench/test_var_from_other_file_python")
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
        context, _ = build_context(
            retrieved_doc=doc,
            id2doc=rag.id2doc,
            direction="both",
            count=count,
        )
        contexts.append(context)

    prompt = """
There is a `print(a)` in the main.py, find the definition of a.

You need to find the definition of the a printed in `print(a)`

Return JSON:
{
    "inference": "The inference of the definition"
    "definition": "The full definition or declaration of a",
    "is_complete": "Yes or No",
}

The is_complete should be Yes if the definition is complete, otherwise No.

If you cannot find the definition statement, return
{
    "inference": None,
    "definition": None,
    "is_complete": None,
}
    """

    prompt = """
Tell me the file structure of the repo
    """
    result = rag.query(prompt, contexts)
    save_query_results(
        result=result,
        bm25_keywords="a",
        faiss_query="the definition of a",
        question=prompt,
    )
    # print(result["response"])
    data = json.loads(result["response"])
    if data.get("definition") is None:
        print("definition is none")
        continue
    if data.get("is_complete") == "No":
        print(f"is not complete")
        continue
    print(data["definition"])
    break
