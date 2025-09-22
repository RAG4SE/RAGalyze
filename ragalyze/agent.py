import json
from typing import List, Dict
from copy import deepcopy
from time import sleep
from json_repair import repair_json

from ragalyze.rag.rag import RAG, GeneratorWrapper
from ragalyze.query import print_result, save_query_results, build_context
from ragalyze.configs import *
from ragalyze.prompts import *
from ragalyze.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class FetchCallerHeaderAgent:

    def __init__(self, debug: bool = False):
        self.repo_path = configs()["repo_path"]
        self.debug = debug

    def __call__(self, callee_name: str, callee_body: str = "") -> List[str]:
        prompt = FETCH_CALLER_HEADERS_TEMPLATE.call(
            callee_name=callee_name,
            callee_body=callee_body,
        )
        bm25_keywords = [f"[CALL]{callee_name}"]
        faiss_query = ""

        # set_global_config_value("generator.json_output", "true")
        generator = GeneratorWrapper()

        rag = RAG()
        caller_headers = []
        retrieved_docs = rag.retrieve(
            bm25_keywords=" ".join(bm25_keywords), faiss_query=faiss_query
        )[0].documents
        for doc in retrieved_docs:
            count = 0
            while count <= 30:
                context, cannot_extend = build_context(
                    retrieved_doc=doc,
                    id2doc=rag.id2doc,
                    direction="previous",
                    count=count,
                )

                response = rag.query(input_str=prompt, contexts=[context])

                if self.debug:
                    save_query_results(response, bm25_keywords, faiss_query, prompt)
                reply = response["response"]
                format_prompt = OUTPUT_FORMAT_TEMPLATE.call(
                    reply=reply,
                    output_format=r"""Response in json format such as
{
    "function_headers": [function_header_1, function_header_2, function_header_3, ...]
}
If you cannot find any functions that call {{callee_name}}, reply
{
    "function_headers": "None"
}
""",
                )
                reply = generator(input_str=format_prompt).data.strip()
                data = json.loads(repair_json(reply))
                if data["function_headers"] == "None":
                    count += 3
                else:
                    caller_headers.extend(data["function_headers"])
                    break
                if cannot_extend:
                    break
                sleep(0.5)  # avoid rate limit

        if not caller_headers:
            logger.warning(f"Cannot find caller names for {callee_name}")

        return list(set(caller_headers))

class FetchCallerNameFromHeaderAgent:
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def __call__(self, header: str) -> List[str]:
        prompt = FETCH_CALLER_NAMES_FROM_HEADER_TEMPLATE.call(
            header=header,
        )
        generator = GeneratorWrapper()
        reply = generator(input_str=prompt).data.strip()
        if (reply == "None"):
            logger.warning(f"Cannot find caller name for {header}")
            return None
        data = json.loads(repair_json(reply))
        return data["full_name"], data["short_name"]

class FetchCallerNameAgent:

    def __init__(self, debug: bool = False):
        self.repo_path = configs()["repo_path"]
        self.debug = debug

    def __call__(self, callee_name: str, callee_body: str = "") -> List[str]:
        prompt = FETCH_CALLER_NAMES_TEMPLATE.call(
            callee_name=callee_name,
            callee_body=callee_body,
        )
        bm25_keywords = [f"[CALL]{callee_name}"]
        faiss_query = ""

        # set_global_config_value("generator.json_output", "true")
        generator = GeneratorWrapper()

        rag = RAG()
        caller_names = []
        retrieved_docs = rag.retrieve(
            bm25_keywords=" ".join(bm25_keywords), faiss_query=faiss_query
        )[0].documents
        for doc in retrieved_docs:
            count = 0
            while count <= 30:
                context, cannot_extend = build_context(
                    retrieved_doc=doc,
                    id2doc=rag.id2doc,
                    direction="previous",
                    count=count,
                )

                response = rag.query(input_str=prompt, contexts=[context])

                if self.debug:
                    save_query_results(response, bm25_keywords, faiss_query, prompt)
                reply = response["response"]
                format_prompt = OUTPUT_FORMAT_TEMPLATE.call(
                    reply=reply,
                    output_format=r"""Response in json format such as
{
    "caller_names": [caller_name_1, caller_name_2, caller_name_3, ...]
}
If you cannot find any functions that call {{callee_name}}, reply
{
    "caller_names": "None"
}
""",
                )
                reply = generator(input_str=format_prompt).data.strip()
                data = json.loads(repair_json(reply))
                if data["caller_names"] == "None":
                    count += 3
                else:
                    caller_names.extend(data["caller_names"])
                    break
                if cannot_extend:
                    break
                sleep(0.5)  # avoid rate limit

        if not caller_names:
            logger.warning(f"Cannot find caller names for {callee_name}")

        return list(set(caller_names))
