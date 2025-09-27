import json
from typing import List, Dict, Tuple
from copy import deepcopy
from time import sleep
from json_repair import repair_json

from ragalyze.rag.rag import RAG, GeneratorWrapper
from ragalyze.query import print_result, save_query_results, build_context
from ragalyze.configs import *
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from adalflow import Prompt

logger = get_tqdm_compatible_logger(__name__)


# Base prompt template that can be extended for various tasks
PROMPT_TEMPLATE = Prompt(
    template=r"""{{task_description}}
{% if context %}
{{context}}
{% endif %}
{% if instructions %}
Before you start, please follow the instructions below:
{{instructions}}
{% endif %}
{% if output_format %}
---------- Output Format ----------
You MUST output EXACTLY in this format, no extra text, no introduction, no summary.
{{output_format}}
{% endif %}
{% if example %}
---------- Examples ----------
Here are some examples:
{{example}}
{% endif %}
"""
)

OUTPUT_FORMAT_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description=r"""
Help extract info from LLM's reply and summarize them in a well-organized format.
{% if info %}
Extract the following information from the reply:
{{info}}
{% endif %}

<REPLY>
{{reply}}
</REPLY>

<OUTPUT_FORMAT>
{{output_format}}
</OUTPUT_FORMAT>
"""
    )
)

class FetchCallerHeaderPipeline:

    """
    Given a function call expr E, return the function header of the function that calls E
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to:
1. Identify all **function definitions** whose **function body contains call(s)** to the function named `{{callee_name}}`.
2. Return the **function header** of these **function definitions**.

{% if callee_body %}
To help disambiguate overloaded functions, here is the definition of the function named `{{callee_name}}`:
{{callee_body}}
Use this to judge whether a call in the code is actually calling this specific overload, when possible.
{% endif %}
            """,
        instructions = r"""
1. Locate every call to the target function.
2. Up-scan from that line until you reach the outermost function header that contains the call.
3. If the snippet is truncated and no header is found, the header is indeterminable—header missing.
4. If the found header is incomplete (e.g., missing return type, parameters, or template parameters), the header is indeterminable—header missing.
5. Skip any local lambdas or nested functions; return the first non-local header. 
6. First determine the language, then check if the header is complete for that language. If not, return drop it.
            """
        )
    )

    def __init__(self, debug: bool = False):
        self.repo_path = configs()["repo_path"]
        self.debug = debug
        if self.debug:
            print('FetchCallerHeaderPipeline')

    def __call__(self, callee_name: str, callee_body: str = "") -> List[str]:
        prompt = self.prompt_template.call(
            callee_name=callee_name,
            callee_body=callee_body,
        )
        bm25_keywords = [f"[CALL]{callee_name}"]
        faiss_query = ""

        set_global_config_value("generator.json_output", True)
        generator = GeneratorWrapper()
        set_global_config_value("generator.json_output", False)
        rag = RAG()
        caller_headers = []
        retrieved_docs = rag.retrieve(
            bm25_keywords=" ".join(bm25_keywords), faiss_query=faiss_query
        )[0].documents
        for doc in retrieved_docs:
            set_global_config_value("generator.json_output", False)
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
                    info="valid function headers that are confirmed by the LLM as header candidates",
                    output_format=r"""Response in json format such as
{
    "function_headers": [function_header_1, function_header_2, function_header_3, ...]
}
If you cannot find any functions that call {{callee_name}}, reply
{
    "function_headers": None
}
""",
                )
                reply = generator(input_str=format_prompt).data.strip()
                if self.debug:
                    generator.save_result()
                data = json.loads(repair_json(reply))
                if not data["function_headers"]:
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

class FetchCallerNameFromHeaderQuery:
    
    """
    Given a function header, return the full_name and short_name of the function
    """
    
    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Extract full_name and short_name from the function header {{header}}:

1. full_name: 
- Includes complete scope (namespace/class prefixes like A::B::)
- Preserves function names, operator names, etc, following the namespace
- Maintains template parameters when present (e.g., func<T>)

2. short_name:
- Removes all scope prefixes (no namespace/class qualifiers)
- Preserves function names, operator names, etc, following the namespace
- Omits template parameters (e.g., func instead of func<T>)
    """,
            output_format=r"""Return JSON:
{"full_name":"...", "short_name":"..."}
Return "None" for invalid function headers
    """,
    example=r"""
- "void MyNamespace::MyClass::doSomething(int)" → {"full_name":"MyNamespace::MyClass::doSomething", "short_name":"doSomething"}
- "bool Container::operator==(const Container&)" → {"full_name":"Container::operator==", "short_name":"operator=="}
- "T Collection::operator[](size_t) const" → {"full_name":"Collection::operator[]", "short_name":"operator[]"}
- "template <typename T> T Calculator::add(T a, T b)" → {"full_name":"Calculator::add<T>", "short_name":"add"}
    """
        )
    )
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        if self.debug:
            print('FetchCallerNameFromHeaderQuery')
    
    def __call__(self, header: str) -> List[str]:
        prompt = self.prompt_template.call(
            header=header,
        )
        set_global_config_value("generator.json_output", True)
        generator = GeneratorWrapper()
        set_global_config_value("generator.json_output", False)
        reply = generator(input_str=prompt).data.strip()
        if self.debug:
            generator.save_result()
        if (reply == "None"):
            logger.warning(f"Cannot find caller name for {header}")
            return None
        data = json.loads(repair_json(reply))
        return data["full_name"], data["short_name"]

class ExtractCallerNameAgent:

    """
    Given a function call expr E, return the function name of the function that calls E
    """

    def __init__(self, debug: bool = False):
        self.header_agent = FetchCallerHeaderPipeline(debug=debug)
        self.name_from_header_agent = FetchCallerNameFromHeaderQuery(debug=debug)
        self.debug = debug

    def __call__(self, callee_name: str, callee_body: str = "") -> List[Tuple[str, str]]:
        """
        Args:
            callee_name (str): The name of the callee function.
            callee_body (str, optional): The body of the callee function. Defaults to "".

        Returns:
            List[Tuple[str, str]]: The full name and short name of the caller functions.
        """
        caller_headers = self.header_agent(callee_name, callee_body)

        if not caller_headers:
            if self.debug:
                logger.warning(f"No caller headers found for {callee_name}")
            return []

        caller_names  = []
        for header in caller_headers:
            result = self.name_from_header_agent(header)
            if result:
                full_name, short_name = result
                caller_names.append((full_name, short_name))

        if not caller_names:
            logger.warning(f"Cannot extract caller names from headers for {callee_name}")

        return list(set(caller_names))

#TODO: FetchCallExprFromCallerBodyAgent



#TODO: _ExtractCalleeName


class IsClassInstanceQuery:
    """
    Check if a function call implicitly uses overloaded operator(), __call__, etc.
    """
    
    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Given the code snippet {{context}}, determine if the callee expression the name of whic is `{{callee_name}}` is a call to a class instance.
{% if call_expr %}
Function call expression for {{callee_name}}: {{call_expr}}
{% endif %}
            """,
            output_format=r"""Return JSON:
{"is_class_instance": true/false/?, "class": "class_name"}
If you can determine whether the callee_name is a class instance, return true or false.
If you cannot determine, return "?".
If it is a class instance, return the class name in the "class" field.
            """,
        )
    )
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        if self.debug:
            print('IsClassInstanceQuery')
    
    def __call__(self, context: str, callee_name: str, call_expr: str = '') -> bool:
        """
        Check if the callee_name is a class instance.

        Args:
            context (str): The context where the function call is made.
            callee_name (str): The name of the callee function.

        Returns:
            bool: True if the callee_name is a class instance, False otherwise.
        """
        prompt = self.prompt_template.call(
            context=context,
            callee_name=callee_name,
            call_expr=call_expr,
        )
        set_global_config_value("generator.json_output", True)
        generator = GeneratorWrapper()
        set_global_config_value("generator.json_output", False)
        reply = generator(input_str=prompt).data.strip()
        if self.debug:
            generator.save_result()
        data = json.loads(repair_json(reply))
        if data["is_class_instance"] == "?":
            logger.error(f'Cannot determine if {callee_name} is a class instance. Check if {callee_name} is in {context}')
            raise
        return data["is_class_instance"], data["class"]


class FetchClassPipeline:
    """
    Given a class name, return the class definition.
    """
    
    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to find the **complete** class definition for the class named `{{class_name}}`. 
            """,
            output_format=r"""If the class definition cannot be found or the class definition is **not complete**, return json:
{"class": None}
Otherwise, return json:
{"class": class_definition}
            """,
        )
    )
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        if self.debug:
            print('FetchClassPipeline')

    def __call__(self, class_name: str) -> str:
        """
        Args:
            class_name (str): The name of the class.

        Returns:
            str: The class definition.
        """
        prompt = self.prompt_template.call(
            class_name=class_name,
        )
        bm25_keywords = [f"[CLASS]{class_name}"]
        faiss_query = ""
        set_global_config_value("generator.json_output", True)
        rag = RAG()
        set_global_config_value("generator.json_output", False)

        retrieved_docs = rag.retrieve(
            bm25_keywords=" ".join(bm25_keywords), faiss_query=faiss_query
        )[0].documents

        if not retrieved_docs:
            if self.debug:
                logger.warning(f"No documents found for {class_name} definition")
            return None

        for doc in retrieved_docs:
            count = 0
            while count <= 30:
                context, cannot_extend = build_context(
                    retrieved_doc=doc,
                    id2doc=rag.id2doc,
                    direction="next",
                    count=count,
                )

                response = rag.query(input_str=prompt, contexts=[context])

                if self.debug:
                    save_query_results(response, bm25_keywords, faiss_query, prompt)

                reply = response["response"]
                data = json.loads(repair_json(reply))
                if data.get("class"):
                    return data["class"]

                if cannot_extend:
                    break
                count += 3
                sleep(0.5)  # avoid rate limit

        return None

class FindCallOperatorQuery:
    """
    Given a class definition body, return the Call Operator definition. This is required since some languages contain over-design language sugars:(
    """
    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a class definition body: {{class_definition}}
Your task is to find the **complete** Call Operator definition in the class definition body.
**Complete** here means the Call Operator definition is **not just a function declaration**, but contain the **function body**.
Call Operator is a function that can make the class instance callable, such as `operator()` in C++ and `__call__` in Python

            """
        )
    )
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        if self.debug:
            print('FindCallOperatorQuery')
    
    def __call__(self, class_definition: str) -> str:
        """
        Args:
            class_name (str): The name of the class.
            class_definition (str): The class definition body.
        
        """
        
        prompt = self.prompt_template.call(
            class_definition=class_definition,
        )
        set_global_config_value("generator.json_output", False)
        generator = GeneratorWrapper()
        reply = generator(input_str=prompt).data.strip()
        set_global_config_value("generator.json_output", True)
        generator = GeneratorWrapper()
        set_global_config_value("generator.json_output", False)
        format_prompt = OUTPUT_FORMAT_TEMPLATE.call(
            reply=reply,
            info="Whether or not the call operator definition exists",
            output_format=r"""If the Call Operator definition cannot be found or the Call Operator definition is **not complete**, return json:
{"call_operator": None}
Otherwise, return json:
{"call_operator": call_operator_definition}
            """,
        )
        reply = generator(input_str=format_prompt).data.strip()
        if self.debug:
            generator.save_result()
        data = json.loads(repair_json(reply))
        return data["call_operator"]

class FetchFunctionDefinitionAgent:
    
    """
    Given a function call expr E, return the function definition of the function called in E.
    """
    
    """TODO
    since function call expr is diverse, containing regular function call, constructor call, member function call, functor call, etc.
    This agent cannot cover all scenarios. For instance, if the function call is a functor call, its definition is presented as
    class_name::operator()(args), or simply operator() inside the class body. But in Python, the definition is __call__(args) inside the
    class body. These differences require a rounter for different language. Currently, we only plan to support C/C++, Java, and Python.
    """
    
    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to find all function definition candidates for the function named `{{callee_name}}`.

{% if callee_expr %}
Function call expression for {{callee_name}}: {{callee_expr}}
{% endif %}

{% if caller_body %}
To help disambiguate overloaded functions, here is the calling context of `{{callee_name}}`:
{{caller_body}}
Use this to determine which specific overload or template specialization is being called.
{% endif %}
""",
            instructions=r"""
1. Identify the **complete function definition** for the function named `{{callee_name}}`.
2. If multiple definition candidates exist (e.g., overloads, template specializations), analyze which ones are actually relevant to the function call `{{callee_name}}` and return all relevant candidates.
3. Ensure the definition includes the complete function header and body, for each candidate, print out if it is complete or not.
"""
        )
    )

    def __init__(self, debug: bool = False):
        self.repo_path = configs()["repo_path"]
        self.debug = debug
        if self.debug:
            print('FetchFunctionDefinitionAgent')

    def __call__(self, callee_name: str, callee_expr: str = "", caller_body: str = "") -> Dict[str, str]:
        """
        Args:
            callee_name (str): The name of the callee function.
            callee_expr (str, optional): The expression of the callee function. Defaults to "".
            caller_body (str, optional): The body of the caller function to help disambiguate. Defaults to "".

        Returns:
            Dict[str, str]: Dictionary containing file_path and function_definition.
        """
        is_class_instance, class_name = IsClassInstanceQuery(self.debug)(context=caller_body, callee_name=callee_name, call_expr=callee_expr)
        if is_class_instance:
            class_definition = FetchClassPipeline(self.debug)(class_name)
            if not class_definition:
                raise ValueError(f'Cannot find class definition for {class_name}.')
            call_operator = FindCallOperatorQuery(self.debug)(class_definition=class_definition)
            if call_operator:
                return [call_operator]
            else:
                # !WARNING: only supports out-of-class call operator in C++ style
                # @haoyang9804: support more language styles
                bm25_keywords = [f"[FUNCDEF]{class_name}::operator()"]
                prompt = self.prompt_template.call(
                    callee_name=f"{class_name}::operator()",
                )
        else:        
            bm25_keywords = [f"[FUNCDEF]{callee_name}"]
            prompt = self.prompt_template.call(
                callee_name=callee_name,
                callee_expr=callee_expr,
                caller_body=caller_body,
            )
        faiss_query = ""
        set_global_config_value("generator.json_output", True)
        generator = GeneratorWrapper()
        set_global_config_value("generator.json_output", False)
        rag = RAG()
        retrieved_docs = rag.retrieve(
            bm25_keywords=" ".join(bm25_keywords), faiss_query=faiss_query
        )[0].documents

        if not retrieved_docs:
            logger.error(f"No documents found for {function_name} definition")
            raise

        function_definitions = []

        for doc in retrieved_docs:
            count = 0
            while count <= 30:
                context, cannot_extend = build_context(
                    retrieved_doc=doc,
                    id2doc=rag.id2doc,
                direction="next",
                    count=count,
                )

                response = rag.query(input_str=prompt, contexts=[context])

                if self.debug:
                    save_query_results(response, bm25_keywords, faiss_query, prompt)

                reply = response["response"]
                format_prompt = OUTPUT_FORMAT_TEMPLATE.call(
                    reply=f'Here is the reply:\n {reply}. ',
                    info="valid function definitions that are confirmed by the LLM as definition candidates",
                    output_format=r"""Response in json format such as
{
    "function_definition": "[Function definition 1, Function definition 2, ...]",
    "complete": "[yes|no for Function definition 1, yes|no for Function definition 2, ...]"
}
""",
                )
                reply = generator(input_str=format_prompt).data.strip()
                if self.debug:
                    generator.save_result()
                if reply == 'None':
                    raise ValueError(f"FetchFunctionDefinitionAgent: Cannot find definition for {function_name}, possibly because the bm25 index is incorrect.")
                data = json.loads(repair_json(reply))
                if data.get("function_definition") != [] and all([complete == "yes" for complete in data.get("complete", [])]):
                    function_definitions.extend(data["function_definition"])
                    break

                if cannot_extend:
                    break
                count += 3
                sleep(0.5)  # avoid rate limit

        return function_definitions


