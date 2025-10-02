import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from copy import deepcopy
from time import sleep
from json_repair import repair_json

from ragalyze.rag.rag import RAG, GeneratorWrapper
from ragalyze.query import print_result, save_query_results, build_context
from ragalyze.configs import *
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from adalflow import Prompt

logger = get_tqdm_compatible_logger(__name__)


_CALL_DEBUG_DEPTH = 0
COUNT_UPPER_LIMIT = 30
"""
CALL_CHAIN_WIDTH: the width of the call chain, i.e., the number of callees the agent will
explore in each step.
"""
CALL_CHAIN_WIDTH = 3


def _format_details(info: Dict[str, Any]) -> str:
    parts = []
    for key, value in info.items():
        if value is None:
            continue
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


@contextmanager
def call_debug_scope(class_name: str, debug: bool, action: str = "__call__", **info: Any):
    global _CALL_DEBUG_DEPTH
    active = bool(debug)
    if active:
        _CALL_DEBUG_DEPTH += 1
        depth = _CALL_DEBUG_DEPTH
        details = _format_details(info)
        message = f"{depth}. {class_name}.{action}"
        if details:
            message += f" [{details}]"
        logger.info(message)
    try:
        yield
    finally:
        if active:
            _CALL_DEBUG_DEPTH = max(0, _CALL_DEBUG_DEPTH - 1)


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
Extract the following information from the reply: {{info}}
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


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VulnerabilityType(str, Enum):
    BUFFER_OVERFLOW = "buffer_overflow"
    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RACE_CONDITION = "race_condition"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_LEAK = "resource_leak"
    GENERIC = "generic"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestType(str, Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    SECURITY = "security"
    FUZZ = "fuzz"


@dataclass
class FunctionKindInfo:
    """Information about different kinds of member functions in classes."""
    kind: str
    description: str


# Core member function types found in classes
FUNCTION_KINDS = [
    FunctionKindInfo(
        kind="constructor",
        description="Initializes object state when an instance is created"
    ),
    FunctionKindInfo(
        kind="destructor",
        description="Cleans up resources when an object is destroyed"
    ),
    FunctionKindInfo(
        kind="member_method",
        description="Regular instance method that operates on object state"
    ),
    FunctionKindInfo(
        kind="operator",
        description="Overloaded operators (+, -, *, /, [], (), etc.)"
    ),
    FunctionKindInfo(
        kind="static_method",
        description="Method that operates on class level without requiring instance"
    ),
    FunctionKindInfo(
        kind="virtual_method",
        description="Method that can be overridden in derived classes"
    ),
    FunctionKindInfo(
        kind="pure_virtual",
        description="Abstract method with declaration only, no implementation"
    ),
    FunctionKindInfo(
        kind="const_method",
        description="Method that promises not to modify object state"
    ),
    FunctionKindInfo(
        kind="template_method",
        description="Method with template parameters for generic programming"
    ),
    FunctionKindInfo(
        kind="conversion_operator",
        description="Method that converts object to other types"
    )
]


@dataclass
class CodeLocation:
    file_path: Optional[str] = None
    line: int = 0
    column: int = 1


@dataclass
class Parameter:
    name: str
    type_hint: Optional[str] = None


@dataclass
class FunctionCall:
    function_name: str
    full_name: Optional[str] = None
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None
    call_site: Optional[CodeLocation] = None
    is_virtual: bool = False
    is_overloaded: bool = False
    template_args: List[str] = field(default_factory=list)
    context: Optional[str] = None
    call_expr: Optional[str] = None


@dataclass
class CallChain:
    entry_function: str
    functions: List[FunctionCall] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    language: Optional[str] = None
    file_path: Optional[str] = None

    
def infer_language_from_definition(definition: Optional[str]) -> Optional[str]:
    if not definition:
        return None
    snippet = definition.strip()
    header = snippet.splitlines()[0] if snippet else ""
    if header.startswith("def ") or " def " in header:
        return "python"
    if "::" in header or header.startswith("template<"):
        return "cpp"
    if header.startswith("function ") or "=>" in snippet:
        return "javascript"
    if snippet.startswith("class ") and "public:" in snippet:
        return "cpp"
    return "c"


def deduplicate_function_calls(calls: Iterable[FunctionCall]) -> List[FunctionCall]:
    deduped: List[FunctionCall] = []
    seen: Set[str] = set()
    for call in calls:
        key = f"{call.function_name}|{call.call_expr or ''}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def deduplicate_strings(items: Iterable[str]) -> List[str]:
    unique: List[str] = []
    seen: Set[str] = set()
    for item in items or []:
        if not item:
            continue
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


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

    def __call__(self, callee_name: str, callee_body: str = "") -> List[str]:
        with call_debug_scope(self.__class__.__name__, self.debug, callee=callee_name):
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
                while count <= COUNT_UPPER_LIMIT:
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
                        info="valid function headers",
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
    
    def __call__(self, header: str) -> List[str]:
        with call_debug_scope(self.__class__.__name__, self.debug, header=header):
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
        with call_debug_scope(self.__class__.__name__, self.debug, callee=callee_name):
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
class IsClassInstanceQuery:
    """
    Check if a function call implicitly uses overloaded operator(), __call__, etc.
    """
    
    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Given the code snippet 
{{context}}
, determine if the callee expression the name of which is `{{callee_name}}` is a call to a class instance.
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
    
    def __call__(self, context: str, callee_name: str, call_expr: str = '') -> bool:
        """
        Check if the callee_name is a class instance.

        Args:
            context (str): The context where the function call is made.
            callee_name (str): The name of the callee function.

        Returns:
            bool: True if the callee_name is a class instance, False otherwise.
        """
        with call_debug_scope(self.__class__.__name__, self.debug, callee=callee_name):
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

    def __call__(self, class_name: str) -> str:
        """
        Args:
            class_name (str): The name of the class.

        Returns:
            str: The class definition.
        """
        with call_debug_scope(self.__class__.__name__, self.debug):
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
                while count <= COUNT_UPPER_LIMIT:
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
    
    def __call__(self, class_definition: str) -> str:
        """
        Args:
            class_name (str): The name of the class.
            class_definition (str): The class definition body.
        
        """
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            action="__call__",
            class_length=len(class_definition) if class_definition else 0,
        ):
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


class AnalyzeCallExpressionPipeline:
    """Classifies a call expression and suggests lookup hints for definition retrieval."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are analyzing a function call inside some source code.

Call expression: {{call_expr}}
Function name token: {{callee_name}}

Review the caller body to understand the surrounding declarations and determine how the call is made.
            """,
            instructions=r"""
Classify the call expression using one of the following types only:
- simple_function
- namespaced_function
- member_method (instance method invoked via `.` or `->`)
- static_method (invoked via `Class::method`)
- call_operator (invokes `operator()` / `__call__` on an object)
- constructor
- destructor
- template_function (explicit template arguments such as `func<int>` or `obj.template func<int>`)
- function_pointer (plain function pointer or callable variable)
- macro
- unknown

Return strict JSON with these keys:
{
  "call_type": TYPE,
  "qualified_names": [ordered list of most specific fully-qualified names to try],
  "search_terms": [optional extra retrieval keywords],
  "class_name": optional class name related to the target,
  "method_name": optional method identifier (without class qualifiers),
  "needs_class_definition": true/false indicating whether inspecting the class body could help,
  "template_arguments": [template arguments if any, else empty list],
  "language_hint": optional language hint (e.g. "cpp", "python"),
  "notes": optional short note if helpful
}

Always provide `qualified_names` (at least the callee name). Only include fields that apply; use null when information is unavailable. Do not add extra keys.
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, callee_name: str, call_expr: str, caller_body: str) -> Dict[str, Any]:
        with call_debug_scope(self.__class__.__name__, self.debug, callee=callee_name):
            prompt = self.prompt_template.call(
                callee_name=callee_name,
                call_expr=call_expr or callee_name,
                caller_body=caller_body or "",
            )
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            if self.debug:
                generator.save_result()
            set_global_config_value("generator.json_output", False)

        try:
            data = json.loads(repair_json(reply))
        except Exception as exc:
            logger.error("AnalyzeCallExpressionPipeline: JSON parse failed for %s: %s", callee_name, exc)
            return {
                "call_type": "unknown",
                "qualified_names": [callee_name],
                "search_terms": [],
                "class_name": None,
                "method_name": None,
                "needs_class_definition": False,
                "template_arguments": [],
                "language_hint": None,
                "notes": None,
            }

        # Ensure fallback values exist for downstream processing
        data.setdefault("call_type", "unknown")
        qualified = data.get("qualified_names")
        if not isinstance(qualified, list) or not qualified:
            data["qualified_names"] = [callee_name]
        search_terms = data.get("search_terms", [])
        if isinstance(search_terms, str):
            data["search_terms"] = [search_terms]
        elif not isinstance(search_terms, list):
            data["search_terms"] = []
        template_args = data.get("template_arguments", [])
        if isinstance(template_args, str):
            data["template_arguments"] = [template_args]
        elif not isinstance(template_args, list):
            data["template_arguments"] = []
        if "needs_class_definition" not in data:
            data["needs_class_definition"] = False

        return data


class ExtractMemberDefinitionFromClassQuery:
    """Extracts inline member, constructor, or destructor definitions from a class body.

    Can extract a specific function by name, all functions, or functions filtered by kind.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a complete class definition:
{{class_definition}}

{% if target_name %}
Your task is to locate the definition of function named `{{target_name}}`.
{% else %}
Your task is to locate ALL function definitions in the class.
{% endif %}

{% if kind %}
Only return functions that are of kind: {{kind}}
{% endif %}

CRITICAL: You MUST return the COMPLETE function definition including:
- ALL attributes (e.g., [[nodiscard]], [[deprecated]], [[noreturn]])
- ALL specifiers (e.g., virtual, static, const, constexpr, inline)
- COMPLETE function signature with all parameters and return types
- FULL function body with all statements
- NO modifications or stripping of any parts

Only return definitions that include the full body (not merely a declaration).
            """,
            output_format=r"""Return strict JSON:
{% if target_name %}
{"definition": definition_text_or_null, "is_complete": true/false}
If the definition is missing or incomplete, set "definition" to null and "is_complete" to false.
{% else %}
{"functions": [{"name": "function_name", "definition": definition_text, "kind": "function_kind", "is_complete": true/false}, ...]}
If no functions are found, return an empty array.
{% endif %}
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    """
    A known bug: if kind is the same as a function name in the class_definition, then it may only return the function definition with that name.
    For instance, there are more than one **const** functions, one of which is named "const_method". If the kind is "const_method", then it may only return the "const_method" function.
    """
    def __call__(self, class_definition: str, target_name: Optional[str] = None, kind: Optional[str] = None) -> Union[Optional[str], List[Dict[str, Any]]]:
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            target=target_name or "all",
            kind=kind,
        ):
            prompt = self.prompt_template.call(
                class_definition=class_definition,
                target_name=target_name,
                kind=kind,
            )
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            if self.debug:
                generator.save_result()
            set_global_config_value("generator.json_output", False)

        try:
            data = json.loads(repair_json(reply))
        except Exception as exc:
            logger.error(
                "ExtractMemberDefinitionFromClassQuery: JSON parse failed for target=%s, kind=%s: %s",
                target_name or "all",
                kind or "all",
                exc,
            )
            return None if target_name else []

        # Handle single function extraction (target_name specified)
        if target_name:
            definition = data.get("definition")
            is_complete = data.get("is_complete")
            if definition and is_complete:
                return definition
            return None

        # Handle multiple functions extraction (no target_name specified)
        functions = data.get("functions", [])
        valid_functions = []

        for func_data in functions:
            definition = func_data.get("definition")
            is_complete = func_data.get("is_complete")
            if definition and is_complete:
                valid_functions.append(func_data)

        return valid_functions


class FetchReturnTypeQuery:
    """Determines the return type from a function header or full definition snippet."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a function signature or definition snippet:
{{function_snippet}}

{% if language %}
The function is written in {{language}}.
{% endif %}

Identify the return type of the function as it would appear in code.
            """,
            instructions=r"""
Return strict JSON:
{
  "return_type": string|None,
  "confidence": "high"|"medium"|"low"
}
- If the function has no return type (e.g., constructors, Python functions with implicit None), return None.
- Include qualifiers such as `const`, `static`, `constexpr`, pointer/reference symbols, namespaces, templates, etc., as they appear in the signature.
- If you cannot infer the return type, return null and set confidence to "low".
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, function_snippet: str, language: str = "") -> Optional[str]:
        with call_debug_scope(self.__class__.__name__, self.debug, snippet_length=len(function_snippet or "")):
            if not function_snippet or not function_snippet.strip():
                return None

            prompt = self.prompt_template.call(function_snippet=function_snippet, language=language)
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            if self.debug:
                generator.save_result()
            set_global_config_value("generator.json_output", False)

        try:
            data = json.loads(repair_json(reply))
        except Exception as exc:
            logger.error("FetchReturnTypeQuery: JSON parse failed: %s", exc)
            return None

        return_type = data.get("return_type")
        if isinstance(return_type, str):
            normalized = return_type.strip()
            return normalized or None
        return None

class FetchCallExpressionsQuery:
    """Uses the LLM to extract direct call expressions from a function body."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Extract function call expressions from the code.
Code:
{{function_body}}
            """,
            output_format=r"""Return strict JSON with this structure:
{
  "call_exprs": [
    CALL_EXPR_1,
    CALL_EXPR_2,
    ...
  ]
}
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, function_body: str, current_function: str) -> List[Dict[str, Any]]:
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            function=current_function,
            length=len(function_body) if function_body else 0,
        ):
            if not function_body.strip():
                return []

            prompt = self.prompt_template.call(
                current_function=current_function or "",
                function_body=function_body,
            )
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            set_global_config_value("generator.json_output", False)
            if self.debug:
                generator.save_result()

            try:
                data = json.loads(repair_json(reply))
            except Exception as exc:
                logger.error(f"FetchCallExpressionsQuery: JSON parse failed: {exc}")
                return []

            call_exprs = data.get("call_exprs") or []
            if not isinstance(call_exprs, list):
                return []

            result: List[Dict[str, Any]] = []
            for call_expr in call_exprs:
                result.append({
                    "call_expr": call_expr,
                    "current_function": current_function,
                    "function_body": function_body,
                })
            return result

    
class FetchFunctionDefinitionFromCallAgent:
    
    """
    Given a function call expr E or the funciton name inside, return the function definition of the function called in E.
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
        self.call_expr_analyzer = AnalyzeCallExpressionPipeline(debug=debug)
        self.class_pipeline = FetchClassPipeline(debug=debug)
        self.call_operator_query = FindCallOperatorQuery(debug=debug)
        self.class_member_extractor = ExtractMemberDefinitionFromClassQuery(debug=debug)

    def __call__(
        self,
        callee_name: str,
        callee_expr: str = "",
        caller_body: str = "",
        file_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Args:
            callee_name (str): The name of the callee function.
            callee_expr (str, optional): The expression of the callee function. Defaults to "".
            caller_body (str, optional): The body of the caller function to help disambiguate. Defaults to "".
            file_path (str, optional): Relative repo path to constrain retrieval.

        Returns:
            Dict[str, str]: Dictionary containing file_path and function_definition.
        """
        with call_debug_scope(self.__class__.__name__, self.debug, callee=callee_name):
            analysis = self.call_expr_analyzer(
                callee_name=callee_name,
                call_expr=callee_expr or callee_name,
                caller_body=caller_body or "",
            )

            call_type = analysis.get("call_type", "unknown")
            qualified_names: List[str] = analysis.get("qualified_names", [])
            search_terms: List[str] = analysis.get("search_terms", [])
            class_name: Optional[str] = analysis.get("class_name")
            method_name: Optional[str] = analysis.get("method_name")

            definitions: List[str] = []

            if call_type == "call_operator":
                definitions.extend(
                    self._resolve_call_operator(
                        analysis=analysis,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        file_path=file_path,
                    )
                )
            elif call_type == "member_method":
                definitions.extend(
                    self._resolve_member_method(
                        analysis=analysis,
                        fallback_name=callee_name,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        file_path=file_path,
                    )
                )
            elif call_type == "static_method":
                definitions.extend(
                    self._resolve_static_method(
                        analysis=analysis,
                        fallback_name=callee_name,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        file_path=file_path,
                    )
                )
            elif call_type == "constructor":
                definitions.extend(
                    self._resolve_constructor(
                        analysis=analysis,
                        fallback_name=callee_name,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        file_path=file_path,
                    )
                )
            elif call_type == "destructor":
                definitions.extend(
                    self._resolve_destructor(
                        analysis=analysis,
                        fallback_name=callee_name,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        file_path=file_path,
                    )
                )
            elif call_type == "template_function":
                definitions.extend(
                    self._resolve_template_function(
                        analysis=analysis,
                        fallback_name=callee_name,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        file_path=file_path,
                    )
                )

            # Fallback when specialised strategies do not yield results
            if not definitions:
                fallback_candidates = list(dict.fromkeys((qualified_names or []) + [callee_name]))
                definitions.extend(
                    self._collect_from_candidates(
                        candidate_names=fallback_candidates,
                        callee_expr=callee_expr,
                        caller_body=caller_body,
                        search_terms=search_terms,
                        file_path=file_path,
                    )
                )

            return deduplicate_strings(definitions)

    def _resolve_call_operator(
        self,
        analysis: Dict[str, Any],
        callee_expr: str,
        caller_body: str,
        file_path: Optional[str],
    ) -> List[str]:
        class_name = analysis.get("class_name")
        definitions: List[str] = []
        if class_name:
            class_definition = self.class_pipeline(class_name)
            if class_definition:
                operator_definition = self.call_operator_query(class_definition=class_definition)
                if operator_definition:
                    definitions.append(operator_definition)

        candidate_names = []
        if class_name:
            candidate_names.append(f"{class_name}::operator()")
        candidate_names.extend(analysis.get("qualified_names", []))

        definitions.extend(
            self._collect_from_candidates(
                candidate_names=candidate_names,
                callee_expr=callee_expr,
                caller_body=caller_body,
                search_terms=analysis.get("search_terms", []),
                file_path=file_path,
            )
        )
        return deduplicate_strings(definitions)

    def _resolve_member_method(
        self,
        analysis: Dict[str, Any],
        fallback_name: str,
        callee_expr: str,
        caller_body: str,
        file_path: Optional[str],
    ) -> List[str]:
        class_name = analysis.get("class_name")
        method_name = analysis.get("method_name") or fallback_name
        definitions: List[str] = []

        if analysis.get("needs_class_definition") and class_name:
            class_definition = self.class_pipeline(class_name)
            if class_definition:
                #! @haoyang9804: change it later
                inline_definition = self.class_member_extractor(
                    class_definition=class_definition,
                    target_name=method_name,
                )[0]
                if inline_definition:
                    definitions.append(inline_definition)

        candidate_names = []
        candidate_names.extend(analysis.get("qualified_names", []))
        if class_name and method_name:
            candidate_names.append(f"{class_name}::{method_name}")

        definitions.extend(
            self._collect_from_candidates(
                candidate_names=candidate_names or [fallback_name],
                callee_expr=callee_expr,
                caller_body=caller_body,
                search_terms=analysis.get("search_terms", []),
                file_path=file_path,
            )
        )
        return deduplicate_strings(definitions)

    def _resolve_static_method(
        self,
        analysis: Dict[str, Any],
        fallback_name: str,
        callee_expr: str,
        caller_body: str,
        file_path: Optional[str],
    ) -> List[str]:
        candidate_names = []
        candidate_names.extend(analysis.get("qualified_names", []))
        class_name = analysis.get("class_name")
        method_name = analysis.get("method_name") or fallback_name
        if class_name and method_name:
            candidate_names.append(f"{class_name}::{method_name}")

        return self._collect_from_candidates(
            candidate_names=candidate_names or [fallback_name],
            callee_expr=callee_expr,
            caller_body=caller_body,
            search_terms=analysis.get("search_terms", []),
            file_path=file_path,
        )

    def _resolve_constructor(
        self,
        analysis: Dict[str, Any],
        fallback_name: str,
        callee_expr: str,
        caller_body: str,
        file_path: Optional[str],
    ) -> List[str]:
        class_name = analysis.get("class_name") or fallback_name
        definitions: List[str] = []
        if class_name:
            class_definition = self.class_pipeline(class_name)
            if class_definition:
                constructor_def = self.class_member_extractor(
                    class_definition=class_definition,
                    target_name=class_name,
                    target_kind="constructor",
                )
                if constructor_def:
                    definitions.append(constructor_def)

        candidate_names = []
        candidate_names.extend(analysis.get("qualified_names", []))
        if class_name:
            candidate_names.append(f"{class_name}::{class_name}")
            candidate_names.append(class_name)

        definitions.extend(
            self._collect_from_candidates(
                candidate_names=list(dict.fromkeys(candidate_names or [fallback_name])),
                callee_expr=callee_expr,
                caller_body=caller_body,
                search_terms=analysis.get("search_terms", []),
                file_path=file_path,
            )
        )
        return deduplicate_strings(definitions)

    def _resolve_destructor(
        self,
        analysis: Dict[str, Any],
        fallback_name: str,
        callee_expr: str,
        caller_body: str,
        file_path: Optional[str],
    ) -> List[str]:
        class_name = analysis.get("class_name") or fallback_name.lstrip("~")
        definitions: List[str] = []
        if class_name:
            class_definition = self.class_pipeline(class_name)
            if class_definition:
                destructor_def = self.class_member_extractor(
                    class_definition=class_definition,
                    target_name=f"~{class_name}",
                    target_kind="destructor",
                )
                if destructor_def:
                    definitions.append(destructor_def)

        candidate_names = []
        candidate_names.extend(analysis.get("qualified_names", []))
        if class_name:
            candidate_names.append(f"{class_name}::~{class_name}")
            candidate_names.append(f"~{class_name}")

        definitions.extend(
            self._collect_from_candidates(
                candidate_names=list(dict.fromkeys(candidate_names or [fallback_name])),
                callee_expr=callee_expr,
                caller_body=caller_body,
                search_terms=analysis.get("search_terms", []),
                file_path=file_path,
            )
        )
        return deduplicate_strings(definitions)

    def _resolve_template_function(
        self,
        analysis: Dict[str, Any],
        fallback_name: str,
        callee_expr: str,
        caller_body: str,
        file_path: Optional[str],
    ) -> List[str]:
        candidate_names = []
        candidate_names.extend(analysis.get("qualified_names", []))
        if fallback_name not in candidate_names:
            candidate_names.append(fallback_name)

        # Include the raw call expression as a search term to capture template arguments.
        search_terms = list(dict.fromkeys((analysis.get("search_terms", []) or []) + ([callee_expr] if callee_expr else [])))

        return self._collect_from_candidates(
            candidate_names=candidate_names,
            callee_expr=callee_expr,
            caller_body=caller_body,
            search_terms=search_terms,
            file_path=file_path,
        )

    def _collect_from_candidates(
        self,
        candidate_names: Iterable[str],
        callee_expr: str,
        caller_body: str,
        search_terms: Optional[Iterable[str]],
        file_path: Optional[str],
    ) -> List[str]:
        collected: List[str] = []
        seen: Set[str] = set()
        for candidate in candidate_names:
            if not candidate:
                continue
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            bm25_terms = [f"[FUNCDEF]{normalized}"]
            for term in search_terms or []:
                term_str = str(term).strip()
                if term_str and term_str not in bm25_terms:
                    bm25_terms.append(term_str)

            try:
                definitions = self._run_retrieval(
                    callee_display_name=normalized,
                    callee_expr=callee_expr,
                    caller_body=caller_body,
                    bm25_terms=bm25_terms,
                    file_path=file_path,
                )
            except Exception as exc:
                if self.debug:
                    logger.warning(
                        "FetchFunctionDefinitionFromCallAgent: retrieval failed for %s: %s",
                        normalized,
                        exc,
                    )
                continue

            for definition in definitions:
                if definition not in collected:
                    collected.append(definition)

        return collected

    def _run_retrieval(
        self,
        callee_display_name: str,
        callee_expr: str,
        caller_body: str,
        bm25_terms: Iterable[str],
        file_path: Optional[str],
    ) -> List[str]:
        prompt = self.prompt_template.call(
            callee_name=callee_display_name,
            callee_expr=callee_expr,
            caller_body=caller_body,
        )
        faiss_query = ""
        terms_list = [str(term).strip() for term in bm25_terms if str(term).strip()]
        if not terms_list:
            terms_list = [callee_display_name]
        set_global_config_value("generator.json_output", True)
        generator = GeneratorWrapper()
        set_global_config_value("generator.json_output", False)
        rag = RAG()
        retrieved = rag.retrieve(
            bm25_keywords=" ".join(terms_list),
            faiss_query=faiss_query,
        )
        documents = retrieved[0].documents if retrieved else []

        if file_path:
            filtered_docs = [
                doc
                for doc in documents
                if getattr(doc, "meta_data", {}).get("file_path") == file_path
            ]
            if filtered_docs:
                documents = filtered_docs
            else:
                logger.warning(
                    "FetchFunctionDefinitionFromCallAgent: no documents matched file_path '%s' for %s; falling back to all candidates.",
                    file_path,
                    callee_display_name,
                )

        if not documents:
            logger.debug("FetchFunctionDefinitionFromCallAgent: no documents found for %s", callee_display_name)
            return []

        function_definitions: List[str] = []
        for doc in documents:
            count = 0
            while count <= COUNT_UPPER_LIMIT:
                context, cannot_extend = build_context(
                    retrieved_doc=doc,
                    id2doc=rag.id2doc,
                    direction="next",
                    count=count,
                )

                response = rag.query(input_str=prompt, contexts=[context])

                if self.debug:
                    save_query_results(response, " ".join(terms_list), faiss_query, prompt)

                reply = response["response"]
                format_prompt = OUTPUT_FORMAT_TEMPLATE.call(
                    reply=f"Here is the reply:\n {reply}. ",
                    info="valid function definitions containing both function header and body",
                    output_format=r"""Response in json format such as
{
    "function_definition": "[Function definition 1, Function definition 2, ...]",
    "complete": "[yes|no for Function definition 1, yes|no for Function definition 2, ...]"
}
""",
                )
                formatted = generator(input_str=format_prompt).data.strip()
                if self.debug:
                    generator.save_result()
                if formatted == "None":
                    raise ValueError(
                        f"FetchFunctionDefinitionFromCallAgent: Cannot find definition for {callee_display_name}, possibly because the bm25 index is incorrect."
                    )
                data = json.loads(repair_json(formatted))
                definitions = data.get("function_definition") or []
                complete_flags = data.get("complete", [])
                complete_definitions = [
                    definition
                    for definition, complete in zip(definitions, complete_flags)
                    if complete == "yes"
                ]
                if not complete_definitions:
                    fallback = extract_first_code_block(reply)
                    if fallback:
                        complete_definitions.append(fallback)
                if complete_definitions:
                    for definition in complete_definitions:
                        if definition not in function_definitions:
                            function_definitions.append(definition)
                    break

                if cannot_extend:
                    break
                count += 3
                sleep(0.5)

        return function_definitions


class FetchFunctionDefinitionFromNameAgent:

    """
    Given a function name, return the complete function definition.
    This is a general-purpose agent that doesn't require call expression context.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to find the **complete function definition** for the function named `{{function_name}}`.

{% if function_signature %}
To help disambiguate overloaded functions, here is the expected function signature:
{{function_signature}}
Use this to determine which specific overload or template specialization is being referenced.
{% endif %}
""",
            instructions=r"""
1. Identify the **complete function definition** for the function named `{{function_name}}`.
2. If multiple definition candidates exist (e.g., overloads, template specializations), return all relevant candidates.
3. Ensure the definition includes the complete function header and body.
4. For each candidate, verify it's a complete definition (not just a declaration).
"""
        )
    )

    def __init__(self, debug: bool = False):
        self.repo_path = configs()["repo_path"]
        self.debug = debug

    def __call__(
        self,
        function_name: str,
        function_signature: str = "",
        file_path: Optional[str] = None,
    ) -> List[str]:
        """
        Args:
            function_name (str): The name of the function to find.
            function_signature (str, optional): Expected signature to disambiguate overloads. Defaults to "".
            file_path (str, optional): Relative path from repo root to constrain search.

        Returns:
            List[str]: List of complete function definitions found.
        """
        with call_debug_scope(self.__class__.__name__, self.debug, function=function_name):
            prompt = self.prompt_template.call(
                function_name=function_name,
                function_signature=function_signature,
            )
            bm25_keywords = [f"[FUNCDEF]{function_name}"]
            faiss_query = ""
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            set_global_config_value("generator.json_output", False)
            rag = RAG()
            retrieved_docs = rag.retrieve(
                bm25_keywords=" ".join(bm25_keywords), faiss_query=faiss_query
            )[0].documents

            if file_path:
                filtered_docs = [
                    doc
                    for doc in retrieved_docs
                    if getattr(doc, "meta_data", {}).get("file_path") == file_path
                ]
                if filtered_docs:
                    retrieved_docs = filtered_docs
                else:
                    logger.warning(
                        "FetchFunctionDefinitionFromNameAgent: no documents matched file_path '%s' for %s; falling back to all candidates.",
                        file_path,
                        function_name,
                    )

            if not retrieved_docs:
                logger.warning(f"No documents found for {function_name} definition")
                return []

            function_definitions = []
            
            for doc in retrieved_docs:
                count = 0
                while count <= COUNT_UPPER_LIMIT:
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
                        info="valid function definitions containing both function header and body",
                        output_format=r"""Response in json format such as
{
    "function_definitions": ["Function definition 1", "Function definition 2", ...],
    "complete": ["yes|no for Function definition 1", "yes|no for Function definition 2", ...]
}
""",
                    )
                    reply = generator(input_str=format_prompt).data.strip()
                    if self.debug:
                        generator.save_result()

                    if reply == 'None':
                        logger.warning(f"Cannot find definition for {function_name}")
                        break

                    data = json.loads(repair_json(reply))
                    if data.get("function_definitions"):
                        # Only include complete definitions
                        complete_definitions = [
                            defn for defn, complete in zip(data["function_definitions"], data.get("complete", []))
                            if complete == "yes"
                        ]
                    else:
                        complete_definitions = []
                    
                    if complete_definitions:
                        for definition in complete_definitions:
                            if definition not in function_definitions:
                                function_definitions.append(definition)
                        break

                    if cannot_extend:
                        break
                    count += 3
                    sleep(0.5)  # avoid rate limit
            return function_definitions

class CallChainRetrievalAgent:
    """Builds call chains starting from an entry function using recursive expansion."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.caller_agent = ExtractCallerNameAgent(debug=debug)
        self.def_from_call_agent = FetchFunctionDefinitionFromCallAgent(debug=debug)
        self.def_from_name_agent = FetchFunctionDefinitionFromNameAgent(debug=debug)
        self.class_agent = FetchClassPipeline(debug=debug)
        self.operator_agent = FindCallOperatorQuery(debug=debug)
        self.is_class_agent = IsClassInstanceQuery(debug=debug)
        self.call_extraction_query = FetchCallExpressionsQuery(debug=debug)
        self.macro_analyzer = MacroAnalysisAgent(debug=debug)
        self._definition_cache: Dict[Tuple[str, str, str, str], List[str]] = {}

    def retrieve_call_chains(
        self,
        entry_function: str,
        max_depth: int = 10,
        file_path: Optional[str] = None,
    ) -> List[CallChain]:
        """Build call chains starting from ``entry_function``.

        Args:
            entry_function: Fully qualified name for the entry point.
            max_depth: Maximum recursion depth while expanding the call graph.
            file_path: Optional path to the file that defines ``entry_function``.
                This should be **relative to the repository root** under analysis.

        Returns:
            A list of call chains discovered for the entry function.
        """
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            action="retrieve_call_chains",
            entry=entry_function,
            max_depth=max_depth,
            file_path=file_path,
        ):
            if file_path:
                logger.info(
                    "CallChainRetrievalAgent: using entry file_path '%s' (relative to repository root)",
                    file_path,
                )
            definitions = self._safe_fetch_definitions(entry_function, file_path=file_path)
            if not definitions:
                logger.warning(f"CallChainRetrievalAgent: no definitions found for {entry_function}")
                return [CallChain(entry_function=entry_function, functions=[], depth=0, file_path=file_path)]
            chains: List[CallChain] = []
            for definition in definitions:
                root_call = FunctionCall(
                    function_name=entry_function,
                    full_name=entry_function,
                    context=definition,
                )
                self._expand_call_chain(
                    current_call=root_call,
                    definition=definition,
                    current_chain=[root_call],
                    collected=chains,
                    max_depth=max_depth,
                )

            for chain in chains:
                if file_path and not chain.file_path:
                    chain.file_path = file_path

            deduped = self._deduplicate_chains(chains, entry_function, entry_file_path=file_path)
            if file_path:
                for chain in deduped:
                    if not chain.file_path:
                        chain.file_path = file_path
            return deduped

    def _expand_call_chain(
        self,
        current_call: FunctionCall,
        definition: str,
        current_chain: List[FunctionCall],
        collected: List[CallChain],
        max_depth: int,
    ) -> None:
        if len(current_chain) >= max_depth:
            collected.append(self._build_chain(current_chain))
            return

        candidate_calls = self._extract_calls(definition, current_call)
        if not candidate_calls:
            collected.append(self._build_chain(current_chain))
            return

        for candidate in candidate_calls:
            if self._is_cycle(candidate.function_name, current_chain):
                continue
            sub_definitions = self._safe_fetch_definitions(
                candidate.function_name,
                caller_context=current_call.context or "",
                call_expr=candidate.call_expr or "",
            )
            if not sub_definitions:
                collected.append(self._build_chain(current_chain + [candidate]))
                continue

            for sub_definition in sub_definitions[:CALL_CHAIN_WIDTH]:
                next_call = FunctionCall(
                    function_name=candidate.function_name,
                    full_name=candidate.full_name or candidate.function_name,
                    call_site=candidate.call_site,
                    context=sub_definition,
                    call_expr=candidate.call_expr,
                )
                self._expand_call_chain(
                    current_call=next_call,
                    definition=sub_definition,
                    current_chain=current_chain + [next_call],
                    collected=collected,
                    max_depth=max_depth,
                )

    def _extract_calls(self, definition: str, current_call: FunctionCall) -> List[FunctionCall]:
        calls: List[FunctionCall] = []
        call_candidates = self.call_extraction_query(
            function_body=definition,
            current_function=current_call.function_name,
        )
        for candidate in call_candidates:
            name = candidate.get("callee")
            if not name or name == current_call.function_name:
                continue
            call_expr = candidate.get("call_expr")
            function_body = candidate.get("function_body")
            calls.append(
                FunctionCall(
                    function_name=name,
                    full_name=name,
                    context=function_body,
                    call_expr=call_expr,
                )
            )

        return deduplicate_function_calls(calls)

    def _safe_fetch_definitions(
        self,
        function_name: str,
        caller_context: str = "",
        call_expr: str = "",
        signature: str = "",
        file_path: Optional[str] = None,
    ) -> List[str]:
        normalized_path = file_path or ""
        cache_key = (function_name, call_expr.strip(), signature.strip(), normalized_path)
        cached = self._definition_cache.get(cache_key)

        if cached:
            return cached

        definitions: List[str] = []
        call_context_available = bool(call_expr.strip() or caller_context.strip())
        if call_context_available:
            try:
                definitions = self.def_from_call_agent(
                    callee_name=function_name,
                    callee_expr=call_expr,
                    caller_body=caller_context,
                    file_path=None,
                )
            except Exception as exc:
                logger.warning(f"Call-context definition lookup failed for {function_name}: {exc}")
                definitions = []
        else:
            try:
                definitions = self.def_from_name_agent(
                    function_name=function_name,
                    function_signature=signature,
                    file_path=file_path,
                )
            except Exception as exc:
                logger.warning(f"Name-based definition lookup failed for {function_name}: {exc}")
                definitions = []

        if definitions:
            self._definition_cache[cache_key] = definitions
        else:
            self._definition_cache.setdefault(cache_key, [])
            empty_key = (function_name, "", "", normalized_path)
            if cache_key != empty_key:
                self._definition_cache.setdefault(empty_key, [])

        return definitions

    def _is_cycle(self, function_name: str, chain: List[FunctionCall]) -> bool:
        return any(func.function_name == function_name for func in chain)

    def _build_chain(self, chain: List[FunctionCall]) -> CallChain:
        if not chain:
            return CallChain(entry_function="", functions=[], depth=0)
        entry = chain[0]
        chain_copy = deepcopy(chain)
        language = infer_language_from_definition(entry.context)
        context: Dict[str, Any] = {"entry_definition": entry.context}
        return CallChain(
            entry_function=entry.function_name,
            functions=chain_copy,
            context=context,
            depth=len(chain_copy),
            language=language,
        )

    def _deduplicate_chains(
        self,
        chains: List[CallChain],
        entry_function: str,
        entry_file_path: Optional[str] = None,
    ) -> List[CallChain]:
        unique: List[CallChain] = []
        seen: Set[Tuple[str, ...]] = set()
        for chain in chains:
            key = tuple(func.function_name for func in chain.functions)
            if key in seen:
                continue
            seen.add(key)
            unique.append(chain)
        if not unique:
            unique.append(
                CallChain(
                    entry_function=entry_function,
                    functions=[],
                    depth=0,
                    file_path=entry_file_path,
                )
            )
        return unique


class FetchMemberVarQuery:
    """Extracts member variables from a class definition, including their names and types.

    This query analyzes a class definition body and extracts all member variables,
    including their names, types, access specifiers, and other relevant information.
    Can optionally extract a specific member variable by name.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a complete class definition:
{{class_definition}}

{% if extraction_target %}
Your task is to locate {{extraction_target}} in the class.
{% else %}
Your task is to locate ALL member variables in the class.
{% endif %}

CRITICAL: You MUST extract {% if extraction_target %}{{extraction_target}}{% else %}ALL member variables{% endif %} including:
- Public, private, and protected member variables
- Static member variables
- Const member variables
- Member variables with default values
- Template member variables
- Member variables with complex types (including pointers, references, templates, etc.)
- Member variables with access specifiers (public:, private:, protected:)
- Member variables with attributes (e.g., [[nodiscard]], [[deprecated]])
            """,
            output_format=r"""Return strict JSON:
{
    "member_variables": [
        {
            "name": "variable_name",
            "type": "variable_type",
            "access": "public|private|protected|static|const",
            "default_value": "default_value_or_null",
            "is_static": true/false,
            "is_const": true/false,
            "attributes": ["attribute1", "attribute2", ...]
        },
        ...
    ]
}
If no member variables are found, return an empty array for "member_variables".
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, class_definition: str, target_name: Optional[str] = None) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Extract member variables from a class definition.

        Args:
            class_definition: The complete class definition as a string
            target_name: Optional name of specific member variable to extract.
                        If provided, returns only that variable or None if not found.

        Returns:
            If target_name is None: List of dictionaries containing all member variable information
            If target_name is provided: Single dictionary for the target variable or None if not found
        """
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            class_length=len(class_definition) if class_definition else 0,
        ):
            if not class_definition or not class_definition.strip():
                return [] if target_name is None else None

            # Prepare template variables
            template_vars = {"class_definition": class_definition}
            if target_name:
                template_vars["extraction_target"] = f"only the member variable named '{target_name}'"

            prompt = self.prompt_template.call(**template_vars)
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            if self.debug:
                generator.save_result()
            set_global_config_value("generator.json_output", False)

        try:
            data = json.loads(repair_json(reply))
        except Exception as exc:
            logger.error(
                "FetchMemberVarQuery: JSON parse failed: %s",
                exc,
            )
            raise

        member_variables = data.get("member_variables", [])
        if not isinstance(member_variables, list):
            return []

        # Validate and normalize the member variable data
        valid_variables = []
        for var in member_variables:
            if not isinstance(var, dict):
                continue

            # Ensure required fields exist
            if "name" not in var or not var["name"]:
                continue

            # Normalize the data
            normalized_var = {
                "name": var["name"].strip(),
                "type": var.get("type", "").strip(),
                "access": var.get("access", "private").strip().lower(),
                "default_value": var.get("default_value"),
                "is_static": bool(var.get("is_static", False)),
                "is_const": bool(var.get("is_const", False)),
                "attributes": var.get("attributes", [])
            }

            # Validate access specifier
            if normalized_var["access"] not in ["public", "private", "protected", "static", "const"]:
                normalized_var["access"] = "private"

            # Ensure attributes is a list
            if not isinstance(normalized_var["attributes"], list):
                normalized_var["attributes"] = []

            valid_variables.append(normalized_var)

        # If target_name is specified, find and return only that variable
        if target_name:
            target_var = next((var for var in valid_variables if var["name"] == target_name), None)
            return target_var

        # Otherwise return all variables
        return valid_variables


class ChainedCallAnalyzerPipeline:
    """Analyzes chained call expressions to determine the final return type."""

    def __init__(self, debug: bool = False):
        """
        Initialize the pipeline with optional debug logging.

        Args:
            debug: Enable debug logging for troubleshooting
        """
        self.debug = debug
        self.is_class_instance_query = IsClassInstanceQuery(debug=debug)
        self.fetch_class_pipeline = FetchClassPipeline(debug=debug)
        self.extract_member_query = ExtractMemberDefinitionFromClassQuery(debug=debug)
        self.fetch_return_type_query = FetchReturnTypeQuery(debug=debug)
        self.fetch_member_var_query = FetchMemberVarQuery(debug=debug)

    def __call__(self, expression: str, context: str) -> str:
        """
        Analyze a chained call expression and return the final class name.

        Args:
            expression: The chained call expression to analyze (e.g., "a.b.c()")
            context: The source code context containing the expression

        Returns:
            The class name of the innermost function call's return type
        """
        with call_debug_scope(self.__class__.__name__, self.debug, expression=expression):
            try:
                # Parse the chained expression into components
                components = self._parse_chained_expression(expression)
                if not components:
                    logger.warning(f"Could not parse chained expression: {expression}")
                    return None

                # Get the initial class name from the outermost instance
                initial_instance = components[0]["name"]
                is_class_instance, class_name = self.is_class_instance_query(
                    context=context, callee_name=initial_instance, call_expr=expression
                )

                if not is_class_instance:
                    logger.warning(f"Initial instance '{initial_instance}' is not a class instance")
                    return None

                # Iteratively resolve each component in the chain
                current_class_name = class_name
                for component in components[1:-1]:  # Skip the initial instance, and ignore the last one
                    current_class_name = self._resolve_component(
                        component, current_class_name, context
                    )
                    if not current_class_name:
                        logger.warning(f"Could not resolve component: {component}")
                        return None
                    
                    current_class_name = current_class_name.strip()
                    
                    in_while = True
                    while in_while:
                        if current_class_name.startswith("const "):
                            current_class_name = current_class_name[6:]
                        elif current_class_name.startswith("static "):
                            current_class_name = current_class_name[7:]
                        elif current_class_name.startswith("constexpr "):
                            current_class_name = current_class_name[10:]
                        elif current_class_name.startswith("volatile "):
                            current_class_name = current_class_name[9:]
                        elif current_class_name.endswith("&"):
                            current_class_name = current_class_name[:-1]
                        elif current_class_name.endswith("*"):
                            current_class_name = current_class_name[:-1]
                        else:
                            in_while = False

                current_class_name = current_class_name.strip()
                return current_class_name

            except Exception as exc:
                logger.error(f"Error analyzing chained expression '{expression}': {exc}")
                if self.debug:
                    raise
                return None

    def _parse_chained_expression(self, expression: str) -> List[Dict[str, Any]]:
        """
        Parse a chained call expression into individual components.

        Args:
            expression: The chained call expression to parse

        Returns:
            List of components with name, type, and operator information
        """
        if not expression or not expression.strip():
            return []

        components = []
        # Remove whitespace for easier parsing
        expr = expression.strip()

        # Split on access operators while preserving parentheses
        current_pos = 0
        i = 0

        while i < len(expr):
            if expr[i] in ['.', '-']:
                # Check if it's -> operator
                if expr[i] == '-' and i + 1 < len(expr) and expr[i + 1] == '>':
                    operator = '->'
                    i += 2
                else:
                    operator = '.'
                    i += 1

                # Extract the component name
                name_start = i
                while i < len(expr) and expr[i] not in ['.', '-', '(', ')', '[', ']', ',', ' ', '\t', '\n']:
                    i += 1

                name = expr[name_start:i]

                # Check if it's a function call
                is_function_call = False
                if i < len(expr) and expr[i] == '(':
                    is_function_call = True
                    # Skip matching parentheses
                    paren_count = 1
                    i += 1
                    while i < len(expr) and paren_count > 0:
                        if expr[i] == '(':
                            paren_count += 1
                        elif expr[i] == ')':
                            paren_count -= 1
                        i += 1

                components.append({
                    "name": name,
                    "type": "function_call" if is_function_call else "member_access",
                    "operator": operator
                })
            else:
                i += 1

        # Add the initial instance if we found components
        if components:
            # Extract the initial instance name
            initial_part = expr.split(components[0]["operator"])[0].strip()
            components.insert(0, {
                "name": initial_part,
                "type": "instance",
                "operator": None
            })

        return components

    def _resolve_component(self, component: Dict[str, Any], current_class_name: str, context: str) -> Optional[str]:
        """
        Resolve a single component in the chain to get its return type.

        Args:
            component: The component to resolve
            current_class_name: The current class name
            context: The source code context

        Returns:
            The class name of the return type, or None if resolution fails
        """
        try:
            if component["type"] == "function_call":
                return self._resolve_function_call(component, current_class_name)
            elif component["type"] == "member_access":
                return self._resolve_member_access(component, current_class_name)
            else:
                logger.warning(f"Unknown component type: {component['type']}")
                return None
        except Exception as exc:
            logger.error(f"Error resolving component {component}: {exc}")
            return None

    def _resolve_function_call(self, component: Dict[str, Any], current_class_name: str) -> Optional[str]:
        """
        Resolve a function call component to get its return type.

        Args:
            component: The function call component
            current_class_name: The current class name

        Returns:
            The class name of the return type, or None if resolution fails
        """
        # Get the class definition
        class_definition = self.fetch_class_pipeline(current_class_name)
        if not class_definition:
            logger.warning(f"Could not find class definition for: {current_class_name}")
            return None

        # Extract the member function definition
        member_definition = self.extract_member_query(
            class_definition=class_definition,
            target_name=component["name"]
        )

        if not member_definition:
            logger.warning(f"Could not find member function: {component['name']} in class {current_class_name}")
            return None

        # Get the return type
        return_type = self.fetch_return_type_query(member_definition)
        return return_type

    def _resolve_member_access(self, component: Dict[str, Any], current_class_name: str) -> Optional[str]:
        """
        Resolve a member access component to get its type.

        Args:
            component: The member access component
            current_class_name: The current class name

        Returns:
            The class name of the member type, or None if resolution fails
        """
        # Get the class definition
        class_definition = self.fetch_class_pipeline(current_class_name)
        if not class_definition:
            logger.warning(f"Could not find class definition for: {current_class_name}")
            return None

        # Extract the member variable information
        member_var = self.fetch_member_var_query(
            class_definition=class_definition,
            target_name=component["name"]
        )

        if not member_var:
            logger.warning(f"Could not find member variable: {component['name']} in class {current_class_name}")
            return None

        # Return the type of the member variable
        return member_var.get("type")
