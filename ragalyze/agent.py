import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union
from copy import deepcopy
from time import sleep
from json_repair import repair_json
from numpy import promote_types

from ragalyze.rag.rag import RAG, GeneratorWrapper
from ragalyze.query import print_result, save_query_results, build_context
from ragalyze.configs import *
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from adalflow import Prompt
from adalflow.core.types import Document

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
def call_debug_scope(
    class_name: str, debug: bool, action: str = "__call__", **info: Any
):
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
        description="Initializes object state when an instance is created",
    ),
    FunctionKindInfo(
        kind="destructor", description="Cleans up resources when an object is destroyed"
    ),
    FunctionKindInfo(
        kind="member_method",
        description="Regular instance method that operates on object state",
    ),
    FunctionKindInfo(
        kind="operator", description="Overloaded operators (+, -, *, /, [], (), etc.)"
    ),
    FunctionKindInfo(
        kind="static_method",
        description="Method that operates on class level without requiring instance",
    ),
    FunctionKindInfo(
        kind="virtual_method",
        description="Method that can be overridden in derived classes",
    ),
    FunctionKindInfo(
        kind="pure_virtual",
        description="Abstract method with declaration only, no implementation",
    ),
    FunctionKindInfo(
        kind="const_method",
        description="Method that promises not to modify object state",
    ),
    FunctionKindInfo(
        kind="template_method",
        description="Method with template parameters for generic programming",
    ),
    FunctionKindInfo(
        kind="conversion_operator",
        description="Method that converts object to other types",
    ),
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


def find_doc(file_path: str, line_number: int):
    """
    Find the document that contains the line_number in the file_path.
    """
    if line_number == None:
        return None
    rag = RAG()
    rag.get_docs()
    doc = rag.id2doc[rag.codePath2beginDocid[file_path]]
    while True:
        if (
            doc.meta_data["start_line"] <= line_number
            and doc.meta_data["end_line"] >= line_number
        ):
            return doc
        if doc.meta_data["next_doc_id"] == None:
            break
        doc = rag.id2doc[doc.meta_data["next_doc_id"]]
        if doc == None:
            break
    return None


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
            instructions=r"""
1. Locate every call to the target function.
2. Up-scan from that line until you reach the outermost function header that contains the call.
3. If the snippet is truncated and no header is found, the header is indeterminable—header missing.
4. If the found header is incomplete (e.g., missing return type, parameters, or template parameters), the header is indeterminable—header missing.
5. Skip any local lambdas or nested functions; return the first non-local header. 
6. First determine the language, then check if the header is complete for that language. If not, return drop it.
            """,
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
                bm25_keywords="[TOKEN_SPLIT]".join(bm25_keywords),
                faiss_query=faiss_query,
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
    """,
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
            if reply == "None":
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

    def __call__(
        self, callee_name: str, callee_body: str = ""
    ) -> List[Tuple[str, str]]:
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

            caller_names = []
            for header in caller_headers:
                result = self.name_from_header_agent(header)
                if result:
                    full_name, short_name = result
                    caller_names.append((full_name, short_name))

            if not caller_names:
                logger.warning(
                    f"Cannot extract caller names from headers for {callee_name}"
                )

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
{"is_class_instance": true/false/"?", "class": "class_name"/"None"}
If you can determine whether the callee_name is a class instance, return true or false.
If you cannot determine, return "?".
If it is a class instance, return the class name in the "class" field.
If the class name is not found, return "None".
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, context: str, callee_name: str, call_expr: str = "") -> bool:
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
                logger.warning(
                    f"Cannot determine if {callee_name} is a class instance. Check if {callee_name} is in {context}"
                )
            return data["is_class_instance"], data["class"]


class BaseClassNameExtractor:
    """
    Extract the base class name and type arguments from templated or over-decorated class names.
    This helps when searching for class definitions that use template parameters.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Given a potentially templated or over-decorated class name, extract the namespace hierarchy, base class name, and any type arguments.

Examples:
- "DataMap<std::string, DataProcessor>" -> {"namespaces": [], "base_class_name": "DataMap", "type_arguments": ["std::string", "DataProcessor"]}
- "List<int>" -> {"namespaces": [], "base_class_name": "List", "type_arguments": ["int"]}
- "GenericRepository<T extends Comparable<T>, ID>" -> {"namespaces": [], "base_class_name": "GenericRepository", "type_arguments": ["T extends Comparable<T>", "ID"]}
- "std::vector<double>" -> {"namespaces": ["std"], "base_class_name": "vector", "type_arguments": ["double"]}
- "Map<String, Integer>" -> {"namespaces": [], "base_class_name": "Map", "type_arguments": ["String", "Integer"]}
- "Array<DataType>" -> {"namespaces": [], "base_class_name": "Array", "type_arguments": ["DataType"]}
- "MyClass<T, U>" -> {"namespaces": [], "base_class_name": "MyClass", "type_arguments": ["T", "U"]}
- "Vector<Map<string, int>>" -> {"namespaces": [], "base_class_name": "Vector", "type_arguments": ["Map<string, int>"]}
- "Engine" -> {"namespaces": [], "base_class_name": "Engine", "type_arguments": []}
- "namespace1::namespace2::Class<T1, T2>" -> {"namespaces": ["namespace1", "namespace2"], "base_class_name": "Class", "type_arguments": ["T1", "T2"]}
- "std::map<int, std::vector<std::string>>" -> {"namespaces": ["std"], "base_class_name": "map", "type_arguments": ["int", "std::vector<std::string>"]}
- "JavaPackage.InnerClass<String, Integer>" -> {"namespaces": ["JavaPackage"], "base_class_name": "InnerClass", "type_arguments": ["String", "Integer"]}
- "boost::filesystem::path" -> {"namespaces": ["boost", "filesystem"], "base_class_name": "path", "type_arguments": []}
- "my.package.Class<T extends Comparable<T>>" -> {"namespaces": ["my", "package"], "base_class_name": "Class", "type_arguments": ["T extends Comparable<T>"]}
- "outer::middle::inner::NestedClass<Type1, Type2>" -> {"namespaces": ["outer", "middle", "inner"], "base_class_name": "NestedClass", "type_arguments": ["Type1", "Type2"]}
- "com.example.util.DataStructure<Key, Value>" -> {"namespaces": ["com", "example", "util"], "base_class_name": "DataStructure", "type_arguments": ["Key", "Value"]}

For the class name "{{class_name}}", return the namespace hierarchy, base class name, and any type arguments.
            """,
            output_format=r"""Return in JSON format
{
    "namespaces": ["namespace1", "namespace2", ...],
    "base_class_name": "base_class_name",
    "type_arguments": ["arg1", "arg2", ...]
}
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, class_name: str) -> dict:
        """
        Args:
            class_name (str): The potentially templated or over-decorated class name.

        Returns:
            dict: Dictionary containing namespaces, base_class_name, and type_arguments.
        """
        with call_debug_scope(
            self.__class__.__name__, self.debug, input_class_name=class_name
        ):
            # Use LLM to handle all extraction logic
            prompt = self.prompt_template.call(class_name=class_name)

            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            set_global_config_value("generator.json_output", False)

            try:
                reply = generator(input_str=prompt).data.strip()
                if self.debug:
                    generator.save_result()

                data = json.loads(repair_json(reply))
                namespaces = data.get("namespaces", [])
                base_class_name = data.get("base_class_name", "").strip()
                type_arguments = data.get("type_arguments", [])

                if base_class_name:
                    if self.debug:
                        logger.info(
                            f"LLM extraction result: {class_name} -> namespaces: {namespaces}, base: {base_class_name}, args: {type_arguments}"
                        )
                    return {
                        "namespaces": namespaces,
                        "base_class_name": base_class_name,
                        "type_arguments": type_arguments,
                    }
                else:
                    if self.debug:
                        logger.warning(
                            f"LLM extraction failed, returning original: {class_name}"
                        )
                    return {
                        "namespaces": [],
                        "base_class_name": class_name,
                        "type_arguments": [],
                    }
            except Exception as e:
                if self.debug:
                    logger.error(f"Error during LLM base class extraction: {e}")
                return {
                    "namespaces": [],
                    "base_class_name": class_name,
                    "type_arguments": [],
                }


class FetchClassPipeline:
    """
    Given a class name, return the class definition along with type arguments.
    Enhanced to handle templated class names and extract type parameters.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to find the **complete** class definition for the class named `{{class_name}}`.

Note that the class name may be:
- A simple class name (e.g., "MyClass")
- A namespace-qualified name (e.g., "std::vector", "boost::filesystem::path", "my.package.Class")
- A templated class name without type arguments (e.g., "vector" from "std::vector<int>")

If you find multiple classes with the same base name but different namespaces, prefer the most specific match.
            """,
            output_format=r"""Return json:
{"class": class_definition_str | None, "is_complete": true|false}
If the class definition is **not complete**, set "is_complete" to false.
If you cannot find the class definition, set "class" to None.
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, class_name: str) -> dict:
        """
        Args:
            class_name (str): The name of the class.

        Returns:
            dict: Dictionary containing class_definition, original_class_name, namespaces, base_class_name, and type_arguments.
        """
        with call_debug_scope(self.__class__.__name__, self.debug):

            # First, try with the original class name
            class_definition = self._try_fetch_class_definition(class_name)
            if class_definition:
                return {
                    "class_definition": class_definition,
                    "original_class_name": class_name,
                    "namespaces": [],
                    "base_class_name": "",
                    "type_arguments": [],
                }

            # Extract namespace, base class name, and type arguments from the original class name
            base_name_extractor = BaseClassNameExtractor(debug=self.debug)
            extracted_info = base_name_extractor(class_name)

            original_class_name = class_name
            namespaces = extracted_info["namespaces"]
            base_class_name = extracted_info["base_class_name"]
            type_arguments = extracted_info["type_arguments"]

            if self.debug:
                logger.info(
                    f"No definition found for '{original_class_name}', trying with base class name: '{base_class_name}'"
                )

            class_definition = self._try_fetch_class_definition(base_class_name)
            if class_definition:
                return {
                    "class_definition": class_definition,
                    "original_class_name": original_class_name,
                    "namespaces": namespaces,
                    "base_class_name": base_class_name,
                    "type_arguments": type_arguments,
                }

            return None

    def _try_fetch_class_definition(self, class_name: str) -> str:
        """Try to fetch class definition for a given class name."""
        if self.debug:
            logger.info(f"Attempting to fetch class definition for: '{class_name}'")

        prompt = self.prompt_template.call(class_name=class_name)
        bm25_keywords = [f"[CLASS]{class_name}"]
        faiss_query = ""
        set_global_config_value("generator.json_output", True)
        rag = RAG()
        set_global_config_value("generator.json_output", False)

        retrieved_docs = rag.retrieve(
            bm25_keywords="[TOKEN_SPLIT]".join(bm25_keywords), faiss_query=faiss_query
        )[0].documents

        if not retrieved_docs:
            if self.debug:
                logger.warning(f"No documents found for {class_name} definition")
            return None

        for doc in retrieved_docs:
            count = 5
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
                class_str = data.get("class")
                is_complete = data.get("is_complete")
                if class_str and class_str != "None" and is_complete:
                    return class_str

                if cannot_extend:
                    break
                count += 5
                sleep(0.5)  # avoid rate limit

        return None


class FetchConstructorQuery:
    """
    Given a class definition body, return the Constructor definition. This is required since some languages contain over-design language sugars:(
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a class definition body: {{class_definition}}

Your task is to find the **complete** Constructor definition in the class definition body.
**Complete** here means the Constructor definition is **not just a function declaration**, but contain the **function body**.
Constructor is a special function that initializes object instances, such as `ClassName()` in C++, `__init__` in Python, or methods with the same name as the class in Java.
            """
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, class_definition: str, class_name: str = "") -> Optional[str]:
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
                info="Whether or not the constructor definition exists",
                output_format=r"""If the Constructor definition cannot be found or the Constructor definition is **not complete**, return json:
{"constructor": None}
Otherwise, return json:
{"constructor": constructor_definition}
            """,
            )
            reply = generator(input_str=format_prompt).data.strip()
            if self.debug:
                generator.save_result()
        data = json.loads(repair_json(reply))
        constructor_def = data.get("constructor")
        if constructor_def == "None":
            constructor_def = None
        if not constructor_def:
            # If cannot find constructor definition inside the class definition,
            # it may be defined outside the function, some languages supports such feature.
            # So, query it with class name prefixed
            if not class_name:
                return None
            #! WARNING: only consider cpp's outside-class function definition
            function_name = f"{class_name}::{class_name}"
            constructor_defs = FetchFunctionDefinitionFromNamePipeline(
                debug=self.debug
            )(function_name)
            #! WARNING: only consider the first function definition found
            if len(constructor_defs) > 0:
                constructor_def = constructor_defs[0]

        return constructor_def


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

    def __call__(self, class_definition: str, class_name: str = "") -> Optional[str]:
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
        call_operator_def = data.get("call_operator")
        if call_operator_def == "None":
            call_operator_def = None
        if not call_operator_def:
            # If cannot find call operator definition inside the class definition,
            # it may be defined outside the function, some languages supports such feature.
            # So, query it with class name prefixed
            if not class_name:
                return None
            #! WARNING: only consider cpp's outside-class function definition
            function_name = f"{class_name}::operator()"
            call_operator_defs = FetchFunctionDefinitionFromNamePipeline(
                debug=self.debug
            )(function_name)
            #! WARNING: only consider the first function definition found
            if len(call_operator_defs) > 0:
                call_operator_def = call_operator_defs[0]

        return call_operator_def


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

    def __call__(
        self, callee_name: str, call_expr: str, caller_body: str
    ) -> Dict[str, Any]:
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
            logger.error(
                "AnalyzeCallExpressionPipeline: JSON parse failed for %s: %s",
                callee_name,
                exc,
            )
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

    def __call__(
        self,
        class_definition: str,
        target_name: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> Union[Optional[str], List[Dict[str, Any]]]:
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


class AnalyzeFunctionTypeQuery:
    """Analyzes function signatures to extract parameter types and return types."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a function signature or definition snippet:
{{function_snippet}}

{% if language %}
The function is written in {{language}}.
{% endif %}

Analyze the function to extract parameter types and return type.
            """,
            instructions=r"""
Return strict JSON:
{
  "parameter_types": a list of string,
  "return_types": a list of string
}
- parameter_types: List of parameter types in order, or None if cannot be determined
- return_types: List of return types in order, or None if cannot be determined
- If the function has no parameters, parameter_types should be an empty list []
- If the function has no return type (e.g., constructors, void functions), return_types should be an empty list []
- If the function has a single return type, return_types should be a list with one element: ["int"]
- If the function can return multiple possible types (e.g., conditional returns), use union syntax as a single string: ["int | string"]
- DO NOT return multiple separate types in the list for union cases - always combine them with " | " syntax
- Include qualifiers such as `const`, `static`, `constexpr`, pointer/reference symbols, namespaces, templates, etc., as they appear in the signature.
- If the given snippet does not contain a complete function signature or definition, return a simple None but not the JSON.
- If the snippet contains non-function code (variables, statements, etc. without function definitions), return None.
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(
        self, function_snippet: str, language: str = ""
    ) -> Optional[Dict[str, Any]]:
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            snippet_length=len(function_snippet or ""),
        ):
            if not function_snippet or not function_snippet.strip():
                return None

            prompt = self.prompt_template.call(
                function_snippet=function_snippet, language=language
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
            logger.error("AnalyzeFunctionTypeQuery: JSON parse failed: %s", exc)
            return None

        # If LLM returned None (not JSON), return None
        if data is None:
            return None

        parameter_types = data.get("parameter_types")
        return_types = data.get("return_types")

        result = {}
        if parameter_types is not None:
            if isinstance(parameter_types, list):
                result["parameter_types"] = [
                    ptype.strip() if isinstance(ptype, str) else ptype
                    for ptype in parameter_types
                ]
            elif isinstance(parameter_types, str):
                result["parameter_types"] = [parameter_types.strip()]
            else:
                result["parameter_types"] = None

        if return_types is not None:
            if isinstance(return_types, str):
                result["return_types"] = [return_types.strip()]
            elif isinstance(return_types, list):
                # Process list of return types
                processed_types = []
                for rtype in return_types:
                    if isinstance(rtype, str):
                        processed_types.append(rtype.strip())
                    else:
                        processed_types.append(str(rtype))
                result["return_types"] = processed_types
            else:
                result["return_types"] = None

        return result if result else None


class ResolveIndexedTypeQuery:
    """Resolves the element type from an indexed container type."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a container type and an index expression:
Container type: {{container_type}}
Index expression: {{index_expression}}

{% if language %}
The code is written in {{language}}.
{% endif %}

Analyze the container type to determine what type of element is returned when indexed.
            """,
            instructions=r"""
Return the element type as a string, or None if cannot be determined.

Examples:
- Container type: "vector<int>", Index expression: "[0]" → Return: "int"
- Container type: "string[]", Index expression: "[i]" → Return: "string"
- Container type: "map<string, int>", Index expression: "[\"key\"]" → Return: "int"
- Container type: "int*", Index expression: "[0]" → Return: "int"
- Container type: "vector<vector<string>>", Index expression: "[0][1]" → Return: "string"
- Container type: "MyClass", Index expression: "[0]" → Return: None (unless MyClass has operator[])

Handle these common patterns:
- Array types: T[] → T
- Pointer types: T* → T
- Template containers: vector<T> → T, array<T, N> → T
- Map containers: map<K, V> → V, unordered_map<K, V> → V
- Nested containers: vector<vector<T>> → vector<T> for first index, T for second

If the container type does not support indexing or the element type cannot be determined, return None.
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(
        self, container_type: str, index_expression: str, language: str = "auto"
    ) -> Optional[str]:
        """Resolve the element type from an indexed container type."""
        # Use a simple fallback for configuration
        try:
            from ragalyze.configs import get_global_config_value

            generator = get_global_config_value("generator.provider", "openai")
            model = get_global_config_value("generator.model", "gpt-4o-mini")
        except ImportError:
            # Fallback to default values
            generator = "openai"
            model = "gpt-4o-mini"

        prompt = self.prompt_template.format(
            container_type=container_type,
            index_expression=index_expression,
            language=language,
        )

        if self.debug:
            print(
                f"[ResolveIndexedTypeQuery] Resolving index {index_expression} on type {container_type}"
            )
            print(f"[ResolveIndexedTypeQuery] Using {generator}/{model}")

        response = None
        if generator == "openai":
            response = call_openai(model, prompt, self.debug)
        elif generator == "deepseek":
            response = call_deepseek(model, prompt, self.debug)
        elif generator == "claude":
            response = call_claude(model, prompt, self.debug)
        else:
            raise ValueError(f"Unsupported generator: {generator}")

        if self.debug:
            print(f"[ResolveIndexedTypeQuery] Raw response: {response}")

        if not response:
            return None

        # Clean up response - remove extra whitespace, quotes, etc.
        cleaned_response = response.strip()
        if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
            cleaned_response = cleaned_response[1:-1]
        if cleaned_response.startswith("'") and cleaned_response.endswith("'"):
            cleaned_response = cleaned_response[1:-1]

        # Return None for ambiguous responses
        if cleaned_response.lower() in ["none", "null", "undefined", "unknown"]:
            return None

        return cleaned_response if cleaned_response else None


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

    def __call__(self, function_body: str) -> List[Dict[str, Any]]:
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            length=len(function_body) if function_body else 0,
        ):
            if not function_body.strip():
                return []

            prompt = self.prompt_template.call(
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
                result.append(
                    {
                        "call_expr": call_expr,
                        "function_body": function_body,
                    }
                )
            return result


class FetchFunctionDefinitionFromCallAgent:
    """
    Given a function call expr E or the funciton name inside, return the function definition of the function called in E.
    """

    """TODO
    since function call expr is diverse, containing regular function call, constructor call, member function call, functor call, etc.
    This agent cannot cover all scenarios. For instance, if the function call is a functor call, its definition is presented as
    class_name::operator()(args), or simply operator() inside the class body. But in Python, the definition is __call__(args) inside the
    class body. These differences require a rounter for different language. Currently, we only plan to support C/C++, Java, Python, and Solidity.
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
""",
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
                fallback_candidates = list(
                    dict.fromkeys((qualified_names or []) + [callee_name])
                )
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
            result = self.class_pipeline(class_name)
            class_definition = (
                result.get("class_definition") if isinstance(result, dict) else result
            )
            if class_definition:
                operator_definition = self.call_operator_query(
                    class_definition=class_definition
                )
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
            result = self.class_pipeline(class_name)
            class_definition = (
                result.get("class_definition") if isinstance(result, dict) else result
            )
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
            result = self.class_pipeline(class_name)
            class_definition = (
                result.get("class_definition") if isinstance(result, dict) else result
            )
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
            result = self.class_pipeline(class_name)
            class_definition = (
                result.get("class_definition") if isinstance(result, dict) else result
            )
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
        search_terms = list(
            dict.fromkeys(
                (analysis.get("search_terms", []) or [])
                + ([callee_expr] if callee_expr else [])
            )
        )

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
            bm25_keywords="[TOKEN_SPLIT]".join(terms_list),
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
            logger.debug(
                "FetchFunctionDefinitionFromCallAgent: no documents found for %s",
                callee_display_name,
            )
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
                    save_query_results(
                        response, " ".join(terms_list), faiss_query, prompt
                    )

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


class FetchFunctionDefinitionFromNamePipeline:
    """
    Given a function name, return the complete function definition.
    This is a general-purpose agent that doesn't require call expression context.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to find the **complete function definition** for the function named `{{function_name}}`.
If the given code snippets contain line numbers, please include them in your response.

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
""",
        )
    )

    def __init__(self, debug: bool = False):
        self.repo_path = configs()["repo_path"]
        self.debug = debug

    def _select_definition_by_line_number(
        self, function_name: str, line_number: int, function_definitions: List[str]
    ) -> str:
        """
        Use LLM to select the correct function definition based on line number.

        Args:
            function_name: Name of the function to find
            line_number: Target line number where the function header should be located
            function_definitions: List of candidate function definitions with line numbers

        Returns:
            Selected function definition that matches the line number
        """
        try:
            # Build a prompt showing all function definitions with their line numbers
            prompt = f"""I need to find the function definition for '{function_name}' that is located at line {line_number}.

Here are the candidate function definitions found:

"""

            for i, func_def in enumerate(function_definitions, 1):
                # Extract line number from function definition if available
                lines = func_def.split("\n")
                line_info = ""
                if lines and any(line.strip().startswith("// Line:") for line in lines):
                    line_comment = next(
                        (line for line in lines if line.strip().startswith("// Line:")),
                        "",
                    )
                    line_info = f" ({line_comment.strip()})"

                prompt += f"Candidate {i}{line_info}:\n{func_def}\n\n"

            prompt += f"""Based on the line number {line_number}, which candidate function definition is the correct one?
Please respond with just the candidate number (1, 2, 3, etc.) that corresponds to the function definition at line {line_number}."""

            # Use generator to get LLM response
            set_global_config_value("generator.json_output", False)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()

            if self.debug:
                logger.info(f"Line number selection prompt:\n{prompt}")
                logger.info(f"LLM response: {reply}")
                generator.save_result()

            # Parse the response to get the candidate number
            try:
                candidate_num = int(reply.strip())
                if 1 <= candidate_num <= len(function_definitions):
                    selected_def = function_definitions[candidate_num - 1]
                    if self.debug:
                        logger.info(
                            f"Selected function definition {candidate_num} based on line number {line_number}"
                        )
                    return selected_def
                else:
                    logger.warning(
                        f"Invalid candidate number {candidate_num}, falling back to first definition"
                    )
                    return function_definitions[0]
            except ValueError:
                logger.warning(
                    f"Could not parse candidate number from response: {reply}, falling back to first definition"
                )
                return function_definitions[0]

        except Exception as exc:
            if self.debug:
                logger.error(f"Error in _select_definition_by_line_number: {exc}")
            # Fall back to first definition if there's an error
            return function_definitions[0] if function_definitions else ""

    def __call__(
        self,
        function_name: str,
        function_signature: str = "",
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> List[str]:
        """
        Args:
            function_name (str): The name of the function to find.
            function_signature (str, optional): Expected signature to disambiguate overloads. Defaults to "".
            file_path (str, optional): Relative path from repo root to constrain search.
            line_number (int, optional): Line number to precisely locate the entry function's header.

        Returns:
            List[str]: List of complete function definitions found.
        """
        with call_debug_scope(
            self.__class__.__name__, self.debug, function=function_name
        ):
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
                bm25_keywords="[TOKEN_SPLIT]".join(bm25_keywords),
                faiss_query=faiss_query,
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
                        "FetchFunctionDefinitionFromNamePipeline: no documents matched file_path '%s' for %s; falling back to all candidates.",
                        file_path,
                        function_name,
                    )

            if not retrieved_docs:
                # the function call may be a constructor or a call operator
                logger.warning(
                    f"No documents found for {function_name} definition, {function_name} may be a class name"
                )
                # step 1: fetch class definition
                class_definition = FetchClassPipeline(debug=self.debug)(function_name)
                if not class_definition:
                    logger.warning(f"Cannot find class definition for {function_name}")
                    return []
                # step 2: fetch call operator definition
                call_operator_definition = FindCallOperatorQuery(debug=self.debug)(
                    class_definition, function_name
                )
                constructor_definition = FetchConstructorQuery(debug=self.debug)(
                    class_definition, function_name
                )
                if not constructor_definition and not call_operator_definition:
                    logger.warning(
                        f"Cannot find constructor definition for {function_name}"
                    )
                    return []
                function_definitions = []
                if constructor_definition:
                    function_definitions.append(constructor_definition)
                if call_operator_definition:
                    function_definitions.append(call_operator_definition)
                return function_definitions

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
                        reply=f"Here is the reply:\n {reply}. ",
                        info="valid function definitions containing both function header and body",
                        output_format=r"""Response in json format such as
{
    "function_definitions": ["Function definition 1", "Function definition 2", ...],
    "complete": ["yes|no for Function definition 1", "yes|no for Function definition 2", ...]
}
If the given function definitions contain line numbers, please include them in your response.
""",
                    )
                    reply = generator(input_str=format_prompt).data.strip()
                    if self.debug:
                        generator.save_result()

                    if reply == "None":
                        logger.warning(f"Cannot find definition for {function_name}")
                        break

                    data = json.loads(repair_json(reply))
                    if data.get("function_definitions"):
                        # Only include complete definitions
                        complete_definitions = [
                            defn
                            for defn, complete in zip(
                                data["function_definitions"], data.get("complete", [])
                            )
                            if complete == "yes"
                        ]

                        # Check if there are any "no" entries in complete list
                        has_incomplete = any(
                            complete == "no" for complete in data.get("complete", [])
                        )

                        if has_incomplete:
                            if self.debug:
                                logger.info(
                                    f"Found incomplete definitions for {function_name}, continuing search..."
                                )
                            # Don't break, continue to next iteration
                        elif complete_definitions:
                            for definition in complete_definitions:
                                if definition not in function_definitions:
                                    function_definitions.append(definition)
                            break
                    else:
                        complete_definitions = []

                    if cannot_extend:
                        break
                    count += 3
                    sleep(0.5)  # avoid rate limit

            logger.info("Finish fetching function definitions for %s", function_name)
            function_definitions = list(set(function_definitions))

            # If line_number is provided and we have multiple definitions, use LLM to select the correct one
            if line_number is not None and len(function_definitions) > 1:
                if self.debug:
                    logger.info(
                        f"Multiple function definitions found for {function_name}, using line number {line_number} to select"
                    )
                selected_definition = self._select_definition_by_line_number(
                    function_name, line_number, function_definitions
                )
                return (
                    [selected_definition]
                    if selected_definition
                    else function_definitions
                )
            else:
                return function_definitions


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

    def __call__(
        self, class_definition: str, target_name: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
                template_vars["extraction_target"] = (
                    f"only the member variable named '{target_name}'"
                )

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
                "attributes": var.get("attributes", []),
            }

            # Validate access specifier
            if normalized_var["access"] not in [
                "public",
                "private",
                "protected",
                "static",
                "const",
            ]:
                normalized_var["access"] = "private"

            # Ensure attributes is a list
            if not isinstance(normalized_var["attributes"], list):
                normalized_var["attributes"] = []

            valid_variables.append(normalized_var)

        # If target_name is specified, find and return only that variable
        if target_name:
            target_var = next(
                (var for var in valid_variables if var["name"] == target_name), None
            )
            return target_var

        # Otherwise return all variables
        return valid_variables


class ExtractClassTemplateParametersQuery:
    """
    Extracts template parameters from class definitions using LLM analysis.
    This query identifies type parameters in template/generic class definitions
    and returns them as a list of parameter names.

    Examples:
    - template<typename T> class MyClass -> ["T"]
    - template<class Key, class Value> class Map -> ["Key", "Value"]
    - template<typename T = int> class DefaultTemplate -> ["T"]
    - class MyClass (no template) -> []
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Extract template parameters from the given class definition.

Class definition:
{{class_definition}}

Your task is to identify and extract all template parameter names from the class definition.
Look for patterns like:
- template<typename T> class MyClass { ... }
- template<class T, class U> struct MyStruct { ... }
- template<typename KeyType, typename ValueType> class Map { ... }
- template<typename T = int> class DefaultTemplate { ... }
- template<class T> concept MyConcept = ... (C++20 concepts)

IMPORTANT: Only extract template parameters from class/struct definitions, not function templates.

Return only the parameter names (e.g., "T", "U", "KeyType", "ValueType") in a list.
If no template parameters are found, return an empty list.

Note: Handle different language syntax:
- C++: template<typename T> class, template<class T> struct
- Java: public class MyClass<T>, public interface MyInterface<T>
- C#: public class MyClass<T>, public struct MyStruct<T>
- Python: class MyClass(Generic[T]), class MyInterface(Generic[T])
- C# with constraints: public class MyClass<T> where T : new()
""",
            output_format=r"""Return JSON:
{
  "template_parameters": ["param1", "param2", ...]
}
""",
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, class_definition: str) -> List[str]:
        """
        Extract template parameters from a class definition.

        Args:
            class_definition: The class definition text to analyze

        Returns:
            List of template parameter names (e.g., ["T", "U"] for template<typename T, typename U>)
        """
        if not class_definition:
            return []

        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            class_length=len(class_definition) if class_definition else 0,
        ):
            prompt = self.prompt_template.call(class_definition=class_definition)

            try:
                # Use generator to get LLM response
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()
                reply = generator(input_str=prompt).data.strip()
                set_global_config_value("generator.json_output", False)

                if self.debug:
                    generator.save_result()

                data = json.loads(repair_json(reply))
                template_params = data.get("template_parameters", [])

                if self.debug:
                    logger.info(f"Extracted template parameters: {template_params}")

                return template_params

            except Exception as exc:
                if self.debug:
                    logger.error(f"Error parsing template parameters: {exc}")
                return []


class ExtractTypeParameterQuery:
    """
    Extracts type parameters from compound type expressions.
    This query identifies type parameters within complex type expressions
    and returns them as a list of parameter names.

    Examples:
    - vector<T> -> ["T"]
    - map<string, vector<int>> -> [] (no type parameters, only concrete types)
    - Optional<MyClass<T>> -> ["T"]
    - Result<T, Error> -> ["T", "Error"] (depends on whether Error is a type parameter)
    - unordered_map<K, V> -> ["K", "V"]
    - int (no type parameters) -> []
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Extract type parameters from the given type expression.

Type expression:
{{type_expression}}

Known type parameters (if any):
{{type_parameters}}

Your task is to identify and extract all type parameter names from the type expression.
Look for type parameters within template instantiations, nested templates, and generic types.
If known type parameters are provided, prioritize those in your extraction, but also include any additional type parameters you find.

Return only the type parameter names (e.g., "T", "U", "K", "V") in a list.
If no type parameters are found, return an empty list.

Examples:
- vector<T> -> ["T"]
- map<string, vector<int>> -> [] (no type parameters, only concrete types)
- Optional<MyClass<T>> -> ["T"]
- Result<T, Error> -> ["T", "Error"] (if Error is a type parameter)
- unordered_map<K, vector<V>> -> ["K", "V"]

When known type parameters are provided:
- Type expression: "vector<T>", Known: ["T"] -> ["T"]
- Type expression: "map<K, V>", Known: ["K", "V", "W"] -> ["K", "V"]
- Type expression: "Optional<SomeType>", Known: ["T", "U"] -> [] (SomeType is not a known parameter)

""",
            output_format=r"""Return JSON:
{
  "type_parameters": ["param1", "param2", ...]
}
""",
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(
        self, type_expression: str, type_parameters: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract type parameters from a type expression.

        Args:
            type_expression: The type expression text to analyze
            type_parameters: Optional list of known type parameter names to help guide extraction

        Returns:
            List of type parameter names (e.g., ["T", "U"] for vector<map<T, U>>)
        """
        if not type_expression:
            return []

        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            type_length=len(type_expression) if type_expression else 0,
        ):
            prompt = self.prompt_template.call(
                type_expression=type_expression, type_parameters=type_parameters or []
            )

            try:
                # Use generator to get LLM response
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()
                reply = generator(input_str=prompt).data.strip()
                set_global_config_value("generator.json_output", False)

                if self.debug:
                    generator.save_result()

                data = json.loads(repair_json(reply))
                type_params = data.get("type_parameters", [])

                if self.debug:
                    logger.info(
                        f"Extracted type parameters from '{type_expression}': {type_params}"
                    )

                return type_params

            except Exception as exc:
                if self.debug:
                    logger.error(f"Error parsing type parameters: {exc}")
                return []


class TypeParameterSubstitutionQuery:
    """
    Substitutes type parameters in compound type expressions with concrete types using LLM analysis.
    This query replaces type parameters in complex type expressions with their corresponding concrete types
    based on provided mappings.

    Examples:
    - Type: "vector<T>", Mapping: {"T": "int"} -> "vector<int>"
    - Type: "map<K, vector<V>>", Mapping: {"K": "string", "V": "int"} -> "map<string, vector<int>>"
    - Type: "Optional<MyClass<T>>", Mapping: {"T": "User"} -> "Optional<MyClass<User>>"
    - Type: "Result<T, Error>", Mapping: {"T": "Data"} -> "Result<Data, Error>" (if Error is not in mapping)
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Substitute type parameters in the given type expression with concrete types using the provided mapping.

Type expression:
{{type_expression}}

Type parameter mapping (parameter -> concrete type):
{{type_mapping}}

Your task is to substitute type parameters in the type expression with their corresponding concrete types from the mapping.
Only substitute parameters that are present in the mapping - leave other parameters unchanged.

Rules:
1. Only substitute exact parameter names (e.g., "T", not partial matches like "T" in "MyClassT")
2. Preserve the original syntax and structure of the type expression
3. Handle nested types correctly (e.g., vector<map<T, U>>)
4. Support multiple languages (C++, Java, Python, C#)
5. If a parameter is not in the mapping, leave it as-is
6. If no substitutions can be made, return the original type expression unchanged

Examples:
- Type: "vector<T>", Mapping: {"T": "int"} -> "vector<int>"
- Type: "map<K, V>", Mapping: {"K": "string", "V": "int"} -> "map<string, int>"
- Type: "Optional<MyClass<T>>", Mapping: {"T": "User"} -> "Optional<MyClass<User>>"
- Type: "Result<T, Error>", Mapping: {"T": "Data"} -> "Result<Data, Error>"
- Type: "vector<ClassT<T>>", Mapping: {"T": "int"} -> "vector<ClassT<int>" (don't replace T in ClassT)
- Type: "list<int>", Mapping: {"T": "string"} -> "list<int>" (no parameters to replace)

Different language syntax support:
- C++: vector<T>, map<K, V>, unique_ptr<T[]>
- Java: List<T>, Map<K, V>, Optional<T>
- Python: List[T], Dict[K, V], Optional[T]
- C#: List<T>, Dictionary<K, V>, Nullable<T>
""",
            output_format=r"""Return JSON:
{
  "concrete_type": "the type expression with parameters substituted"
}
""",
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, type_expression: str, type_mapping: Dict[str, str]) -> str:
        """
        Substitute type parameters in a type expression with concrete types.

        Args:
            type_expression: The type expression containing parameters (e.g., "vector<T>")
            type_mapping: Dictionary mapping parameter names to concrete types (e.g., {"T": "int"})

        Returns:
            The type expression with parameters substituted (e.g., "vector<int>")
        """
        if not type_expression or not type_mapping:
            return type_expression

        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            type_length=len(type_expression) if type_expression else 0,
        ):
            prompt = self.prompt_template.call(
                type_expression=type_expression, type_mapping=str(type_mapping)
            )

            try:
                # Use generator to get LLM response
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()
                reply = generator(input_str=prompt).data.strip()
                set_global_config_value("generator.json_output", False)

                if self.debug:
                    generator.save_result()

                data = json.loads(repair_json(reply))
                concrete_type = data.get("concrete_type", type_expression)

                if self.debug:
                    logger.info(
                        f"Substituted type parameters in '{type_expression}' -> '{concrete_type}' using mapping: {type_mapping}"
                    )

                return concrete_type

            except Exception as exc:
                if self.debug:
                    logger.error(f"Error substituting type parameters: {exc}")
                return type_expression


class ExtractFunctionTemplateParametersQuery:
    """
    Extracts template parameters from function definitions using LLM analysis.
    This query identifies type parameters in template/generic function definitions
    and returns them as a list of parameter names.

    Examples:
    - template<typename T> void myFunction(T param) -> ["T"]
    - template<class T, class U> T createObject(U input) -> ["T", "U"]
    - template<typename Iter> void sort(Iter begin, Iter end) -> ["Iter"]
    - void regularFunction(int param) -> []
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Extract template parameters from the given function definition.

Function definition:
{{function_definition}}

Your task is to identify and extract all template parameter names from the function definition.
Look for patterns like:
- template<typename T> void myFunction(T param) { ... }
- template<class T, class U> T createObject(U input) { ... }
- template<typename Iter> void sort(Iter begin, Iter end) { ... }
- template<typename Ret, typename... Args> Ret callFunction(Args... args) { ... }
- template<template<typename> class Container, typename T> void process(Container<T>& container) { ... }

IMPORTANT: Only extract template parameters from function definitions, not class templates.

Return only the parameter names (e.g., "T", "U", "Iter", "Ret", "Args") in a list.
If no template parameters are found, return an empty list.

Note: Handle different language syntax:
- C++: template<typename T> function, template<class T> function
- Java: public <T> void myMethod(T param), public static <T> T createInstance()
- C#: public T Create<T>(), public void Process<T, U>(T item, U other)
- Python: def func[T](param: T) -> T (Python 3.12+), generic type hints
- C# with constraints: public T Create<T>() where T : new()
""",
            output_format=r"""Return JSON:
{
  "template_parameters": ["param1", "param2", ...]
}
""",
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, function_definition: str) -> List[str]:
        """
        Extract template parameters from a function definition.

        Args:
            function_definition: The function definition text to analyze

        Returns:
            List of template parameter names (e.g., ["T", "U"] for template<typename T, typename U>)
        """
        if not function_definition:
            return []

        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            function_length=len(function_definition) if function_definition else 0,
        ):
            prompt = self.prompt_template.call(function_definition=function_definition)

            try:
                # Use generator to get LLM response
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()
                reply = generator(input_str=prompt).data.strip()
                set_global_config_value("generator.json_output", False)

                if self.debug:
                    generator.save_result()

                data = json.loads(repair_json(reply))
                template_params = data.get("template_parameters", [])

                if self.debug:
                    logger.info(
                        f"Extracted function template parameters: {template_params}"
                    )

                return template_params

            except Exception as exc:
                if self.debug:
                    logger.error(f"Error parsing function template parameters: {exc}")
                return []


class FunctionNameExtractor:
    """Extracts base function names and type arguments from function calls using LLM across multiple programming languages."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Extract the base function name and type arguments from the following function call:

Function call: {{function_name}}

Your task is to:
1. Extract the base function name (without type arguments)
2. Extract all type arguments as a list
3. Handle different language format, for instance
   - C++: namespace::func<param1, param2>(args) -> base: "namespace::func", type_args: ["param1", "param2"]; or func<param1, param2>(args) -> base: "func", type_args: ["param1", "param2"]
   - Java: GenericExample.<String>getFirstElement(args) -> base: "getFirstElement", type_args: ["String"]

            """,
            output_format=r"""Return strict JSON:
{
    "base_function_name": "extracted_base_name",
    "type_arguments": ["param1", "param2", ...]
}

If there are no type arguments, return an empty list for type_arguments.
Always return valid JSON with both fields present.""",
        )
    )

    def __init__(self, debug: bool = False):
        """
        Initialize the FunctionNameExtractor.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

    def __call__(self, function_name: str) -> tuple[str, list[str]]:
        """
        Extract the base function name and type arguments from a function call string.

        Handles various formats across multiple languages:
        - C++: func<param1, param2>(args)
        - Java: GenericExample.<String>getFirstElement(args)
        - TypeScript: func<string, number>(args)

        Args:
            function_name: The function name that may contain type arguments

        Returns:
            A tuple of (base_function_name, type_arguments)
        """
        if not function_name or ("<" not in function_name and ">" not in function_name):
            return function_name, []

        with call_debug_scope(
            self.__class__.__name__, self.debug, function_name=function_name
        ):
            prompt = self.prompt_template.call(function_name=function_name)
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            if self.debug:
                generator.save_result()
            set_global_config_value("generator.json_output", False)

            try:
                data = json.loads(repair_json(reply))
                base_name = data.get("base_function_name", function_name)
                type_args = data.get("type_arguments", [])

                # Clean up the type arguments
                type_args = [self._clean_type_name(arg) for arg in type_args]

                if self.debug:
                    logger.info(
                        f"Extracted function name '{base_name}' with type arguments: {type_args}"
                    )

                return base_name, type_args
            except Exception as e:
                if self.debug:
                    logger.error(f"Error parsing function name extraction result: {e}")
                # Fallback: return original function name if parsing fails
                return function_name, []

    def _clean_type_name(self, type_name: str) -> str:
        """
        Clean the type name by removing any leading or trailing whitespace or type decorators.

        Args:
            type_name: The type name to clean

        Returns:
            The cleaned type name
        """
        type_name = type_name.strip()
        # Remove common type decorators
        for decorator in ["const ", "static ", "constexpr ", "volatile ", "final "]:
            if type_name.startswith(decorator):
                type_name = type_name[len(decorator) :]
        # Remove trailing references and pointers, but preserve array brackets
        while type_name.endswith(("&", "*", "?")):
            type_name = type_name[:-1].strip()
        return type_name.strip()


class IsChainedCallQuery:
    """Determines if a statement or expression is a chained call."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Analyze the given code expression to determine if it is a chained call.

Expression: {{expression}}
{% if context %}
Context: {{context}}
{% endif %}

A chained call is an expression that involves member access operations ending with a function call, such as:
- obj.method() (single method call)
- obj->method() (single method call through pointer)
- obj.method1().method2().method3() (multiple sequential method calls)
- obj->method1()->method2() (multiple sequential method calls through pointers)
- obj.property.method() (property access followed by method call)

Key indicators of chained calls:
- Access operators (., ->) connecting objects to their members
- Expression ends with a function call ()
- May involve single or multiple access operations in sequence
- Function calls may return objects which are immediately used for further access
- Can include property access patterns that end with method calls
            """,
            output_format=r"""Return strict JSON:
{
    "is_chained_call": true/false,
    "confidence": "high"|"medium"|"low",
    "chain_components": [
        {
            "component": "component_text",
            "type": "method_call"|"member_access"|"function_call"|"property_access",
            "operator": "."|"->"|"::"|"()"
        }
    ],
    "explanation": "brief explanation of the analysis"
}

If is_chained_call is false, chain_components should be empty.
If is_chained_call is true, break down the chain into its components in order.
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, expression: str, context: str = "") -> Dict[str, Any]:
        """
        Determine if the given expression is a chained call.

        Args:
            expression: The expression to analyze
            context: Optional surrounding code context for better analysis

        Returns:
            Dictionary containing:
            - is_chained_call: Boolean indicating if it's a chained call
            - confidence: Confidence level in the analysis
            - chain_components: List of components if it's a chain
            - explanation: Brief explanation of the analysis
        """
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            expression=expression,
            context_length=len(context) if context else 0,
        ):
            prompt = self.prompt_template.call(
                expression=expression,
                context=context,
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
                "IsChainedCallQuery: JSON parse failed for '%s': %s", expression, exc
            )
            return {
                "is_chained_call": False,
                "confidence": "low",
                "chain_components": [],
                "explanation": f"Failed to analyze expression: {exc}",
            }

        # Validate and normalize the response
        result = {
            "is_chained_call": bool(data.get("is_chained_call", False)),
            "confidence": data.get("confidence", "low"),
            "chain_components": data.get("chain_components", []) or [],
            "explanation": data.get("explanation", ""),
        }

        # Ensure confidence is valid
        if result["confidence"] not in ["high", "medium", "low"]:
            result["confidence"] = "low"

        # Ensure chain_components is a list of valid objects
        valid_components = []
        if isinstance(result["chain_components"], list):
            for component in result["chain_components"]:
                if isinstance(component, dict):
                    valid_component = {
                        "component": component.get("component", ""),
                        "type": component.get("type", ""),
                        "operator": component.get("operator", ""),
                    }
                    valid_components.append(valid_component)
        result["chain_components"] = valid_components

        return result


class AssignmentQuery:
    """Finds the latest assignment of a variable from the given context."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Your task is to analyze the given code context and find the latest assignment to the specified variable.

Context:
{{context}}

Target Variable: {{variable_name}}

Please find the most recent assignment to this variable and return the assignment expression.
If the variable is declared with a type (e.g., "int x = 5;"), return the expression after the equals sign (e.g., "5").
If the variable is reassigned (e.g., "x = 10;"), return the expression after the equals sign (e.g., "10").
If no assignment is found, return None.
            """,
            output_format=r"""Return strict JSON:
{"assignment": (assignment_expr, line_number)}
If there is no line number info in the context, set the line_number to None
If no assignment is found, return 
{"assignment": None}
            """,
        )
    )

    def __init__(self, debug: bool = False):
        """
        Initialize the assignment query.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

    def __call__(self, context: str, variable_name: str) -> Optional[str]:
        """
        Find the latest assignment of the variable from the context.

        Args:
            context: The code context to search
            variable_name: The variable name to find assignments for

        Returns:
            The assignment expression if found, None otherwise
        """
        with call_debug_scope(
            self.__class__.__name__, self.debug, variable=variable_name
        ):
            try:
                prompt = self.prompt_template.call(
                    context=context, variable_name=variable_name
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
                        f"AssignmentQuery: JSON parse failed for '{variable_name}': {exc}"
                    )
                    return None

                assignment = data.get("assignment")
                return assignment if assignment else None

            except Exception as exc:
                logger.error(
                    f"AssignmentQuery: Error finding assignment for '{variable_name}': {exc}"
                )
                return None


class VarTypeInferenceQuery:
    """Resolves the type of variables considering both static and dynamic types."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Your task is to analyze the given code context and find the static type declaration for the specified variable.

Context:
{{context}}

Target Variable: {{variable_name}}

Please find the variable declaration and determine its static type:
- For explicit declarations like "int x = 5;", the static type is "int"
- For "auto x = 5;" (C++), infer the type from the initial expression as "int"
- For "var x = 5;" (Java/JavaScript), infer the type from the initial expression as appropriate
- For "const x = 5;" or "let x = 5;", infer the type from the initial expression
- If no declaration is found, static_type should be None

Return only the static type information.
            """,
            output_format=r"""Return strict JSON:
{
    "static_type": "declared_type"
}
If you cannot decide the static type, return
{
    "static_type": None
}
""",
        )
    )

    def __init__(
        self,
        query_dynamic_type=None,
        debug: bool = False,
        expression_type_inference_query=None,
    ):
        self.debug = debug
        self.assignment_query = AssignmentQuery(debug=debug)
        self.query_dynamic_type = query_dynamic_type
        self.expression_type_inference_query = expression_type_inference_query

    def set_expression_type_inference_query(self, expression_type_inference_query):
        """Set the ExpressionTypeInferenceQuery to resolve circular dependency."""
        self.expression_type_inference_query = expression_type_inference_query

    def __call__(
        self,
        variable_name: str,
        file_path: str,
        expression: str,
        line_number: int,
        language: str = "",
    ) -> Optional[str]:
        """Determine the type of a variable from context.
        Args:
            context: The code context to search
            variable_name: The variable name to find assignments for
            file_path: The path of the file that contains context
            expression: The expression that uses the variable
            line_number: The line number of the expression in the file
            language: The programming language of the expression

        Returns:
            The final runtime type as a string, or None if type cannot be determined.
        """
        # Track recursion depth to prevent infinite loops
        with call_debug_scope(
            self.__class__.__name__, self.debug, variable=variable_name
        ):

            rag = RAG()
            rag.get_docs()
            doc = rag.id2doc[rag.codePath2beginDocid[file_path]]
            while True:
                if (
                    doc.meta_data["start_line"] <= line_number
                    and doc.meta_data["end_line"] >= line_number
                ):
                    self.doc = doc
                    break
                if doc.meta_data["next_doc_id"] == None:
                    break
                doc = rag.id2doc[doc.meta_data["next_doc_id"]]
                if doc == None:
                    break
            language = self.doc.meta_data["programming_language"]

            if not self.doc:
                raise ValueError(
                    f"Could not find the document for line number {line_number} in file {file_path}"
                )
            #!WARNING: The following logic is built on the assumption that the var decl is in the same file of the current call
            response = None
            count = 3
            while count <= COUNT_UPPER_LIMIT:
                context, cannot_extend = build_context(
                    retrieved_doc=self.doc,
                    id2doc=rag.id2doc,
                    direction="previous",
                    count=count,
                )

                #! Following the above warning, the `file_path` should be changed in the future
                prompt = self.prompt_template.call(
                    context=context, variable_name=variable_name
                )
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()
                response = generator(input_str=prompt).data.strip()
                if self.debug:
                    generator.save_result()
                set_global_config_value("generator.json_output", False)
                if cannot_extend:
                    break
                if response:
                    break
                else:
                    count += 3
                    sleep(0.5)

            if not response:
                # the var decl is not in the current file (where the var is used in the expression)
                retrieved_docs = rag.retrieve(
                    bm25_keywords=f"[VARDECL]{variable_name}",
                    faiss_query="",
                )[0].documents

                if not retrieved_docs:
                    return None

            try:
                data = json.loads(repair_json(response))
                static_type = data.get("static_type")
                logger.info(
                    f"VarTypeInferenceQuery: Static type for '{variable_name}': {static_type}"
                )
                if static_type == "None":
                    static_type = None
            except Exception as exc:
                logger.error(
                    f"VarTypeInferenceQuery: JSON parse failed for '{variable_name}': {exc}"
                )
                static_type = None
            if static_type and not self.query_dynamic_type:
                return static_type
            # Step 2: Find latest assignment using AssignmentQuery
            assignment = self.assignment_query(context, variable_name)
            if assignment == "None":
                assignment = None
            # Step 3: Determine dynamic type from assignment expression
            dynamic_type = None
            if assignment and self.expression_type_inference_query:
                assignment_expr, line_number = assignment
                try:
                    dynamic_type = self.expression_type_inference_query(
                        assignment_expr, file_path, line_number, language
                    )
                    logger.info(
                        f"VarTypeInferenceQuery: Dynamic type for '{variable_name}': {dynamic_type}"
                    )
                except Exception as exc:
                    if self.debug:
                        logger.debug(
                            f"VarTypeInferenceQuery: Could not infer dynamic type for '{variable_name}': {exc}"
                        )
                    dynamic_type = None
            # Step 4: Apply rules for final runtime type
            if static_type is None:
                final_type = dynamic_type
            elif dynamic_type is None:
                final_type = static_type
            else:
                # Both types exist, dynamic type takes precedence
                final_type = dynamic_type
            if self.debug:
                logger.debug(
                    f"VarTypeInferenceQuery: {variable_name} -> static={static_type}, dynamic={dynamic_type}, final={final_type}"
                )
            return final_type


class ExpressionSegmentationQuery:
    """Segments expressions into literals, identifiers, and function calls using LLM-based analysis."""

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""You are given a code expression: {{expression}} Your task is to segment this expression into its fundamental components.""",
            instructions=r"""Analyze the expression and break it down into segments based on operators.

Expression Segmentation Philosophy:
1. Every expression is either:
   - A literal (e.g., 123, "hello", true, 3.14)
   - An identifier (e.g., variable names, function names)
   - A function call (e.g., func(), obj.method())
   - A combination of the above connected by operators

2. Member access operators (., ->, ::) are part of function calls/identifiers, not separate operators
3. Focus on breaking at actual operators that separate distinct expression parts
4. Classify each segment by its type: "literal", "identifier", or "function_call"
5. Return segments as tuples with their type classifications

Type Classification Rules:
- "literal": Numeric literals, string literals, boolean literals (true/false), null/nil literals
- "identifier": Variable names, function names (without parentheses), object names
- "function_call": Any expression ending with parentheses, including method calls

Examples:
- "1 + a->f().g()->m" -> [["1", "literal"], ["a->f().g()->m", "function_call"]]
- "a + b * c" -> [["a", "identifier"], ["b", "identifier"], ["c", "identifier"]]
- "obj.method() + func(x)" -> [["obj.method()", "function_call"], ["func(x)", "function_call"]]
- "arr[i] + len" -> [["arr[i]", "identifier"], ["len", "identifier"]]
- "x = true ? y : z" -> [["x", "identifier"], ["true", "literal"], ["y", "identifier"], ["z", "identifier"]]
- "123 + variable" -> [["123", "literal"], ["variable", "identifier"]]
- '"string" + name' -> [['"string"', "literal"], ["name", "identifier"]]
- "function_call()" -> [["function_call()", "function_call"]]

Important: Return segments as tuples with their type classifications, not just the segments alone.""",
            output_format=r"""Return JSON: {"result": [["segment1", "type1"], ["segment2", "type2"], ["segment3", "type3"], ...]}""",
        )
    )

    def __init__(self, debug: bool = False):
        """
        Initialize the expression segmentation query.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

    def __call__(self, expression: str) -> Optional[List[Tuple[str, str]]]:
        """
        Segment an expression into its fundamental components with type classifications.

        Args:
            expression: The expression to segment

        Returns:
            List of tuples containing (segment, type) where type is "literal", "identifier", or "function_call",
            or None if segmentation fails
        """
        try:
            with call_debug_scope(
                self.__class__.__name__, self.debug, expression=expression[:50]
            ):
                if self.debug:
                    logger.debug(
                        f"ExpressionSegmentationQuery: Segmenting expression: {expression}"
                    )

                # Generate prompt for the LLM
                prompt = self.prompt_template.call(expression=expression)

                # Set up generator for LLM call
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()

                # Call LLM
                reply = generator(input_str=prompt).data.strip()

                if self.debug:
                    logger.debug(f"ExpressionSegmentationQuery: Raw LLM reply: {reply}")

                # Parse the JSON response
                try:
                    result = json.loads(reply)

                    # Extract the result array from the JSON object
                    if isinstance(result, dict) and "result" in result:
                        segments = result["result"]
                        if isinstance(segments, list):
                            # Convert list of lists to list of tuples
                            segment_tuples = []
                            for segment in segments:
                                if isinstance(segment, list) and len(segment) == 2:
                                    segment_tuples.append((segment[0], segment[1]))
                                else:
                                    logger.error(
                                        f"ExpressionSegmentationQuery: Invalid segment format: {segment}"
                                    )
                                    return None

                            if self.debug:
                                logger.debug(
                                    f"ExpressionSegmentationQuery: Segments: {segment_tuples}"
                                )
                            return segment_tuples
                        else:
                            logger.error(
                                "ExpressionSegmentationQuery: Result field is not a list"
                            )
                            return None
                    else:
                        logger.error(
                            "ExpressionSegmentationQuery: Response missing 'result' field"
                        )
                        return None

                except json.JSONDecodeError as exc:
                    logger.error(
                        f"ExpressionSegmentationQuery: JSON parse failed: {exc}"
                    )
                    return None

        except Exception as exc:
            logger.error(
                f"ExpressionSegmentationQuery: Error segmenting expression: {exc}"
            )
            return None


class LiteralTypeInferenceQuery:
    """
    Infers the type of literal expressions using LLM-based analysis.
    This class handles type inference for literal values using language model intelligence.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""You are given a literal expression: {{literal}} Your task is to determine its type.""",
            instructions=r"""Analyze the literal expression and determine its most appropriate type.

Literal Type Classification Rules:
1. String literals: Values enclosed in single or double quotes (e.g., "hello", 'world')
2. Numeric literals: Integer, floating-point, hexadecimal, binary, or octal values
3. Boolean literals: true, false (case-insensitive)
4. Null/nil literals: null, nil, None (case-insensitive)
5. Character literals: Single characters in single quotes (e.g., 'a', '1')
6. Other literals: Any other literal values that can be typed

Type Naming Conventions:
- Use common programming language type names: "string", "int", "float", "bool", "char", "null"
- For complex literals, use the most specific type possible
- If unsure, use the most general applicable type

Examples:
- "hello world" -> "string"
- 'test' -> "string"
- 42 -> "int"
- 3.14 -> "float"
- 0xFF -> "int"
- 0b1010 -> "int"
- true -> "bool"
- false -> "bool"
- null -> "null"
- 'a' -> "char"
- '1' -> "char"
- None -> "null"
- 100L -> "long" (if language supports long integers)
- 3.14f -> "float" (float literal)
- 2.718281828459045 -> "double" (high precision float)

Important: Return only the type name as a string, not the literal value.""",
            output_format=r"""Return JSON: {"type": "<type_name>"}""",
        )
    )

    def __init__(self, debug: bool = False):
        """
        Initialize the literal type inference query.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

    def __call__(self, literal: str) -> Optional[str]:
        """
        Infer the type of a literal expression using LLM analysis.

        Args:
            literal: The literal expression to analyze

        Returns:
            The inferred type as a string, or None if the literal cannot be typed
        """
        try:
            with call_debug_scope(
                self.__class__.__name__, self.debug, literal=literal[:50]
            ):
                if self.debug:
                    logger.debug(
                        f"LiteralTypeInferenceQuery: Analyzing literal: {literal}"
                    )

                # Generate prompt for the LLM
                prompt = self.prompt_template.call(literal=literal)

                # Set up generator for LLM call
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()

                # Call LLM
                reply = generator(input_str=prompt).data.strip()

                if self.debug:
                    logger.debug(f"LiteralTypeInferenceQuery: Raw LLM reply: {reply}")

                # Parse the JSON response
                try:
                    result = json.loads(reply)

                    # Extract the type from the JSON object
                    if isinstance(result, dict) and "type" in result:
                        literal_type = result["type"]
                        if isinstance(literal_type, str) and literal_type.strip():
                            if self.debug:
                                logger.debug(
                                    f"LiteralTypeInferenceQuery: Inferred type: {literal_type}"
                                )
                            return literal_type.strip()
                        else:
                            logger.error(
                                "LiteralTypeInferenceQuery: Type field is empty or not a string"
                            )
                            return None
                    else:
                        logger.error(
                            "LiteralTypeInferenceQuery: Response missing 'type' field"
                        )
                        return None

                except json.JSONDecodeError as exc:
                    logger.error(f"LiteralTypeInferenceQuery: JSON parse failed: {exc}")
                    return None

        except Exception as exc:
            logger.error(
                f"LiteralTypeInferenceQuery: Error analyzing literal '{literal}': {exc}"
            )
            return None


class ExpressionTypeInferenceQuery:
    """A simple proxy for ChainedExpressionTypeAnalyzerPipeline that infers expression types."""

    def __init__(self, debug: bool = False, query_dynamic_type: bool = False):
        """
        Initialize the expression type inference query as a proxy.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.callee_definition_fetch = FetchCalleeInfoPipeline(debug=debug)
        # Set up circular dependency
        self.callee_definition_fetch.set_expression_type_inference_query(self)
        self.var_type_inference_query = VarTypeInferenceQuery(
            debug=debug,
            expression_type_inference_query=self,
            query_dynamic_type=query_dynamic_type,
        )
        self.expression_segmentation_query = ExpressionSegmentationQuery(debug=debug)
        self.literal_type_inference_query = LiteralTypeInferenceQuery(debug=debug)

    def __call__(
        self, expression: str, file_path: str, line_number: int, language: str = ""
    ) -> Optional[str]:
        """
        Determine the type of an expression by delegating to ChainedExpressionTypeAnalyzerPipeline.

        This is a simple proxy that forwards all expression type inference requests to the
        more sophisticated ChainedExpressionTypeAnalyzerPipeline.

        Args:
            expression: The expression to infer type for
            file_path: The file path where the expression is located
            line_number: The line number where the expression is located
            language: The programming language of the code context (optional)

        Returns:
            The inferred type as a string, or None if type cannot be determined
        """

        try:
            with call_debug_scope(
                self.__class__.__name__, self.debug, expression=expression[:50]
            ):
                if self.debug:
                    logger.debug(
                        f"ExpressionTypeInferenceQuery: Proxying to ChainedExpressionTypeAnalyzerPipeline for expression: {expression}"
                    )
                count = 3

                segments = self.expression_segmentation_query(expression)
                if segments is None:
                    return None
                types = []
                for segment in segments:
                    if segment[1] == "literal":
                        types.append(self.literal_type_inference_query(segment[0]))
                    elif segment[1] == "function_call":
                        # expression: str, file_path: str, line_number: int
                        func_type, func_def = self.callee_definition_fetch(
                            segment[0], file_path, line_number
                        )
                        if func_type is None:
                            types.append(None)
                        else:
                            types.append(func_type)
                    elif segment[1] == "variable":
                        res = self.var_type_inference_query(
                            segment[0], file_path, expression, line_number, language
                        )
                        if res:
                            types.append(res)

                if self.debug:
                    logger.debug(
                        f"ExpressionTypeInferenceQuery: Got result from proxy: {result}"
                    )

                return result

        except Exception as exc:
            logger.error(
                f"ExpressionTypeInferenceQuery: Error in proxy call for '{expression}': {exc}"
            )
            return None


class ElementTypeAnalyzerQuery:
    """
    Determines the element type given a compound type and an expression.

    This query uses LLM-based reasoning to determine the return type of indexing operations
    on both standard containers and custom data structures. It analyzes the compound type
    and expression to deduce the element type without using traditional parsing or regex tricks.

    Examples:
        - Compound type: List<int>, Expression: a[0] → Result: int
        - Compound type: Map<int, string>, Expression: a[12345] → Result: string
        - Compound type: MyCustomList<float>, Expression: a[2] → Result: float
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
The type of {{base_expression}} is {{compound_type}}.

Your task is to determine what type the expression {{indexed_expression}}.

Analyze this step by step:
1. Identify the indexing operation in {{indexed_expression}}
2. Determine the compound type structure (container/mapping) of {{compound_type}}
3. Deduce the element type based on the indexing operation
4. Consider both standard containers and custom data structures

Rules for deduction:
- For standard containers like List<T>, Vector<T>, Array<T>: T is the element type
- For mapping types like Map<K, V>, Dictionary<K, V>: V is the element type when indexed by K
- For custom data structures: analyze the structure name and any provided context
- If it's a custom container ending in 'List', 'Array', 'Vector': the template parameter is likely the element type
- If it's a custom mapping ending in 'Map', 'Dictionary': the second template parameter is likely the value type

Examples:
- List<int> with base 'a' and indexed 'a[0]' → int
- Map<int, string> with base 'm' and indexed 'm[123]' → string
- Vector<double> with base 'v' and indexed 'v[i]' → double
- MyCustomList<float> with base 'data' and indexed 'data[2]' → float
- CustomMap<string, bool> with base 'config' and indexed 'config["key"]' → bool

{% if context %}
Additional Context:
The following code context may help with custom data structure analysis:
{{context}}
{% endif %}

            """,
            output_format=r"""Return JSON:
{
  "element_type": "<deduced_type>",
  "reasoning": "<brief explanation of your deduction>"
}
""",
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(
        self,
        compound_type: str,
        base_expression: str,
        indexed_expression: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Determine the element type given a compound type and an expression.

        Args:
            compound_type: The compound type (e.g., "List<int>", "Map<int, string>")
            base_expression: The base expression (e.g., "a", "m", "data")
            indexed_expression: The indexed expression (e.g., "a[0]", "m[12345]", "data[2]")
            context: Additional context about the codebase or custom data structures

        Returns:
            The deduced element type as a string
        """
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            compound_type=compound_type,
            base_expression=base_expression,
            indexed_expression=indexed_expression,
        ):
            try:
                prompt = self.prompt_template.call(
                    compound_type=compound_type,
                    base_expression=base_expression,
                    indexed_expression=indexed_expression,
                    context=context,
                )

                # Set JSON output mode
                set_global_config_value("generator.json_output", True)
                generator = GeneratorWrapper()
                reply = generator(input_str=prompt).data.strip()
                set_global_config_value("generator.json_output", False)

                if self.debug:
                    generator.save_result()

                # Parse JSON response
                try:
                    data = json.loads(repair_json(reply))
                    element_type = data.get("element_type", "unknown")
                    reasoning = data.get("reasoning", "No reasoning provided")

                    if self.debug:
                        logger.info(f"Deduced element type: {element_type}")
                        logger.info(f"Reasoning: {reasoning}")

                    return element_type
                except Exception as exc:
                    logger.error(f"ElementTypeAnalyzerQuery: JSON parse failed: {exc}")
                    return "unknown"

            except Exception as exc:
                logger.error(
                    f"ElementTypeAnalyzerQuery: Error analyzing '{compound_type}' with '{indexed_expression}': {exc}"
                )
                return "unknown"


class BuiltinFunctionReturnTypeQuery:
    """
    Determines the return type of built-in/standard library functions using LLM inference.

    This query can work with just the function name, but provides more accurate results
    when class name and context are also provided.

    Examples:
        - function_name="find", class_name="std::map<K, V>" → iterator type
        - function_name="size", class_name="std::vector<T>" → size_t
        - function_name="c_str" (no class_name) → const char*
        - function_name="empty" (no class_name) → bool
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
Determine the return type of a built-in/standard library function.

Function name: {{function_name}}
{% if class_name %}
Class/Container type: {{class_name}}
{% endif %}

Your task is to determine the return type of the function call {% if class_name %}{{class_name}}.{% endif %}{{function_name}}().
This appears to be a built-in function from standard library or common container types.

{% if context %}
Additional context that might help:
{{context}}
{% endif %}

""",
            output_format=r"""Return JSON:
{
  "return_type": "<deduced_return_type>",
  "reasoning": "<brief_explanation_of_reasoning>",
}

If the return type cannot be determined with reasonable confidence, return "unknown" for return_type.
""",
        )
    )

    def __init__(self, debug: bool = False):
        """Initialize the query with optional debug logging."""
        self.debug = debug

    def __call__(
        self, function_name: str, class_name: Optional[str] = None, context: str = ""
    ) -> Optional[str]:
        """
        Determine the return type of a built-in function.

        Args:
            function_name: The name of the built-in function (required)
            class_name: The class/container type (optional)
            context: Additional source code context (optional)

        Returns:
            The deduced return type, or None if cannot be determined
        """
        try:
            if not function_name or not function_name.strip():
                logger.warning(
                    "BuiltinFunctionReturnTypeQuery: function_name is required"
                )
                return None

            prompt = self.prompt_template.call(
                function_name=function_name.strip(),
                class_name=class_name.strip() if class_name else None,
                context=context,
            )

            # Use generator to get LLM response
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            set_global_config_value("generator.json_output", False)

            if self.debug:
                generator.save_result()

            try:
                data = json.loads(repair_json(reply))
                return_type = data.get("return_type")
                reasoning = data.get("reasoning", "")

                if return_type and return_type != "unknown":
                    if self.debug:
                        logger.info(
                            f"Built-in function inference: {function_name}() → {return_type}"
                        )
                        logger.info(f"Reasoning: {reasoning}")
                    return return_type
                else:
                    if self.debug:
                        logger.warning(
                            f"Could not determine return type for built-in function: {function_name}()"
                        )
                        if reasoning:
                            logger.info(f"Reasoning: {reasoning}")
                    return None

            except Exception as exc:
                if self.debug:
                    logger.error(
                        f"Error parsing built-in function inference result: {exc}"
                    )
                return None

        except Exception as exc:
            if self.debug:
                logger.error(
                    f"Error in BuiltinFunctionReturnTypeQuery for {function_name}: {exc}"
                )


class FunctionCallExtractorFromEntryFuncPipeline:
    """
    Pipeline to extract function call chains from an entry function.

    Follows the 6-step process:
    1. Use FetchFunctionDefinitionFromNamePipeline to fetch the definition of the entry function
    2. Use FetchCallExpressionsQuery to extract function calls from the entry function
    3. For each function call expression, use FetchFunctionDefinitionFromNamePipeline to fetch the definition of the called function
    4. Store the current function call globally as entry_function -> called_function1 -> called_function2 -> ...
    5. Recursively go to step 1 to extract function call chains starting from each called function
    6. Handle recursive calls properly
    """

    def __init__(self, max_depth: int = 10, debug: bool = False):
        """Initialize the pipeline with depth limit and debug logging."""
        self.max_depth = max_depth
        self.debug = debug
        self.call_chains = []
        self.visited_functions = set()
        self.fetch_function_def_pipeline = FetchFunctionDefinitionFromNamePipeline(
            debug=debug
        )
        self.fetch_call_expressions_query = FetchCallExpressionsQuery(debug=debug)
        self.function_name_extractor = FunctionNameExtractor(debug=debug)

    def __call__(
        self,
        entry_function_name: str,
        file_path: str,
        line_number: Optional[int] = None,
    ) -> List[CallChain]:
        """
        Extract function call chains starting from the entry function.

        Args:
            entry_function_name: The name of the entry function
            file_path: The path to the file that contains the entry function
            line_number: Optional line number to precisely locate the entry function's header

        Returns:
            List of call chains starting from the entry function
        """
        try:
            self.call_chains = []
            self.visited_functions = set()

            if self.debug:
                logger.info(
                    f"Starting function call extraction from: {entry_function_name} in {file_path}"
                )

            # Step 1: Fetch the definition of the entry function
            entry_defs = self.fetch_function_def_pipeline(
                entry_function_name, file_path=file_path, line_number=line_number
            )
            if not entry_defs:
                if self.debug:
                    logger.warning(
                        f"Could not find definition for entry function: {entry_function_name}"
                    )
                return []

            # Start recursive extraction
            self._extract_call_chain_recursive(
                function_name=entry_function_name,
                file_path=file_path,
                current_chain=[entry_function_name],
                depth=0,
            )

            if self.debug:
                logger.info(f"Extracted {len(self.call_chains)} call chains")

            return self.call_chains

        except Exception as exc:
            if self.debug:
                logger.error(
                    f"Error in FunctionCallExtractorFromEntryFuncPipeline: {exc}"
                )
            return []

    def _extract_call_chain_recursive(
        self,
        function_name: str,
        file_path: str,
        current_chain: List[str],
        depth: int,
        line_number: Optional[int] = None,
    ) -> None:
        """
        Recursively extract function call chains.

        Args:
            function_name: Current function to analyze
            file_path: Path to file containing the function
            current_chain: Current call chain being built
            depth: Current recursion depth
            line_number: Optional line number to precisely locate the function's header
        """
        try:
            # Check recursion limits
            if depth >= self.max_depth:
                if self.debug:
                    logger.warning(
                        f"Max depth {self.max_depth} reached at {function_name}"
                    )
                return

            # Check for recursive calls
            function_key = f"{function_name}:{file_path}"
            if function_key in self.visited_functions:
                if self.debug:
                    logger.info(f"Detected recursive call to {function_name}")
                return

            self.visited_functions.add(function_key)

            if self.debug:
                logger.info(f"Analyzing function: {function_name} at depth {depth}")

            # Step 1: Fetch function definition
            function_defs = self.fetch_function_def_pipeline(
                function_name, file_path=file_path, line_number=line_number
            )
            if not function_defs:
                if self.debug:
                    logger.warning(
                        f"Could not find definition for function: {function_name}"
                    )
                return

            # Take the first function definition
            function_def = function_defs[0]

            # Step 2: Extract call expressions from the function
            call_expressions = self.fetch_call_expressions_query(
                function_def.source_code
            )
            if not call_expressions:
                if self.debug:
                    logger.info(
                        f"No call expressions found in function: {function_name}"
                    )
                # Store the current chain as a complete call chain
                self._store_call_chain(current_chain.copy())
                return

            if self.debug:
                logger.info(
                    f"Found {len(call_expressions)} call expressions in {function_name}"
                )

            # Step 3: For each call expression, extract function name and fetch definition
            for call_expr in call_expressions:
                # Use FunctionNameExtractor to extract function name from call expression
                function_names = self.function_name_extractor(call_expr)

                if function_names and len(function_names) > 0:
                    # Take the first function name found
                    called_function_name = function_names[0]

                    if self.debug:
                        logger.info(
                            f"Extracted function name: {called_function_name} from call: {call_expr}"
                        )

                    # Step 4: Store the current function call globally
                    new_chain = current_chain.copy() + [called_function_name]

                    # Step 5: Recursively process the called function
                    # Try to find the called function in the same file first
                    called_defs = self.fetch_function_def_pipeline(
                        called_function_name, file_path=file_path
                    )

                    if called_defs:
                        # Found in same file, continue recursion
                        self._extract_call_chain_recursive(
                            function_name=called_function_name,
                            file_path=file_path,
                            current_chain=new_chain,
                            depth=depth + 1,
                            line_number=line_number,
                        )
                    else:
                        # Could not find called function definition, store current chain
                        if self.debug:
                            logger.info(
                                f"Could not find definition for called function: {called_function_name}"
                            )
                        self._store_call_chain(new_chain)
                else:
                    if self.debug:
                        logger.warning(
                            f"Could not extract function name from call expression: {call_expr}"
                        )

            # If no valid function calls found, store the current chain
            if not any(
                function_names
                for function_names in [
                    self.function_name_extractor(call_expr)
                    for call_expr in call_expressions
                ]
            ):
                self._store_call_chain(current_chain.copy())

        except Exception as exc:
            if self.debug:
                logger.error(
                    f"Error in _extract_call_chain_recursive for {function_name}: {exc}"
                )

    def _store_call_chain(self, chain: List[str]) -> None:
        """Store a call chain if it's valid and not already stored."""
        try:
            if len(chain) >= 2:  # Only store chains with at least one function call
                # Check if this chain already exists
                chain_str = " -> ".join(chain)
                if not any(
                    " -> ".join(c.functions) == chain_str for c in self.call_chains
                ):
                    call_chain = CallChain(functions=chain)
                    self.call_chains.append(call_chain)
                    if self.debug:
                        logger.info(f"Stored call chain: {chain_str}")

        except Exception as exc:
            if self.debug:
                logger.error(f"Error storing call chain: {exc}")

    def print_call_chains(self) -> None:
        """Print all extracted call chains in a readable format."""
        try:
            if not self.call_chains:
                print("No call chains found.")
                return

            print(f"\nFound {len(self.call_chains)} function call chain(s):")
            print("=" * 60)

            for i, call_chain in enumerate(self.call_chains, 1):
                chain_str = " -> ".join(call_chain.functions)
                print(f"Chain {i}: {chain_str}")

            print("=" * 60)

        except Exception as exc:
            print(f"Error printing call chains: {exc}")


class FetchCalleeInfoPipeline:
    """
    Fetches the definition of a called function and its return type
    A callee can be
    - simple function call: func()
    - complicated function call chain:
        + namespace::class::func1().member->func2()
        + package.class.call()
        + etc.
    """

    def __init__(self, debug: bool = False, dynamic_type_inference: bool = False):
        """
        Initialize the pipeline with optional debug logging.

        Args:
            debug: Enable debug logging for troubleshooting
        """
        self.debug = debug
        # self.is_class_instance_query = IsClassInstanceQuery(debug=debug)
        self.fetch_class_pipeline = FetchClassPipeline(debug=debug)
        self.extract_member_query = ExtractMemberDefinitionFromClassQuery(debug=debug)
        self.fetch_func_type_query = AnalyzeFunctionTypeQuery(debug=debug)
        self.fetch_member_var_query = FetchMemberVarQuery(debug=debug)
        self.resolve_indexed_type_query = ResolveIndexedTypeQuery(debug=debug)
        self.fetch_function_definition_by_name = (
            FetchFunctionDefinitionFromNamePipeline(debug=debug)
        )
        self.element_type_analyzer_query = ElementTypeAnalyzerQuery(debug=debug)
        self.builtin_function_query = BuiltinFunctionReturnTypeQuery(debug=debug)
        self.extract_class_template_parameters_query = (
            ExtractClassTemplateParametersQuery(debug=debug)
        )
        self.extract_function_template_parameters_query = (
            ExtractFunctionTemplateParametersQuery(debug=debug)
        )
        self.function_name_extractor = FunctionNameExtractor(debug=debug)
        self.extract_type_parameter_query = ExtractTypeParameterQuery(debug=debug)
        self.type_parameter_substitution_query = TypeParameterSubstitutionQuery(
            debug=debug
        )

        # Initialize VarTypeInferenceQuery without circular dependency initially
        self.var_type_inference_query = VarTypeInferenceQuery(
            debug=debug, query_dynamic_type=dynamic_type_inference
        )

        # Set up the circular dependency with ExpressionTypeInferenceQuery
        # This will be called after the ExpressionTypeInferenceQuery is created
        self._expression_type_inference_query = None

        # Type parameter mapping for template/generic type inference
        # Maps: type_parameter -> type_argument and type_argument -> type_parameter
        self._type_parameter_mappings = (
            {}
        )  # {class_name: {"param_to_arg": {param: arg}, "arg_to_param": {arg: param}}}
        self.doc = None

    def set_expression_type_inference_query(self, expression_type_inference_query):
        """Set the ExpressionTypeInferenceQuery to resolve circular dependency."""
        self._expression_type_inference_query = expression_type_inference_query
        # Update the VarTypeInferenceQuery with the expression type inference query
        self.var_type_inference_query.set_expression_type_inference_query(
            expression_type_inference_query
        )

    def _create_class_type_parameter_mapping(
        self, class_name: str, type_parameters: List[str], type_arguments: List[str]
    ) -> None:
        """
        Create a bidirectional mapping between type parameters and arguments for a class.

        Args:
            class_name: The name of the class (e.g., "std::vector<int>")
            type_parameters: List of type parameter names from the class definition (e.g., ["T"])
            type_arguments: List of type argument values from the usage (e.g., ["int"])
        """
        if len(type_parameters) != len(type_arguments):
            if self.debug:
                logger.warning(
                    f"Type parameter/argument count mismatch for {class_name}: "
                    f"parameters={type_parameters}, arguments={type_arguments}"
                )
            return

        type_arguments = [self._clean_type_name(arg) for arg in type_arguments]

        param_to_arg = dict(zip(type_parameters, type_arguments))
        arg_to_param = dict(zip(type_arguments, type_parameters))

        self._type_parameter_mappings[class_name] = {
            "param_to_arg": param_to_arg,
            "arg_to_param": arg_to_param,
        }

        if self.debug:
            logger.info(
                f"Created type parameter mapping for {class_name}: "
                f"param_to_arg={param_to_arg}, arg_to_param={arg_to_param}"
            )

    def _get_type_parameter_mapping(
        self, class_name: str
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Get the type parameter mapping for a class.

        Args:
            class_name: The name of the class

        Returns:
            Dictionary with "param_to_arg" and "arg_to_param" mappings, or None if not found
        """
        return self._type_parameter_mappings.get(class_name)

    def _infer_type_from_parameter(self, type_name: str, class_name: str) -> str:
        """
        Infer the actual type from a type parameter using the mapping.

        Args:
            type_name: The type name that might be a parameter (e.g., "T")
            class_name: The class context to use for mapping

        Returns:
            The inferred type (e.g., "int"), or the original type if no mapping exists
        """
        mapping = self._get_type_parameter_mapping(class_name)
        if mapping and "param_to_arg" in mapping:
            return mapping["param_to_arg"].get(type_name, type_name)
        return type_name

    def _infer_parameter_from_type(self, type_name: str, class_name: str) -> str:
        """
        Infer the type parameter from a concrete type using the mapping.

        Args:
            type_name: The concrete type name (e.g., "int")
            class_name: The class context to use for mapping

        Returns:
            The inferred parameter (e.g., "T"), or the original type if no mapping exists
        """
        mapping = self._get_type_parameter_mapping(class_name)
        if mapping and "arg_to_param" in mapping:
            return mapping["arg_to_param"].get(type_name, type_name)
        return type_name

    def _clean_type_name(self, type_name: str) -> str:
        """
        Clean the type name by removing any leading or trailing whitespace or type decorators such as &, *, const, volatile, etc.

        Args:
            type_name: The type name to clean

        Returns:
            The cleaned type name
        """
        type_name = type_name.strip()
        in_while = True
        while in_while:
            if type_name.startswith("const "):
                type_name = type_name[6:]
            elif type_name.startswith("static "):
                type_name = type_name[7:]
            elif type_name.startswith("constexpr "):
                type_name = type_name[10:]
            elif type_name.startswith("volatile "):
                type_name = type_name[9:]
            elif type_name.endswith("&"):
                type_name = type_name[:-1]
            elif type_name.endswith("*"):
                type_name = type_name[:-1]
            else:
                in_while = False
        type_name = type_name.strip()
        return type_name

    def _component_type_decision(self, component_name: str):
        """
        Decide the type of a component in a chained expression.

        Args:
            component_name: The name of the component (e.g., "a", "b", "c")

        Returns:
            The type of the component
        """
        set_global_config_value("generator.json_output", True)
        rag = RAG()
        set_global_config_value("generator.json_output", False)

        for bm25_keyword in ["CLASS", "VARDECL"]:
            retrieved_docs = rag.retrieve(
                bm25_keywords="[TOKEN_SPLIT]".join(bm25_keyword),
                faiss_query="",
            )[0].documents
            for doc in retrieved_docs:
                context, _ = build_context(
                    retrieved_doc=doc,
                    id2doc=rag.id2doc,
                    direction="both",
                    count=1,
                )
                prompt = f"""
                {context}
                Based on the above context, please infer the type of the component {component_name}.
                """

    # TODO: Currently, it supports cpp, java, solidity
    # TODO: However, python's namespace is actually module name, so using llm to getting repo structure is important
    # TODO: In Python, extracting package names and label them is important and we'll do it later
    def _is_namespace(self, component: dict, expression: str) -> int:
        """
        Check if a component name is a namespace.

        Args:
            component_name: The name of the component (e.g., "a", "b", "c")

        Returns:
            1 if the component name is a namespace, 0 if the component name is a class, -1 if the component name is a member, -2 if the component name is a function call
        """

        if component["type"] == "function_call":
            return -2
        if component["type"] == "member_access":
            return -1

        if component["type"] == "namespace_or_class" or component["type"] == "scope":
            rag = RAG()
            namespace_bm25_keywords = [["NAMESPACE"] + component["name"]]
            namespace_retrieved_docs = rag.retrieve(
                bm25_keywords="[TOKEN_SPLIT]".join(namespace_bm25_keywords),
                faiss_query="",
            )[0].documents
            class_bm25_keywords = [["CLASS"] + component["name"]]
            class_retrieved_docs = rag.retrieve(
                bm25_keywords="[TOKEN_SPLIT]".join(class_bm25_keywords),
                faiss_query="",
            )[0].documents
            if len(class_retrieved_docs) == 0 and len(namespace_retrieved_docs) > 0:
                return True
            if len(class_retrieved_docs) > 0 and len(namespace_retrieved_docs) == 0:
                return False
            if len(class_retrieved_docs) == 0 and len(namespace_retrieved_docs) == 0:
                return -1
            if len(class_retrieved_docs) > 0 and len(namespace_retrieved_docs) > 0:
                raise ValueError(
                    "Both class and namespace retrieved docs are not empty."
                )
                retrieved_docs = list(
                    set(class_retrieved_docs + namespace_retrieved_docs)
                )
                doc_prompt = ""
                for doc in retrieved_docs:
                    context, _ = build_context(
                        retrieved_doc=doc,
                        id2doc=rag.id2doc,
                        direction="previous",
                        count=5,
                    )
                    doc_prompt += f"document ID {doc.id}:\n{context}\n"
                prompt = PROMPT_TEMPLATE.call(
                    task_description=f"""
Below are code snippets that may contain the component {component["name"]} in the expression {expression}.
{doc_prompt}
Based on the above code snippet, infer which code snippet is the one that contains the component {component["name"]} in the expression {expression}. Return the document ID of the code snippet.
                    """,
                    output_format="""Return JSON
{
    "document_id": "doc_id",
}
                    """,
                )

        return False

    def __call__(
        self, expression: str, file_path: str, line_number: int, language: str = ""
    ) -> Tuple[str, str]:
        """
        Analyze a chained call expression and return the final class name.

        Args:
            expression: The chained call expression to analyze (e.g., "a.b.c()")
            file_path: The path to the source file containing the expression
            line_number: The line number in the source file where the expression appears
            language: The programming language of the expression

        Returns:
            The type of the final element in the expression chain
        """
        if line_number == None:
            self.doc = None
        rag = RAG()
        rag.get_docs()
        doc = rag.id2doc[rag.codePath2beginDocid[file_path]]
        while True:
            if (
                doc.meta_data["start_line"] <= line_number
                and doc.meta_data["end_line"] >= line_number
            ):
                self.doc = doc
                break
            if doc.meta_data["next_doc_id"] == None:
                break
            doc = rag.id2doc[doc.meta_data["next_doc_id"]]
            if doc == None:
                break

        language = self.doc.meta_data["programming_language"]

        if not self.doc:
            raise ValueError(
                f"Could not find the document for line number {line_number} in file {file_path}"
            )

        if not self._expression_type_inference_query:
            self.set_expression_type_inference_query(
                ExpressionTypeInferenceQuery(debug=self.debug)
            )

        with call_debug_scope(
            self.__class__.__name__, self.debug, expression=expression
        ):
            # Parse the chained expression into components
            components = self._parse_chained_expression(expression)
            if not components:
                logger.warning(f"Could not parse chained expression: {expression}")
                return None

            if len(components) > 1:  # components may contain namespace, class, member
                # Get the initial class name from the outermost instance
                initial_instance = components[0]["name"]
                initial_instance_type = self.var_type_inference_query(
                    initial_instance, file_path, expression, line_number, language
                )
                current_class_name = (
                    self._clean_type_name(initial_instance_type)
                    if initial_instance_type
                    else None
                )
                if not current_class_name:
                    logger.warning(
                        f"Could not resolve initial instance type: {initial_instance_type}"
                    )
                    return None

                # Process all components except the initial instance
                for component in components[1:-1]:
                    prev_class_name = current_class_name
                    current_class_name = self._resolve_component(
                        component, current_class_name
                    )
                    if not current_class_name:
                        logger.warning(
                            f"Could not resolve component: {component} from class {prev_class_name}"
                        )
                        return None

                    current_class_name = self._clean_type_name(current_class_name)

                current_class_name = current_class_name.strip()
                # Get the function body
                return_type, function_body = self._resolve_member_function_call(
                    components[-1], current_class_name
                )

            else:
                return_type, function_body = self._resolve_nonmember_function_call(
                    components[-1], file_path=file_path, line_number=line_number
                )

            return return_type, function_body

    def _resolve_nonmember_function_call(
        self, func_name: str, file_path: str, line_number: int
    ) -> Tuple[str, str]:
        """
        Resolve a non-member function call.

        Args:
            func_name: The name of the function to resolve
            file_path: The path to the file containing the function call
            line_number: The line number of the function call in the file

        Returns:
            A tuple containing the return type and function body
        """
        # Extract function name and type arguments
        base_function_name, function_type_args = self.function_name_extractor(func_name)

        if self.debug and function_type_args:
            logger.info(
                f"Extracted function type arguments: {function_type_args} from {component['name']}"
            )

        function_def = self.fetch_function_definition_by_name(
            function_name=base_function_name,
            file_path=file_path,
            line_number=line_number,
        )

        #! WARNING: cannot handle the following case:
        """
        class MyClass:
            def my_method(self):
                pass

        def external_method(self):
            print(1)

        MyClass.my_method = external_method
        """

        # Create local function type parameter mapping if we have function type arguments
        function_type_mapping = None
        function_template_params = self.extract_function_template_parameters_query(
            function_def
        )
        if function_type_args and function_def:
            # Extract template parameters from function definition
            if function_template_params and len(function_template_params) == len(
                function_type_args
            ):
                # Create local bidirectional mapping between function template parameters and type arguments
                param_to_arg = dict(zip(function_template_params, function_type_args))
                arg_to_param = dict(zip(function_type_args, function_template_params))
                function_type_mapping = {
                    "param_to_arg": param_to_arg,
                    "arg_to_param": arg_to_param,
                }
                if self.debug:
                    logger.info(
                        f"Created function type parameter mapping for {base_function_name}: {function_template_params} -> {function_type_args}"
                    )

        # Get the return type
        result = self.fetch_func_type_query(function_def)
        #! WARNING: can only hold single return scenario
        if result and result.get("return_types"):
            # Extract the first return type from the list
            return_types = result["return_types"]
            return_types = [
                self._clean_type_name(ret_type) for ret_type in return_types
            ]
            return_type = return_types[0] if return_types else None

            # Use type parameter mapping to infer concrete types if return type is a parameter
            if return_type:
                # First try to infer using function type mapping (more specific)
                if (
                    function_type_mapping
                    and return_type in function_type_mapping["param_to_arg"]
                ):
                    inferred_type = function_type_mapping["param_to_arg"][return_type]
                    if self.debug:
                        logger.info(
                            f"Inferred concrete type '{inferred_type}' from function parameter '{return_type}' for {base_function_name}"
                        )
                    return_type = inferred_type
                else:
                    # Check if return type contains embedded type parameters (e.g., vector<T>)
                    # Collect all known type parameters from class and function to improve extraction
                    all_known_type_params = set()

                    # Add function type parameters
                    if function_type_mapping:
                        all_known_type_params.update(
                            function_type_mapping["param_to_arg"].keys()
                        )

                    # Add function template parameters
                    if function_template_params:
                        all_known_type_params.update(function_template_params)

                    # Convert to list and pass to extract_type_parameter_query
                    known_params_list = (
                        list(all_known_type_params) if all_known_type_params else None
                    )
                    type_params = self.extract_type_parameter_query(
                        return_type, known_params_list
                    )
                    if type_params:
                        # Collect all known type parameters from function and class mappings
                        all_known_params = {}

                        # Add function type parameters (highest priority)
                        if function_type_mapping:
                            all_known_params.update(
                                function_type_mapping["param_to_arg"]
                            )

                        # Filter to only include parameters that are actually in the return type
                        relevant_mapping = {
                            param: all_known_params[param]
                            for param in type_params
                            if param in all_known_params
                        }

                        if relevant_mapping:
                            # Use TypeParameterSubstitutionQuery for intelligent replacement
                            inferred_type = self.type_parameter_substitution_query(
                                return_type, relevant_mapping
                            )
                            if inferred_type != return_type:
                                if self.debug:
                                    logger.info(
                                        f"Inferred concrete type '{inferred_type}' from compound return type '{return_type}' for {base_function_name}"
                                    )
                                return_type = inferred_type

        return return_type, function_def

    def _parse_chained_expression(self, expression: str) -> List[Dict[str, Any]]:
        """
        Parse a chained call expression into individual components based on language.

        Args:
            expression: The chained call expression to parse

        Returns:
            List of components with name, type, and operator information
        """

        # Use language-specific parsing
        if self.doc.meta_data["programming_language"] == "cpp":
            return self._parse_cpp_chained_expression(expression)
        elif self.doc.meta_data["programming_language"] in [
            "python",
            "java",
            "solidity",
        ]:
            return self._parse_dot_language_chained_expression(expression)
        else:
            # Fallback to generic parsing
            raise ValueError(
                f"Unsupported language '{language}' for expression: {expression}"
            )

    def _parse_cpp_chained_expression(self, expression: str) -> List[Dict[str, Any]]:
        """
        Parse C++ chained expressions like A::B::c.f()->m.g().

        This handles the pattern where:
        - A, B are namespaces/classes (separated by ::)
        - c is a member of class B
        - f is a function in the class c belongs to
        - m is a member of the class type returned by f
        - g is a member function in m's class

        Args:
            expression: The C++ chained expression to parse

        Returns:
            List of components with proper C++ namespace and member access handling
        """
        components = []
        chunks = expression.strip().split("::")
        for namespace_or_class in chunks[:-1]:
            components.append(
                {
                    "name": namespace_or_class.strip(),
                    "type": "namespace_or_class",
                }
            )

        front_pointer = 0
        for i in range(len(chunks[-1])):
            if chunks[-1][i] == ".":
                _name = chunks[-1][front_pointer:i].strip()
                _type = "member_access" if "(" not in _name else "function_call"
                if _type == "function_call":
                    _name = _name.split("(")[0].strip()
                components.append(
                    {
                        "name": _name,
                        "type": _type,
                    }
                )
                front_pointer = i + 1
            elif (
                chunks[-1][i] == "-"
                and i + 1 < len(chunks[-1])
                and chunks[-1][i + 1] == ">"
            ):
                _name = chunks[-1][front_pointer:i].strip()
                _type = "member_access" if "(" not in _name else "function_call"
                if _type == "function_call":
                    _name = _name.split("(")[0].strip()
                components.append(
                    {
                        "name": _name,
                        "type": _type,
                    }
                )
                front_pointer = i + 2

        _name = chunks[-1][front_pointer:].strip()
        _type = "member_access" if "(" not in _name else "function_call"
        if _type == "function_call":
            _name = _name.split("(")[0].strip()
        components.append(
            {
                "name": _name,
                "type": _type,
            }
        )
        return components

    def _parse_dot_language_chained_expression(
        self, expression: str
    ) -> List[Dict[str, Any]]:
        """
        Parse Python/Java/Solidity chained expressions like namespace1.class_instance.func1().member1.func2().

        Args:
            expression: The dot-language chained expression to parse

        Returns:
            List of components with proper namespace and member access handling
        """
        components = []
        expr = expression.strip()

        # First, handle namespace parts (everything before the first method call)
        # Split on dots and find where methods start
        parts = expr.split(".")
        for part in parts:
            if "(" in part:  # Function call
                components.append(
                    {
                        "name": part.split("(")[0].strip(),
                        "type": "function_call",
                    }
                )
            else:  # Member access
                components.append(
                    {
                        "name": part.strip(),
                        "type": "scope",
                    }
                )
        return components

    def _resolve_component(
        self, component: Dict[str, Any], current_class_name: str
    ) -> Optional[str]:
        """
        Resolve a single component in the chain to get its return type.

        Args:
            component: The component to resolve
            current_class_name: The current class name

        Returns:
            The class name of the return type, or None if resolution fails
        """
        try:
            if component["type"] == "function_call":
                return self._resolve_member_function_call(
                    component, current_class_name
                )[0]
            elif component["type"] == "index_access":
                return self._resolve_index_access(
                    component, current_class_name, component["name"]
                )
            else:  # member_access
                return self._resolve_member_access(component, current_class_name)

        except Exception as exc:
            logger.error(f"Error resolving component {component}: {exc}")
            return None

    def _resolve_member_function_call(
        self, component: Dict[str, Any], current_class_name: str
    ) -> Tuple[Optional[str], str]:
        """
        Resolve a function call component to get its return type.

        Args:
            component: The function call component
            current_class_name: The current class name

        Returns:
            The return type, or None if resolution fails
            The function body
        """

        # Get the class definition
        result = self.fetch_class_pipeline(current_class_name)
        class_definition = (
            result.get("class_definition") if isinstance(result, dict) else result
        )

        # Create type parameter mapping if we have type arguments and class definition
        if isinstance(result, dict) and class_definition and class_definition != "None":
            type_arguments = result.get("type_arguments", [])
            type_arguments = [self._clean_type_name(arg) for arg in type_arguments]
            if type_arguments:
                # Extract template parameters from class definition
                template_params = self.extract_class_template_parameters_query(
                    class_definition
                )
                if template_params:
                    # Create bidirectional mapping between parameters and arguments
                    self._create_class_type_parameter_mapping(
                        current_class_name, template_params, type_arguments
                    )
                    if self.debug:
                        logger.info(
                            f"Created type parameter mapping for {current_class_name}: {template_params} -> {type_arguments}"
                        )

        if not class_definition or class_definition == "None":
            logger.warning(
                f"Could not find class definition for: {current_class_name}, query builtin_function_query for help"
            )
            # Try to handle built-in functions using LLM inference
            return (
                self.builtin_function_query(
                    function_name=component["name"],
                    class_name=current_class_name,
                ),
                "",
            )

        # Extract function name and type arguments
        base_function_name, function_type_args = self.function_name_extractor(
            component["name"]
        )

        if self.debug and function_type_args:
            logger.info(
                f"Extracted function type arguments: {function_type_args} from {component['name']}"
            )

        # Extract the member function definition
        member_definition = self.extract_member_query(
            class_definition=class_definition, target_name=base_function_name
        )

        if not member_definition:
            logger.warning(
                f"Could not find member function: {component['name']} in class {current_class_name}:\n {class_definition}"
            )

            # Try to fetch the function definition by name using cpp syntax
            #! WARNING: only support cpp syntax now because cpp allows member function definition outside class
            member_definitions = self.fetch_function_definition_by_name(
                current_class_name + "::" + base_function_name
            )
            if len(member_definitions) == 0:
                logger.warning(
                    f"Could not find function definition for: {current_class_name + '::' + base_function_name}"
                )
                # Try to fetch the function definition by name using solidity syntax
                member_definitions = self.fetch_function_definition_by_name(
                    current_class_name + "." + base_function_name
                )
                if len(member_definitions) == 0:
                    logger.warning(
                        f"Could not find function definition for: {current_class_name + '.' + base_function_name}"
                    )
                    # Try to handle built-in functions using LLM inference
                    return (
                        self.builtin_function_query(
                            function_name=base_function_name,
                            class_name=current_class_name,
                        ),
                        "",
                    )
                else:
                    member_definition = member_definitions[0]
            else:
                member_definition = member_definitions[0]

        #! WARNING: cannot handle the following case:
        """
        class MyClass:
            def my_method(self):
                pass

        def external_method(self):
            print(1)

        MyClass.my_method = external_method
        """

        # Create local function type parameter mapping if we have function type arguments
        function_type_mapping = None
        function_template_params = self.extract_function_template_parameters_query(
            member_definition
        )
        if function_type_args and member_definition:
            # Extract template parameters from function definition
            if function_template_params and len(function_template_params) == len(
                function_type_args
            ):
                # Create local bidirectional mapping between function template parameters and type arguments
                param_to_arg = dict(zip(function_template_params, function_type_args))
                arg_to_param = dict(zip(function_type_args, function_template_params))
                function_type_mapping = {
                    "param_to_arg": param_to_arg,
                    "arg_to_param": arg_to_param,
                }
                if self.debug:
                    logger.info(
                        f"Created function type parameter mapping for {current_class_name}::{base_function_name}: {function_template_params} -> {function_type_args}"
                    )

        # Get the return type
        result = self.fetch_func_type_query(member_definition)
        #! WARNING: can only hold single return scenario
        if result and result.get("return_types"):
            # Extract the first return type from the list
            return_types = result["return_types"]
            return_types = [
                self._clean_type_name(ret_type) for ret_type in return_types
            ]
            return_type = return_types[0] if return_types else None

            # Use type parameter mapping to infer concrete types if return type is a parameter
            if return_type:
                # First try to infer using function type mapping (more specific)
                if (
                    function_type_mapping
                    and return_type in function_type_mapping["param_to_arg"]
                ):
                    inferred_type = function_type_mapping["param_to_arg"][return_type]
                    if self.debug:
                        logger.info(
                            f"Inferred concrete type '{inferred_type}' from function parameter '{return_type}' for {current_class_name}::{base_function_name}"
                        )
                    return_type = inferred_type
                else:
                    # Check if return type contains embedded type parameters (e.g., vector<T>)
                    # Collect all known type parameters from class and function to improve extraction
                    all_known_type_params = set()

                    # Add class type parameters
                    class_mapping = self._type_parameter_mappings.get(
                        current_class_name, {}
                    ).get("param_to_arg", {})
                    all_known_type_params.update(class_mapping.keys())

                    # Add function type parameters
                    if function_type_mapping:
                        all_known_type_params.update(
                            function_type_mapping["param_to_arg"].keys()
                        )

                    # Add function template parameters
                    if function_template_params:
                        all_known_type_params.update(function_template_params)

                    # Convert to list and pass to extract_type_parameter_query
                    known_params_list = (
                        list(all_known_type_params) if all_known_type_params else None
                    )
                    type_params = self.extract_type_parameter_query(
                        return_type, known_params_list
                    )
                    if type_params:
                        # Collect all known type parameters from function and class mappings
                        all_known_params = {}

                        # Add function type parameters (highest priority)
                        if function_type_mapping:
                            all_known_params.update(
                                function_type_mapping["param_to_arg"]
                            )

                        # Add class type parameters (lower priority)
                        class_mapping = self._type_parameter_mappings.get(
                            current_class_name, {}
                        ).get("param_to_arg", {})
                        all_known_params.update(class_mapping)

                        # Filter to only include parameters that are actually in the return type
                        relevant_mapping = {
                            param: all_known_params[param]
                            for param in type_params
                            if param in all_known_params
                        }

                        if relevant_mapping:
                            # Use TypeParameterSubstitutionQuery for intelligent replacement
                            inferred_type = self.type_parameter_substitution_query(
                                return_type, relevant_mapping
                            )
                            if inferred_type != return_type:
                                if self.debug:
                                    logger.info(
                                        f"Inferred concrete type '{inferred_type}' from compound return type '{return_type}' for {current_class_name}::{base_function_name}"
                                    )
                                return_type = inferred_type
                    else:
                        # Fall back to class type mapping for simple type parameters
                        inferred_type = self._infer_type_from_parameter(
                            return_type, current_class_name
                        )
                        return_type = inferred_type

            return return_type, member_definition

        # If traditional analysis fails, try built-in function inference
        return (
            self.builtin_function_query(
                function_name=component["name"],
                class_name=current_class_name,
            ),
            "",
        )

    def _resolve_member_access(
        self, component: Dict[str, Any], current_class_name: str
    ) -> Optional[str]:
        """
        Resolve a member access component to get its type.

        Args:
            component: The member access component
            current_class_name: The current class name

        Returns:
            The class name of the member type, or None if resolution fails
        """
        # Get the class definition
        result = self.fetch_class_pipeline(current_class_name)
        class_definition = (
            result.get("class_definition") if isinstance(result, dict) else result
        )

        # Create type parameter mapping if we have type arguments and class definition
        if isinstance(result, dict) and class_definition:
            type_arguments = result.get("type_arguments", [])
            if type_arguments:
                # Extract template parameters from class definition
                template_params = self.extract_class_template_parameters_query(
                    class_definition
                )
                if template_params:
                    # Create bidirectional mapping between parameters and arguments
                    self._create_class_type_parameter_mapping(
                        current_class_name, template_params, type_arguments
                    )
                    if self.debug:
                        logger.info(
                            f"Created type parameter mapping for {current_class_name}: {template_params} -> {type_arguments}"
                        )

        if not class_definition:
            logger.warning(f"Could not find class definition for: {current_class_name}")
            return None

        # Extract the member variable information
        member_var = self.fetch_member_var_query(
            class_definition=class_definition, target_name=component["name"]
        )

        if not member_var:
            logger.warning(
                f"Could not find member variable: {component['name']} in class {current_class_name}"
            )
            return None

        # Get the member variable type
        member_type = member_var.get("type")

        # Use type parameter mapping to infer concrete types if member type is a parameter
        if member_type:
            inferred_type = self._infer_type_from_parameter(
                member_type, current_class_name
            )
            if inferred_type and inferred_type != member_type:
                if self.debug:
                    logger.info(
                        f"Inferred concrete type '{inferred_type}' from parameter '{member_type}' for {current_class_name}"
                    )
                return inferred_type

        return member_type

    def _resolve_index_access(
        self, component: Dict[str, Any], base_class: str, base_expr: str
    ) -> Optional[str]:
        """
        Resolve an index access component to get its element type.

        Args:
            component: The index access component
            base_class: The base expr's class name
            base_expr: The base expression

        Returns:
            The class name of the element type, or None if resolution fails
        """
        try:
            # Extract the index value from the component
            index_value = component.get("index_value", "")
            if not index_value:
                logger.warning("Index access component missing index_value")
                return None

            index_expr = base_expr + "[" + index_value + "]"

            element_type = self.element_type_analyzer_query(
                compound_type=base_class,
                base_expression=base_expr,
                indexed_expression=index_expr,
            )

            return element_type

        except Exception as exc:
            logger.error(f"Error resolving index access component {component}: {exc}")
            return None

    def _return_type_contains_type_parameters(
        self, return_type: str, type_parameters: Optional[List[str]] = None
    ) -> bool:
        """
        Determine if a function return type contains type parameters.

        Args:
            return_type: The return type to analyze (e.g., "vector<T>", "std::map<K, V>", "int")
            type_parameters: Optional list of known type parameter names to help guide the analysis

        Returns:
            bool: True if the return type contains type parameters, False otherwise
        """
        try:
            # Use ExtractTypeParameterQuery to get all type parameters from the return type
            found_type_parameters = self.extract_type_parameter_query(
                return_type, type_parameters
            )

            # If we found any type parameters, the return type contains them
            if found_type_parameters:
                if self.debug:
                    logger.info(
                        f"Return type '{return_type}' contains type parameters: {found_type_parameters}"
                    )
                return True
            else:
                if self.debug:
                    logger.info(
                        f"Return type '{return_type}' does not contain type parameters"
                    )
                return False

        except Exception as exc:
            if self.debug:
                logger.error(
                    f"Error checking if return type '{return_type}' contains type parameters: {exc}"
                )
            # Default to False if we can't determine
            return False


class LanguageDetector:
    """
    Detects the programming language of a code snippet using LLM analysis.
    """

    prompt_template = Prompt(
        PROMPT_TEMPLATE.call(
            task_description=r"""
You are given a code snippet:

{{code_snippet}}

Your task is to determine the programming language of this code snippet.
            """,
            instructions=r"""
Analyze the code snippet and identify the programming language based on:
1. Syntax patterns (e.g., semicolons, braces, indentation)
2. Keywords (e.g., 'def', 'function', 'class', 'public', 'private')
3. Comment styles (e.g., '//', '#', '/* */', '--')
4. String literals and escape sequences
5. Import/include patterns (e.g., '#include', 'import', 'using')
6. Variable declaration patterns (e.g., 'var', 'let', 'const', type annotations)
7. Function declaration patterns (e.g., 'def', 'function', '=>')
8. Memory management patterns (e.g., pointers, references, garbage collection)

Common languages to identify: C, C++, Java, Python, JavaScript, TypeScript, C#, Go, Rust, Swift, Kotlin, PHP, Ruby, Perl, Lua, R, MATLAB, Shell, SQL, HTML, CSS, JSON, XML, YAML, etc.

Return the most likely language as a single string.
            """,
            output_format=r"""Return strict JSON:
{"language": "language_name", "confidence": "high|medium|low", "reasoning": "brief explanation of key indicators"}

Examples:
{"language": "python", "confidence": "high", "reasoning": "Uses 'def' for functions, colons, and indentation-based blocks"}
{"language": "cpp", "confidence": "high", "reasoning": "Uses '#include' headers, 'std::' namespace, and semicolons"}
{"language": "javascript", "confidence": "medium", "reasoning": "Uses '=>' arrow function and semicolons, but syntax is ambiguous"}
            """,
        )
    )

    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, code_snippet: str) -> Dict[str, Any]:
        """
        Determine the programming language of a code snippet.

        Args:
            code_snippet: The code snippet to analyze

        Returns:
            Dictionary containing language detection results
        """
        with call_debug_scope(
            self.__class__.__name__,
            self.debug,
            snippet_length=len(code_snippet or ""),
        ):
            if not code_snippet or not code_snippet.strip():
                return {
                    "language": "unknown",
                    "confidence": "low",
                    "reasoning": "Empty code snippet",
                }

            # Use LLM-based detection
            prompt = self.prompt_template.call(code_snippet=code_snippet)
            set_global_config_value("generator.json_output", True)
            generator = GeneratorWrapper()
            reply = generator(input_str=prompt).data.strip()
            if self.debug:
                generator.save_result()
            set_global_config_value("generator.json_output", False)

            try:
                data = json.loads(repair_json(reply))
                return data
            except Exception as exc:
                logger.error("LanguageDetector: JSON parse failed: %s", exc)
                return {
                    "language": "unknown",
                    "confidence": "low",
                    "reasoning": "Failed to parse LLM response",
                }
