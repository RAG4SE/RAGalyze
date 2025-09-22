from adalflow import Prompt

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
Help summarize a reply from a large language model and turn it into a well-organized format.

<REPLY>
{{reply}}
</REPLY>

<OUTPUT_FORMAT>
{{output_format}}
</OUTPUT_FORMAT>
"""
    )
)

FETCH_CALLER_NAMES_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to:
1. Identify all **function definitions** whose **function body contains call(s)** to the function named `{{callee_name}}`.
2. Return the **function names** of these **caller functions**.

{% if callee_body %}
To help disambiguate overloaded functions, here is the definition of the function named `{{callee_name}}`:
{{callee_body}}
Use this to judge whether a call in the code is actually calling this specific overload, when possible.
{% endif %}

Important:
- Only return headers of functions that **call** `{{callee_name}}`.
- The full function header must appear in the code snippet.
- Only count **direct calls** to `{{callee_name}}` written inside the function body.
            """,
    )
)


FETCH_CALLER_NAMES_FROM_HEADER_TEMPLATE = Prompt(
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


FETCH_CALLER_HEADERS_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description=r"""
You are given code snippets between <START_OF_CONTEXT> and <END_OF_CONTEXT>.

Your task is to:
1. Identify all **function definitions** whose **function body contains call(s)** to the function named `{{callee_name}}`.
2. Return the **function headers** of these **caller functions**.

{% if callee_body %}
To help disambiguate overloaded functions, here is the definition of the function named `{{callee_name}}`:
{{callee_body}}
Use this to judge whether a call in the code is actually calling this specific overload, when possible.
{% endif %}

Important:
- Only return headers of functions that **call** `{{callee_name}}`.
- The full function header must appear in the code snippet.
- Only count **direct calls** to `{{callee_name}}` written inside the function body.
            """,
        #         output_format=r"""Response in json format such as
        # {
        #     "function_headers": [function_header_1, function_header_2, function_header_3, ...]
        # }
        # If you cannot find any functions that call {{callee_name}}, reply
        # {
        #     "function_headers": "None"
        # }
        # """,
    )
)

# Prompt for finding definition for a function call in codebase
FETCH_CALLEE_DEF_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description=f"Find the true definition of {{callee_name}} that is being called in function {{caller_name}}",
        context=r"{{caller_body}}",
        #         instructions=r"""1. Analyze the calling function context to understand how {{caller_name}} is being used
        # 2. Examine the retrieved documents between <START_OF_CONTEXT> and <END_OF_CONTEXT> to identify all candidate definitions of {{caller_name}}
        # 3. Determine which specific definition is being called based on:
        #    - Function parameter types and counts
        #    - Template arguments (if applicable)
        #    - Object type and inheritance hierarchy
        #    - Function signature matching
        #    - Context of the call site
        # 4. If multiple valid candidates exist, explain why one is more likely than others
        # 5. Provide the exact code snippet with line numbers if available
        # 6. Include the file path for each found declaration/definition
        # 7. If the target cannot be resolved with the given context, state this clearly""",
        output_format=r"""Response in json format such as
{
    "file_path": "[File path]",
    "function_name": "[Function name]",
    "code_snippet": "[Code snippet]"
}
If you cannot find the definition, reply "None".
""",
        #         example=r"""Target: process_data
        # Calling Function: handle_request
        # Calling Function Context:
        # void handle_request(Request req) {
        #     // ...
        #     DataProcessor processor;
        #     processor.process_data(req.get_payload());
        #     // ...
        # }
        # Retrieved Documents:
        # data_processor.h:25: virtual void DataProcessor::process_data(const std::string& data);
        # data_processor.cpp:50: void DataProcessor::process_data(const std::string& data) { /* generic implementation */ }
        # json_processor.h:30: class JsonProcessor : public DataProcessor {
        #     void process_data(const std::string& data) override;
        # }
        # json_processor.cpp:40: void JsonProcessor::process_data(const std::string& data) { /* JSON-specific implementation */ }
        # Expected Output:
        # Resolved Definition:
        # data_processor.cpp:50: void DataProcessor::process_data(const std::string& data) { /* generic implementation */ }
        # Analysis:
        # The calling function creates a DataProcessor object (not JsonProcessor), so the base class implementation is called. The parameter type matches (std::string), and there's no template specialization or override that would change this resolution.
        # Other Candidates:
        # json_processor.cpp:40: void JsonProcessor::process_data(const std::string& data) { /* JSON-specific implementation */ }
        # This override would only be called if the object was of type JsonProcessor.
        # Call Site Context:
        # process_data is called on a DataProcessor object with a std::string parameter from req.get_payload()""",
    )
)

# Prompt for querying complete function implementation
FETCH_FUNC_IMPLEMENTATION_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Find and retrieve the complete implementation of function {{function_name}}",
        context=r"""{% if parameters %}The parameter of this function includes: {{parameters}}
{% endif %}""",
        #         instructions=r"""1. Search for the complete implementation of function {{function_name}} in the codebase
        # 2. Include the full function signature and body
        # 3. Provide line numbers and file path
        # 4. If multiple implementations exist (overloads, different classes), list all of them
        # 5. If the function is templated, include the template declaration
        # 6. If the function has inline implementation in a header, include that as well
        # 7. Include any relevant comments or documentation within the function
        # 8. If the function cannot be found, state this clearly""",
        output_format=r"""
{
    "file_path": "[File path]",
    "function_definition": "[Function definition]"
}
""",
    )
)

# Prompt for fetching function definition with optional header and parameters
FETCH_FUNCTION_DEFINITION_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Find the complete definition of function {{function_name}}",
        context=r"""{% if header %}Function header: {{header}}
{% endif %}{% if parameters %}Parameters: {{parameters}}
{% endif %}""",
        output_format=r"""
{
    "file_path": "[File path]",
    "function_definition": "[Function definition]"
}
If you cannot find the definition, reply "None".
""",
    )
)

# Prompt for querying arguments inside a function call
GET_FUNCTION_ARGUMENTS_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Find and extract the arguments passed to function calls of {{function_name}}",
        context=r"""{% if context %}The context of this function call is: {{context}}{% endif %}""",
        output_format=r"""
{
    "arguments": "[Argument1] [Argument2] [Argument3] ..."
}
""",
    )
)
