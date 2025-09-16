from adalflow import Prompt

# Base prompt template that can be extended for various tasks
PROMPT_TEMPLATE = Prompt(
    template=r"""{{task_description}}
{% if context %}
{{context}}
{% endif %}
{% if instructions %}
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

# Generic prompt template for finding function calls - user specifies the API/function to search for
FIND_FUNCTION_CALL_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description=r"List every function **definition** (not declaration) where the **body** contains a **CallExpr** to  `{{function_name}}`",
        output_format=r"""
<file_name>:
<the lined code of the function definition including the function signature and body>
...
If no such caller exists, only reply "None".
""",
        example=r"""
List every function **definition** (not declaration) where the **body** contains a **CallExpr** to  bar
67: void foo() {
68:     bar();
69:     // other code
70: }

Expected Output:
file1.cpp:
67: void foo() {
68:     bar();
69:     // other code
70: }
        """,
    )
)

# Prompt for finding declaration and definition in codebase
FIND_DECLARATION_DEFINITION_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Find the true definition of {{target_name}} that is being called in function {{calling_function}}",
        context=r"{{calling_function_body}}",
        instructions=r"""1. Analyze the calling function context to understand how {{target_name}} is being used
2. Examine the retrieved documents between <START_OF_CONTEXT> and <END_OF_CONTEXT> to identify all candidate definitions of {{target_name}}
3. Determine which specific definition is being called based on:
   - Function parameter types and counts
   - Template arguments (if applicable)
   - Object type and inheritance hierarchy
   - Function signature matching
   - Context of the call site
4. If multiple valid candidates exist, explain why one is more likely than others
5. Provide the exact code snippet with line numbers if available
6. Include the file path for each found declaration/definition
7. If the target cannot be resolved with the given context, state this clearly""",
        output_format=r"""
{
    "file_path": "[File path]",
    "function_name": "[Function name]",
    "code_snippet": "[Code snippet]"
}
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
GET_FUNCTION_IMPLEMENTATION_TEMPLATE = Prompt(
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

# Prompt for querying arguments inside a function call
GET_FUNCTION_ARGUMENTS_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Find and extract the arguments passed to function calls of {{function_name}}",
        context = r"""{% if context %}The context of this function call is: {{context}}{% endif %}""",
        output_format=r"""
{
    "arguments": "[Argument1] [Argument2] [Argument3] ..."
}
""",
    )
)


if __name__ == "__main__":   
    print(PROMPT_TEMPLATE.call(
        task_description="Summarize the following text", context='1'))