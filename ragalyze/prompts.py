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
{{output_format}}
{% endif %}
{% if example %}
---------- Examples ----------
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

# Prompt for code review tasks
CODE_REVIEW_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Perform a code review focusing on best practices and potential issues",
        context="Reviewing a code snippet for potential improvements, bugs, and adherence to best practices",
        instructions=r"""1. Identify any potential bugs or logical errors
2. Check for adherence to coding standards and best practices
3. Look for potential performance improvements
4. Identify any security vulnerabilities
5. Suggest improvements to code readability and maintainability
6. Highlight any anti-patterns or code smells""",
        output_format="List of issues and suggestions, each with a severity level (critical, high, medium, low)",
        example="""Issue 1 (High): Potential null pointer dereference in function processData()
Suggestion: Add null check before accessing pointer

Issue 2 (Medium): Magic number used in calculateTax() function
Suggestion: Replace with named constant""",
    )
)

# Prompt for documentation generation
GENERATE_DOCS_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Generate documentation for code components",
        context="Creating comprehensive documentation for functions, classes, or modules",
        instructions=r"""1. Analyze the provided code to understand its purpose and functionality
2. Generate clear, concise documentation that explains:
   - What the code does
   - How to use it
   - Parameters/inputs (if any)
   - Return values/outputs (if any)
   - Exceptions that might be thrown
   - Example usage
3. Use a consistent documentation style
4. Focus on clarity and completeness""",
        output_format="Structured documentation in a standard format (e.g., docstring, markdown)",
    )
)

# Prompt for code explanation
EXPLAIN_CODE_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Explain complex code in simple terms",
        context="Breaking down complex code for better understanding",
        instructions=r"""1. Analyze the provided code to understand its functionality
2. Explain the code in simple, easy-to-understand language
3. Identify the main components and their roles
4. Describe the flow of execution
5. Highlight any important algorithms or patterns used
6. Explain why certain approaches were taken""",
        output_format="Plain language explanation with technical accuracy",
    )
)

# Prompt for test generation
GENERATE_TESTS_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Generate unit tests for code",
        context=r"""Generate unit tests for the following code:

Code:
{{code}}

Testing Framework: {{testing_framework}}""",
        instructions=r"""1. Analyze the provided code to understand its functionality
2. Identify different execution paths and edge cases
3. Generate unit tests that cover:
   - Normal/expected inputs
   - Edge cases
   - Error conditions
   - Boundary values
4. Follow the specified testing framework best practices
5. Include descriptive test names
6. Ensure tests are independent and repeatable""",
        output_format="Runnable test code in the specified testing framework",
    )
)

# Prompt for code optimization
OPTIMIZE_CODE_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Optimize code for performance or readability",
        context=r"""Optimize the following code for the specified focus area:

Code:
{{code}}

Focus Area: {{focus_area}}""",
        instructions=r"""1. Analyze the provided code for potential optimizations
2. Identify bottlenecks or inefficient patterns
3. Suggest improvements for the specified focus area
4. Ensure optimizations don't change the code's behavior
5. Explain the reasoning behind each optimization""",
        output_format="Optimized code with explanations of changes",
    )
)

# Prompt for bug detection
BUG_DETECTION_TEMPLATE = Prompt(
    PROMPT_TEMPLATE.call(
        task_description="Identify potential bugs in the code",
        context=r"""Analyze the following code to find common programming errors and edge cases:

Code:
{{code}}""",
        instructions=r"""1. Identify potential null pointer dereferences
2. Look for off-by-one errors in loops
3. Check for proper error handling
4. Identify resource leaks (memory, file handles, etc.)
5. Look for race conditions in concurrent code
6. Check for proper input validation
7. Identify any logical errors in conditionals or calculations""",
        output_format="List of potential bugs with severity levels and suggested fixes",
    )
)


"""
How to use?
result = FIND_DECLARATION_DEFINITION_TEMPLATE.call(
      target_name="process_data",
      calling_function="handle_request",
      calling_function_body="void handle_request() { DataProcessor p; p.process_data(data); }",
  )

"""
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
        output_format=r"""Resolved Definition:
[File path]: [Line numbers]
[Code snippet]
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


if __name__ == "__main__":   
    print(PROMPT_TEMPLATE.call(
        task_description="Summarize the following text", context='1'))