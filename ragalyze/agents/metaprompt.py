"""
MetaPrompt module for combining prompt templates with search strategies.

This module provides a MetaPrompt class that automatically assigns prompt templates
based on task names and maps inputs to BM25 keywords and FAISS queries.
"""

from typing import Optional, List, Dict, Any, Literal
import re
from ragalyze.prompts import (
    FETCH_CALLEE_DEF_TEMPLATE,
    FETCH_FUNC_IMPLEMENTATION_TEMPLATE,
    GET_FUNCTION_ARGUMENTS_TEMPLATE,
)


class MetaPrompt:
    """
    MetaPrompt class that automatically assigns prompt templates based on task names
    and maps inputs to BM25 keywords and FAISS queries.
    """

    TASK_TO_TEMPLATE = {
        "fetch_callee_def": FETCH_CALLEE_DEF_TEMPLATE,
        "fetch_func_impl": FETCH_FUNC_IMPLEMENTATION_TEMPLATE,
        "fetch_func_args": GET_FUNCTION_ARGUMENTS_TEMPLATE,
    }

    def __init__(self, task_name: Literal["fetch_callee_def", "fetch_func_impl", "fetch_func_args"]):
        """
        Initialize MetaPrompt with a task name.

        Args:
            task_name: Name of the task to get the appropriate prompt template

        Raises:
            ValueError: If task_name is not supported
        """
        if task_name not in self.TASK_TO_TEMPLATE:
            available_tasks = list(self.TASK_TO_TEMPLATE.keys())
            raise ValueError(f"Task '{task_name}' not supported. Available tasks: {available_tasks}")

        self.task_name = task_name
        self.prompt_template = self.TASK_TO_TEMPLATE[task_name]

    def call(self, *args, **kwargs) -> str:
        """
        Call the prompt template to instantiate it and create BM25 keywords and FAISS query.

        Args:
            *args: Positional arguments for the prompt template
            **kwargs: Keyword arguments for the prompt template

        Returns:
            Combined prompt with search strategy information
        """
        # Generate the base prompt from template
        prompt = self.prompt_template.call(*args, **kwargs)

        # Create BM25 keywords and FAISS query based on task and inputs
        bm25_keywords = self._create_bm25_keywords(*args, **kwargs)
        faiss_query = self._create_faiss_query(*args, **kwargs)

        # Add search strategy if we have keywords or query
        if bm25_keywords or faiss_query:
            search_context = self._create_search_context(bm25_keywords, faiss_query)
            return f"{prompt}\n\n{search_context}"
        else:
            return prompt

    def _create_bm25_keywords(self, *args, **kwargs) -> List[str]:
        """
        Create BM25 keywords based on task and inputs.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of BM25 keywords
        """
        if self.task_name == "fetch_callee_def":
            callee_name, caller_name = kwargs["callee_name"], kwargs["caller_name"]
            keywords = [f"[FUNCDEF]{callee_name}", f"[FUNCDEF]{caller_name}"]

        elif self.task_name == "fetch_func_impl":
            function_name, parameters = kwargs["function_name"], kwargs.get("parameters", "")
            keywords = [f"[FUNCDEF]{function_name}"]

        elif self.task_name == "function_implementation":
            # Extract function name from kwargs
            function_name = kwargs.get("function_name", "")
            if function_name:
                keywords.append(f"[FUNCDEF]{function_name}")
                keywords.append("implementation")
                keywords.append("function")

        elif self.task_name == "function_arguments":
            # Extract function name from kwargs
            function_name = kwargs.get("function_name", "")
            if function_name:
                keywords.append(f"[CALL]{function_name}")
                keywords.append("arguments")
                keywords.append("parameters")

        elif self.task_name == "custom":
            # Extract task description and create general keywords
            task_description = kwargs.get("task_description", str(args[0]) if args else "")
            if task_description:
                # Extract programming-related terms
                important_terms = [
                    "function", "class", "method", "variable", "parameter", "argument",
                    "call", "declaration", "definition", "implementation", "signature",
                    "return", "type", "object", "instance", "inherit", "override"
                ]

                for term in important_terms:
                    if term.lower() in task_description.lower():
                        keywords.append(term)

                # Extract potential function names
                function_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                function_matches = re.findall(function_pattern, task_description)
                for func_name in function_matches:
                    keywords.append(f"[CALL]{func_name}")

        return keywords

    def _create_faiss_query(self, *args, **kwargs) -> str:
        """
        Create FAISS query based on task and inputs.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            FAISS query string
        """
        if self.task_name == "function_call":
            function_name = kwargs.get("function_name", "")
            return f"functions that call {function_name}"

        elif self.task_name == "declaration_definition":
            target_name = kwargs.get("target_name", "")
            calling_function = kwargs.get("calling_function", "")
            return f"definition of {target_name} called by {calling_function}"

        elif self.task_name == "function_implementation":
            function_name = kwargs.get("function_name", "")
            parameters = kwargs.get("parameters", "")
            if parameters:
                return f"complete implementation of function {function_name} with parameters {parameters}"
            return f"complete implementation of function {function_name}"

        elif self.task_name == "function_arguments":
            function_name = kwargs.get("function_name", "")
            context = kwargs.get("context", "")
            return f"arguments passed to {function_name} function calls in context: {context}"

        elif self.task_name == "custom":
            # Use task description as FAISS query
            task_description = kwargs.get("task_description", str(args[0]) if args else "")
            return task_description

        return ""