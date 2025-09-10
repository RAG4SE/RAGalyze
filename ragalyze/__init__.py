"""
DeepWiki CLI - Repository Analysis and Query Tool

This tool provides code repository analysis and question-answering capabilities
using RAG (Retrieval-Augmented Generation) technology.
"""

# Lazy imports to avoid circular dependency issues
# def __getattr__(name):
#     """Lazy import to avoid module loading order issues."""
#     if name == "query_repository":
#         from .query import query_repository
#         return query_repository
#     elif name == "analyze_repository":
#         from .query import analyze_repository
#         return analyze_repository
#     elif name == "save_query_results":
#         from .query import save_query_results
#         return save_query_results
#     elif name == "print_result":
#         from .query import print_result
#         return print_result
#     elif name == "set_global_configs":
#         from .configs import set_global_configs
#         return set_global_configs
#     elif name == "load_default_config":
#         from .configs import load_default_config
#         return load_default_config
#     elif name == "configs":
#         from .configs import configs
#         return configs
#     else:
#         raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .query import (
    save_query_results,
    print_result,
    query_repository,
    query_repository_with_format_find_then_do,
    analyze_repository,
)
from .configs import (
    configs,
    set_global_configs,
    load_default_config,
    set_global_config_value,
)

__all__ = [
    "query_repository",
    "query_repository_with_format_find_then_do",
    "analyze_repository",
    "save_query_results",
    "set_global_configs",
    "print_result",
    "load_default_config",
    "configs",
    "set_global_config_value",
]
