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
#     elif name == "load_all_configs":
#         from .configs import load_all_configs
#         return load_all_configs
#     elif name == "load_default_config":
#         from .configs import load_default_config
#         return load_default_config
#     elif name == "configs":
#         from .configs import configs
#         return configs
#     else:
#         raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .query import save_query_results, print_result, query_repository, analyze_repository
from .configs import configs, load_all_configs, load_default_config

__all__ = [
    "query_repository",
    "analyze_repository",
    "save_query_results",
    "load_all_configs",
    "print_result",
    "load_default_config",
    "configs",
]