# This file makes the rag directory a Python package

# Import the treesitter_parse module if available
try:
    from . import treesitter_parse
except ImportError:
    pass