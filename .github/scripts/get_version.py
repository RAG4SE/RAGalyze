#!/usr/bin/env python3
"""Script to extract version from pyproject.toml"""

import re
import sys
from pathlib import Path


def get_version(file_path="pyproject.toml"):
    """Extract version from pyproject.toml file."""
    try:
        content = Path(file_path).read_text()
        match = re.search(r'version\s*=\s*["\'](.*?)["\']', content)
        if match:
            return match.group(1)
        else:
            return "0.0.0"
    except Exception:
        return "0.0.0"


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml"
    print(get_version(file_path))
