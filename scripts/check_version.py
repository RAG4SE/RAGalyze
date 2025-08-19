#!/usr/bin/env python3
"""
Script to check if version has changed and if it exists on PyPI.
Useful for testing the logic locally before pushing.
"""

import re
import subprocess
import sys
import requests
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        return None

    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*["\'](.*?)["\']', content)
    if match:
        return match.group(1)

    print("Error: Version not found in pyproject.toml")
    return None


def get_previous_version():
    """Get version from previous git commit."""
    try:
        # Check if we have any commits
        result = subprocess.run(
            ["git", "rev-parse", "HEAD~1"], capture_output=True, text=True, check=True
        )

        # Get previous pyproject.toml
        result = subprocess.run(
            ["git", "show", "HEAD~1:pyproject.toml"],
            capture_output=True,
            text=True,
            check=True,
        )

        content = result.stdout
        match = re.search(r'version\s*=\s*["\'](.*?)["\']', content)
        if match:
            return match.group(1)

        return "0.0.0"

    except subprocess.CalledProcessError:
        # No previous commit or file
        return "0.0.0"


def check_pypi_exists(version):
    """Check if version exists on PyPI."""
    url = f"https://pypi.org/pypi/deepwiki-cli/{version}/json"
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_testpypi_exists(version):
    """Check if version exists on TestPyPI."""
    url = f"https://test.pypi.org/pypi/deepwiki-cli/{version}/json"
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    print("🔍 Checking version status...")
    print("=" * 50)

    # Get current version
    current_version = get_current_version()
    if not current_version:
        sys.exit(1)

    # Get previous version
    previous_version = get_previous_version()

    print(f"📌 Previous version: {previous_version}")
    print(f"📌 Current version:  {current_version}")
    print()

    # Check if version changed
    version_changed = current_version != previous_version
    print(f"🔄 Version changed: {'Yes' if version_changed else 'No'}")

    if not version_changed:
        print("   ➡️ No action needed - version unchanged")
        return

    print()
    print("🌐 Checking PyPI status...")

    # Check PyPI
    pypi_exists = check_pypi_exists(current_version)
    print(f"📦 PyPI:     {'Exists' if pypi_exists else 'Available'}")

    # Check TestPyPI
    testpypi_exists = check_testpypi_exists(current_version)
    print(f"🧪 TestPyPI: {'Exists' if testpypi_exists else 'Available'}")

    print()
    print("📋 Recommendation:")

    if pypi_exists:
        print("   ⚠️ Version already exists on PyPI - no action needed")
    elif version_changed:
        print("   ✅ Ready to publish to PyPI!")
        print("   💡 Steps:")
        print("      1. Test locally: python scripts/release.py --test-only")
        print("      2. Publish to TestPyPI: python scripts/release.py --testpypi")
        print("      3. Push to main branch to trigger auto-publish")
        print("      4. Or use: python scripts/release.py --version", current_version)

    print()
    print("🔗 Links:")
    print(f"   PyPI: https://pypi.org/project/deepwiki-cli/{current_version}/")
    print(f"   TestPyPI: https://test.pypi.org/project/deepwiki-cli/{current_version}/")


if __name__ == "__main__":
    main()
