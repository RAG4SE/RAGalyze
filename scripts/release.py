#!/usr/bin/env python3
"""
Release script for deepwiki-cli package.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)

    with open(pyproject_path, "r") as f:
        for line in f:
            if line.startswith("version = "):
                version = line.split('"')[1]
                return version

    print("Error: Version not found in pyproject.toml")
    sys.exit(1)


def update_version(new_version):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Find and replace version line
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("version = "):
            lines[i] = f'version = "{new_version}"'
            break

    pyproject_path.write_text("\n".join(lines))
    print(f"Updated version to {new_version}")


def build_package():
    """Build the package."""
    print("Building package...")
    run_command("python -m build")
    print("Package built successfully!")


def test_package():
    """Test the built package."""
    print("Testing package...")

    # Test installation
    run_command("pip install dist/*.whl --force-reinstall")

    # Test imports
    test_commands = [
        "python -c \"from deepwiki_cli import query_repository; print('Core import successful')\"",
        "python -c \"from deepwiki_cli.rag.splitter.natural_language_splitter import NaturalLanguageSplitter; print('NaturalLanguageSplitter import successful')\"",
        "python -c \"from deepwiki_cli.rag.splitter.code_splitter import CodeSplitter; print('CodeSplitter import successful')\"",
    ]

    for cmd in test_commands:
        run_command(cmd)

    print("Package tests passed!")


def publish_to_testpypi():
    """Publish to TestPyPI."""
    print("Publishing to TestPyPI...")
    run_command("twine upload --repository testpypi dist/* --skip-existing")
    print("Published to TestPyPI!")


def publish_to_pypi():
    """Publish to PyPI."""
    print("Publishing to PyPI...")
    run_command("twine upload dist/* --skip-existing")
    print("Published to PyPI!")


def create_git_tag(version):
    """Create and push git tag."""
    tag = f"v{version}"
    run_command(f"git add .")
    run_command(f'git commit -m "Release {version}"')
    run_command(f"git tag {tag}")
    run_command(f"git push origin main")
    run_command(f"git push origin {tag}")
    print(f"Created and pushed tag {tag}")


def main():
    parser = argparse.ArgumentParser(description="Release deepwiki-cli package")
    parser.add_argument("--version", help="New version number")
    parser.add_argument(
        "--build-only", action="store_true", help="Only build, don't publish"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only test, don't publish"
    )
    parser.add_argument(
        "--testpypi", action="store_true", help="Publish to TestPyPI instead of PyPI"
    )
    parser.add_argument("--no-tag", action="store_true", help="Don't create git tag")

    args = parser.parse_args()

    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Update version if specified
    if args.version:
        update_version(args.version)
        version = args.version
    else:
        version = current_version

    # Clean previous builds
    run_command("rm -rf dist/ build/ *.egg-info", check=False)

    # Build package
    build_package()

    if args.build_only:
        print("Build complete!")
        return

    # Test package
    test_package()

    if args.test_only:
        print("Tests complete!")
        return

    # Check if twine is installed
    result = run_command("which twine", check=False)
    if result.returncode != 0:
        print("Installing twine...")
        run_command("pip install twine")

    # Publish
    if args.testpypi:
        publish_to_testpypi()
    else:
        publish_to_pypi()

    # Create git tag
    if not args.no_tag and args.version:
        create_git_tag(version)

    print(f"Release {version} completed successfully!")


if __name__ == "__main__":
    main()
