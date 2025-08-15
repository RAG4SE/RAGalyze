#!/bin/bash

# DeepWiki CLI Build and Publish Script

set -e

echo "🚀 DeepWiki CLI Build and Publish Script"
echo "=========================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Consider using one."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade pip build twine

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build the package
echo "🔨 Building package..."
python -m build

# Check the package
echo "🔍 Checking package..."
python -m twine check dist/*

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "📁 Built files:"
ls -la dist/

echo ""
echo "🚀 To publish to PyPI:"
echo "   Test PyPI: python -m twine upload --repository testpypi dist/*"
echo "   Real PyPI: python -m twine upload dist/*"
echo ""
echo "📦 To install locally:"
echo "   pip install dist/*.whl"
echo ""
echo "🖥️  To test the executable:"
echo "   pip install -e ."
echo "   deepwiki --help"
