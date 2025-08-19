# Release Scripts

This directory contains scripts to help with releasing the `deepwiki-cli` package.

## Release Script (`release.py`)

A comprehensive script for building, testing, and publishing the package.

### Usage

```bash
# Build and test only
python scripts/release.py --build-only

# Test the built package
python scripts/release.py --test-only

# Publish to TestPyPI for testing
python scripts/release.py --testpypi

# Full release to PyPI with version bump
python scripts/release.py --version 1.0.1

# Release without creating git tag
python scripts/release.py --version 1.0.1 --no-tag
```

### What it does

1. **Version Management**: Updates version in `pyproject.toml`
2. **Build**: Creates wheel and source distributions
3. **Test**: Installs and tests the built package
4. **Publish**: Uploads to PyPI or TestPyPI
5. **Git Tags**: Creates and pushes version tags

### Prerequisites

```bash
pip install build twine
```

### Environment Variables

For publishing, you'll need:

```bash
# For PyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your_pypi_token

# For TestPyPI  
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your_testpypi_token
```

Or configure in `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = your_pypi_token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = your_testpypi_token
```

## Manual Release Process

If you prefer manual steps:

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml

# 2. Clean and build
rm -rf dist/ build/ *.egg-info
python -m build

# 3. Check package
twine check dist/*

# 4. Test locally
pip install dist/*.whl
python -c "from deepwiki_cli import query_repository; print('Success')"

# 5. Publish to TestPyPI first
twine upload --repository testpypi dist/*

# 6. Test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ deepwiki-cli

# 7. Publish to PyPI
twine upload dist/*

# 8. Create git tag
git tag v1.0.1
git push origin v1.0.1
```

## GitHub Actions

The GitHub workflows will automatically handle releases when you:

1. **Push a tag**: `git push origin v1.0.1`
2. **Create a GitHub release**
3. **Trigger manually** from Actions tab

See `.github/PUBLISHING.md` for more details.
