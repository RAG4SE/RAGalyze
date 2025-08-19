#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""


def test_core_imports():
    """Test core package imports."""
    print("Testing core imports...")

    try:
        from deepwiki_cli import query_repository, load_all_configs

        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

    return True


def test_splitter_imports():
    """Test splitter imports."""
    print("Testing splitter imports...")

    try:
        from deepwiki_cli.rag.splitter.natural_language_splitter import NaturalLanguageSplitter
        from deepwiki_cli.rag.splitter.code_splitter import CodeSplitter

        print("✅ Splitter imports successful")
    except ImportError as e:
        print(f"❌ Splitter import failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic functionality...")

    try:
        from deepwiki_cli.rag.splitter.natural_language_splitter import NaturalLanguageSplitter
        from adalflow.core.types import Document

        # Test word splitting
        splitter = NaturalLanguageSplitter(split_by="word", chunk_size=5, chunk_overlap=1)
        doc = Document(text="This is a test document with multiple words", id="test")
        result = splitter.call([doc])

        if len(result) >= 1:
            print(f"✅ Word splitting successful: {len(result)} chunks created")
        else:
            print("❌ Word splitting failed: no chunks created")
            return False

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("🧪 Running import and functionality tests...")
    print("=" * 50)

    tests = [
        test_core_imports,
        test_splitter_imports,
        test_basic_functionality,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Package is ready for publishing.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before publishing.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
