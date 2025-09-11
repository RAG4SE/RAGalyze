#!/usr/bin/env python3
"""
Unified setup script for RAGalyze - includes BM25 and complete Tree-sitter integration
"""

from setuptools import setup, Extension
import numpy as np
import subprocess
import os
import urllib.request
import tarfile
import tempfile
import shutil
# import sys  # Will be used if needed later

# Configuration flags
ENABLE_TREESITTER = True  # Enable tree-sitter integration
ENABLE_LANGUAGE_PARSERS = True  # Download language parsers (set to False for now)

# def get_pcre2_config():
#     """Get PCRE2 configuration"""
#     try:
#         pcre2_prefix = subprocess.check_output(['brew', '--prefix', 'pcre2'], text=True).strip()
#     except Exception:
#         pcre2_prefix = '/opt/homebrew/opt/pcre2'
    
#     return {
#         'include_dirs': [os.path.join(pcre2_prefix, 'include')],
#         'library_dirs': [os.path.join(pcre2_prefix, 'lib')],
#         'libraries': ['pcre2-8']
#     }

tree_sitter_version = '0.25.9'  # Updated to a more recent version

def download_tree_sitter_core():
    """Download tree-sitter core library"""
    tree_sitter_lib_dir = "tree-sitter-lib"
    
    if not os.path.exists(tree_sitter_lib_dir):
        os.makedirs(tree_sitter_lib_dir)
    
    # Download tree-sitter core library
    core_url = f"https://github.com/tree-sitter/tree-sitter/archive/v{tree_sitter_version}.tar.gz"
    core_archive = os.path.join(tree_sitter_lib_dir, f"tree-sitter-{tree_sitter_version}.tar.gz")
    core_extract_dir = os.path.join(tree_sitter_lib_dir, f"tree-sitter-{tree_sitter_version}")

    if not os.path.exists(core_extract_dir) and not os.path.exists(core_archive):
        try:
            print("Downloading tree-sitter core library...")
            urllib.request.urlretrieve(core_url, core_archive)
            
            with tarfile.open(core_archive, "r:gz") as tar:
                tar.extractall(tree_sitter_lib_dir)
            
            # Clean up archive
            os.remove(core_archive)
            
        except Exception as e:
            print(f"Failed to download tree-sitter core library: {e}")
            return None
    
    return core_extract_dir

def build_tree_sitter_core(core_dir):
    """Build tree-sitter core library using make"""
    if not core_dir or not os.path.exists(core_dir):
        print("Tree-sitter core directory not found")
        return None
        
    try:
        # Run make in the tree-sitter core directory
        print(f"Building tree-sitter core library in {core_dir}")
        makefile_path = os.path.join(core_dir, "Makefile")
        if os.path.exists(makefile_path):
            result = subprocess.run(["make"], cwd=core_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Make failed with error: {result.stderr}")
                return None
            print("Tree-sitter core library built successfully")
        else:
            print("No Makefile found, skipping build step")
            
        return core_dir
    except Exception as e:
        print(f"Failed to build tree-sitter core library: {e}")
        return None

parser_versions = {
    'python': '0.25.0',
    'cpp': '0.23.4',
    'java': '0.23.5',
    'c': '0.24.1',
    'javascript': '0.25.0',
    'go': '0.25.0',
    'rust': '0.24.0'
}

def download_tree_sitter_languages():
    """Download tree-sitter language parsers"""
    tree_sitter_dir = "tree-sitter-languages"
    
    if not os.path.exists(tree_sitter_dir):
        os.makedirs(tree_sitter_dir)
    
    # Language parsers to download (using more reliable URLs)
    languages = {
        'python': f'https://github.com/tree-sitter/tree-sitter-python/archive/v{parser_versions["python"]}.tar.gz',
        'cpp': f'https://github.com/tree-sitter/tree-sitter-cpp/archive/v{parser_versions["cpp"]}.tar.gz',
        'java': f'https://github.com/tree-sitter/tree-sitter-java/archive/v{parser_versions["java"]}.tar.gz',
        'c': f'https://github.com/tree-sitter/tree-sitter-c/archive/v{parser_versions["c"]}.tar.gz',
        'javascript': f'https://github.com/tree-sitter/tree-sitter-javascript/archive/v{parser_versions["javascript"]}.tar.gz',
        'go': f'https://github.com/tree-sitter/tree-sitter-go/archive/v{parser_versions["go"]}.tar.gz',
        'rust': f'https://github.com/tree-sitter/tree-sitter-rust/archive/v{parser_versions["rust"]}.tar.gz'
    }
    
    # Download and extract language parsers
    for lang, url in languages.items():
        lang_archive = os.path.join(tree_sitter_dir, f"tree-sitter-{lang}.tar.gz")
        lang_extract_dir = os.path.join(tree_sitter_dir, f"tree-sitter-{lang}")
        
        if not os.path.exists(lang_extract_dir) and not os.path.exists(lang_archive):
            try:
                print(f"Downloading tree-sitter {lang} parser...")
                urllib.request.urlretrieve(url, lang_archive)
                
                with tarfile.open(lang_archive, "r:gz") as tar:
                    tar.extractall(tree_sitter_dir)
                
                # Clean up archive
                os.remove(lang_archive)
                
            except Exception as e:
                print(f"Failed to download {lang} parser: {e}")
                continue
    
    return tree_sitter_dir

def get_tree_sitter_config():
    """Get tree-sitter configuration using downloaded libraries"""
    # Download tree-sitter core library
    core_dir = download_tree_sitter_core()
    if not core_dir:
        print("Failed to download tree-sitter core library")
        return {'include_dirs': [], 'sources': [], 'libraries': []}
    
    # Build tree-sitter core library
    build_tree_sitter_core(core_dir)
    
    # Set up include directories for tree-sitter
    include_dirs = [
        os.path.join(core_dir, 'lib', 'include'),
        os.path.join(core_dir, 'lib', 'src')
    ]
    
    # Download language parsers if enabled
    sources = []
    if ENABLE_LANGUAGE_PARSERS:
        # lang_dir = download_tree_sitter_languages() # haoyang
        lang_dir = "tree-sitter-languages"
        
        # Find downloaded language parsers
        for lang in ['python', 'cpp', 'java', 'c', 'javascript', 'go', 'rust']:
            # Try both versioned and unversioned directory names
            possible_dirs = [
                os.path.join(lang_dir, f"tree-sitter-{lang}-{get_version_for_lang(lang)}"),
                os.path.join(lang_dir, f"tree-sitter-{lang}")
            ]
            
            lang_parser_dir = None
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    lang_parser_dir = dir_path
                    break
            
            if lang_parser_dir:
                parser_file = os.path.join(lang_parser_dir, "src", "parser.c")
                if os.path.exists(parser_file):
                    sources.append(parser_file)
                    print(f"Added parser for {lang}: {parser_file}")
                else:
                    print(f"Parser file not found for {lang}: {parser_file}")
                
                # Check for scanner files
                scanner_c = os.path.join(lang_parser_dir, "src", "scanner.c")
                scanner_cc = os.path.join(lang_parser_dir, "src", "scanner.cc")
                if os.path.exists(scanner_c):
                    sources.append(scanner_c)
                    print(f"Added scanner for {lang}: {scanner_c}")
                elif os.path.exists(scanner_cc):
                    sources.append(scanner_cc)
                    print(f"Added scanner.cc for {lang}: {scanner_cc}")
        
        print(f"Found {len(sources)} language parser source files")
    
    # Add tree-sitter library source
    core_source = os.path.join(core_dir, 'lib', 'src', 'lib.c')
    assert os.path.exists(core_source), f"Core source not found: {core_source}"
    sources.append(core_source)
    print(f"Added tree-sitter core source: {core_source}")
    
    return {
        'include_dirs': include_dirs,
        'sources': sources,
        'libraries': []
    }

def get_version_for_lang(lang):
    """Get version string for language parser"""
    if lang not in parser_versions:
        raise ValueError(f"Unsupported language: {lang}")
    return parser_versions[lang]

# Get configurations
# pcre2_config = get_pcre2_config()
tree_sitter_config = get_tree_sitter_config() if ENABLE_TREESITTER else {'include_dirs': [], 'sources': []}

# # BM25 C extension with PCRE2
# bm25_c_extension = Extension(
#     'ragalyze.rag.bm25_c_extension',
#     sources=['ragalyze/rag/bm25_c_extension.c'],
#     include_dirs=[np.get_include()] + pcre2_config['include_dirs'],
#     library_dirs=pcre2_config['library_dirs'],
#     libraries=pcre2_config['libraries'],
#     language='c',
#     extra_compile_args=['-std=c99', '-O3']
# )

# Tree-sitter extension with real tree-sitter integration
treesitter_sources = ['ragalyze/rag/treesitter_parse.c']

treesitter_include_dirs = [np.get_include()]
if 'include_dirs' in tree_sitter_config:
    treesitter_include_dirs.extend(tree_sitter_config['include_dirs'])

# Add language parser sources
if 'sources' in tree_sitter_config:
    treesitter_sources.extend(tree_sitter_config['sources'])

print("treesitter_sources:", treesitter_sources)
print("treesitter_include_dirs:", treesitter_include_dirs)

treesitter_extension = Extension(
    'ragalyze.rag.treesitter_parse',
    sources=treesitter_sources,
    include_dirs=treesitter_include_dirs,
    library_dirs=[],
    libraries=[],
    language='c',
    define_macros=[("TS_DEBUG", "1")],  # 👈 开启 LOG
)

# Setup configuration
setup(
    name='RAGalyze-unified',
    version='1.0',
    description='Unified C extensions for RAGalyze with tree-sitter integration',
    # ext_modules=[bm25_c_extension, treesitter_extension],
    ext_modules=[treesitter_extension],
    cmdclass={},
    zip_safe=False,
    install_requires=['numpy'],
)