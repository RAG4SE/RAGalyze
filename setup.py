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

def get_pcre2_config():
    """Get PCRE2 configuration"""
    try:
        pcre2_prefix = subprocess.check_output(['brew', '--prefix', 'pcre2'], text=True).strip()
    except Exception:
        pcre2_prefix = '/opt/homebrew/opt/pcre2'
    
    return {
        'include_dirs': [os.path.join(pcre2_prefix, 'include')],
        'library_dirs': [os.path.join(pcre2_prefix, 'lib')],
        'libraries': ['pcre2-8']
    }

def get_tree_sitter_system_config():
    """Get tree-sitter configuration from system installation"""
    try:
        # Try to get tree-sitter from Homebrew
        tree_sitter_prefix = subprocess.check_output(['brew', '--prefix', 'tree-sitter'], text=True).strip()
        
        include_dirs = [
            os.path.join(tree_sitter_prefix, 'include')
        ]
        
        library_dirs = [
            os.path.join(tree_sitter_prefix, 'lib')
        ]
        
        libraries = ['tree-sitter']
        
        print(f"Found system tree-sitter at: {tree_sitter_prefix}")
        
        return {
            'include_dirs': include_dirs,
            'library_dirs': library_dirs,
            'libraries': libraries
        }
    except Exception as e:
        print(f"System tree-sitter not found: {e}")
        return None

def download_tree_sitter_languages():
    """Download tree-sitter language parsers only (not core library)"""
    tree_sitter_dir = "tree-sitter-languages"
    
    if not os.path.exists(tree_sitter_dir):
        os.makedirs(tree_sitter_dir)
    
    # Language parsers to download (using more reliable URLs)
    languages = {
        'python': 'https://github.com/tree-sitter/tree-sitter-python/archive/v0.20.4.tar.gz',
        'cpp': 'https://github.com/tree-sitter/tree-sitter-cpp/archive/v0.20.3.tar.gz',
        'java': 'https://github.com/tree-sitter/tree-sitter-java/archive/v0.20.2.tar.gz',
        'c': 'https://github.com/tree-sitter/tree-sitter-c/archive/v0.20.7.tar.gz',
        'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript/archive/v0.20.1.tar.gz',
        'go': 'https://github.com/tree-sitter/tree-sitter-go/archive/v0.20.0.tar.gz',
        'rust': 'https://github.com/tree-sitter/tree-sitter-rust/archive/v0.20.4.tar.gz'
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
    """Get tree-sitter configuration using system installation and downloaded languages"""
    
    # First try system tree-sitter
    system_config = get_tree_sitter_system_config()
    if system_config:
        # Download language parsers if enabled
        if ENABLE_LANGUAGE_PARSERS:
            lang_dir = download_tree_sitter_languages()
            
            # Add language parser sources
            sources = []
            
            # Find downloaded language parsers
            for lang in ['python', 'cpp', 'java', 'c', 'javascript', 'go', 'rust']:
                lang_parser_dir = os.path.join(lang_dir, f"tree-sitter-{lang}-{get_version_for_lang(lang)}")
                if os.path.exists(lang_parser_dir):
                    parser_file = os.path.join(lang_parser_dir, "src", "parser.c")
                    if os.path.exists(parser_file):
                        sources.append(parser_file)
                    
                    # Check for scanner files
                    scanner_c = os.path.join(lang_parser_dir, "src", "scanner.c")
                    scanner_cc = os.path.join(lang_parser_dir, "src", "scanner.cc")
                    if os.path.exists(scanner_c):
                        sources.append(scanner_c)
                    elif os.path.exists(scanner_cc):
                        sources.append(scanner_cc)
            
            system_config['sources'] = sources
        else:
            system_config['sources'] = []
        return system_config
    
    # Fallback: use minimal implementation without tree-sitter
    print("Tree-sitter not available, using minimal implementation")
    return {'include_dirs': [], 'sources': [], 'libraries': []}

def get_version_for_lang(lang):
    """Get version string for language parser"""
    versions = {
        'python': '0.20.4',
        'cpp': '0.20.3',
        'java': '0.20.2',
        'c': '0.20.7',
        'javascript': '0.20.1',
        'go': '0.20.0',
        'rust': '0.20.4'
    }
    return versions.get(lang, '0.20.0')

# Get configurations
pcre2_config = get_pcre2_config()
tree_sitter_config = get_tree_sitter_config() if ENABLE_TREESITTER else {'include_dirs': [], 'sources': []}

# BM25 C extension with PCRE2
bm25_c_extension = Extension(
    'ragalyze.rag.bm25_c_extension',
    sources=['ragalyze/rag/bm25_c_extension.c'],
    include_dirs=[np.get_include()] + pcre2_config['include_dirs'],
    library_dirs=pcre2_config['library_dirs'],
    libraries=pcre2_config['libraries'],
    language='c',
    extra_compile_args=['-std=c99', '-O3']
)

# Tree-sitter extension with real tree-sitter integration
treesitter_sources = ['ragalyze/rag/treesitter_parse.c']

treesitter_include_dirs = [np.get_include()]
if 'include_dirs' in tree_sitter_config:
    treesitter_include_dirs.extend(tree_sitter_config['include_dirs'])

# Add tree-sitter include directory
treesitter_include_dirs.append('tree-sitter-lib/tree-sitter-0.20.8/lib/include')

treesitter_extension = Extension(
    'ragalyze.rag.treesitter_parse',
    sources=treesitter_sources,
    include_dirs=treesitter_include_dirs,
    library_dirs=tree_sitter_config.get('library_dirs', []),
    libraries=tree_sitter_config.get('libraries', []),
    language='c',
    extra_compile_args=['-std=c99', '-O3']
)

# Setup configuration
setup(
    name='RAGalyze-unified',
    version='1.0',
    description='Unified C/C++ extensions for RAGalyze with tree-sitter integration',
    ext_modules=[bm25_c_extension, treesitter_extension],
    cmdclass={},
    zip_safe=False,
    install_requires=['numpy'],
)