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

# Configuration flags
ENABLE_TREESITTER = True  # Enable tree-sitter integration
ENABLE_LANGUAGE_PARSERS = True  # Download language parsers (set to False for now)
ENABLE_PCRE2 = True  # Enable PCRE2 integration for regex parsing

def get_pcre2_config():
    """Download and configure PCRE2 with automatic build pipeline"""
    if not ENABLE_PCRE2:
        return {'include_dirs': [], 'library_dirs': [], 'libraries': []}
    
    pcre2_version = '10.46'
    pcre2_dir = 'pcre2-lib'
    
    if not os.path.exists(pcre2_dir):
        os.makedirs(pcre2_dir)
    
    pcre2_url = f'https://github.com/PCRE2Project/pcre2/archive/refs/tags/pcre2-{pcre2_version}.tar.gz'
    pcre2_archive = os.path.join(pcre2_dir, f'pcre2-{pcre2_version}.tar.gz')
    pcre2_extract_dir = os.path.join(pcre2_dir, f'pcre2-pcre2-{pcre2_version}')
    
    # PCRE2 build pipeline
    def build_pcre2():
        """PCRE2 build pipeline"""
        try:
            # Step 1: Configure
            configure_path = os.path.join(pcre2_extract_dir, 'configure')
            if os.path.exists(configure_path):
                print("Configuring PCRE2...")
                result = subprocess.run(['./configure', '--disable-shared', '--enable-static'], 
                                      cwd=pcre2_extract_dir, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"PCRE2 configure failed: {result.stderr}")
                    return False
            else:
                print("Configure script not found, skipping configuration")
                return False
            
            # Step 2: Build
            print("Building PCRE2...")
            result = subprocess.run(['make'], cwd=pcre2_extract_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"PCRE2 build failed: {result.stderr}")
                return False
            
            print("PCRE2 built successfully")
            return True
            
        except Exception as e:
            print(f"PCRE2 build pipeline failed: {e}")
            return False
    
    # Download PCRE2 if not already present
    if not os.path.exists(pcre2_extract_dir) and not os.path.exists(pcre2_archive):
        try:
            print(f"Downloading PCRE2 {pcre2_version}...")
            urllib.request.urlretrieve(pcre2_url, pcre2_archive)
            
            with tarfile.open(pcre2_archive, 'r:gz') as tar:
                tar.extractall(pcre2_dir)
            
            os.remove(pcre2_archive)
            print("PCRE2 downloaded successfully")
            
        except Exception as e:
            print(f"Failed to download PCRE2: {e}")
            return {'include_dirs': [], 'library_dirs': [], 'libraries': []}
    
    # Extract if archive exists but directory doesn't
    if os.path.exists(pcre2_archive) and not os.path.exists(pcre2_extract_dir):
        try:
            print("Extracting PCRE2...")
            with tarfile.open(pcre2_archive, 'r:gz') as tar:
                tar.extractall(pcre2_dir)
            os.remove(pcre2_archive)
            print("PCRE2 extracted successfully")
        except Exception as e:
            print(f"Failed to extract PCRE2: {e}")
            return {'include_dirs': [], 'library_dirs': [], 'libraries': []}
    
    # Run build pipeline
    if os.path.exists(pcre2_extract_dir):
        # Check if already built
        lib_path = os.path.join(pcre2_extract_dir, '.libs', 'libpcre2-8.a')
        if not os.path.exists(lib_path):
            print("PCRE2 build pipeline starting...")
            if not build_pcre2():
                return {'include_dirs': [], 'library_dirs': [], 'libraries': []}
        else:
            print("PCRE2 already built, skipping build step")
    else:
        print("PCRE2 extract directory not found")
        return {'include_dirs': [], 'library_dirs': [], 'libraries': []}
    
    return {
        'include_dirs': [os.path.join(pcre2_extract_dir, 'src')],
        'library_dirs': [os.path.join(pcre2_extract_dir, '.libs')],
        'libraries': ['pcre2-8'],
        'extra_compile_args': ['-I' + os.path.join(pcre2_extract_dir, 'src')]
    }

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
    'rust': '0.24.0',
    'xml': '0.7.0'
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
        'rust': f'https://github.com/tree-sitter/tree-sitter-rust/archive/v{parser_versions["rust"]}.tar.gz',
        'xml': f'https://github.com/tree-sitter-grammars/tree-sitter-xml/archive/refs/tags/v{parser_versions["xml"]}.tar.gz'
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
        # lang_dir = download_tree_sitter_languages()
        lang_dir = "tree-sitter-languages"
        
        # Find downloaded language parsers
        for lang in ['python', 'cpp', 'java', 'c', 'javascript', 'go', 'rust', 'xml']:
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
            assert lang_parser_dir, f"Language parser directory not found for {lang}"

            # Add language-specific include directories
            if lang == 'xml':
                # XML has nested structure, add the xml/src and common directories
                xml_src_dir = os.path.join(lang_parser_dir, "xml", "src")
                common_dir = os.path.join(lang_parser_dir, "common")
                if os.path.exists(xml_src_dir):
                    include_dirs.append(xml_src_dir)
                if os.path.exists(common_dir):
                    include_dirs.append(common_dir)
            else:
                # Other languages use direct src structure
                src_dir = os.path.join(lang_parser_dir, "src")
                if os.path.exists(src_dir):
                    include_dirs.append(src_dir)

            # Check for nested src structure (like XML) or direct src
            possible_parser_paths = [
                os.path.join(lang_parser_dir, "src", "parser.c"),
                os.path.join(lang_parser_dir, lang, "src", "parser.c")
            ]

            parser_file = None
            for path in possible_parser_paths:
                if os.path.exists(path):
                    parser_file = path
                    break

            if parser_file:
                sources.append(parser_file)
                print(f"Added parser for {lang}: {parser_file}")
            else:
                print(f"Parser file not found for {lang}")

            assert parser_file, f"Parser file not found for {lang}"

            # Check for scanner files
            base_dir = os.path.dirname(parser_file)
            scanner_c = os.path.join(base_dir, "scanner.c")
            scanner_cc = os.path.join(base_dir, "scanner.cc")
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
pcre2_config = get_pcre2_config()
tree_sitter_config = get_tree_sitter_config() if ENABLE_TREESITTER else {'include_dirs': [], 'sources': []}

# Tree-sitter extension with real tree-sitter integration
treesitter_sources = ['ragalyze/rag/treesitter_parse.c']

treesitter_include_dirs = [np.get_include()]
if 'include_dirs' in tree_sitter_config:
    treesitter_include_dirs.extend(tree_sitter_config['include_dirs'])
if 'include_dirs' in pcre2_config:
    treesitter_include_dirs.extend(pcre2_config['include_dirs'])

# Add language parser sources
if 'sources' in tree_sitter_config:
    treesitter_sources.extend(tree_sitter_config['sources'])

print("treesitter_sources:", treesitter_sources)
print("treesitter_include_dirs:", treesitter_include_dirs)

treesitter_library_dirs = []
treesitter_libraries = []
treesitter_extra_compile_args = []
if 'library_dirs' in pcre2_config:
    treesitter_library_dirs.extend(pcre2_config['library_dirs'])
if 'libraries' in pcre2_config:
    treesitter_libraries.extend(pcre2_config['libraries'])
if 'extra_compile_args' in pcre2_config:
    treesitter_extra_compile_args.extend(pcre2_config['extra_compile_args'])

# Add PCRE2 macro definitions for 8-bit characters
if ENABLE_PCRE2:
    treesitter_extra_compile_args.extend(['-DPCRE2_CODE_UNIT_WIDTH=8'])

treesitter_extension = Extension(
    'ragalyze.rag.treesitter_parse',
    sources=treesitter_sources,
    include_dirs=treesitter_include_dirs,
    library_dirs=treesitter_library_dirs,
    libraries=treesitter_libraries,
    extra_compile_args=treesitter_extra_compile_args,
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