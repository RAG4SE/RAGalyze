import os
import json
from logger.logging_config import get_tqdm_compatible_logger
import re
from pathlib import Path
from typing import List, Union, Dict, Any

logger = get_tqdm_compatible_logger(__name__)

from core.openai_client import OpenAIClient

#! Though the following imports are not directly used, they are stored in globals() and will be used implicitly. So DO NOT REMOVE THEM!!
from core.huggingface_embedder_client import HuggingfaceClient, HuggingfaceEmbedder
from core.dashscope_client import DashScopeClient, DashScopeEmbedder
from adalflow import GoogleGenAIClient

# Get API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY')
DASHSCOPE_WORKSPACE_ID = os.environ.get('DASHSCOPE_WORKSPACE_ID')

# Set keys in environment (in case they're needed elsewhere in the code)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
if DASHSCOPE_WORKSPACE_ID:
    os.environ["DASHSCOPE_WORKSPACE_ID"] = DASHSCOPE_WORKSPACE_ID

# Wiki authentication settings
raw_auth_mode = os.environ.get('RAGalyze_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']
WIKI_AUTH_CODE = os.environ.get('RAGalyze_AUTH_CODE', '')

# Get configuration directory from environment variable, or use default if not set
CONFIG_DIR = os.environ.get('RAGalyze_CONFIG_DIR', None)

PROVIDER_NAME_TO_CLASS = {
    "google": GoogleGenAIClient,
    "openai": OpenAIClient,
    "dashscope": DashScopeClient,
}

def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    Recursively replace placeholders like "${ENV_VAR}" in string values
    within a nested configuration structure (dicts, lists, strings)
    with environment variable values. Logs a warning if a placeholder is not found.
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"Environment variable placeholder '{original_placeholder}' was not found in the environment. "
                f"The placeholder string will be used as is."
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        # Handles numbers, booleans, None, etc.
        return config

# Load JSON configuration file
def load_json_config(filename):
    try:
        # If environment variable is set, use the directory specified by it
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            # Otherwise use default directory
            config_path = Path(__file__).parent / "config" / filename

        logger.info(f"Loading configuration from {config_path}")

        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} does not exist")
            return {}

        with open(config_path, 'r') as f:
            config = json.load(f)
            config = replace_env_placeholders(config)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration file {filename}: {str(e)}")
        return {}

# Load generator model configuration
#TODO: immitate load_code_understanding_config to rewrite load_generator_config
def load_generator_config():
    generator_config = load_json_config("generator.json")

    # Add client classes to each provider
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            # Try to set client class from client_class
            if provider_id in ["google", "openai", "dashscope"]:
                provider_config["model_client"] = PROVIDER_NAME_TO_CLASS[provider_id]
            else:
                logger.warning(f"Unknown provider or client class: {provider_id}")

    return generator_config

# Load embedder configuration
def load_embedder_config():
    embedder_config = load_json_config("embedder.json")

    # Process client classes
    for key in ["embedder"]:
        if key in embedder_config and "client_class" in embedder_config[key]:
            class_name = embedder_config[key]["client_class"]
            assert class_name in globals(), f"load_embedder_config: {class_name} not in globals()  {globals()}"
            embedder_config[key]["model_client"] = globals()[class_name]

    return embedder_config

def get_embedder_config():
    """
    Get the current embedder configuration.

    Returns:
        dict: The embedder configuration with model_client resolved
    """
    return configs.get("embedder", {})

# Load repository and file filters configuration
def load_repo_config():
    return load_json_config("repo.json")

# Load code understanding configuration
def load_code_understanding_config():
    """Load code understanding specific configuration"""
    return load_json_config("code_understanding.json")


# Default excluded directories and files
DEFAULT_EXCLUDED_DIRS: List[str] = [
    # Virtual environments and package managers
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    # Version control
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    # Cache and compiled files
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    # Build and distribution
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    # Documentation
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    # IDE specific
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    # Logs and temporary files
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]

# Initialize empty configuration
configs = {}

# Load all configuration files
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()
code_understanding_config = load_code_understanding_config()

# Update configuration
if generator_config:
    configs["default_provider"] = generator_config.get("default_provider", "google")
    configs["providers"] = generator_config.get("providers", {})

# Update embedder configuration
if embedder_config:
    for key in ["embedder", "sketch_filling", "force_embedding", "retriever", "text_splitter", "hybrid"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

# Update repository configuration
if repo_config:
    for key in ["file_filters", "repository", "file_extensions"]:
        if key in repo_config:
            configs[key] = repo_config[key]

# Update code understanding configuration
if code_understanding_config:
    configs["code_understanding"] = code_understanding_config

def get_model_config(provider=None, model=None):
    """
    Get configuration for the specified provider and model

    Parameters:
        provider (str): Model provider ('google', 'openai', 'dashscope'). If None, uses default from config.
        model (str): Model name, or None to use default model

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    # Use default provider if not specified
    if provider is None:
        provider = configs.get("default_provider", "dashscope")
    
    # Get provider configuration
    if "providers" not in configs:
        raise ValueError("Provider configuration not loaded")

    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError(f"Model client not specified for provider '{provider}'")

    # If model not provided, use default model for the provider
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}'")

    # Get model parameters (if present)
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        default_model = provider_config.get("default_model")
        model_params = provider_config["models"][default_model]

    # Prepare base configuration
    result = {
        "model_client": model_client,
    }

    # Provider-specific adjustments
    result["model_kwargs"] = {"model": model, **model_params}

    return result


def get_code_understanding_config(provider=None, model=None):
    """
    Get configuration for code understanding with the specified provider and model

    Parameters:
        provider (str): Model provider ('dashscope'). If None, uses default from config.
        model (str): Model name, or None to use default model

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    # Get code understanding configuration
    code_understanding_config = configs.get("code_understanding", {})
    if not code_understanding_config:
        raise ValueError("Code understanding configuration not found")

    # Use default provider if not specified
    if provider is None:
        provider = code_understanding_config.get("default_provider", "dashscope")
    
    # Get provider configuration
    providers_config = code_understanding_config.get("providers", {})
    if provider not in providers_config:
        available_providers = list(providers_config.keys())
        raise ValueError(f"Provider '{provider}' not found in code understanding configuration. Available providers: {available_providers}")

    provider_config = providers_config[provider]

    # Get client class
    client_class = provider_config.get("client_class")
    if not client_class:
        raise ValueError(f"Client class not specified for provider '{provider}' in code understanding configuration")

    # Map client class to actual class
    model_client = globals().get(client_class)
    if not model_client:
        raise ValueError(f"Unknown client class: {client_class}")

    # If model not provided, use default model for the provider
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}' in code understanding configuration")

    # Get model parameters
    models_config = provider_config.get("models", {})
    if model not in models_config:
        available_models = list(models_config.keys())
        raise ValueError(f"Model '{model}' not found for provider '{provider}' in code understanding configuration. Available models: {available_models}")

    model_params = models_config[model]

    # Prepare result
    result = {
        "provider": provider,
        "model_client": model_client,
        "model": model,
        "model_kwargs": {"model": model, **model_params},
        "provider_config": provider_config,
        "model_config": model_params
    }

    return result