"""Factory for creating appropriate text splitters based on document type."""

import os

from adalflow.components.data_process import TextSplitter

from deepwiki_cli.rag.splitter import *
from deepwiki_cli.configs import configs
from deepwiki_cli.logger.logging_config import get_tqdm_compatible_logger

logger = get_tqdm_compatible_logger(__name__)


class SplitterFactory:
    """Factory class for creating appropriate splitters based on document type."""


    def __init__(self):
        """Initialize the splitter factory."""
        self._natural_language_splitter = None
        self._code_splitter = None
        # File extensions that are considered code files
        self.CODE_EXTENSIONS = configs()["repo"]["file_extensions"]["code_extensions"]
        # File extensions that are considered text/documentation files
        self.TEXT_EXTENSIONS = configs()["repo"]["file_extensions"]["doc_extensions"]

    def _get_txt_splitter(self) -> NaturalLanguageSplitter:
        """Get or create text splitter instance.

        Returns:
            SmartTextSplitter: Configured smart text splitter
        """
        if self._natural_language_splitter is None:
            natural_language_splitter_config = configs()["rag"]["dynamic_splitter"]["natural_language_splitter"].copy()
            # Add smart splitting parameters for text content
            natural_language_splitter_config["smart_boundary_ratio"] = 0.8
            self._natural_language_splitter = NaturalLanguageSplitter(**natural_language_splitter_config)
        return self._natural_language_splitter

    def _get_code_splitter(self, extension: str) -> CodeSplitter:
        """Get or create code splitter instance.

        Returns:
            SmartTextSplitter: Configured smart code splitter with custom tokenizer
        """
        if self._code_splitter is None:
            code_splitter_config = configs()["rag"]["dynamic_splitter"]["code_splitter"].copy()

            # Add smart splitting parameters for code content
            code_splitter_config["smart_boundary_ratio"] = (
                0.75  # Slightly more aggressive for code
            )
            code_splitter_config["file_extension"] = extension
            # Create the smart code splitter
            self._code_splitter = CodeSplitter(**code_splitter_config)

        return self._code_splitter

    def detect_document_type(self, file_path: str) -> tuple[str, str]:
        """Detect document type and extension based on file path.

        Args:
            file_path (str): Path to the file

        Returns:
            tuple: (document_type, extension)
        """
        if not file_path:
            # return 'unknown', ''
            raise ValueError("file_path is required for document type detection")

        # Get file extension
        _, ext = os.path.splitext(file_path.lower())

        if ext in self.CODE_EXTENSIONS:
            return "code", ext
        elif ext in self.TEXT_EXTENSIONS:
            return "text", ext
        else:
            # For unknown extensions, try to detect based on content patterns
            raise ValueError(f"Unknown file extension: {ext}")

    def detect_content_type(self, content: str, file_path: str = "") -> str:
        """Detect content type based on file path and content analysis.

        Args:
            content (str): File content
            file_path (str): File path (optional)

        Returns:
            str: 'code', 'text', or 'unknown'
        """
        # First try file extension
        if file_path:
            doc_type, _ = self.detect_document_type(file_path)
            if doc_type != "unknown":
                return doc_type

        else:
            # TODO: currently, file_path is required. The case where file_path == "" will be supported later.
            raise ValueError("file_path is required for content type detection")

        # Fallback to content analysis
        if not content.strip():
            return "unknown"

        # Simple heuristics for content detection
        content_lower = content.lower()

        # Code indicators
        code_indicators = [
            "def ",
            "function ",
            "class ",
            "import ",
            "from ",
            "#include",
            "namespace ",
            "package ",
            "public class",
            "private ",
            "protected ",
            "const ",
            "let ",
            "var ",
            "fn ",
            "func ",
            "impl ",
            "trait ",
            "#!/",
            "<?php",
            "<%",
            "{",
            "}",
            ";\n",
            "->",
            "=>",
            "::",
        ]

        # Text/documentation indicators
        text_indicators = [
            "# ",
            "## ",
            "### ",
            "- ",
            "* ",
            "1. ",
            "2. ",
            "3. ",
            "http://",
            "https://",
            "[",
            "](",
            "**",
            "__",
            "*",
            "_",
        ]

        # Markdown-specific indicators
        markdown_indicators = [
            "# ",
            "## ",
            "### ",
            "#### ",
            "##### ",
            "###### ",
            "```",
            "~~~",
            "[",
            "](",
            "**",
            "__",
            "*",
            "_",
            "- [ ]",
            "- [x]",
            "> ",
            "---",
            "***",
            "| ",
        ]

        # JSON-specific indicators
        json_indicators = [
            "{",
            "}",
            "[",
            "]",
            '":',
            '",',
            "null",
            "true",
            "false",
            '"id":',
            '"name":',
            '"value":',
            '"data":',
            '"type":',
        ]

        # RST-specific indicators
        rst_indicators = [
            ".. ",
            ":::",
            "::",
            "====",
            "----",
            "~~~~",
            "^^^^",
            ".. code-block::",
            ".. note::",
            ".. warning::",
            ".. image::",
            ".. toctree::",
            ".. automodule::",
            ".. autoclass::",
            ".. autofunction::",
        ]

        # YAML-specific indicators
        yaml_indicators = [
            "---",
            "...",
            "- ",
            ": ",
            "|",
            ">",
            "&",
            "*",
            "apiVersion:",
            "kind:",
            "metadata:",
            "spec:",
            "data:",
            "version:",
            "services:",
            "volumes:",
            "networks:",
            "environment:",
            "ports:",
            "image:",
            "build:",
        ]

        code_score = sum(
            1 for indicator in code_indicators if indicator in content_lower
        )
        text_score = sum(
            1 for indicator in text_indicators if indicator in content_lower
        )
        markdown_score = sum(
            1 for indicator in markdown_indicators if indicator in content_lower
        )
        json_score = sum(
            1 for indicator in json_indicators if indicator in content_lower
        )
        rst_score = sum(1 for indicator in rst_indicators if indicator in content_lower)
        yaml_score = sum(
            1 for indicator in yaml_indicators if indicator in content_lower
        )

        # Calculate ratios
        total_chars = len(content)
        if total_chars > 0:
            brace_ratio = (content.count("{") + content.count("}")) / total_chars
            semicolon_ratio = content.count(";") / total_chars

            # Code files typically have more braces and semicolons
            if brace_ratio > 0.01 or semicolon_ratio > 0.02:
                code_score += 2

        # Determine content type based on scores
        max_score = max(
            code_score, text_score, markdown_score, json_score, rst_score, yaml_score
        )

        if max_score == 0:
            return "unknown"
        elif yaml_score == max_score and yaml_score > 0:
            return "yaml"
        elif rst_score == max_score and rst_score > 0:
            return "rst"
        elif json_score == max_score and json_score > 0:
            return "json"
        elif markdown_score == max_score and markdown_score > 0:
            return "markdown"
        elif code_score == max_score:
            return "code"
        elif text_score == max_score:
            return "text"
        else:
            return "unknown"

    def get_splitter(
        self, content: str = "", file_path: str = "", force_type: str = None
    ):
        """Get appropriate splitter based on content and file path.

        Args:
            content (str): File content (optional)
            file_path (str): File path (optional)
            force_type (str): Force specific splitter type ('code', 'text', or 'markdown')

        Returns:
            TextSplitter: Appropriate splitter instance (SmartTextSplitter or MarkdownTextSplitter)
        """
        if force_type:
            doc_type = force_type
            _, ext = (
                self.detect_document_type(file_path) if file_path else ("unknown", "")
            )
        else:
            doc_type, ext = self.detect_document_type(file_path)
            if doc_type == "unknown":
                doc_type = self.detect_content_type(content, file_path)

        # Return appropriate splitter based on document type
        if doc_type == "code":
            return self._get_code_splitter(ext)
        else:  # 'text' or fallback
            return self._get_txt_splitter()


# Global factory instance
_splitter_factory = None


def get_splitter_factory() -> SplitterFactory:
    """Get global splitter factory instance.

    Returns:
        SplitterFactory: Global factory instance
    """
    global _splitter_factory
    if _splitter_factory is None:
        _splitter_factory = SplitterFactory()
    return _splitter_factory
