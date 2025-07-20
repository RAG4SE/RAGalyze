import logging
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Union, Optional
from uuid import uuid4
import adalflow as adal
from adalflow.core.types import RetrieverOutput, RetrieverOutputType
from adalflow.core.types import Document
from server.tools.embedder import get_embedder
from server.dual_vector_pipeline import DualVectorRetriever

# Create our own implementation of the conversation classes
@dataclass
class UserQuery:
    query_str: str

@dataclass
class AssistantResponse:
    response_str: str

@dataclass
class DialogTurn:
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse

class CustomConversation:
    """Custom implementation of Conversation to fix the list assignment index out of range error"""

    def __init__(self):
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn):
        """Safely append a dialog turn to the conversation"""
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)

# Import other adalflow components
from adalflow.components.retriever.faiss_retriever import FAISSRetriever, FAISSRetrieverQueriesType
from server.config import configs
from server.data_pipeline import DatabaseManager
from server.dual_vector import DualVectorDocument

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit

class Memory(adal.core.component.DataComponent):
    """Simple conversation management with a list of dialog turns."""

    def __init__(self):
        super().__init__()
        # Use our custom implementation instead of the original Conversation class
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        """Return the conversation history as a dictionary."""
        all_dialog_turns = {}
        try:
            # Check if dialog_turns exists and is a list
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"Memory content: {len(self.current_conversation.dialog_turns)} turns")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"Added turn {i+1} with ID {turn.id} to memory")
                        else:
                            logger.warning(f"Skipping invalid turn object in memory: {turn}")
                else:
                    logger.info("Dialog turns list exists but is empty")
            else:
                logger.info("No dialog_turns attribute in current_conversation")
                # Try to initialize it
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            # Try to recover
            try:
                self.current_conversation = CustomConversation()
                logger.info("Recovered by creating new conversation")
            except Exception as e2:
                logger.error(f"Failed to recover: {str(e2)}")

        logger.info(f"Returning {len(all_dialog_turns)} dialog turns from memory")
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        Add a dialog turn to the conversation history.

        Args:
            user_query: The user's query
            assistant_response: The assistant's response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a new dialog turn using our custom implementation
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )

            # Make sure the current_conversation has the append_dialog_turn method
            if not hasattr(self.current_conversation, 'append_dialog_turn'):
                logger.warning("current_conversation does not have append_dialog_turn method, creating new one")
                # Initialize a new conversation if needed
                self.current_conversation = CustomConversation()

            # Ensure dialog_turns exists
            if not hasattr(self.current_conversation, 'dialog_turns'):
                logger.warning("dialog_turns not found, initializing empty list")
                self.current_conversation.dialog_turns = []

            # Safely append the dialog turn
            self.current_conversation.dialog_turns.append(dialog_turn)
            logger.info(f"Successfully added dialog turn, now have {len(self.current_conversation.dialog_turns)} turns")
            return True

        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            # Try to recover by creating a new conversation
            try:
                self.current_conversation = CustomConversation()
                dialog_turn = DialogTurn(
                    id=str(uuid4()),
                    user_query=UserQuery(query_str=user_query),
                    assistant_response=AssistantResponse(response_str=assistant_response),
                )
                self.current_conversation.dialog_turns.append(dialog_turn)
                logger.info("Recovered from error by creating new conversation")
                return True
            except Exception as e2:
                logger.error(f"Failed to recover from error: {str(e2)}")
                return False

system_prompt = r"""
You are a code assistant which answers user questions on a Github Repo or a local repo.
You will receive user query, relevant context, and past conversation history.

LANGUAGE DETECTION AND RESPONSE:
- Detect the language of the user's query
- Respond in the SAME language as the user's query
- IMPORTANT:If a specific language is requested in the prompt, prioritize that language over the query language

FORMAT YOUR RESPONSE USING MARKDOWN:
- Use proper markdown syntax for all formatting
- For code blocks, use triple backticks with language specification (```python, ```javascript, etc.)
- Use ## headings for major sections
- Use bullet points or numbered lists where appropriate
- Format tables using markdown table syntax when presenting structured data
- Use **bold** and *italic* for emphasis
- When referencing file paths, use `inline code` formatting

IMPORTANT FORMATTING RULES:
1. DO NOT include ```markdown fences at the beginning or end of your answer
2. Start your response directly with the content
3. The content will already be rendered as markdown, so just provide the raw markdown content

Think step by step and ensure your answer is well-structured and visually organized.
"""

# Template for RAG
RAG_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
{{output_format_str}}
<END_OF_SYS_PROMPT>
{# OrderedDict of DialogTurn #}
{% if conversation_history %}
<START_OF_CONVERSATION_HISTORY>
{% for key, dialog_turn in conversation_history.items() %}
{{key}}.
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index}}.
File Path: {{context.meta_data.get('file_path', 'unknown')}}
{% if context.meta_data.get('understanding_text') %}
Code Summary:
---
{{context.meta_data.get('understanding_text')}}
---
{% endif %}
Code Snippet:
```
{{context.text}}
```
{% endfor %}
<END_OF_CONTEXT>
{% endif %}
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""

from dataclasses import dataclass, field

@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Chain of thoughts for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query, formatted in markdown for beautiful rendering with react-markdown. DO NOT include ``` triple backticks fences at the beginning or end of your answer."})

    __output_fields__ = ["rationale", "answer"]

class SingleVectorRetriever(FAISSRetriever):
    """A wrapper of FAISSRetriever with the additional feature of supporting docoments feature"""
    def __init__(self, documents, *args, **kwargs):
        self.original_doc = documents
        super().__init__(documents=documents, *args, **kwargs)
    
    def call(
        self,
        input: FAISSRetrieverQueriesType,
        top_k: Optional[int] = None,
    ) -> RetrieverOutputType:
        retriever_output = super().call(input, top_k)

        # Extract the first result from the list
        if not retriever_output:
            return []

        first_output = retriever_output[0]

        # Get the documents based on the indices
        retrieved_docs = [self.original_doc[i] for i in first_output.doc_indices]

        # Create a new RetrieverOutput with the documents
        return [
            RetrieverOutput(
                doc_indices=first_output.doc_indices,
                doc_scores=first_output.doc_scores,
                query=first_output.query,
                documents=retrieved_docs,
            )
        ]

class RAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_path) first."""

    def __init__(self, provider=None, model=None, is_huggingface_embedder: bool = False, use_dual_vector: bool = False):  # noqa: F841 - use_s3 is kept for compatibility
        """
        Initialize the RAG component.

        Args:
            provider: Model provider to use (google, openai, dashscope, siliconflow, deepseek). If None, uses default from config.
            model: Model name to use with the provider. If None, uses default from config.
            use_s3: Whether to use S3 for database storage (default: False)
        """
        super().__init__()

        # Import configs to get defaults
        from server.config import configs
        
        # Use provided provider or fall back to config default
        if provider is None:
            provider = configs.get("default_provider", "dashscope")
        
        # Use provided model or fall back to config default for the provider
        if model is None and "providers" in configs and provider in configs["providers"]:
            model = configs["providers"][provider].get("default_model")

        self.provider = provider
        self.model = model
        self.use_dual_vector = use_dual_vector
        # Initialize components
        self.memory = Memory()
        self.embedder = get_embedder(is_huggingface_embedder)

        # Format instructions for natural language output (no structured parsing)
        format_instructions = """
Please provide a comprehensive answer to the user's question based on the provided context.

IMPORTANT FORMATTING RULES:
1. Respond in the same language as the user's question
2. Format your response using markdown for better readability
3. Use code blocks, bullet points, headings, and other markdown features as appropriate
4. Be clear, concise, and helpful
5. If you use code examples, make sure they are properly formatted with language-specific syntax highlighting
6. Structure your answer logically with clear sections if the question is complex"""

        # Get model configuration based on provider and model
        from server.config import get_model_config
        generator_config = get_model_config(self.provider, self.model)

        # Set up the main generator (no output processors to avoid JSON parsing issues)
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": None, # No conversation history
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=generator_config["model_client"](),
            model_kwargs=generator_config["model_kwargs"],
        )

        self.retriever = None
        self.db_manager = None
        self.documents = None

    def initialize_db_manager(self, repo_path: str, file_count_upperlimit: int = None, use_dual_vector: bool = False):
        """Initialize the database manager with local storage"""
        self.db_manager = DatabaseManager(repo_path, file_count_upperlimit, use_dual_vector)
        self.documents = []

    def _validate_and_filter_embeddings(self, documents: List[Document|DualVectorDocument]) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched embedding sizes.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning("No documents provided for embedding validation")
            return []

        valid_documents = []
        embedding_sizes = {}
        code_embedding_sizes = {}
        understanding_embedding_sizes = {}
        is_dual_vector = False
        # First pass: collect all embedding sizes and count occurrences
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                if not hasattr(doc, 'vector') or doc.vector is None:
                    logger.warning(f"❓Document {i} has no embedding vector, skipping...\n doc: {doc}")
                    continue

                try:
                    if isinstance(doc.vector, list):
                        embedding_size = len(doc.vector)
                    elif hasattr(doc.vector, 'shape'):
                        embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                    elif hasattr(doc.vector, '__len__'):
                        embedding_size = len(doc.vector)
                    else:
                        logger.warning(f"Document {i} has invalid embedding vector type: {type(doc.vector)}, skipping")
                        continue

                    if embedding_size == 0:
                        logger.warning(f"❓Document {i} has empty embedding vector, skipping...\n doc: {doc}")
                        continue

                    embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

                except Exception as e:
                    logger.warning(f"Error checking embedding size for document {i}: {str(e)}, skipping")
                    continue
            elif isinstance(doc, DualVectorDocument):
                is_dual_vector = True
                if hasattr(doc, 'code_embedding'):
                    code_embedding_sizes[len(doc.code_embedding)] = code_embedding_sizes.get(len(doc.code_embedding), 0) + 1
                if hasattr(doc, 'understanding_embedding'):
                    understanding_embedding_sizes[len(doc.understanding_embedding)] = understanding_embedding_sizes.get(len(doc.understanding_embedding), 0) + 1
            else:
                raise ValueError(f"❓Document {i} has invalid type: {type(doc)}, skipping...\n")

        if not is_dual_vector:
            if not embedding_sizes:
                logger.error("No valid embeddings found in any documents")
                return []

            # Find the most common embedding size (this should be the correct one)
            target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
            logger.info(f"Target embedding size: {target_size} (found in {embedding_sizes[target_size]} documents)")

            # Log all embedding sizes found
            for size, count in embedding_sizes.items():
                if size != target_size:
                    logger.warning(f"Found {count} documents with incorrect embedding size {size}, will be filtered out")

            # Second pass: filter documents with the target embedding size
            for i, doc in enumerate(documents):
                if not hasattr(doc, 'vector') or doc.vector is None:
                    continue

                try:
                    if isinstance(doc.vector, list):
                        embedding_size = len(doc.vector)
                    elif hasattr(doc.vector, 'shape'):
                        embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                    elif hasattr(doc.vector, '__len__'):
                        embedding_size = len(doc.vector)
                    else:
                        continue

                    if embedding_size == target_size:
                        valid_documents.append(doc)
                    else:
                        # Log which document is being filtered out
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning(f"Filtering out document '{file_path}' due to embedding size mismatch: {embedding_size} != {target_size}")

                except Exception as e:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Error validating embedding for document '{file_path}': {str(e)}, skipping")
                    continue

            logger.info(f"Embedding validation complete: {len(valid_documents)}/{len(documents)} documents have valid embeddings")

            if len(valid_documents) == 0:
                logger.error("No documents with valid embeddings remain after filtering")
            elif len(valid_documents) < len(documents):
                filtered_count = len(documents) - len(valid_documents)
                logger.warning(f"Filtered out {filtered_count} documents due to embedding issues")

            return valid_documents
        
        else:
            if not code_embedding_sizes or not understanding_embedding_sizes:
                logger.error("No valid embeddings found in any documents")
                return []
            target_code_embedding_size = max(code_embedding_sizes.keys(), key=lambda k: code_embedding_sizes[k])
            # Some understanding text is "" and its embedding size is 0. We need to remove them when calculating the primary embedding size
            target_understanding_embedding_size = max({k: v for k, v in understanding_embedding_sizes.items() if k > 0}.keys(), key=lambda k: understanding_embedding_sizes[k])
            logger.info(f"Target code embedding size: {target_code_embedding_size} (found in {code_embedding_sizes[target_code_embedding_size]} documents)")
            logger.info(f"Target understanding embedding size: {target_understanding_embedding_size} (found in {understanding_embedding_sizes[target_understanding_embedding_size]} documents)")
            for i, doc in enumerate(documents):
                assert isinstance(doc, DualVectorDocument)
                if len(doc.code_embedding) != target_code_embedding_size or len(doc.understanding_embedding) != target_understanding_embedding_size and len(doc.understanding_embedding) > 0:
                    logger.warning(f"Filtering out document '{doc.file_path}' due to embedding size mismatch: {len(doc.code_embedding)} != {target_code_embedding_size} or {len(doc.understanding_embedding)} != {target_understanding_embedding_size}")
                else:
                    valid_documents.append(doc)
            return valid_documents

    def prepare_retriever(self, repo_path: str,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None, force_recreate_db: bool = False, file_count_upperlimit: Union[int, None] = None,
                      is_huggingface_embedder: bool = False):
        """
        Prepare the retriever for a repository.
        Will load database from local storage if available.

        Args:
            repo_path: URL or local path to the repository
            access_token: Optional access token for private repositories
            excluded_dirs: Optional list of directories to exclude from processing
            excluded_files: Optional list of file patterns to exclude from processing
            included_dirs: Optional list of directories to include exclusively
            included_files: Optional list of file patterns to include exclusively
            force_recreate_db: Whether to force recreate the database
            file_count_upperlimit: Upper limit of the number of files to process
            is_huggingface_embedder: Whether to use huggingface embedder
        """
        self.initialize_db_manager(repo_path, file_count_upperlimit, self.use_dual_vector)

        logger.info(f'\n🔍 Build up database...')
        # self.documents is a list of Document or DualVectorDocument
        self.documents = self.db_manager.prepare_database(
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
            force_recreate=force_recreate_db,
            is_huggingface_embedder=is_huggingface_embedder
        )
        logger.info(f"✅ Loaded {len(self.documents)} documents for retrieval")
        # Validate and filter embeddings to ensure consistent sizes
        self.documents = self._validate_and_filter_embeddings(self.documents)
        logger.info(f"🎉Validated and filtered {len(self.documents)} documents")
        if not self.documents:
            raise ValueError("No valid documents with embeddings found. Cannot create retriever.")

        try:
            if self.use_dual_vector:
                self.retriever = DualVectorRetriever(self.documents, self.embedder, **configs["retriever"])
                logger.info("✅ Dual vector retriever created successfully")
            else:
                self.retriever = SingleVectorRetriever(
                    **configs["retriever"],
                    embedder=self.embedder,
                    documents=self.documents,
                    document_map_func=lambda doc: doc.vector,
                )
                logger.info(f"✅ SingleVectorRetriever created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create retriever: {e}")
            raise

    def call(self, query: str) -> List[RetrieverOutput]:
        """
        Query the RAG system.
        """
        if not self.retriever:
            raise ValueError("Retriever not prepared. Call prepare_retriever first.")
        
        logger.info(f"🏃 Running RAG for query: '{query}'")
        
        try:
            retrieved_docs = self.retriever.call(query)
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")
            # Return an empty RetrieverOutput to indicate an error
            return [RetrieverOutput(documents=[])]
