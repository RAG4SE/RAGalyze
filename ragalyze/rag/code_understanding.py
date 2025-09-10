import os
import asyncio
import logging
from typing import List, Optional, Union
from openai.types.chat import ChatCompletion
from tqdm import tqdm

import adalflow as adal
from adalflow.core.types import (
    ModelType,
)
from ragalyze.logger.logging_config import get_tqdm_compatible_logger
from ragalyze.configs import get_code_understanding_client

logger = get_tqdm_compatible_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# System prompt designed specifically for the code understanding task
CODE_UNDERSTANDING_SYSTEM_PROMPT = """
## Role
Act as a senior software engineer conducting a code review.

## Task
1. For each line of the supplied code snippet, provide a clear, concise natural language explanation of what that line does.
2. Number each explanation with the corresponding line number.
3. After all line explanations, provide a concise executive summary of the entire code snippet.

## Output Format Requirements
Follow this exact format:
```
File Path: <absolute or relative path>
---------------------------
<line number>: <natural language explanation of what this line does>
... (repeat for every line)
---------------------------
Summary: <concise summary of the entire code snippet in natural language>
```

## Important Guidelines
- Provide meaningful explanations, not just restating the code
- Be concise but clear in your explanations
- Focus on the purpose and function of each line
- For the summary, capture the main purpose of the entire code snippet

## Example
For input code:
def add(a, b):
    return a + b

Expected output:
File Path: example.py
---------------------------
1: Define a function named 'add' that takes two parameters 'a' and 'b'
2: Return the sum of parameters 'a' and 'b'
---------------------------
Summary: This code defines a simple function that adds two numbers together and returns the result.
"""

# Template for Code Understanding
CODE_UNDERSTANDING_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
<END_OF_SYS_PROMPT>
<START_OF_USER_PROMPT>
Analyze the following code snippet:

File Path: {{file_path}}

Code Snippet:
```
{{code_snippet}}
```

Please provide your analysis in the exact format specified in the instructions.
<END_OF_USER_PROMPT>
"""


class CodeUnderstandingGenerator:
    """
    Uses the Dashscope model to generate natural language summaries for code.
    """

    def __init__(self, **kwargs):
        """
        Initializes the code understanding generator.

        """
        assert (
            "model" in kwargs
        ), f"rag/dual_vector_pipeline.py:model not found in code_understanding_generator_config"
        self.model = kwargs["model"]
        assert (
            "batch_size" in kwargs
        ), f"batch_size not found in the hydra config of code_understanding in rag.yaml."
        self.batch_size = kwargs["batch_size"]
        # Extract configuration
        assert (
            "model_kwargs" in kwargs
        ), f"model_kwargs not found in the hydra config of code_understanding in rag.yaml."
        self.model_kwargs = kwargs["model_kwargs"]
        
        # Set up the adal.Generator
        self.generator = adal.Generator(
            template=CODE_UNDERSTANDING_TEMPLATE,
            prompt_kwargs={
                "system_prompt": CODE_UNDERSTANDING_SYSTEM_PROMPT,
                "file_path": None,
                "code_snippet": None,
            },
            model_client=get_code_understanding_client(),
            model_kwargs={"model": self.model, **self.model_kwargs},
        )

    def call(
        self, code: Union[str, List[str]], file_path: Union[str, List[str]]
    ) -> Union[List[str], None]:
        """
        Generates a summary for the given code snippet.

        Args:
            code: The code string to be summarized, or a list of code strings to be summarized.
            file_path: The file path where the code is located (optional).

        Returns:
            A list of generated code summary strings.
        """
        if isinstance(code, str):
            code = [code]
        if isinstance(file_path, str):
            file_path = [file_path]
        assert len(code) == len(
            file_path
        ), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"
        summaries = []
        for i, code_snippet in enumerate(code):
            try:
                result = self.generator.call(
                    prompt_kwargs={
                        "file_path": file_path[i],
                        "code_snippet": code_snippet,
                    }
                )

                # Extract content from GeneratorOutput data field
                summary = result.data
                summaries.append(summary.strip())
            except Exception as e:
                logger.error(
                    f"Failed to generate code understanding for {file_path[i]}: {e}"
                )
                # Return an empty or default summary on error
                return None

        return summaries

    async def acall(
        self, code: Union[str, List[str]], file_path: Union[str, List[str]]
    ) -> Union[List[str], None]:
        """
        Generates a summary for the given code snippet.

        Args:
            code: The code string to be summarized, or a list of code strings to be summarized.
            file_path: The file path where the code is located (optional).

        Returns:
            A list of generated code summary strings.
        """
        if isinstance(code, str):
            code = [code]
        if isinstance(file_path, str):
            file_path = [file_path]
        assert len(code) == len(
            file_path
        ), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"

        async def _acall(code_snippet, file_path):
            try:
                result = await self.generator.acall(
                    prompt_kwargs={
                        "file_path": file_path,
                        "code_snippet": code_snippet,
                    }
                )

                # Extract content from GeneratorOutput data field
                summary = result.data
                return summary.strip()
            except Exception as e:
                logger.error(
                    f"Failed to generate code understanding for {file_path}: {e}"
                )
                # Re-raise the exception to let caller handle it
                raise

        tasks = [_acall(c, f) for c, f in zip(code, file_path)]
        try:
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            logger.error(f"One or more code understanding tasks failed: {e}")
            return None

    def call_in_sequence(self, code: List[str], file_path: List[str]) -> List[str]:
        assert len(code) == len(
            file_path
        ), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"
        understanding_texts = []
        for i, code_snippet in enumerate(
            tqdm(
                code,
                desc="Generating code understanding",
                disable=False,
                total=len(code),
            )
        ):
            understanding_text = self.call(code_snippet, file_path[i])
            if understanding_text:
                understanding_texts.extend(understanding_text)
        return understanding_texts

    async def batch_call(
        self, code: List[str], file_path: List[str], batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Process code understanding in batches to avoid overwhelming the API.

        Args:
            code: List of code strings to process
            file_path: List of corresponding file paths
            batch_size: Number of concurrent requests per batch (uses config if None)

        Returns:
            List of understanding texts
        """
        assert len(code) == len(
            file_path
        ), f"The number of code snippets ({len(code)}) should be the same as the number of file paths ({len(file_path)})"

        # Use configured batch size if not provided
        if batch_size is None:
            batch_size = self.batch_size

        logger.info(f"Processing {len(code)} documents with batch size: {batch_size}")

        # Divide tasks into batches
        all_results = []
        total_items = len(code)

        with tqdm(
            total=total_items,
            desc="Generating code understanding (batched)",
            disable=False,
        ) as pbar:

            for i in range(0, total_items, batch_size):
                # Create batch
                batch_end = min(i + batch_size, total_items)
                batch_code = code[i:batch_end]
                batch_paths = file_path[i:batch_end]

                logger.debug(
                    f"Processing batch {i//batch_size + 1}: items {i+1}-{batch_end} of {total_items}"
                )

                # Process batch with acall which handles multiple inputs
                try:
                    batch_results = await self.acall(batch_code, batch_paths)
                    
                    # Collect results from this batch
                    if batch_results:
                        all_results.extend(batch_results)
                    pbar.update(len(batch_code))

                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Update progress bar even on error
                    pbar.update(len(batch_code))
                    # Re-raise exception to handle at higher level
                    raise

                # Optional: Add small delay between batches to be API-friendly
                if batch_end < total_items:  # Not the last batch
                    await asyncio.sleep(0.1)  # 100ms delay between batches

        logger.info(
            f"Completed batch processing: {len(all_results)} understanding texts generated"
        )
        return all_results
