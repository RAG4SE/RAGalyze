# Instructions on building the FunctionCallExtractorFromEntryFuncPipeline

## Description

The FunctionCallExtractorFromEntryFuncPipeline is a pipeline that extracts function calls from the entry function.

## Implementation

You can refer to other classes in the agent.py and immitate their implementation logic.

The __call__ function should take in the following parameters:

1. entry_function_name: The name of the entry function.
2. file_path : The path to the file that contains the entry function.

## Material

1. FetchFunctionDefinitionFromCallAgent
2. FetchCallExpressionsQuery
3. FetchFunctionDefinitionFromNamePipeline

## Steps

1. Use FetchFunctionDefinitionFromNamePipeline to fetch the definition of the entry function.
2. Use FetchCallExpressionsQuery to extract function calls from the entry function.
3. For each function call expression, use FetchFunctionDefinitionFromNamePipeline to fetch the definition of the called function.
4. Store the current function call globally as 
   ```
    entry_function -> called_function1
                   -> called_function2
                   -> called_function3
                   ...
   ```
6. recursively go to the 1st step to extract function call chains starting from each called function. and pay attention to recursive calls.
