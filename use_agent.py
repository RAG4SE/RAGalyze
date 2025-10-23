from ragalyze import *
from ragalyze.configs import *
from ragalyze.agent import *
from ragalyze.rag import retriever
from pathlib import Path
import os
import traceback

# set_global_config_value("repo_path", "/Users/mac/repo/RAGalyzeBench/solidity/")
set_global_config_value("repo_path", "./bench/call_maze_cpp")
print(os.path.abspath("./bench/single_file_solc/"))
set_global_config_value("repo_path", os.path.abspath("./bench/single_file_solc/"))
# set_global_config_value("repo_path", os.path.abspath("./bench/python_no_entry_func/"))
set_global_config_value("repo.file_filters.extra_excluded_patterns", ["*txt"])
# set_global_config_value("repo_path", "./bench/python_no_entry_func")
# set_global_config_value("repo_path", "./bench/test_var_from_other_file_python")
set_global_config_value("rag.recreate_db", True)
# set_global_config_value(
#     "repo.file_filters.extra_excluded_patterns",
#     ["*deps/*", "*test*/*", "workspace/*", "*/tags"],
# )
# set_global_config_value("generator.provider", "kimi")
# set_global_config_value("generator.model", "kimi-k2-0905-preview")
set_global_config_value("generator.provider", "dashscope")
# set_global_config_value("generator.model", "qwen3-coder-plus")
# set_global_config_value("generator.model", "qwen3-30b-a3b-instruct-2507")
set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")

set_global_config_value("generator.provider", "deepseek")
set_global_config_value("generator.model", "deepseek-chat")


# r = VarTypeInferencePipeline(debug=True)
# var = "a"
# expr = "print(a)"
# file_path = "main.py"
# line_number = 3
# result = r(
#     variable_name=var,
#     file_path=file_path,
#     line_number=line_number,
#     expression=expr,
# )
# print(result)

# r = FunctionCallExtractorFromEntryFuncPipeline(debug=True)
# entry_function_name = "execute"
# file_path = "parity_wallet_bug_1.sol"
# line_number = 65
# call_chain = r(
#     entry_function_name=entry_function_name,
#     file_path=file_path,
#     line_number=line_number,
# )
# if call_chain:
#     call_chain.print_call_chains()

r = FindAllFuncHeaderPipeline(debug=True)
all_func_infos = r()
print(all_func_infos)
r = FunctionCallExtractorFromEntryFuncsPipeline(debug=True)
call_chain_forest = r(all_func_infos)
call_chain_forest.serialize("call_chain_forest.pkl")
call_chain_forest.write2file_call_chains("call_chains.txt")
call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
call_chain_forest.print_call_chains()

# call_chain_forest = CallChainForest.deserialize("call_chain_forest.pkl")
# call_chain_forest.write2file_call_chains("call_chains.txt")
# call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
# call_chain_forest.print_call_chains()

# r = FunctionCallExtractorFromEntryFuncWithLSPPipeline(debug=True)
# call_chain_tree = r('initWallet', 'parity_wallet_bug_1.sol', 222)
# call_chain_tree.print_nocode_call_chains()

# r = FetchCallExpressionsQuery(debug=True)
# function_body = """
# 0: function setDailyLimit(uint _newLimit) onlymanyowners(sha3(msg.data)) external {
# 1:   m_dailyLimit = _newLimit;
# 2: }
# """
# print(r(function_body=function_body))

# r = FetchCalleeInfoPipeline(debug=True)
# expression = "isOwner(_to)"
# file_path = "parity_wallet_bug_1.sol"
# line_number = 140
# result = r(
#     expression=expression,
#     file_path=file_path,
#     line_number=line_number,
#     language="solidity",
# )
# print(result)


