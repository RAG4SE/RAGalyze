from ragalyze import *
from ragalyze.configs import *
from ragalyze.agent import *
from ragalyze.rag import retriever
from pathlib import Path

# set_global_config_value("repo_path", "/Users/mac/repo/RAGalyzeBench/solidity/")
set_global_config_value("repo_path", "./bench/call_maze_cpp")
set_global_config_value("repo_path", "./bench/solidity_vulnerable_contracts")
set_global_config_value("repo_path", "./bench/call_fetch_cpp")
# set_global_config_value("repo_path", "./bench/same_function_name_in_a_file")
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

# set_global_config_value("generator.provider", "deepseek")
# set_global_config_value("generator.model", "deepseek-chat")


# r = FetchFunctionDefinitionFromNamePipeline(debug=True)
# result = r("f", file_path="test.py", line_number=5)
# print(result)

# r = FetchFunctionDefinitionFromNamePipeline(debug=True)
# result = r("f", file_path="test2.py", line_number=1)
# print(result)
# result = r("f", file_path="test2.py", line_number=103)
# print(result)
# result = r("f", file_path="test2.py", line_number=152)
# print(result)

function_body = """
function launchReentrancyAttack(address token, uint256 amount) external onlyOwner {
    require(!attackInProgress, "Attack already in progress");

    attackInProgress = true;
    attackCount++;
    emit AttackStarted("Reentrancy Attack");

    // First, deposit tokens to the protocol
    targetToken.approve(address(targetProtocol), amount);

    // Call withdraw to trigger reentrancy
    targetProtocol.withdraw(token, amount);

    attackInProgress = false;
    emit AttackCompleted("Reentrancy Attack", stolenAmount);
}
"""

# r = FetchCalleeInfoPipeline(debug=True)
# expr = "targetToken.approve(address(targetProtocol), amount);"
# file_path = "AttackContract.sol"
# line_number = 50
# result = r(expr, file_path, line_number)
# print(result)

# r = FetchFunctionDefinitionFromNamePipeline(debug=True)
# print(r("VulnerableDeFiProtocol"))

# r = FetchCalleeInfoPipeline(debug=True)
# expr = "VulnerableDeFiProtocol(_protocol);"
# file_path = "AttackContract.sol"
# line_number = 34
# result = r(expr, file_path, line_number)
# print(result)

r = FetchCalleeInfoPipeline(debug=True)
expr = "cptr->func().func();"
file_path = "test.cpp"
line_number = 29
result = r(expr, file_path, line_number)
print(result)
