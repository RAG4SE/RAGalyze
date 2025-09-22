from ragalyze import *
from ragalyze.configs import *
from ragalyze.agent import *

set_global_config_value("repo_path", "/Users/mac/repo/test-solidity/")
set_global_config_value("rag.recreate_db", True)
# set_global_config_value(
#     "repo.file_filters.extra_excluded_patterns",
#     ["*deps/*", "*test*/*", "workspace/*", "*/tags"],
# )
# set_global_config_value("generator.provider", "kimi")
# set_global_config_value("generator.model", "kimi-k2-0905-preview")
set_global_config_value("generator.provider", "dashscope")
set_global_config_value("generator.model", "qwen3-coder-plus")
# set_global_config_value("generator.model", "qwen3-30b-a3b-instruct-2507")
# set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")

# set_global_config_value("generator.provider", "modelscope")
# set_global_config_value("generator.model", "Qwen/Qwen3-Next-80B-A3B-Instruct")


agent = FetchCallerHeaderAgent(debug=True)

headers = agent(
    callee_name="builtin",
    callee_body="""
    BuiltinFunctionForEVM const& EVMDialect::builtin(BuiltinHandle const& _handle) const
    """,
)

print(headers)

# agent2 = FetchCallerNameFromHeaderAgent(debug=True)


# for header in headers:
#     print(">>", header)
#     name = agent2(header)
#     print(name)

# agent = FetchCallerNameAgentV2(debug=True)
# print(agent(
#     callee_name="builtin",
#     # callee_body="""
#     # BuiltinFunctionForEVM const& EVMDialect::builtin(BuiltinHandle const& _handle) const
#     # """,
# ))

# agent = FetchFunctionDefinitionAgent(debug=True)
# print(
#     agent(
#         function_name="containsNonContinuingFunctionCall",
#         # callee_body="""
#         # BuiltinFunctionForEVM const& EVMDialect::builtin(BuiltinHandle const& _handle) const
#         # """,
#     )
# )
