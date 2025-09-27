from ragalyze import *
from ragalyze.configs import *
from ragalyze.agent import *

set_global_config_value("repo_path", "/Users/mac/repo/RAGalyzeBench/solidity/")
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

# set_global_config_value("generator.provider", "modelscope")
# set_global_config_value("generator.model", "Qwen/Qwen3-Next-80B-A3B-Instruct")

agent = FetchFunctionDefinitionAgent(debug=True)

functions = agent(
    callee_name="sc",
    caller_body="""
void MovableChecker::operator()(Identifier const& _identifier)
{
	SideEffectsCollector sc;
	sc(_identifier);
	SideEffectsCollector::operator()(_identifier);
	m_variableReferences.emplace(_identifier.name);
}
"""
)
print('==========print==========')
for function in functions:
    print(function)

# agent = IsClassInstance(debug=True)
# result = agent(
#     context="""
# void MovableChecker::operator()(Identifier const& _identifier)
# {
#     SideEffectsCollector sc;
# 	sc(_identifier);
# 	m_variableReferences.emplace(_identifier.name);
# }
# """,
#     callee_name="sc",
# )
# print('====')
# print(result)

# agent = FetchClassPipeline(debug=True)
# result = agent("SideEffectsCollector")
# print('=====result=====')
# print(result)

# agent = FindCallOperatorQuery(debug=True)
# result = agent(
#     class_definition="""
# class A {
#     public:
#         A() {}
#         ~A() {}
#         void operator()();
# };
# """,
# )

# print('=====result=====')
# print(result)
