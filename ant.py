from ragalyze.configs import *
from ragalyze.agent import *
import os

# 一定要用绝对路径，否则lsp会报错
set_global_config_value("repo_path", os.path.abspath("./bench/single_file_solc/"))
set_global_config_value("rag.recreate_db", True)

# set_global_config_value("rag.embedder.provider", "local_server")
# set_global_config_value("rag.embedder.model", "text-embedding-qwen3-embedding-8b")

# set_global_config_value("rag.code_understanding.provider", "lingxi")
# set_global_config_value("rag.code_understanding.model", "qwen3-8b")

set_global_config_value("generator.provider", "lingxi")
# set_global_config_value("generator.model", "qwen3-coder-480b-a35b-instruct")
# set_global_config_value("generator.model", "qwen3-next-80b-a3b-instruct")
# 别用qwen，lingxi上的qwen不支持强制json输出，会报错，ling是可以的
set_global_config_value("generator.model", "ling-1t")

set_global_config_value("rag.retriever.bm25.top_k", 100)


r = FindAllFuncHeaderPipeline(debug=True)
all_func_infos = r()
print(all_func_infos)
r = FunctionCallExtractorFromEntryFuncsPipeline(debug=True)
call_chain_forest = r(all_func_infos)
call_chain_forest.serialize("call_chain_forest.pkl")
call_chain_forest.write2file_call_chains("call_chains.txt")
call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
call_chain_forest.print_call_chains()

# 序列化call chain后，可以按照以下方式deserailize
# call_chain_forest = CallChainForest.deserialize("call_chain_forest.pkl")
# call_chain_forest.write2file_call_chains("call_chains.txt")
# call_chain_forest.write2file_no_code_call_chains("call_chains_no_code.txt")
# call_chain_forest.print_call_chains()


# 以上方法是将所有函数都当作入口函数处理，如果你指定入口函数，按照以下方式，给函数名，文件对于仓库的相对路径，和代码行(0-based)
# r = FunctionCallExtractorFromEntryFuncWithLSPPipeline(debug=True)
# call_chain_tree = r('initWallet', 'parity_wallet_bug_1.sol', 222)
# call_chain_tree.print_nocode_call_chains()