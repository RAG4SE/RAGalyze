"""
Function Call
"""
from ragalyze import *
from ragalyze.prompts import FETCH_CALLEE_DEF_TEMPLATE

repo_path = "/Users/mac/repo/RAGalyzeBench/mybatis-3"

dict_config = load_default_config()

# dict_config.repo.file_filters.extra_excluded_patterns = ["*deps/*", "*test*/*", "workspace/*", "*/tags", "docs/*"]

dict_config.generator.provider = "kimi"
dict_config.generator.model = "kimi-k2-0905-preview"

# retriever的top_k越大，传入大模型的代码片段越多，思考依据更多，但同时消耗更多token
dict_config.rag.retriever.top_k = 25
# bm25 weight越大，rag在搜索中更倾向于稀疏检索而非稠密检索。稀疏检索更依赖于关键词匹配，而稠密检索则更依赖于语义理解。因此，如果问题关于查找而非分析，可以把weight调大，最大至1.
dict_config.rag.retriever.bm25.weight = 0.5
# 按需向量化的前提是用一个合适的检索方式搜索出大量代码片段，然后在此基础上进一步检索，query_driven.top_k指定了第一次检索的量
dict_config.rag.query_driven.top_k = 30
# 排序策略
dict_config.rag.retriever.fusion = "normal_add"

dict_config.rag.recreate_db = True

set_global_configs(dict_config)

'''
void encoding1() {
	try (SqlSession sqlSession = sqlSessionFactory.openSession()) {
		EncodingMapper mapper = sqlSession.getMapper(EncodingMapper.class);
		String answer = mapper.select1();
		assertEquals("Mara\u00f1\u00f3n", answer);
	}
}
'''
bm25_keywords = "[FUNCDEF]encoding1 [FUNCDEF]select1"
faiss_query = ""
# question = FIND_FUNCTION_CALL_TEMPLATE.call(function_name="EVMDialect::builtin")
question = FETCH_CALLEE_DEF_TEMPLATE.call(
    calling_function="encoding1",
    target_name="select1",
#     calling_function_body="""
# void encoding1() {
# 	try (SqlSession sqlSession = sqlSessionFactory.openSession()) {
# 		EncodingMapper mapper = sqlSession.getMapper(EncodingMapper.class);
# 		String answer = mapper.select1();
# 		assertEquals("Mara\u00f1\u00f3n", answer);
# 	}
# }
# """
)

result = query_repository(
    repo_path=repo_path,
    bm25_keywords=bm25_keywords,
    faiss_query=faiss_query,
    question=question,
)

print_result(result)
save_query_results(result, repo_path, bm25_keywords, faiss_query, question)

