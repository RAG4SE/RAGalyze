import ragalyze
from ragalyze.configs import *


def test_deepwiki_cli():
    repo_path = "/home/lyr/test_RAGalyze"
    question = "What is the main function of the project?"
    result = ragalyze.query_repository(repo_path=repo_path, question=question)
    ragalyze.save_query_results(result, repo_path, question)
    ragalyze.print_result(result)


def test_deepwiki_cli_with_config_modification():
    dict_config = ragalyze.load_default_config()
    dict_config.rag.hybrid.enabled = False
    dict_config.rag.embedder.sketch_filling = False
    dict_config.generator.model = "qwen-plus"
    ragalyze.set_global_configs(dict_config)
    print(ragalyze.configs())
    repo_path = "/home/lyr/test_RAGalyze"
    question = "What is the main function of the project?"
    result = ragalyze.query_repository(repo_path=repo_path, question=question)
    ragalyze.save_query_results(result, repo_path, question)
    ragalyze.print_result(result)


if __name__ == "__main__":
    # test_deepwiki_cli()
    test_deepwiki_cli_with_config_modification()
