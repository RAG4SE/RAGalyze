import deepwiki_cli
from deepwiki_cli.configs import *

def test_deepwiki_cli():
    repo_path = "/home/lyr/test_RAGalyze"
    question = "What is the main function of the project?"
    result = deepwiki_cli.query_repository(repo_path=repo_path, question=question)
    deepwiki_cli.save_query_results(result, repo_path, question)
    deepwiki_cli.print_result(result)


def test_deepwiki_cli_with_config_modification():
    dict_config = deepwiki_cli.load_default_config()
    dict_config.rag.hybrid.enabled = False
    dict_config.rag.embedder.sketch_filling = False
    dict_config.generator.model = "qwen-plus"
    deepwiki_cli.load_all_configs(dict_config)
    print(deepwiki_cli.configs())
    repo_path = "/home/lyr/test_RAGalyze"
    question = "What is the main function of the project?"
    result = deepwiki_cli.query_repository(repo_path=repo_path, question=question)
    deepwiki_cli.save_query_results(result, repo_path, question)
    deepwiki_cli.print_result(result)


if __name__ == "__main__":
    # test_deepwiki_cli()
    test_deepwiki_cli_with_config_modification()
