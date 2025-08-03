import tiktoken

def count_tokens(file_path):
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer
    
    try:
        # Read file contents
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Calculate tokens
        tokens = enc.encode(content)
        token_count = len(tokens)
        
        print(f"Number of tokens in {file_path}: {token_count}")
        return token_count
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

# Example usage
file_path = "/home/lyr/test_RAGalyze/analyze_repo.py"  # Replace with your file path
count_tokens(file_path)
file_path = "/home/lyr/test_RAGalyze/README.md"
count_tokens(file_path)
