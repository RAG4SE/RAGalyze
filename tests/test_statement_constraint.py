#!/usr/bin/env python3
"""
测试SmartTextSplitter的statement约束功能
"""

import sys
sys.path.append('/home/lyr/RAGalyze')

from rag.splitter_factory import SplitterFactory

def test_statement_constraint():
    """测试statement约束是否正确工作"""
    
    # 创建一个包含括号和其他非statement元素的Python代码
    python_code = '''
def calculate(x, y):
    result = (x + y) * 2
    if result > 10:
        print("Large result:", result)
    return result

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b != 0:
            return a / b
        return None
'''
    
    print("测试Python代码:")
    print(python_code)
    print("=" * 50)
    
    # 创建splitter
    factory = SplitterFactory()
    splitter = factory.get_splitter(file_path='test.py', force_type='code')
    
    # 设置很小的chunk_size来强制分割
    splitter.chunk_size = 50  # 非常小的chunk size
    splitter.chunk_overlap = 10
    
    # 分割代码
    chunks = splitter.split_text(python_code)
    
    print(f"生成了 {len(chunks)} 个chunks:")
    print("=" * 50)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        # Handle both string and Document objects
        chunk_text = chunk.text if hasattr(chunk, 'text') else chunk
        print(repr(chunk_text))
        print(f"长度: {len(chunk_text)}")
        print("-" * 30)
    
    # 检查是否有任何chunk以单个符号开始或结束
    problematic_chunks = []
    for i, chunk in enumerate(chunks):
        text = (chunk.text if hasattr(chunk, 'text') else chunk).strip()
        if text and (text[0] in '()[]{}' or text[-1] in '()[]{}'):
            problematic_chunks.append((i, text))
    
    if problematic_chunks:
        print("发现问题chunks (以单个符号开始或结束):")
        for i, text in problematic_chunks:
            print(f"Chunk {i+1}: {repr(text[:50])}...")
        return False
    else:
        print("✓ 所有chunks都以完整的语句开始和结束")
        return True

def test_direct_tree_sitter():
    """直接测试tree-sitter的statement检测功能"""
    
    print("\n直接测试tree-sitter statement检测:")
    print("=" * 50)
    
    # 创建splitter来访问内部方法
    factory = SplitterFactory()
    splitter = factory.get_splitter(file_path='test.py', force_type='code')
    
    # 测试代码
    test_code = b'''
def test():
    x = (1 + 2)
    return x
'''
    
    try:
        parser = splitter.parsers['python']
        tree = parser.parse(test_code)
        
        def print_nodes(node, depth=0):
            indent = "  " * depth
            is_statement = splitter._is_statement_node(node, 'python')
            marker = "[STATEMENT]" if is_statement else ""
            print(f"{indent}{node.type} {marker} - bytes {node.start_byte}-{node.end_byte}")
            
            for child in node.children:
                print_nodes(child, depth + 1)
        
        print("AST结构:")
        print_nodes(tree.root_node)
        
        # 测试边界查找
        print("\n测试边界查找:")
        boundary = splitter._find_code_boundary_with_treesitter(test_code, 0, 30, 'python')
        print(f"在位置0-30范围内找到的边界: {boundary}")
        print(f"边界处的文本: {repr(test_code[:boundary].decode())}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试statement约束功能...")
    
    success = True
    
    try:
        success &= test_statement_constraint()
        success &= test_direct_tree_sitter()
        
        if success:
            print("\n✓ 所有测试通过！statement约束功能正常工作。")
        else:
            print("\n✗ 部分测试失败。")
            
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)