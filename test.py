import tree_sitter_python as tspython
from tree_sitter import Language, Parser

parser = Parser(Language(tspython.language()))
with open('/home/lyr/test_RAGalyze/1.py', 'r') as file:
    content = file.read()
    tree = parser.parse(bytes(content, 'utf8'))

node = tree.root_node

def traverse(node):
    print(node.type, node.start_point, node.end_point, node.start_byte, node.end_byte, node.text)
    for child in node.children:
        traverse(child)

traverse(node)
