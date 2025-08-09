import sys
from tree_sitter import Parser, Node, Language
import tree_sitter_markdown as tsmarkdown

def print_node(node: Node, source_code: bytes, indent: int = 0) -> None:
    """Recursively print all nodes in the AST with their details"""
    # Get node type and text content
    node_type = node.type
    node_text = source_code[node.start_byte:node.end_byte].decode("utf-8").strip()
    
    # Format output with indentation
    indent_str = "  " * indent
    print(f"{indent_str}[{node_type}]")
    # print(f"{indent_str}  Range: {node.start_byte}-{node.end_byte}")
    # print(f"{indent_str}  Text: {node_text[:50]}...")  # Truncate long text
    # print(f"{indent_str}  ---")
    
    # Recurse into child nodes
    for child in node.children:
        print_node(child, source_code, indent + 1)

def parse_markdown(markdown_text: str) -> None:
    """Parse Markdown text and print the AST nodes"""
    # Initialize parser with Markdown grammar
    language = Language(tsmarkdown.language())
    parser = Parser(language)
    
    # Convert text to bytes (required by Tree-sitter)
    source_bytes = markdown_text.encode("utf-8")
    
    # Parse into AST
    tree = parser.parse(source_bytes)
    root_node = tree.root_node
    
    print("Markdown AST Nodes:")
    print("====================")
    print_node(root_node, source_bytes)

if __name__ == "__main__":
    # Example Markdown content
    sample_markdown = """
# Markdown 全特性示例文档

这是一个包含 **Markdown 所有核心语法** 的示例，用于展示其丰富的格式化能力。

## 1. 文本格式

- 斜体文本：*这是斜体* 或 _这也是斜体_
- 粗体文本：**这是粗体** 或 __这也是粗体__
- 粗斜体：***粗斜体组合*** 或 ___粗斜体组合___
- 删除线：~~这段文字被删除了~~
- 下划线（部分解析器支持）：<u>下划线文本</u>
- 行内代码：`print("Hello, Markdown!")`
- 脚注引用[^1]
- 高亮文本（扩展语法）：==这段文字被高亮==


## 2. 标题层级

# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题


## 3. 列表

### 3.1 无序列表
- 项目 1
- 项目 2
  - 子项目 2.1
  - 子项目 2.2
    - 孙项目 2.2.1

### 3.2 有序列表
1. 第一步
2. 第二步
   1. 子步骤 2.1
   2. 子步骤 2.2
3. 第三步

### 3.3 任务列表（扩展语法）
- [x] 已完成任务
- [ ] 未完成任务
- [ ] 待办事项


## 4. 代码块

### 4.1 单行代码
`def add(a, b): return a + b`

### 4.2 多行代码块（带语言标识）
```python
# 这是 Python 代码
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        print(f"Hello, {self.name}!")

person = Person("Markdown")
person.greet()
    """
    
    # Parse and print nodes
    parse_markdown(sample_markdown)
