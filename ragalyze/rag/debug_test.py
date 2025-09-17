#!/usr/bin/env python3
import treesitter_parse

treesitter_parse.set_debug_mode(1)
code = 'struct Point { int x, y; };'
result = treesitter_parse.tokenize_for_bm25(code, 'cpp')
print('Result:', result)