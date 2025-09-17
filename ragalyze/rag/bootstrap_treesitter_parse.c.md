# Booststrap treesitter_parse.c

This file offers instruction of how to boostrap `treesitter_parse.c`
`treesitter_parse.c`'s `traverse_tree` function tokenizes relevant tokens from python/c/cpp/java codes and prefix them for better recall during bm25 search.

Currently, all test cases in `test_codes_bm25_funcs` in `treesitter_parse_interface.py` are passed.
So you need to 
1. add new test cases for cpp
2. run `treesitter_parse_interface.py` to check if the test cases are passed
3. if some test cases fail, modify `traverse_tree` and `extract_bm25_tokens_treesitter` to make them pass.
4. go to step 1 to continue this process until you think all commonly-used language features are tested.

To help you understand `treesitter_parse.c`, read `extract_bm25_tokens_treesitter` to know what tokens we want to extract from codes and how we want to prefix them

You can run `python setup.py build_ext --inplace` to build `treesitter_parse.c` 


For any modification in `treesitter_parse.c`, use `======================= AGENT START =======================` and `======================= AGENT END =======================` to surround your provided code, just like
```
// ======================= AGENT START =======================
// Check for field initializer list (common in constructors) and skip it
uint32_t child_count = ts_node_child_count(node);
debug_printf("DEBUG: Function definition has %d children\n", child_count);
for (uint32_t i = 0; i < child_count; i++) {
    TSNode child = ts_node_child(node, i);
    const char* child_type = ts_node_type(child);
    debug_printf("DEBUG: Child %d type: '%s'\n", i, child_type);
    if (strcmp(child_type, "field_initializer_list") == 0) {
        debug_printf("DEBUG: Found field initializer list, skipping it to avoid duplicate field identifiers\n");
        debug_print_child(child, source_code);
        // Skip this child - don't traverse field initializer lists
        continue;
    }
}
// ======================= AGENT END =======================

```