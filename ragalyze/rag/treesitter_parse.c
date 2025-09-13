#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "pcre2.h"
// Use local tree-sitter headers
#include "tree_sitter/api.h"

// Global debug flag
static int debug_enabled = 0;

// Debug printf function that only prints when debug mode is enabled
static void debug_printf(const char* format, ...) {
    if (debug_enabled) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

static void debug_print_child(TSNode node, const char* source_code) {
    if (debug_enabled) {
        int child_count = ts_node_child_count(node);
        for (uint32_t i = 0; i < child_count; i++) {
            TSNode child = ts_node_child(node, i);
            const char* child_type = ts_node_type(child);
            uint32_t child_start = ts_node_start_byte(child);
            uint32_t child_end = ts_node_end_byte(child);
            size_t child_length = child_end - child_start;
            char* child_text = malloc(child_length + 1);
            if (child_text) {
                memcpy(child_text, source_code + child_start, child_length);
                child_text[child_length] = '\0';
                
                // Get field name if this child has one
                const char* field_name = ts_node_field_name_for_child(node, i);
                if (field_name) {
                    debug_printf("  Child %u: field='%s', type='%s', text='%s'\n", i, field_name, child_type, child_text);
                } else {
                    debug_printf("  Child %u: type='%s', text='%s'\n", i, child_type, child_text);
                }
                free(child_text);
            }
        }
    }
}

static char* normalize_function_name(const char* func_name) {
    // Check if it's a method call (contains dot or arrow)
    char* method_start = func_name;
    if (strchr(func_name, '.') || strchr(func_name, '>')) {
        // Extract just the method name after the last dot or arrow
        char* last_dot = strrchr(func_name, '.');
        char* last_arrow = strrchr(func_name, '>');
        method_start = (last_arrow > last_dot) ? last_arrow + 1 : (last_dot ? last_dot + 1 : func_name);
    }
    if (strchr(method_start, '(')) {
        // Trim off parameters if present
        char* paren = strchr(method_start, '(');
        *paren = '\0';
    }
    return method_start;
}

// Simple logging function for tree-sitter parser
static void treesitter_log_callback(void *payload, TSLogType type, const char *message) {
    (void)payload; // Unused
    (void)type;    // Unused for now
    printf("[TREESITTER] %s\n", message);
}

const TSLanguage* tree_sitter_cpp(void);
const TSLanguage* tree_sitter_python(void);
const TSLanguage* tree_sitter_c(void);
const TSLanguage* tree_sitter_java(void);
const TSLanguage* tree_sitter_javascript(void);

// Helper to get language parser
static const TSLanguage* get_language(const char* language_name) {
    if (language_name == NULL) {
        return NULL;
    }
    
    // Import language parsers - these would be linked at compile time
    
    // Return the appropriate language parser
    if (strcmp(language_name, "cpp") == 0) {
        debug_printf("Using C++ parser\n");
        return tree_sitter_cpp();
    } else if (strcmp(language_name, "python") == 0) {
        debug_printf("Using Python parser\n");
        return tree_sitter_python();
    } else if (strcmp(language_name, "c") == 0) {
        debug_printf("Using C parser\n");
        return tree_sitter_c();
    } else if (strcmp(language_name, "java") == 0) {
        debug_printf("Using Java parser\n");
        return tree_sitter_java();
    } else if (strcmp(language_name, "javascript") == 0) {
        debug_printf("Using JavaScript parser\n");
        return tree_sitter_javascript();
    }
    
    return NULL;
}

// Structure to represent parsing results
typedef struct {
    char** node_types;
    char** node_texts;
    size_t count;
    size_t capacity;
} ParseResults;

// Structure to represent tokenized results for BM25
typedef struct {
    char** tokens;
    size_t count;
    size_t capacity;
} TokenResults;

// Language parser mapping
typedef struct {
    const char* name;
    const TSLanguage* (*parser_func)(void);
} LanguageMapping;

// Function to initialize parse results
static ParseResults* init_parse_results() {
    ParseResults* results = malloc(sizeof(ParseResults));
    if (!results) {
        return NULL;
    }
    
    results->capacity = 50;
    results->count = 0;
    results->node_types = malloc(results->capacity * sizeof(char*));
    results->node_texts = malloc(results->capacity * sizeof(char*));
    
    if (!results->node_types || !results->node_texts) {
        free(results->node_types);
        free(results->node_texts);
        free(results);
        return NULL;
    }
    
    return results;
}

// Function to add a node to results
static void add_node(ParseResults* results, const char* type, const char* text) {
    if (results->count >= results->capacity) {
        results->capacity *= 2;
        results->node_types = realloc(results->node_types, results->capacity * sizeof(char*));
        results->node_texts = realloc(results->node_texts, results->capacity * sizeof(char*));
    }
    
    results->node_types[results->count] = strdup(type);
    results->node_texts[results->count] = strdup(text);
    results->count++;
}

// Function to free parse results
static void free_parse_results(ParseResults* results) {
    for (size_t i = 0; i < results->count; i++) {
        free(results->node_types[i]);
        free(results->node_texts[i]);
    }
    free(results->node_types);
    free(results->node_texts);
    free(results);
}

// Function to initialize token results
static TokenResults* init_token_results() {
    TokenResults* results = malloc(sizeof(TokenResults));
    results->capacity = 100;
    results->count = 0;
    results->tokens = malloc(results->capacity * sizeof(char*));
    return results;
}

// Function to add a token
static void add_token(TokenResults* results, const char* token) {
    if (results->count >= results->capacity) {
        results->capacity *= 2;
        results->tokens = realloc(results->tokens, results->capacity * sizeof(char*));
    }
    results->tokens[results->count] = strdup(token);
    results->count++;
}

// Function to free token results
static void free_token_results(TokenResults* results) {
    for (size_t i = 0; i < results->count; i++) {
        free(results->tokens[i]);
    }
    free(results->tokens);
    free(results);
}

// Recursive function to traverse the AST
static void traverse_tree(TSNode node, const char* source_code, ParseResults* results) {
    uint32_t child_count = ts_node_child_count(node);
    // Get node type and text
    const char* node_type = ts_node_type(node);
    
    // Debug: print all node types to see what we're getting
    // Get start and end points
    uint32_t start_byte = ts_node_start_byte(node);
    uint32_t end_byte = ts_node_end_byte(node);
    // Extract text
    size_t text_length = end_byte - start_byte;
    char* node_text = malloc(text_length + 1);
    //@haoyang9804: comment it off later
    // Print node text and child count (for debugging)
    if (node_text == NULL) {
        // Handle allocation failure
        exit(1);
    }
    memcpy(node_text, source_code + start_byte, text_length);
    node_text[text_length] = '\0';
    debug_printf("DEBUG: Node type: '%s', NODE TEXT: '%s'\n", node_type, node_text);
    
    if (
        strcmp(node_type, "function_definition") == 0 || 
        strcmp(node_type, "method_definition") == 0 ||
        strcmp(node_type, "function_declarator") == 0 ||
        strcmp(node_type, "method_declaration") == 0 ||
        strcmp(node_type, "function_declaration") == 0) {
        // Extract function name
        debug_printf("DEBUG: Function declaration/definition child nodes:\n");
        debug_print_child(node, source_code);
        if (!ts_node_is_null(ts_node_child_by_field_name(node, "name", 4))) {
            TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
            uint32_t name_start = ts_node_start_byte(name_node);
            uint32_t name_end = ts_node_end_byte(name_node);
            size_t name_length = name_end - name_start;
            char* func_name = malloc(name_length + 1);
            if (func_name == NULL) {
                // Handle allocation failure
                free(node_text);
                exit(1);
            }
            memcpy(func_name, source_code + name_start, name_length);
            func_name[name_length] = '\0';
            debug_printf("DEBUG: Found function definition via name field: '%s'\n", func_name);
            add_node(results, "function_definition", func_name);
            free(func_name);
        }
        else if (!ts_node_is_null(ts_node_child_by_field_name(node, "declarator", 10))) {
            TSNode declarator_node = ts_node_child_by_field_name(node, "declarator", 10);
            if (ts_node_type(declarator_node) == "reference_declarator" ||
                ts_node_type(declarator_node) == "function_declarator") {
            }
            else{
                uint32_t decl_start = ts_node_start_byte(declarator_node);
                uint32_t decl_end = ts_node_end_byte(declarator_node);
                size_t decl_length = decl_end - decl_start;
                char* decl_text = malloc(decl_length + 1);
                if (decl_text == NULL) {
                    // Handle allocation failure
                    free(node_text);
                    exit(1);
                }
                memcpy(decl_text, source_code + decl_start, decl_length);
                decl_text[decl_length] = '\0';
                debug_printf("DEBUG: Found function definition via declarator field: '%s'\n", decl_text);
                add_node(results, "function_definition", decl_text);
                free(decl_text);
                if (ts_node_type(declarator_node) == "qualified_identifier") {
                    return;
                }
            }
        }
        else {
            // Fallback: look for identifier child nodes for C++
            debug_printf("DEBUG: No name field found, looking for identifier children\n");
            int found = 0;
            
            // First, look for function_declarator which contains the function name
            for (uint32_t i = 0; i < child_count; i++) {
                TSNode child = ts_node_child(node, i);
                const char* child_type = ts_node_type(child);
                debug_printf("DEBUG: Child %u type: '%s'\n", i, child_type);
                
                if (strcmp(child_type, "function_declarator") == 0) {
                    // Look inside function_declarator for identifier
                    uint32_t sub_child_count = ts_node_child_count(child);
                    for (uint32_t j = 0; j < sub_child_count; j++) {
                        TSNode sub_child = ts_node_child(child, j);
                        const char* sub_child_type = ts_node_type(sub_child);
                        debug_printf("DEBUG: Sub-child %u type: '%s'\n", j, sub_child_type);
                        
                        if (strcmp(sub_child_type, "identifier") == 0) {
                            uint32_t name_start = ts_node_start_byte(sub_child);
                            uint32_t name_end = ts_node_end_byte(sub_child);
                            size_t name_length = name_end - name_start;
                            char* func_name = malloc(name_length + 1);
                            if (func_name == NULL) {
                                free(node_text);
                                exit(1);
                            }
                            memcpy(func_name, source_code + name_start, name_length);
                            func_name[name_length] = '\0';
                            debug_printf("DEBUG: Found function definition via function_declarator: '%s'\n", func_name);
                            add_node(results, "function_definition", func_name);
                            free(func_name);
                            found = 1;
                            break;
                        }
                    }
                }
                
                if (found) break;
            }
            
            // If not found in function_declarator, try direct identifier (for other languages)
            if (!found) {
                for (uint32_t i = 0; i < child_count; i++) {
                    TSNode child = ts_node_child(node, i);
                    const char* child_type = ts_node_type(child);
                    if (strcmp(child_type, "identifier") == 0) {
                        uint32_t name_start = ts_node_start_byte(child);
                        uint32_t name_end = ts_node_end_byte(child);
                        size_t name_length = name_end - name_start;
                        char* func_name = malloc(name_length + 1);
                        if (func_name == NULL) {
                            free(node_text);
                            exit(1);
                        }
                        memcpy(func_name, source_code + name_start, name_length);
                        func_name[name_length] = '\0';
                        debug_printf("DEBUG: Found function definition via direct identifier: '%s'\n", func_name);
                        add_node(results, "function_definition", func_name);
                        free(func_name);
                        break;
                    }
                }
            }
        }
    }
    else if (strcmp(node_type, "class_definition") == 0) {
        // Extract class name
        TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
        if (!ts_node_is_null(name_node)) {
            uint32_t name_start = ts_node_start_byte(name_node);
            uint32_t name_end = ts_node_end_byte(name_node);
            size_t name_length = name_end - name_start;
            char* class_name = malloc(name_length + 1);
            if (class_name == NULL) {
                // Handle allocation failure
                free(node_text);
                exit(1);
            }
            memcpy(class_name, source_code + name_start, name_length);
            class_name[name_length] = '\0';
            add_node(results, "class_definition", class_name);
            free(class_name);
        }
    }
    else if (strcmp(node_type, "call") == 0 || strcmp(node_type, "function_call") == 0 || strcmp(node_type, "call_expression") == 0 || strcmp(node_type, "method_invocation") == 0) {
        // Extract function call name
        
        // Debug: print all child nodes' names
        debug_printf("DEBUG: Call expression child nodes:\n");
        debug_print_child(node, source_code);
        

        if (!ts_node_is_null(ts_node_child_by_field_name(node, "function", 8))) {
            TSNode function_node = ts_node_child_by_field_name(node, "function", 8);
            debug_printf("DEBUG: Call expression's function node's child nodes:\n");
            debug_print_child(function_node, source_code);
            uint32_t func_start = ts_node_start_byte(function_node);
            uint32_t func_end = ts_node_end_byte(function_node);
            size_t func_length = func_end - func_start;
            char* func_name = malloc(func_length + 1);
            if (func_name == NULL) {
                // Handle allocation failure
                free(node_text);
                exit(1);
            }
            memcpy(func_name, source_code + func_start, func_length);
            func_name[func_length] = '\0';
            debug_printf("DEBUG: Extracted function name: '%s'\n", func_name);
            
            // For field expressions like s->add, we need to extract just the field name
            if (strcmp(ts_node_type(function_node), "field_expression") == 0) {
                TSNode field_node = ts_node_child_by_field_name(function_node, "field", 5);
                if (!ts_node_is_null(field_node)) {
                    uint32_t field_start = ts_node_start_byte(field_node);
                    uint32_t field_end = ts_node_end_byte(field_node);
                    size_t field_length = field_end - field_start;
                    char* field_name = malloc(field_length + 1);
                    if (field_name == NULL) {
                        free(func_name);
                        free(node_text);
                        exit(1);
                    }
                    memcpy(field_name, source_code + field_start, field_length);
                    field_name[field_length] = '\0';
                    debug_printf("DEBUG: Extracted field name: '%s'\n", field_name);
                    add_node(results, "call", field_name);
                    free(field_name);
                    free(func_name);
                    // TSNode argument_node = ts_node_child_by_field_name(function_node, "argument", 8);
                    // if (!ts_node_is_null(argument_node)) {
                    //     // Traverse arguments to find nested calls
                    //     traverse_tree(argument_node, source_code, results);
                    // }
                }
            }
            else if (strcmp(ts_node_type(function_node), "identifier") == 0) {
                add_node(results, "call", func_name);
            }
            else if (strcmp(ts_node_type(function_node), "attribute") == 0) {
                // For attribute nodes, extract the attribute name
                TSNode attr_node = ts_node_child_by_field_name(function_node, "attribute", 9);
                if (!ts_node_is_null(attr_node)) {
                    uint32_t attr_start = ts_node_start_byte(attr_node);
                    uint32_t attr_end = ts_node_end_byte(attr_node);
                    size_t attr_length = attr_end - attr_start;
                    char* attr_name = malloc(attr_length + 1);
                    if (attr_name == NULL) {
                        free(func_name);
                        free(node_text);
                        exit(1);
                    }
                    memcpy(attr_name, source_code + attr_start, attr_length);
                    attr_name[attr_length] = '\0';
                    debug_printf("DEBUG: Extracted attribute name: '%s'\n", attr_name);
                    add_node(results, "call", attr_name);
                    free(attr_name);
                    TSNode objects_node = ts_node_child_by_field_name(function_node, "object", 6);
                    if (!ts_node_is_null(objects_node)) {
                        // Traverse object to find nested calls
                        uint32_t obj_start = ts_node_start_byte(objects_node);
                        uint32_t obj_end = ts_node_end_byte(objects_node);
                        size_t obj_length = obj_end - obj_start;
                        char* obj_name = malloc(obj_length + 1);
                        if (obj_name == NULL) {
                            free(func_name);
                            free(node_text);
                            exit(1);
                        }
                        memcpy(obj_name, source_code + obj_start, obj_length);
                        obj_name[obj_length] = '\0';
                        if (strchr(obj_name, '(')) {
                            obj_name = normalize_function_name(obj_name);
                            debug_printf("DEBUG: Extracted object name: '%s'\n", obj_name);
                            add_node(results, "identifier", obj_name);
                            free(func_name);
                        }
                        else {
                            free(obj_name);
                            free(func_name);
                            traverse_tree(objects_node, source_code, results);
                        }
                    }
                }
            }
            else if (strcmp(ts_node_type(function_node), "member_expression") == 0) {
                // For member expressions, extract the member name
                TSNode property_node = ts_node_child_by_field_name(function_node, "property", 8);
                if (!ts_node_is_null(property_node)) {
                    uint32_t prop_start = ts_node_start_byte(property_node);
                    uint32_t prop_end = ts_node_end_byte(property_node);
                    size_t prop_length = prop_end - prop_start;
                    char* prop_name = malloc(prop_length + 1);
                    if (prop_name == NULL) {
                        free(func_name);
                        free(node_text);
                        exit(1);
                    }
                    memcpy(prop_name, source_code + prop_start, prop_length);
                    prop_name[prop_length] = '\0';
                    debug_printf("DEBUG: Extracted member name: '%s'\n", prop_name);
                    add_node(results, "call", prop_name);
                    free(prop_name);
                    // TSNode object_node = ts_node_child_by_field_name(function_node, "object", 6);
                    // if (!ts_node_is_null(object_node)) {
                    //     // Traverse object to find nested calls
                    //     traverse_tree(object_node, source_code, results);
                    // }
                }
            }
            else {
                // Fallback: just add the whole function text
                add_node(results, "call", func_name);
            }

        } 
        else if (!ts_node_is_null(ts_node_child_by_field_name(node, "name", 4))) {
            TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
            uint32_t name_start = ts_node_start_byte(name_node);
            uint32_t name_end = ts_node_end_byte(name_node);
            size_t name_length = name_end - name_start;
            char* func_name = malloc(name_length + 1);
            if (func_name == NULL) {
                // Handle allocation failure
                free(node_text);
                exit(1);
            }
            memcpy(func_name, source_code + name_start, name_length);
            func_name[name_length] = '\0';
            debug_printf("DEBUG: Found call via name field: '%s'\n", func_name);
            add_node(results, "call", func_name);
            free(func_name);
        }
        else {
            debug_enabled = 1;
            debug_print_child(node, source_code);
            exit(1);
        }
    }
    else if (strcmp(node_type, "qualified_identifier") == 0 && text_length > 0) {
        // Handle qualified identifiers like CLS::operator()
        if (!ts_node_is_null(ts_node_child_by_field_name(node, "name", 4))) {
            TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
            uint32_t name_start = ts_node_start_byte(name_node);
            uint32_t name_end = ts_node_end_byte(name_node);
            size_t name_length = name_end - name_start;
            char* qual_name = malloc(name_length + 1);
            if (qual_name == NULL) {
                // Handle allocation failure
                free(node_text);
                exit(1);
            }
            memcpy(qual_name, source_code + name_start, name_length);
            qual_name[name_length] = '\0';
            debug_printf("DEBUG: Found qualified identifier via name field: '%s'\n", qual_name);
            if (strcmp(qual_name, "operator()") == 0) {
                // Skip generic operator names
                add_node(results, "call", qual_name);
                TSNode scope_node = ts_node_child_by_field_name(node, "scope", 5);
                if (!ts_node_is_null(scope_node)) {
                    uint32_t scope_start = ts_node_start_byte(scope_node);
                    uint32_t scope_end = ts_node_end_byte(scope_node);
                    size_t scope_length = scope_end - scope_start;
                    char* scope_name = malloc(scope_length + 1);
                    if (scope_name == NULL) {
                        free(qual_name);
                    }
                    memcpy(scope_name, source_code + scope_start, scope_length);
                    scope_name[scope_length] = '\0';
                    debug_printf("DEBUG: Found scope name: '%s'\n", scope_name);

                    add_node(results, "call", node_text);
                }
            }
            free(qual_name);
        }
    }
    else if (strcmp(node_type, "identifier") == 0 && text_length > 0) {
        // Add meaningful identifiers
        add_node(results, "identifier", node_text);
    }
    else if (strcmp(node_type, "string") == 0 && text_length > 0) {
        add_node(results, "string_literal", node_text);
    }
    else if (strcmp(node_type, "number") == 0 && text_length > 0) {
        add_node(results, "number", node_text);
    }
    else if (strcmp(node_type, "return_statement") == 0) {
        add_node(results, "keyword", "return");
    }

    free(node_text);
    // Recursively traverse children
    for (uint32_t i = 0; i < child_count; i++) {
        TSNode child = ts_node_child(node, i);
        traverse_tree(child, source_code, results);
    }
}

// Fallback parser equivalent to tokens = re.findall(r"\b\w+\b", text) using PCRE2
static ParseResults* fallback_parse(const char* code, const char* language_name) {
    ParseResults* results = init_parse_results();
    
    // PCRE2 pattern equivalent to Python's \b\w+\b
    // Using word boundary and word character with proper options
    const char* pattern = "\\b[a-zA-Z0-9_]+\\b";
    PCRE2_SPTR subject = (PCRE2_SPTR)code;
    PCRE2_SPTR pattern_ptr = (PCRE2_SPTR)pattern;
    
    int errorcode;
    PCRE2_SIZE erroroffset;
    pcre2_code *re = pcre2_compile(
        pattern_ptr,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UCP | PCRE2_UTF | PCRE2_ALT_BSUX,
        &errorcode,
        &erroroffset,
        NULL
    );
    
    if (!re) {
        // Get error message and raise error
        PCRE2_UCHAR error_message[256];
        pcre2_get_error_message(errorcode, error_message, sizeof(error_message));
        fprintf(stderr, "PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, error_message);
        exit(1);
    }
    
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);
    if (!match_data) {
        pcre2_code_free(re);
        fprintf(stderr, "Failed to create PCRE2 match data\n");
        exit(1);
    }
    
    int rc;
    size_t subject_len = strlen(code);
    size_t start_offset = 0;
    
    while (start_offset < subject_len &&
           (rc = pcre2_match(
               re,
               subject,
               subject_len,
               start_offset,
               0,
               match_data,
               NULL
           )) >= 0) {
        
        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
        
        if (rc > 0) {
            for (int i = 0; i < rc; i++) {
                PCRE2_SIZE start = ovector[2*i];
                PCRE2_SIZE end = ovector[2*i+1];
                
                if (start != PCRE2_UNSET && end != PCRE2_UNSET) {
                    size_t len = end - start;
                    char* word = malloc(len + 1);
                    if (word) {
                        memcpy(word, code + start, len);
                        word[len] = '\0';
                        add_node(results, "regex_node", word);
                        free(word);
                    }
                }
            }
        }
        
        start_offset = ovector[1];
    }
    
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
    
    return results;
}

// Tree-sitter based parsing function
static ParseResults* parse_with_treesitter(const char* code, const char* language_name) {
    ParseResults* results = init_parse_results();
    
    // Get the appropriate language parser
    const TSLanguage* language = get_language(language_name);
    if (!language) {
        // Fall back to basic parsing when no language parser is available
        free_parse_results(results);
        return fallback_parse(code, language_name);
    }

    // Create parser
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, language);

    // Enable logging for tree-sitter parser
    TSLogger logger = {
        .payload = NULL,
        .log = treesitter_log_callback
    };
    // ts_parser_set_logger(parser, logger);

    // Set a timeout to prevent infinite loops (10 seconds)
    ts_parser_set_timeout_micros(parser, 10000000); // 10 seconds in microseconds
    
    // Parse the code
    TSTree* tree = ts_parser_parse_string(parser, NULL, code, strlen(code));
    
    // Check if parsing failed or timed out
    if (!tree) {
        // Fall back to basic parsing when tree-sitter fails
        free_parse_results(results);
        ts_parser_delete(parser);
        return fallback_parse(code, language_name);
    }
    
    TSNode root_node = ts_tree_root_node(tree);
    
    // Check if the root node is null (parsing failed)
    if (ts_node_is_null(root_node)) {
        free_parse_results(results);
        ts_tree_delete(tree);
        ts_parser_delete(parser);
        return fallback_parse(code, language_name);
    }
    
    // Traverse the tree and collect nodes
    traverse_tree(root_node, code, results);

    // Clean up
    ts_tree_delete(tree);
    ts_parser_delete(parser);
    
    return results;
}

// Helper to check if a name is in a function names list
static int is_function_name(const char* name, char** function_names, size_t function_count) {
    for (size_t i = 0; i < function_count; i++) {
        if (strcmp(function_names[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

// Helper to extract just the method name from qualified calls
static char* extract_method_name(const char* qualified_name) {
    const char* last_dot = strrchr(qualified_name, '.');
    if (last_dot) {
        char* result = strdup(last_dot + 1);
        if (result == NULL) {
            // Handle allocation failure
            return NULL;
        }
        return result;
    } else {
        char* result = strdup(qualified_name);
        if (result == NULL) {
            // Handle allocation failure
            return NULL;
        }
        return result;
    }
}

// Smart BM25 tokenization using tree-sitter AST
static TokenResults* extract_bm25_tokens_treesitter(const char* code, const char* language) {
    TokenResults* tokens = init_token_results();
    
    // Parse the code using tree-sitter to get comprehensive AST nodes
    ParseResults* ast_nodes = parse_with_treesitter(code, language);
    
    // Track function and class names we've seen to avoid duplicates
    size_t max_names = ast_nodes->count;  // Initial capacity based on AST node count
    char** function_names = (char**)malloc(max_names * sizeof(char*));
    char** class_names = (char**)malloc(max_names * sizeof(char*));
    char** regex_node_names = (char**)malloc(max_names * sizeof(char*));
    
    if (!function_names || !class_names || !regex_node_names) {
        free(function_names);
        free(class_names);
        free(regex_node_names);
        free_parse_results(ast_nodes);
        return tokens;
    }
    
    size_t function_count = 0;
    size_t class_count = 0;
    size_t regex_node_count = 0;

    // First pass: collect all function and class definitions
    debug_printf("DEBUG: Processing %zu AST nodes\n", ast_nodes->count);
    for (size_t i = 0; i < ast_nodes->count; i++) {
        debug_printf("DEBUG: AST node %zu: type='%s', text='%s'\n", i, ast_nodes->node_types[i], ast_nodes->node_texts[i]);
        
        if (strcmp(ast_nodes->node_types[i], "function_definition") == 0) {
            function_names[function_count] = strdup(ast_nodes->node_texts[i]);
            if (!function_names[function_count]) {
                // Allocation failed, clean up and return
                for (size_t j = 0; j < function_count; j++) free(function_names[j]);
                for (size_t j = 0; j < class_count; j++) free(class_names[j]);
                for (size_t j = 0; j < regex_node_count; j++) free(regex_node_names[j]);
                free(function_names);
                free(class_names);
                free(regex_node_names);
                free_parse_results(ast_nodes);
                return tokens;
            }
            function_count++;
            debug_printf("DEBUG: Found function definition: '%s'\n", ast_nodes->node_texts[i]);
        }
        else if (strcmp(ast_nodes->node_types[i], "class_definition") == 0) {
            class_names[class_count] = strdup(ast_nodes->node_texts[i]);
            if (!class_names[class_count]) {
                // Allocation failed, clean up and return
                for (size_t j = 0; j < function_count; j++) free(function_names[j]);
                for (size_t j = 0; j < class_count; j++) free(class_names[j]);
                for (size_t j = 0; j < regex_node_count; j++) free(regex_node_names[j]);
                free(function_names);
                free(class_names);
                free(regex_node_names);
                free_parse_results(ast_nodes);
                return tokens;
            }
            class_count++;
        }
        else if (strcmp(ast_nodes->node_types[i], "regex_node") == 0) {
            regex_node_names[regex_node_count] = strdup(ast_nodes->node_texts[i]);
            if (!regex_node_names[regex_node_count]) {
                // Allocation failed, clean up and return
                for (size_t j = 0; j < function_count; j++) free(function_names[j]);
                for (size_t j = 0; j < class_count; j++) free(class_names[j]);
                for (size_t j = 0; j < regex_node_count; j++) free(regex_node_names[j]);
                free(function_names);
                free(class_names);
                free(regex_node_names);
                free_parse_results(ast_nodes);
                return tokens;
            }
            regex_node_count++;
        }
    }
    
    // Second pass: include function/class definitions and calls
    for (size_t i = 0; i < ast_nodes->count; i++) {
        const char* node_type = ast_nodes->node_types[i];
        const char* node_text = ast_nodes->node_texts[i];
        
        // Function definitions - always include with prefix
        if (strcmp(node_type, "function_definition") == 0) {
            char prefixed_token[256];
            snprintf(prefixed_token, sizeof(prefixed_token), "[FUNC]%s", node_text);
            add_token(tokens, prefixed_token);
            continue;
        }
        
        // Class definitions - always include with prefix
        if (strcmp(node_type, "class_definition") == 0) {
            char prefixed_token[256];
            snprintf(prefixed_token, sizeof(prefixed_token), "[CLASS]%s", node_text);
            add_token(tokens, prefixed_token);
            continue;
        }
        
        // Function calls - include with prefix, but skip if it's a function/class we already defined
        if (strcmp(node_type, "call") == 0) {
            if (!is_function_name(node_text, function_names, function_count) && 
                !is_function_name(node_text, class_names, class_count)) {
                char prefixed_token[256];
                snprintf(prefixed_token, sizeof(prefixed_token), "[CALL]%s", node_text);
                add_token(tokens, prefixed_token);
            }
            continue;
        }

        
        if (strcmp(node_type, "regex_node") == 0) {
            char prefixed_token[256];
            snprintf(prefixed_token, sizeof(prefixed_token), "%s", node_text);
            add_token(tokens, prefixed_token);
            continue;
        }
    }
    
    // Clean up and return
    for (size_t i = 0; i < function_count; i++) free(function_names[i]);
    for (size_t i = 0; i < class_count; i++) free(class_names[i]);
    for (size_t i = 0; i < regex_node_count; i++) free(regex_node_names[i]);
    free(function_names);
    free(class_names);
    free(regex_node_names);
    free_parse_results(ast_nodes);
    return tokens;
}

// Main parsing function - exposed to Python
static PyObject* parse_code_with_treesitter(PyObject* self, PyObject* args) {
    const char* code;
    const char* language_name = NULL;
    
    if (!PyArg_ParseTuple(args, "s|z", &code, &language_name)) {
        return NULL;
    }
    
    if (!language_name) {
        printf("ERROR: Language name is required for parsing\n");
        exit(1);
    }
    
    // Parse the code using tree-sitter
    ParseResults* results = parse_with_treesitter(code, language_name);
    
    // Convert results to Python list of tuples
    PyObject* py_results = PyList_New(results->count);
    for (size_t i = 0; i < results->count; i++) {
        PyObject* tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyUnicode_FromString(results->node_types[i]));
        PyTuple_SetItem(tuple, 1, PyUnicode_FromString(results->node_texts[i]));
        PyList_SetItem(py_results, i, tuple);
    }
    
    // Cleanup
    free_parse_results(results);
    
    return py_results;
}

// BM25 tokenization function - exposed to Python
static PyObject* tokenize_for_bm25(PyObject* self, PyObject* args) {
    const char* code;
    const char* language_name = NULL;
    
    if (!PyArg_ParseTuple(args, "s|z", &code, &language_name)) {
        return NULL;
    }
    if (!language_name) {
        printf("ERROR: Language name is required for BM25 tokenization\n");
        exit(1);
    }
    
    // Extract BM25 tokens using tree-sitter
    TokenResults* results = extract_bm25_tokens_treesitter(code, language_name);
    // Convert results to Python list
    PyObject* py_results = PyList_New(results->count);
    for (size_t i = 0; i < results->count; i++) {
        PyList_SetItem(py_results, i, PyUnicode_FromString(results->tokens[i]));
    }
    
    // Cleanup
    free_token_results(results);
    
    return py_results;
}

// Method definition table
// Function to enable/disable debug mode
static PyObject* set_debug_mode(PyObject* self, PyObject* args) {
    int enabled;
    if (!PyArg_ParseTuple(args, "i", &enabled)) {
        return NULL;
    }
    debug_enabled = enabled;
    Py_RETURN_NONE;
}

// Function to get current debug mode
static PyObject* get_debug_mode(PyObject* self, PyObject* args) {
    return PyLong_FromLong(debug_enabled);
}

// Python wrapper for fallback_parse function
static PyObject* fallback_parse_py(PyObject* self, PyObject* args) {
    const char* code;
    const char* language;
    
    if (!PyArg_ParseTuple(args, "ss", &code, &language)) {
        return NULL;
    }
    
    // Call the C fallback_parse function
    ParseResults* results = fallback_parse(code, language);
    if (!results) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to parse code");
        return NULL;
    }
    
    // Convert results to Python list of dictionaries
    PyObject* py_results = PyList_New(0);
    if (!py_results) {
        free_parse_results(results);
        return NULL;
    }
    
    for (size_t i = 0; i < results->count; i++) {
        PyObject* node_dict = PyDict_New();
        if (!node_dict) {
            Py_DECREF(py_results);
            free_parse_results(results);
            return NULL;
        }
        
        // Add type and name to dictionary
        PyObject* type_str = PyUnicode_FromString(results->node_types[i]);
        PyObject* name_str = PyUnicode_FromString(results->node_texts[i]);
        
        if (!type_str || !name_str) {
            Py_XDECREF(type_str);
            Py_XDECREF(name_str);
            Py_DECREF(node_dict);
            Py_DECREF(py_results);
            free_parse_results(results);
            return NULL;
        }
        
        PyDict_SetItemString(node_dict, "type", type_str);
        PyDict_SetItemString(node_dict, "name", name_str);
        
        Py_DECREF(type_str);
        Py_DECREF(name_str);
        
        // Add node to results list
        if (PyList_Append(py_results, node_dict) != 0) {
            Py_DECREF(node_dict);
            Py_DECREF(py_results);
            free_parse_results(results);
            return NULL;
        }
        
        Py_DECREF(node_dict);
    }
    
    free_parse_results(results);
    return py_results;
}

static PyMethodDef TreeSitterMethods[] = {
    {"parse_code_with_treesitter", parse_code_with_treesitter, METH_VARARGS, 
     "Parse code and return AST nodes as (type, text) tuples"},
    {"tokenize_for_bm25", tokenize_for_bm25, METH_VARARGS,
     "Tokenize code for BM25 search with [FUNC] and [CALL] prefixes"},
    {"fallback_parse", fallback_parse_py, METH_VARARGS,
     "Fallback parser equivalent to re.findall(r'\\b\\w+\\b', text)"},
    {"set_debug_mode", set_debug_mode, METH_VARARGS,
     "Enable or disable debug mode (1=enabled, 0=disabled)"},
    {"get_debug_mode", get_debug_mode, METH_NOARGS,
     "Get current debug mode status"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef treesittermodule = {
    PyModuleDef_HEAD_INIT,
    "treesitter_parse",  // Module name
    "Enhanced code parser with BM25 tokenization",  // Module description
    -1,  // Module keeps state in global variables
    TreeSitterMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_treesitter_parse(void) {
    return PyModule_Create(&treesittermodule);
}