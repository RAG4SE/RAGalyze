#define PCRE2_CODE_UNIT_WIDTH 8
#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <pcre2.h>

// Pre-compiled regex patterns to avoid recompilation
typedef struct {
    pcre2_code** function_definition_patterns;
    pcre2_code** class_definition_patterns;
    pcre2_code* obj_full_calls_pattern;
    pcre2_code* ptr_full_calls_pattern;
    pcre2_code* obj_method_calls_pattern;
    pcre2_code* ptr_method_calls_pattern;
    pcre2_code* class_method_calls_pattern;
    pcre2_code* all_func_calls_pattern;
    pcre2_code* double_quote_pattern;
    pcre2_code* single_quote_pattern;
    pcre2_code* identifiers_pattern;
    size_t function_def_count;
    size_t class_def_count;
} PrecompiledPatterns;

// Global pre-compiled patterns (initialized once)
static PrecompiledPatterns* g_patterns = NULL;
static pcre2_general_context* g_general_context = NULL;
static pcre2_compile_context* g_compile_context = NULL;

// Keyword sets
static const char* cpp_keywords[] = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
    "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept",
    "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local",
    "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", NULL
};

static const char* java_keywords[] = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "_", "true", "false",
    "null", "var", "record", "sealed", "non-sealed", "permits", "module", "open", "requires",
    "exports", "opens", "uses", "provides", "to", "with", "transitive", NULL
};

static const char* python_keywords[] = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield", "match", "case", "type", NULL
};

static const char* js_ts_keywords[] = {
    "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete",
    "do", "else", "export", "extends", "false", "finally", "for", "function", "if", "import",
    "in", "instanceof", "new", "null", "return", "super", "switch", "this", "throw", "true",
    "try", "typeof", "var", "void", "while", "with", "let", "static", "yield", "await", "enum",
    "implements", "interface", "package", "private", "protected", "public", "arguments", "as",
    "async", "eval", "from", "get", "of", "set", "type", "declare", "namespace", "module",
    "abstract", "any", "boolean", "constructor", "symbol", "readonly", "keyof", "infer", "is",
    "asserts", "global", "bigint", "object", "number", "string", "undefined", "unknown",
    "never", "override", "intrinsic", NULL
};

// Simple hash set implementation for keywords
#define HASH_SIZE 1024
typedef struct KeywordNode {
    const char* keyword;
    struct KeywordNode* next;
} KeywordNode;

static KeywordNode* keyword_hash_table[HASH_SIZE];

// Hash function for strings
static unsigned int hash_string(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    return hash % HASH_SIZE;
}

// Add keyword to hash table
static void add_keyword(const char* keyword) {
    unsigned int index = hash_string(keyword);
    KeywordNode* node = (KeywordNode*)malloc(sizeof(KeywordNode));
    node->keyword = keyword;
    node->next = keyword_hash_table[index];
    keyword_hash_table[index] = node;
}

// Check if keyword exists in hash table
static int is_keyword(const char* keyword) {
    unsigned int index = hash_string(keyword);
    KeywordNode* node = keyword_hash_table[index];
    while (node) {
        if (strcmp(node->keyword, keyword) == 0) {
            return 1;
        }
        node = node->next;
    }
    return 0;
}

// Initialize keyword hash table
static void initialize_keywords() {
    // Clear existing hash table
    for (int i = 0; i < HASH_SIZE; i++) {
        KeywordNode* node = keyword_hash_table[i];
        while (node) {
            KeywordNode* next = node->next;
            free(node);
            node = next;
        }
        keyword_hash_table[i] = NULL;
    }
    
    // Add all keywords
    for (int i = 0; cpp_keywords[i]; i++) {
        add_keyword(cpp_keywords[i]);
    }
    for (int i = 0; java_keywords[i]; i++) {
        add_keyword(java_keywords[i]);
    }
    for (int i = 0; python_keywords[i]; i++) {
        add_keyword(python_keywords[i]);
    }
    for (int i = 0; js_ts_keywords[i]; i++) {
        add_keyword(js_ts_keywords[i]);
    }
}

// Compile a regex pattern
static pcre2_code* compile_pattern(const char* pattern) {
    int errorcode;
    PCRE2_SIZE erroroffset;
    pcre2_code* re = pcre2_compile(
        (PCRE2_SPTR)pattern, 
        PCRE2_ZERO_TERMINATED, 
        0, 
        &errorcode, 
        &erroroffset, 
        g_compile_context
    );
    return re;
}

// Initialize pre-compiled patterns
static PrecompiledPatterns* initialize_patterns() {
    PrecompiledPatterns* patterns = (PrecompiledPatterns*)calloc(1, sizeof(PrecompiledPatterns));
    
    // Initialize PCRE2 contexts
    if (!g_general_context) {
        g_general_context = pcre2_general_context_create(NULL, NULL, NULL);
    }
    if (!g_compile_context) {
        g_compile_context = pcre2_compile_context_create(g_general_context);
    }
    
    // Function definition patterns
    const char* function_def_patterns[] = {
        "def\\s+(\\w+)",
        "function\\s+(\\w+)",
        // Pattern for operator() methods - must come before general patterns
        "(?:public|private|protected)?\\s*(?:static)?\\s*((?:\\w+::)*operator\\s*\\([^)]*\\))\\s*(?:const)?\\s*\\{",
        // Pattern for regular C++/Java methods with return types
        "(?:public|private|protected)?\\s*(?:static)?\\s*(?:[\\w&*<>:\\-]+\\s+)*(\\w+)\\s*\\([^)]*\\)(?:\\s*const)?\\s*\\{",
        // Pattern for methods without explicit return types (constructors, etc.)
        "(?:public|private|protected)?\\s*(?:static)?\\s*(?:\\w+(?:\\s*\\*)*\\s+)?(\\w+)\\s*\\([^)]*\\)\\s*\\{",
        // Pattern for method declarations
        "(?:public|private|protected)?\\s*(?:static)?\\s*\\w+\\s+(\\w+)\\s*\\([^)]*\\)\\s*;",
        // Pattern for methods that throw exceptions
        "(?:public|private|protected)?\\s*(?:static)?\\s*(?:\\w+(?:\\s*\\*)*\\s+)?(\\w+)\\s*\\([^)]*\\)\\s*throws",
        NULL
    };
    
    patterns->function_def_count = 7;
    patterns->function_definition_patterns = (pcre2_code**)malloc(patterns->function_def_count * sizeof(pcre2_code*));
    for (size_t i = 0; i < patterns->function_def_count; i++) {
        patterns->function_definition_patterns[i] = compile_pattern(function_def_patterns[i]);
    }
    
    // Class definition patterns
    const char* class_def_patterns[] = {
        "class\\s+(\\w+)",
        NULL
    };
    
    patterns->class_def_count = 1;
    patterns->class_definition_patterns = (pcre2_code**)malloc(patterns->class_def_count * sizeof(pcre2_code*));
    for (size_t i = 0; i < patterns->class_def_count; i++) {
        patterns->class_definition_patterns[i] = compile_pattern(class_def_patterns[i]);
    }
    
    // Other patterns
    patterns->obj_full_calls_pattern = compile_pattern("(\\w+(?:\\.\\w+)*\\.\\w+)\\s*\\(");
    patterns->ptr_full_calls_pattern = compile_pattern("(\\w+(?:\\.\\w+)*(?:->\\w+)+)\\s*\\(");
    patterns->obj_method_calls_pattern = compile_pattern("\\.(\\w+)\\s*\\(");
    patterns->ptr_method_calls_pattern = compile_pattern("->(\\w+)\\s*\\(");
    // Updated pattern to handle operator() correctly
    // This pattern matches class method calls like Class::method()
    // For operator() methods, it will match "Class::operator()" but we need to distinguish definition vs call
    patterns->class_method_calls_pattern = compile_pattern("((?:\\w+::)*\\w+(?:<[^>]*>)?)::(operator\\s*\\([^)]*\\)|\\w+)\\s*\\(");
    patterns->all_func_calls_pattern = compile_pattern("\\b(\\w+)\\s*\\(");
    patterns->double_quote_pattern = compile_pattern("\"([^\"]*)\"");
    patterns->single_quote_pattern = compile_pattern("'([^']*)'");
    patterns->identifiers_pattern = compile_pattern("\\b(\\w+)\\b");
    
    return patterns;
}

// Function to convert C vector to Python list
static PyObject* vector_to_py_list(char** vec, size_t size) {
    PyObject* list = PyList_New(size);
    for (size_t i = 0; i < size; i++) {
        if (vec[i]) {
            PyObject* str = PyUnicode_FromString(vec[i]);
            PyList_SetItem(list, i, str);
        } else {
            Py_INCREF(Py_None);
            PyList_SetItem(list, i, Py_None);
        }
    }
    // Free the C strings and array
    for (size_t i = 0; i < size; i++) {
        if (vec[i]) {
            free(vec[i]);
        }
    }
    free(vec);
    return list;
}

// Split string into lines
static char** split_lines(const char* code, size_t* line_count) {
    *line_count = 0;
    size_t len = strlen(code);
    for (size_t i = 0; i < len; i++) {
        if (code[i] == '\n') {
            (*line_count)++;
        }
    }
    if (len > 0 && code[len - 1] != '\n') {
        (*line_count)++;
    }
    
    char** lines = (char**)malloc((*line_count + 1) * sizeof(char*));
    size_t line_idx = 0;
    const char* start = code;
    for (size_t i = 0; i <= len; i++) {
        if (code[i] == '\n' || code[i] == '\0') {
            size_t line_len = code + i - start;
            lines[line_idx] = (char*)malloc(line_len + 1);
            strncpy(lines[line_idx], start, line_len);
            lines[line_idx][line_len] = '\0';
            line_idx++;
            start = code + i + 1;
        }
    }
    lines[*line_count] = NULL; // Null terminate the array
    
    return lines;
}

// Helper function to run PCRE2 match and extract captures
static int run_pcre2_match(pcre2_code* re, const char* subject, PCRE2_SIZE subject_length, 
                          pcre2_match_data* match_data, char*** captures, size_t* capture_count) {
    int rc = pcre2_match(re, (PCRE2_SPTR)subject, subject_length, 0, 0, match_data, NULL);
    
    if (rc <= 0) {
        *captures = NULL;
        *capture_count = 0;
        return rc;
    }
    
    *capture_count = rc;
    *captures = (char**)malloc(rc * sizeof(char*));
    
    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
    
    for (int i = 0; i < rc; i++) {
        PCRE2_SIZE start = ovector[2*i];
        PCRE2_SIZE end = ovector[2*i+1];
        if (start != PCRE2_UNSET && end != PCRE2_UNSET) {
            size_t len = end - start;
            (*captures)[i] = (char*)malloc(len + 1);
            strncpy((*captures)[i], subject + start, len);
            (*captures)[i][len] = '\0';
        } else {
            (*captures)[i] = NULL;
        }
    }
    
    return rc;
}

// Helper function to free captures
static void free_captures(char** captures, size_t capture_count) {
    if (captures) {
        for (size_t i = 0; i < capture_count; i++) {
            if (captures[i]) {
                free(captures[i]);
            }
        }
        free(captures);
    }
}

// Simplified main function that replicates the core tokenization logic
static PyObject* build_bm25_index_c(PyObject* self, PyObject* args) {
    const char* code_cstr;
    if (!PyArg_ParseTuple(args, "s", &code_cstr)) {
        return NULL;
    }
    
    // Ensure patterns are initialized
    if (!g_patterns) {
        g_patterns = initialize_patterns();
        initialize_keywords();
    }
    
    // Initialize tokens vector
    size_t tokens_capacity = 200;
    size_t tokens_count = 0;
    char** tokens = (char**)malloc(tokens_capacity * sizeof(char*));
    
    // Release GIL before doing CPU-intensive work
    Py_BEGIN_ALLOW_THREADS
    
    // Split code into lines
    size_t line_count;
    char** lines = split_lines(code_cstr, &line_count);
    
    // Create match data for PCRE2
    pcre2_match_data* match_data = pcre2_match_data_create(32, g_general_context);
    
    // Process each line
    for (size_t line_idx = 0; line_idx < line_count; line_idx++) {
        const char* line = lines[line_idx];
        size_t line_len = strlen(line);
        
        // Make sure we have enough capacity for tokens
        if (tokens_count + 50 >= tokens_capacity) {
            tokens_capacity *= 2;
            tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
        }
        
        // Process class definitions and add [CLASSDEF] prefix
        for (size_t i = 0; i < g_patterns->class_def_count; i++) {
            char** class_captures;
            size_t class_capture_count;
            int rc = run_pcre2_match(g_patterns->class_definition_patterns[i], line, line_len, match_data, &class_captures, &class_capture_count);
            if (rc > 1 && class_captures && class_captures[1]) {
                // Add [CLASSDEF] token
                if (tokens_count >= tokens_capacity) {
                    tokens_capacity *= 2;
                    tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                }
                size_t len = strlen("[CLASSDEF]") + strlen(class_captures[1]) + 1;
                tokens[tokens_count] = (char*)malloc(len);
                sprintf(tokens[tokens_count], "[CLASSDEF]%s", class_captures[1]);
                tokens_count++;
                free_captures(class_captures, class_capture_count);
                break;  // Only process the first match
            }
            free_captures(class_captures, class_capture_count);
        }
        
        // Process function definitions and add [FUNCDEF] prefix
        char* func_name = NULL;
        for (size_t i = 0; i < g_patterns->function_def_count; i++) {
            char** func_captures;
            size_t func_capture_count;
            int rc = run_pcre2_match(g_patterns->function_definition_patterns[i], line, line_len, match_data, &func_captures, &func_capture_count);
            if (rc > 1 && func_captures && func_captures[1]) {
                const char* potential_func_name = func_captures[1];
                // Special handling for operator() methods - always treat them as function definitions
                if (strncmp(potential_func_name, "operator", 8) == 0 || !is_keyword(potential_func_name)) {
                    func_name = strdup(potential_func_name);
                    // Add [FUNCDEF] token
                    if (tokens_count >= tokens_capacity) {
                        tokens_capacity *= 2;
                        tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                    }
                    size_t len = strlen("[FUNCDEF]") + strlen(potential_func_name) + 1;
                    tokens[tokens_count] = (char*)malloc(len);
                    sprintf(tokens[tokens_count], "[FUNCDEF]%s", potential_func_name);
                    tokens_count++;
                    free_captures(func_captures, func_capture_count);
                    break;  // Only process the first match
                }
            }
            free_captures(func_captures, func_capture_count);
        }
        
        // Extract full function calls (rule 3)
        char** full_calls = NULL;
        size_t full_calls_count = 0;
        
        // Object method calls: a.f(
        char** obj_captures;
        size_t obj_capture_count;
        int rc = run_pcre2_match(g_patterns->obj_full_calls_pattern, line, line_len, match_data, &obj_captures, &obj_capture_count);
        if (rc > 1 && obj_captures && obj_captures[1]) {
            full_calls = (char**)realloc(full_calls, (full_calls_count + 1) * sizeof(char*));
            full_calls[full_calls_count] = strdup(obj_captures[1]);
            full_calls_count++;
        }
        free_captures(obj_captures, obj_capture_count);
        
        // Pointer method calls: a->f(
        char** ptr_captures;
        size_t ptr_capture_count;
        rc = run_pcre2_match(g_patterns->ptr_full_calls_pattern, line, line_len, match_data, &ptr_captures, &ptr_capture_count);
        if (rc > 1 && ptr_captures && ptr_captures[1]) {
            full_calls = (char**)realloc(full_calls, (full_calls_count + 1) * sizeof(char*));
            full_calls[full_calls_count] = strdup(ptr_captures[1]);
            full_calls_count++;
        }
        free_captures(ptr_captures, ptr_capture_count);
        
        // Object method calls: a.f(
        // Find all matches, not just the first one
        pcre2_match_data* obj_method_match_data = pcre2_match_data_create(32, g_general_context);
        PCRE2_SIZE* obj_method_ovector = pcre2_get_ovector_pointer(obj_method_match_data);
        int obj_method_rc;
        PCRE2_SIZE obj_method_offset = 0;
        
        // Loop to find all matches
        while ((obj_method_rc = pcre2_match(g_patterns->obj_method_calls_pattern, (PCRE2_SPTR)line, line_len, 
                                           obj_method_offset, 0, obj_method_match_data, NULL)) > 0) {
            // Get the captured method name (group 1)
            PCRE2_SIZE func_start = obj_method_ovector[2];  // Start of group 1
            PCRE2_SIZE func_end = obj_method_ovector[3];    // End of group 1
            
            if (func_start != PCRE2_UNSET && func_end != PCRE2_UNSET) {
                size_t func_len = func_end - func_start;
                char* func = (char*)malloc(func_len + 1);
                if (func) {
                    strncpy(func, line + func_start, func_len);
                    func[func_len] = '\0';
                    
                    // Add [OBJCALL] token
                    if (tokens_count >= tokens_capacity) {
                        tokens_capacity *= 2;
                        char** new_tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                        if (new_tokens) {
                            tokens = new_tokens;
                        } else {
                            // Allocation failed, continue with what we have
                            free(func);
                            break;
                        }
                    }
                    size_t len = strlen("[OBJCALL]") + strlen(func) + 1;
                    tokens[tokens_count] = (char*)malloc(len);
                    if (tokens[tokens_count]) {
                        sprintf(tokens[tokens_count], "[OBJCALL]%s", func);
                        tokens_count++;
                    }
                    free(func);
                }
            }
            
            // Move offset past this match
            obj_method_offset = obj_method_ovector[1];  // End of full match
            if (obj_method_offset <= obj_method_ovector[0]) {  // Prevent infinite loop
                obj_method_offset = obj_method_ovector[0] + 1;
            }
            
            // Break if we've reached the end of the string
            if (obj_method_offset >= line_len) {
                break;
            }
        }
        pcre2_match_data_free(obj_method_match_data);
        
        // Pointer method calls: a->f(
        char** ptr_method_captures;
        size_t ptr_method_capture_count;
        rc = run_pcre2_match(g_patterns->ptr_method_calls_pattern, line, line_len, match_data, &ptr_method_captures, &ptr_method_capture_count);
        if (rc > 1 && ptr_method_captures && ptr_method_captures[1]) {
            // Add [PTRCALL] token
            if (tokens_count >= tokens_capacity) {
                tokens_capacity *= 2;
                tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
            }
            size_t len = strlen("[PTRCALL]") + strlen(ptr_method_captures[1]) + 1;
            tokens[tokens_count] = (char*)malloc(len);
            sprintf(tokens[tokens_count], "[PTRCALL]%s", ptr_method_captures[1]);
            tokens_count++;
        }
        free_captures(ptr_method_captures, ptr_method_capture_count);
        
        // Class method calls: Class::method(
        char** class_method_captures;
        size_t class_method_capture_count;
        rc = run_pcre2_match(g_patterns->class_method_calls_pattern, line, line_len, match_data, &class_method_captures, &class_method_capture_count);
        if (rc > 2 && class_method_captures && class_method_captures[1] && class_method_captures[2]) {
            const char* full_class_name = class_method_captures[1];
            const char* method = class_method_captures[2];
            
            // Check if this is an operator() method call (not definition)
            // If we see "Class::operator()" followed by "()" somewhere later in the line, it's a call
            int is_operator_call = 0;
            if (strncmp(method, "operator", 8) == 0) {
                // Look for the pattern "operator()...()" to distinguish call from definition
                const char* after_operator = strstr(line, "operator()");
                if (after_operator) {
                    after_operator += 10; // Skip past "operator()"
                    // Look for another "()" which would indicate a function call
                    if (strstr(after_operator, "()")) {
                        is_operator_call = 1;
                    }
                }
            }
            
            // Only add [CLASSCALL] if it's actually a method call, not a definition
            if (is_operator_call || strncmp(method, "operator", 8) != 0) {
                // Add [CLASSCALL] marker for method
                char* classcall_token = "[CLASSCALL]";
                if (tokens_count >= tokens_capacity) {
                    tokens_capacity *= 2;
                    tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                }
                size_t len = strlen(classcall_token) + strlen(method) + 1;
                tokens[tokens_count] = (char*)malloc(len);
                sprintf(tokens[tokens_count], "%s%s", classcall_token, method);
                tokens_count++;
            }
            
            // Extract base class/namespace names
            // Instead of adding the full class name, extract individual components
            // Split by '::' but be careful with templates
            char* class_name_copy = strdup(full_class_name);
            if (class_name_copy) {
                char* part = strtok(class_name_copy, "::");
                while (part) {
                    // For template classes, extract the base name
                    char* template_part = strdup(part);
                    if (template_part) {
                        char* template_start = strchr(template_part, '<');
                        if (template_start) {
                            *template_start = '\0';  // Truncate at '<'
                        }
                        
                        // Add the base name if it's not empty and not already processed
                        if (strlen(template_part) > 0) {
                            // Check if this identifier was already processed
                            int already_processed = 0;
                            for (size_t i = 0; i < tokens_count; i++) {
                                if (strcmp(tokens[i], template_part) == 0) {
                                    already_processed = 1;
                                    break;
                                }
                            }
                            
                            // Only add if not already processed
                            if (!already_processed) {
                                if (tokens_count >= tokens_capacity) {
                                    tokens_capacity *= 2;
                                    char** new_tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                                    if (new_tokens) {
                                        tokens = new_tokens;
                                    } else {
                                        free(template_part);
                                        free(class_name_copy);
                                        break;  // Allocation failed
                                    }
                                }
                                tokens[tokens_count] = strdup(template_part);
                                tokens_count++;
                            }
                        }
                        free(template_part);
                    }
                    part = strtok(NULL, "::");
                }
                free(class_name_copy);
            }
        }
        free_captures(class_method_captures, class_method_capture_count);
        
        // Regular function calls: f( but exclude keywords and function definitions
        // Find all matches, not just the first one
        pcre2_match_data* func_call_match_data = pcre2_match_data_create(32, g_general_context);
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(func_call_match_data);
        int func_call_rc;
        int func_call_offset = 0;
        
        while ((func_call_rc = pcre2_match(g_patterns->all_func_calls_pattern, (PCRE2_SPTR)line, line_len, 
                                           func_call_offset, 0, func_call_match_data, NULL)) > 0) {
            // Get the captured function name (group 1)
            PCRE2_SIZE func_start = ovector[2];  // Start of group 1
            PCRE2_SIZE func_end = ovector[3];    // End of group 1
            
            if (func_start != PCRE2_UNSET && func_end != PCRE2_UNSET) {
                size_t func_len = func_end - func_start;
                char* func = (char*)malloc(func_len + 1);
                if (func) {
                    strncpy(func, line + func_start, func_len);
                    func[func_len] = '\0';
                    
                    // Exclude keywords - keywords should not be treated as function calls
                    if (!is_keyword(func)) {
                        // Check if this is actually a function definition rather than a call
                        int is_func_def = 0;
                        
                        // Check for Python/JavaScript function definitions
                        char func_def_pattern[256];
                        snprintf(func_def_pattern, sizeof(func_def_pattern), "(?:def|function)\\s+%s\\s*\\(", func);
                        pcre2_code* func_def_re = compile_pattern(func_def_pattern);
                        if (func_def_re) {
                            char** temp_captures;
                            size_t temp_count;
                            int temp_rc = run_pcre2_match(func_def_re, line, line_len, match_data, &temp_captures, &temp_count);
                            if (temp_rc > 0) {
                                is_func_def = 1;
                            }
                            free_captures(temp_captures, temp_count);
                            pcre2_code_free(func_def_re);
                        }
                        
                        // If not found as Python/JavaScript function definition, check for C++/Java/C# function definitions
                        if (!is_func_def) {
                            // Check if this function appears in any of the pre-compiled function definition patterns
                            for (size_t i = 0; i < g_patterns->function_def_count; i++) {
                                char** temp_captures;
                                size_t temp_count;
                                int temp_rc = run_pcre2_match(g_patterns->function_definition_patterns[i], line, line_len, match_data, &temp_captures, &temp_count);
                                if (temp_rc > 1 && temp_captures && temp_captures[1]) {
                                    // Check if the captured function name matches our function
                                    if (strcmp(temp_captures[1], func) == 0) {
                                        is_func_def = 1;
                                    }
                                }
                                free_captures(temp_captures, temp_count);
                                if (is_func_def) {
                                    break;
                                }
                            }
                        }
                        
                        // If it's not a function definition, treat it as a function call
                        if (!is_func_def) {
                            // Check if this function call has already been marked by a specific pattern
                            int already_marked = 0;
                            for (size_t i = 0; i < tokens_count; i++) {
                                // Check if this function is already marked with a specific call pattern
                                if ((strncmp(tokens[i], "[CALL]", 6) == 0 && strcmp(tokens[i] + 6, func) == 0) ||
                                    (strncmp(tokens[i], "[OBJCALL]", 9) == 0 && strcmp(tokens[i] + 9, func) == 0) ||
                                    (strncmp(tokens[i], "[PTRCALL]", 9) == 0 && strcmp(tokens[i] + 9, func) == 0) ||
                                    (strncmp(tokens[i], "[CLASSCALL]", 11) == 0 && strcmp(tokens[i] + 11, func) == 0)) {
                                    already_marked = 1;
                                    break;
                                }
                            }
                            
                            // Only add generic [CALL] marker if not already marked
                            if (!already_marked) {
                                // Add [CALL] marker
                                if (tokens_count >= tokens_capacity) {
                                    tokens_capacity *= 2;
                                    char** new_tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                                    if (new_tokens) {
                                        tokens = new_tokens;
                                    } else {
                                        // Allocation failed, continue with what we have
                                        free(func);
                                        break;
                                    }
                                }
                                size_t len = strlen("[CALL]") + strlen(func) + 1;
                                tokens[tokens_count] = (char*)malloc(len);
                                if (tokens[tokens_count]) {
                                    sprintf(tokens[tokens_count], "[CALL]%s", func);
                                    tokens_count++;
                                }
                            }
                        }
                    }
                    free(func);
                }
            }
            
            // Move offset past this match
            func_call_offset = ovector[1];  // End of full match
            if (func_call_offset <= ovector[0]) {  // Prevent infinite loop
                func_call_offset = ovector[0] + 1;
            }
            
            // Break if we've reached the end of the string
            if (func_call_offset >= line_len) {
                break;
            }
        }
        pcre2_match_data_free(func_call_match_data);
        
        // Add full function call strings as tokens (rule 3)
        for (size_t i = 0; i < full_calls_count; i++) {
            if (tokens_count >= tokens_capacity) {
                tokens_capacity *= 2;
                tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
            }
            tokens[tokens_count] = strdup(full_calls[i]);
            tokens_count++;
        }
        
        // Free full_calls
        if (full_calls) {
            for (size_t i = 0; i < full_calls_count; i++) {
                free(full_calls[i]);
            }
            free(full_calls);
        }
        
        // Process string literals
        char** double_quote_captures;
        size_t double_quote_capture_count;
        rc = run_pcre2_match(g_patterns->double_quote_pattern, line, line_len, match_data, &double_quote_captures, &double_quote_capture_count);
        if (rc > 1 && double_quote_captures && double_quote_captures[1]) {
            // Clean string literals, only keep alphanumeric and underscore
            char* literal = double_quote_captures[1];
            char* clean_literal = (char*)malloc(strlen(literal) + 1);
            size_t j = 0;
            for (size_t i = 0; literal[i]; i++) {
                if (isalnum(literal[i]) || literal[i] == '_') {
                    clean_literal[j++] = literal[i];
                }
            }
            clean_literal[j] = '\0';
            
            if (strlen(clean_literal) > 0) {
                if (tokens_count >= tokens_capacity) {
                    tokens_capacity *= 2;
                    tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                }
                tokens[tokens_count] = strdup(clean_literal);
                tokens_count++;
            }
            free(clean_literal);
        }
        free_captures(double_quote_captures, double_quote_capture_count);
        
        char** single_quote_captures;
        size_t single_quote_capture_count;
        rc = run_pcre2_match(g_patterns->single_quote_pattern, line, line_len, match_data, &single_quote_captures, &single_quote_capture_count);
        if (rc > 1 && single_quote_captures && single_quote_captures[1]) {
            // Clean string literals, only keep alphanumeric and underscore
            char* literal = single_quote_captures[1];
            char* clean_literal = (char*)malloc(strlen(literal) + 1);
            size_t j = 0;
            for (size_t i = 0; literal[i]; i++) {
                if (isalnum(literal[i]) || literal[i] == '_') {
                    clean_literal[j++] = literal[i];
                }
            }
            clean_literal[j] = '\0';
            
            if (strlen(clean_literal) > 0) {
                if (tokens_count >= tokens_capacity) {
                    tokens_capacity *= 2;
                    tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                }
                tokens[tokens_count] = strdup(clean_literal);
                tokens_count++;
            }
            free(clean_literal);
        }
        free_captures(single_quote_captures, single_quote_capture_count);
        
        // Extract all identifiers, including keywords
        // Use a simple loop to find all \b(\w+)\b patterns
        const char* pos = line;
        while (pos < line + line_len) {
            // Skip non-alphanumeric characters
            while (pos < line + line_len && !isalnum(*pos) && *pos != '_') {
                pos++;
            }
            
            if (pos < line + line_len) {
                // Found start of identifier
                const char* start = pos;
                
                // Find end of identifier
                while (pos < line + line_len && (isalnum(*pos) || *pos == '_')) {
                    pos++;
                }
                
                // Extract identifier
                size_t len = pos - start;
                if (len > 0) {
                    char* identifier = (char*)malloc(len + 1);
                    if (identifier) {
                        strncpy(identifier, start, len);
                        identifier[len] = '\0';
                        
                        // Special handling for function names - avoid duplicating them as regular identifiers
                        int is_function_name = (func_name && strcmp(identifier, func_name) == 0);
                        
                        // Skip function names as regular identifiers since we already added [FUNCDEF]name
                        // Also skip function calls that were already marked with [CALL]
                        int is_function_call = 0;
                        for (size_t i = 0; i < tokens_count; i++) {
                            if (strncmp(tokens[i], "[CALL]", 6) == 0 && strcmp(tokens[i] + 6, identifier) == 0) {
                                is_function_call = 1;
                                break;
                            }
                        }
                        
                        // Also skip class names that were already marked with [CLASSDEF]
                        int is_class_name = 0;
                        for (size_t i = 0; i < tokens_count; i++) {
                            if (strncmp(tokens[i], "[CLASSDEF]", 10) == 0 && strcmp(tokens[i] + 10, identifier) == 0) {
                                is_class_name = 1;
                                break;
                            }
                        }
                        
                        // Also skip class names that were already processed in class method calls
                        // For example, in CodeTransform::operator(), we should not add CodeTransform as a regular identifier
                        int is_class_method_class_name = 0;
                        if (func_name && strcmp(identifier, func_name) == 0) {
                            is_class_method_class_name = 1;
                        }
                        
                        if (!is_function_name && !is_function_call && !is_class_name && !is_class_method_class_name) {
                            // Make sure we have enough capacity
                            if (tokens_count >= tokens_capacity) {
                                tokens_capacity *= 2;
                                char** new_tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                                if (new_tokens) {
                                    tokens = new_tokens;
                                } else {
                                    free(identifier);
                                    break;  // Allocation failed
                                }
                            }
                            tokens[tokens_count] = identifier;
                            tokens_count++;
                        } else {
                            // Skip this identifier since it's already been processed
                            free(identifier);
                        }
                    }
                }
            }
        }
        
        
        // Add parameters (if it's a function definition)
        if (func_name) {
            // Get parameter part using a simple approach
            char* func_start = strstr(line, func_name);
            if (func_start) {
                char* paren_start = strchr(func_start, '(');
                if (paren_start) {
                    char* paren_end = strchr(paren_start, ')');
                    if (paren_end) {
                        // Extract parameters substring
                        size_t param_len = paren_end - paren_start - 1;
                        if (param_len > 0) {
                            char* params_str = (char*)malloc(param_len + 1);
                            strncpy(params_str, paren_start + 1, param_len);
                            params_str[param_len] = '\0';
                            
                            // Split parameters by comma
                            char* param = strtok(params_str, ",");
                            while (param) {
                                // Remove leading/trailing whitespace
                                while (*param == ' ' || *param == '\t') param++;
                                char* end = param + strlen(param) - 1;
                                while (end > param && (*end == ' ' || *end == '\t')) end--;
                                *(end + 1) = '\0';
                                
                                // Get last word (remove type declaration like int a, String b)
                                char* last_space = strrchr(param, ' ');
                                if (last_space) {
                                    param = last_space + 1;
                                }
                                
                                // Remove default values (like a=1)
                                char* equals_pos = strchr(param, '=');
                                if (equals_pos) {
                                    *equals_pos = '\0';
                                }
                                
                                // Remove pointer symbols etc.
                                char* clean_param = (char*)malloc(strlen(param) + 1);
                                size_t j = 0;
                                for (size_t i = 0; param[i]; i++) {
                                    if (param[i] != '&' && param[i] != '*') {
                                        clean_param[j++] = param[i];
                                    }
                                }
                                clean_param[j] = '\0';
                                
                                // Clean parameter name, remove punctuation
                                char* final_param = (char*)malloc(strlen(clean_param) + 1);
                                j = 0;
                                for (size_t i = 0; clean_param[i]; i++) {
                                    if (isalnum(clean_param[i]) || clean_param[i] == '_') {
                                        final_param[j++] = clean_param[i];
                                    }
                                }
                                final_param[j] = '\0';
                                
                                // Add parameter name as token
                                if (strlen(final_param) > 0 && strcmp(final_param, "void") != 0) {  // void is not a parameter name
                                    if (tokens_count >= tokens_capacity) {
                                        tokens_capacity *= 2;
                                        tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
                                    }
                                    tokens[tokens_count] = strdup(final_param);
                                    tokens_count++;
                                }
                                
                                free(clean_param);
                                free(final_param);
                                
                                param = strtok(NULL, ",");
                            }
                            
                            free(params_str);
                        }
                    }
                }
            }
            free(func_name);
        }
        
        // Free the line
        free(lines[line_idx]);
    }
    
    // Free lines array
    free(lines);
    
    // Free match data
    pcre2_match_data_free(match_data);
    
    // Re-acquire GIL before returning to Python
    Py_END_ALLOW_THREADS
    
    // Convert to Python list and return
    return vector_to_py_list(tokens, tokens_count);
}

// Method definition table
static PyMethodDef Bm25Methods[] = {
    {"build_bm25_index_c", build_bm25_index_c, METH_VARARGS, "Build BM25 index from code using C implementation with PCRE2"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef bm25module = {
    PyModuleDef_HEAD_INIT,
    "bm25_c_extension",
    "BM25 C extension module with PCRE2",
    -1,
    Bm25Methods
};

// Module initialization
PyMODINIT_FUNC PyInit_bm25_c_extension(void) {
    // Initialize patterns and keywords on module load
    if (!g_patterns) {
        g_patterns = initialize_patterns();
        initialize_keywords();
    }
    
    return PyModule_Create(&bm25module);
}