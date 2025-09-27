#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include "ragalyze_common.h"

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

// Reference structures from text_splitter_fast.c
typedef struct {
    char* token;
    int line;
    int col;
} BM25Index;

typedef struct {
    char* text;
    size_t text_len;
    char* id;
    char* parent_doc_id;
    int order;
    float score;
    int estimated_num_tokens;
    BM25Index* bm25_indexes;
    size_t bm25_count;
} CDocument;

// Parse Python document dictionary into CDocument structure
static int parse_python_document(PyObject* doc_obj, CDocument* cdoc) {
    memset(cdoc, 0, sizeof(CDocument));

    if (!PyDict_Check(doc_obj)) {
        debug_printf("parse_python_document: Input is not a dictionary\n");
        return 0;
    }

    // Get text
    PyObject* text_obj = PyDict_GetItemString(doc_obj, "text");
    if (!text_obj || !PyUnicode_Check(text_obj)) {
        debug_printf("parse_python_document: Invalid or missing text field\n");
        return 0;
    }
    cdoc->text = ragalyze_strdup(PyUnicode_AsUTF8(text_obj));
    RAGALYZE_ASSERT(cdoc->text != NULL, "Failed to duplicate text", NULL);
    cdoc->text_len = strlen(cdoc->text);

    // Get id
    PyObject* id_obj = PyDict_GetItemString(doc_obj, "id");
    if (id_obj && PyUnicode_Check(id_obj)) {
        cdoc->id = ragalyze_strdup(PyUnicode_AsUTF8(id_obj));
    }

    // Get parent_doc_id
    PyObject* parent_obj = PyDict_GetItemString(doc_obj, "parent_doc_id");
    if (parent_obj && PyUnicode_Check(parent_obj)) {
        cdoc->parent_doc_id = ragalyze_strdup(PyUnicode_AsUTF8(parent_obj));
    }

    // Get order
    PyObject* order_obj = PyDict_GetItemString(doc_obj, "order");
    if (order_obj && PyLong_Check(order_obj)) {
        cdoc->order = PyLong_AsLong(order_obj);
    } else {
        cdoc->order = -1;
    }

    // Get metadata and BM25 indexes
    PyObject* meta_obj = PyDict_GetItemString(doc_obj, "meta_data");
    if (meta_obj && PyDict_Check(meta_obj)) {
        PyObject* bm25_obj = PyDict_GetItemString(meta_obj, "bm25_indexes");
        if (bm25_obj && PyList_Check(bm25_obj)) {
            cdoc->bm25_count = PyList_Size(bm25_obj);
            if (cdoc->bm25_count > 0) {
                cdoc->bm25_indexes = RAGALYZE_MALLOC(cdoc->bm25_count * sizeof(BM25Index));
                ASSERT_ALLOC(cdoc->bm25_indexes, NULL);

                for (size_t i = 0; i < cdoc->bm25_count; i++) {
                    PyObject* bm25_item = PyList_GetItem(bm25_obj, i);

                    // Check if it's a BM25Index object with token and position attributes
                    if (PyObject_HasAttrString(bm25_item, "token") && PyObject_HasAttrString(bm25_item, "position")) {
                        // Get token attribute
                        PyObject* token_obj = PyObject_GetAttrString(bm25_item, "token");
                        if (token_obj && PyUnicode_Check(token_obj)) {
                            cdoc->bm25_indexes[i].token = ragalyze_strdup(PyUnicode_AsUTF8(token_obj));
                        } else {
                            cdoc->bm25_indexes[i].token = NULL;
                        }
                        Py_XDECREF(token_obj);

                        // Get position attribute (tuple of (line, col))
                        PyObject* position_obj = PyObject_GetAttrString(bm25_item, "position");
                        if (position_obj && PyTuple_Check(position_obj) && PyTuple_Size(position_obj) >= 2) {
                            PyObject* line_obj = PyTuple_GetItem(position_obj, 0);
                            PyObject* col_obj = PyTuple_GetItem(position_obj, 1);

                            if (PyLong_Check(line_obj)) {
                                cdoc->bm25_indexes[i].line = PyLong_AsLong(line_obj);
                            } else {
                                cdoc->bm25_indexes[i].line = 0;
                            }
                            if (PyLong_Check(col_obj)) {
                                cdoc->bm25_indexes[i].col = PyLong_AsLong(col_obj);
                            } else {
                                cdoc->bm25_indexes[i].col = 0;
                            }
                        } else {
                            cdoc->bm25_indexes[i].line = 0;
                            cdoc->bm25_indexes[i].col = 0;
                        }
                        Py_XDECREF(position_obj);
                    } else {
                        // Fallback: try to parse as tuple (token, line, col)
                        if (PyTuple_Check(bm25_item) && PyTuple_Size(bm25_item) >= 3) {
                            PyObject* token_obj = PyTuple_GetItem(bm25_item, 0);
                            PyObject* line_obj = PyTuple_GetItem(bm25_item, 1);
                            PyObject* col_obj = PyTuple_GetItem(bm25_item, 2);

                            if (PyUnicode_Check(token_obj)) {
                                cdoc->bm25_indexes[i].token = ragalyze_strdup(PyUnicode_AsUTF8(token_obj));
                            } else {
                                cdoc->bm25_indexes[i].token = NULL;
                            }
                            if (PyLong_Check(line_obj)) {
                                cdoc->bm25_indexes[i].line = PyLong_AsLong(line_obj);
                            } else {
                                cdoc->bm25_indexes[i].line = 0;
                            }
                            if (PyLong_Check(col_obj)) {
                                cdoc->bm25_indexes[i].col = PyLong_AsLong(col_obj);
                            } else {
                                cdoc->bm25_indexes[i].col = 0;
                            }
                        } else {
                            // Initialize with default values if object is invalid
                            cdoc->bm25_indexes[i].token = NULL;
                            cdoc->bm25_indexes[i].line = 0;
                            cdoc->bm25_indexes[i].col = 0;
                        }
                    }
                }
            }
        }
    }

    debug_printf("parse_python_document: Successfully parsed document with %zu BM25 indexes\n", cdoc->bm25_count);
    return 1;
}


// Create line-numbered text
static char* create_line_numbered_text(const char* text, int start_line) {
    if (!text) {
        return NULL;
    }

    int line_count = 1;
    int text_len = strlen(text);
    for (int i = 0; i < text_len; i++) {
        if (text[i] == '\n') line_count++;
    }

    int buffer_size = text_len + line_count * 20;
    char* result = malloc(buffer_size);
    if (!result) {
        return NULL;
    }

    int current_line = start_line;
    int result_pos = 0;
    int line_start = 0;

    for (int i = 0; i <= text_len; i++) {
        if (i == text_len || text[i] == '\n') {
            int prefix_len = snprintf(result + result_pos, buffer_size - result_pos, "%d: ", current_line);
            result_pos += prefix_len;

            int line_len = i - line_start;
            if (line_len > 0) {
                strncpy(result + result_pos, text + line_start, line_len);
                result_pos += line_len;
            }

            if (i < text_len) {
                result[result_pos++] = '\n';
            }

            current_line++;
            line_start = i + 1;
        }
    }

    result[result_pos] = '\0';
    return result;
}

// Pure C structure for GIL-free chunk processing results
typedef struct {
    char* text;                    // Line-numbered chunk text
    char* original_text;           // Original chunk text (without line numbers)
    int bytes_processed;
    int start_line;
    int chunk_idx;
    size_t relevant_count;
    // BM25 data - store as raw C data, convert to Python objects later
    struct {
        char* token;
        int line;
        int col;
    }* bm25_data;
    size_t bm25_data_count;
    // Document metadata
    char* doc_id;
    char* parent_doc_id;
} chunk_result_t;

// Convert chunk_result_t to Python dictionary (must be called with GIL held)
static PyObject* convert_chunk_result(chunk_result_t* result) {
    PyObject* chunk_dict = PyDict_New();
    if (!chunk_dict) return NULL;

    // Set text fields
    PyDict_SetItemString(chunk_dict, "text", PyUnicode_FromString(result->text));
    PyDict_SetItemString(chunk_dict, "bytes_processed", PyLong_FromLong(result->bytes_processed));

    // Create metadata
    PyObject* meta_dict = PyDict_New();
    if (!meta_dict) {
        Py_DECREF(chunk_dict);
        return NULL;
    }

    PyDict_SetItemString(meta_dict, "start_line", PyLong_FromLong(result->start_line));
    PyDict_SetItemString(meta_dict, "original_text", PyUnicode_FromString(result->original_text));

    // Add BM25 indexes to metadata
    if (result->bm25_data_count > 0 && result->bm25_data) {
        PyObject* bm25_list = PyList_New(result->bm25_data_count);
        for (size_t i = 0; i < result->bm25_data_count; i++) {
            PyObject* bm25_tuple = PyTuple_New(3);
            PyTuple_SetItem(bm25_tuple, 0, PyUnicode_FromString(result->bm25_data[i].token));
            PyTuple_SetItem(bm25_tuple, 1, PyLong_FromLong(result->bm25_data[i].line));
            PyTuple_SetItem(bm25_tuple, 2, PyLong_FromLong(result->bm25_data[i].col));
            PyList_SetItem(bm25_list, i, bm25_tuple);
        }
        PyDict_SetItemString(meta_dict, "bm25_indexes", bm25_list);
        Py_DECREF(bm25_list);
    }

    PyDict_SetItemString(chunk_dict, "meta_data", meta_dict);
    PyDict_SetItemString(chunk_dict, "order", PyLong_FromLong(result->chunk_idx));
    PyDict_SetItemString(chunk_dict, "parent_doc_id",
                        PyUnicode_FromString(result->parent_doc_id ? result->parent_doc_id : ""));
    PyDict_SetItemString(chunk_dict, "vector", PyList_New(0)); // Empty vector

    // Generate UUID for the new document
    char* new_id = ragalyze_generate_uuid();
    if (new_id) {
        PyDict_SetItemString(chunk_dict, "id", PyUnicode_FromString(new_id));
        free(new_id);
    }

    Py_DECREF(meta_dict);
    return chunk_dict;
}

// Free chunk_result_t memory
static void free_chunk_results(chunk_result_t* results, int count) {
    if (!results) return;

    for (int i = 0; i < count; i++) {
        free(results[i].text);
        free(results[i].original_text);
        free(results[i].doc_id);
        free(results[i].parent_doc_id);

        // Free BM25 data
        if (results[i].bm25_data) {
            for (size_t j = 0; j < results[i].bm25_data_count; j++) {
                free(results[i].bm25_data[j].token);
            }
            free(results[i].bm25_data);
        }
    }
    free(results);
}

// Structure to hold split units with their positions
typedef struct {
    char* text;
    int start_pos;
    int length;
    int line;
    int col;
} text_unit_t;

// Structure to hold text units collection
typedef struct {
    text_unit_t* units;
    size_t count;
    size_t capacity;
} text_units_t;

// Structure to hold chunks collection
typedef struct {
    char** texts;
    int* start_lines;
    int* start_positions;
    int* end_positions;
    size_t count;
    size_t capacity;
} chunks_t;

// Free text units collection
static void free_text_units(text_units_t* units) {
    if (!units) return;
    if (units->units) {
        for (size_t i = 0; i < units->count; i++) {
            free(units->units[i].text);
        }
        free(units->units);
    }
}

// Free chunks collection
static void free_chunks(chunks_t* chunks) {
    if (!chunks) return;
    if (chunks->texts) {
        for (size_t i = 0; i < chunks->count; i++) {
            free(chunks->texts[i]);
        }
        free(chunks->texts);
    }
    if (chunks->start_lines) free(chunks->start_lines);
    if (chunks->start_positions) free(chunks->start_positions);
    if (chunks->end_positions) free(chunks->end_positions);
}

// Split text into units based on separator (similar to Python's str.split)
// This function properly tracks positions and handles UTF-8
static text_units_t split_text_into_units(const char* text, const char* separator) {
    text_units_t units = {0};
    if (!text || !separator) return units;

    size_t text_len = strlen(text);
    size_t sep_len = strlen(separator);

    // Initial capacity
    units.capacity = 64;
    units.units = calloc(units.capacity, sizeof(text_unit_t));
    if (!units.units) return units;

    int current_pos = 0;
    int current_line = 0;
    int current_col = 0;

    while (current_pos < (int)text_len) {
        // Find next separator
        char* sep_pos = strstr(text + current_pos, separator);
        int unit_end_pos = (sep_pos) ? (sep_pos - text) : text_len;
        int unit_length = unit_end_pos - current_pos;
        // Calculate line and column for this unit
        int unit_line = current_line;
        int unit_col = current_col;

        // Extract unit text
        char* unit_text = safe_utf8_strncpy(text + current_pos, text_len - current_pos, unit_length);
        if (!unit_text) break;

        // Add unit to collection
        if (units.count >= units.capacity) {
            units.capacity *= 2;
            text_unit_t* new_units = realloc(units.units, units.capacity * sizeof(text_unit_t));
            if (!new_units) {
                free(unit_text);
                break;
            }
            units.units = new_units;
        }

        units.units[units.count].text = unit_text;
        units.units[units.count].start_pos = current_pos;
        units.units[units.count].length = strlen(unit_text);
        units.units[units.count].line = unit_line;
        units.units[units.count].col = unit_col;
        units.count++;

        // Update line/column tracking for the processed unit (before updating position)
        for (int i = current_pos; i < unit_end_pos; i++) {
            if (text[i] == '\n') {
                current_line++;
                current_col = 0;
            } else {
                current_col++;
            }
        }

        // Update position
        current_pos = unit_end_pos;

        // Skip separator if found
        if (sep_pos && current_pos + sep_len <= text_len) {
            current_pos += sep_len;
            // Update line/column for separator
            for (size_t i = 0; i < sep_len; i++) {
                if (text[current_pos - sep_len + i] == '\n') {
                    current_line++;
                    current_col = 0;
                } else {
                    current_col++;
                }
            }
        } else {
            break;
        }
    }

    return units;
}

// Merge text units into chunks with proper overlap (similar to Python's _merge_units_to_chunks)
static chunks_t merge_units_to_chunks(text_units_t* units, int chunk_size, int chunk_overlap, const char* separator) {
    chunks_t chunks = {0};
    if (!units || units->count == 0 || chunk_size <= 0) return chunks;

    // Initial capacity
    chunks.capacity = 16;
    chunks.texts = calloc(chunks.capacity, sizeof(char*));
    chunks.start_lines = calloc(chunks.capacity, sizeof(int));
    chunks.start_positions = calloc(chunks.capacity, sizeof(int));
    chunks.end_positions = calloc(chunks.capacity, sizeof(int));

    if (!chunks.texts || !chunks.start_lines || !chunks.start_positions || !chunks.end_positions) {
        free_chunks(&chunks);
        return chunks;
    }

    int step = chunk_size - chunk_overlap;
    if (step <= 0) step = chunk_size; // No overlap

    for (size_t idx = 0; idx < units->count; idx += step) {
        // Check if this would be the last chunk
        if (idx + chunk_size >= units->count) {
            // Last chunk: merge remaining units

            // Calculate total length needed
            size_t total_length = 0;
            for (size_t i = idx; i < units->count; i++) {
                total_length += strlen(units->units[i].text) + (i < units->count - 1 ? strlen(separator) : 0);
            }

            // Allocate chunk text
            char* chunk_text = malloc(total_length + 1);
            if (!chunk_text) continue;

            // Build chunk text
            chunk_text[0] = '\0';
            for (size_t i = idx; i < units->count; i++) {
                strcat(chunk_text, units->units[i].text);
                if (i < units->count - 1) {
                    strcat(chunk_text, separator);
                }
            }
            // For sentence splitting, add the separator at the end like Python does
            if (strlen(separator) > 0 && strcmp(separator, ".") == 0) {
                strcat(chunk_text, separator);
            }

            // Add to chunks
            if (chunks.count >= chunks.capacity) {
                chunks.capacity *= 2;
                chunks.texts = realloc(chunks.texts, chunks.capacity * sizeof(char*));
                chunks.start_lines = realloc(chunks.start_lines, chunks.capacity * sizeof(int));
                chunks.start_positions = realloc(chunks.start_positions, chunks.capacity * sizeof(int));
                chunks.end_positions = realloc(chunks.end_positions, chunks.capacity * sizeof(int));

                if (!chunks.texts || !chunks.start_lines || !chunks.start_positions || !chunks.end_positions) {
                    free(chunk_text);
                    break;
                }
            }

            chunks.texts[chunks.count] = chunk_text;
            chunks.start_lines[chunks.count] = units->units[idx].line;
            chunks.start_positions[chunks.count] = units->units[idx].start_pos;
            chunks.end_positions[chunks.count] = units->units[units->count - 1].start_pos + units->units[units->count - 1].length;
            chunks.count++;

            break; // This was the last chunk
        } else {
            // Regular chunk: merge chunk_size units
            size_t end_idx = idx + chunk_size;

            // Calculate total length needed
            size_t total_length = 0;
            for (size_t i = idx; i < end_idx; i++) {
                total_length += strlen(units->units[i].text) + (i < end_idx - 1 ? strlen(separator) : 0);
            }

            // Allocate chunk text
            char* chunk_text = malloc(total_length + 1);
            if (!chunk_text) continue;

            // Build chunk text with separator (add separator to all units except the last one)
            chunk_text[0] = '\0';
            for (size_t i = idx; i < end_idx; i++) {
                strcat(chunk_text, units->units[i].text);
                if (i < end_idx - 1) {
                    strcat(chunk_text, separator);
                }
            }
            // For sentence splitting, add the separator at the end like Python does
            if (strlen(separator) > 0 && strcmp(separator, ".") == 0) {
                strcat(chunk_text, separator);
            }

            // Add to chunks
            if (chunks.count >= chunks.capacity) {
                chunks.capacity *= 2;
                chunks.texts = realloc(chunks.texts, chunks.capacity * sizeof(char*));
                chunks.start_lines = realloc(chunks.start_lines, chunks.capacity * sizeof(int));
                chunks.start_positions = realloc(chunks.start_positions, chunks.capacity * sizeof(int));
                chunks.end_positions = realloc(chunks.end_positions, chunks.capacity * sizeof(int));

                if (!chunks.texts || !chunks.start_lines || !chunks.start_positions || !chunks.end_positions) {
                    free(chunk_text);
                    break;
                }
            }

            chunks.texts[chunks.count] = chunk_text;
            chunks.start_lines[chunks.count] = units->units[idx].line;
            chunks.start_positions[chunks.count] = units->units[idx].start_pos;
            chunks.end_positions[chunks.count] = units->units[end_idx - 1].start_pos + units->units[end_idx - 1].length;
            chunks.count++;
        }
    }

    return chunks;
}

// Free CDocument memory
static void free_cdocument(CDocument* cdoc) {
    if (cdoc->text) free(cdoc->text);
    if (cdoc->id) free(cdoc->id);
    if (cdoc->parent_doc_id) free(cdoc->parent_doc_id);
    if (cdoc->bm25_indexes) {
        for (size_t i = 0; i < cdoc->bm25_count; i++) {
            if (cdoc->bm25_indexes[i].token) {
                free(cdoc->bm25_indexes[i].token);
            }
        }
        free(cdoc->bm25_indexes);
    }
}


// Core document processing logic (GIL released)
// Pure GIL-free document processing - CPU intensive work only
static chunk_result_t* _process_document_impl(CDocument* cdoc, int chunk_size, int chunk_overlap, const char* separator, int* result_count) {
    if (!cdoc || !cdoc->text || chunk_size <= 0) {
        return NULL;
    }

    // First, split text into units based on separator
    text_units_t units = split_text_into_units(cdoc->text, separator);
    debug_printf("split_text_into_units count: %d\n", units.count);
    if (units.count == 0) {
        return NULL;
    }

    // Then merge units into chunks with proper overlap
    chunks_t chunks = merge_units_to_chunks(&units, chunk_size, chunk_overlap, separator);
    if (chunks.count == 0) {
        free_text_units(&units);
        return NULL;
    }

    // Allocate results array
    chunk_result_t* results = calloc(chunks.count, sizeof(chunk_result_t));
    if (!results) {
        free_text_units(&units);
        free_chunks(&chunks);
        return NULL;
    }

    // Convert chunks to results
    for (size_t i = 0; i < chunks.count; i++) {
        chunk_result_t* result = &results[i];

        // Create line-numbered text
        char* line_numbered_text = create_line_numbered_text(chunks.texts[i], chunks.start_lines[i]);

        // Calculate BM25 indexes for this chunk
        int chunk_end = chunks.end_positions[i];
        int chunk_end_line = 0, chunk_end_col = 0;
        get_line_column(cdoc->text, chunk_end, &chunk_end_line, &chunk_end_col);

        // Count relevant BM25 indexes
        size_t relevant_count = 0;
        for (size_t j = 0; j < cdoc->bm25_count; j++) {
            int token_line = cdoc->bm25_indexes[j].line;
            int token_col = cdoc->bm25_indexes[j].col;

            if (token_line < chunk_end_line ||
                (token_line == chunk_end_line && token_col <= chunk_end_col)) {
                relevant_count++;
            }
        }

        // Allocate BM25 data array
        if (relevant_count > 0) {
            result->bm25_data = calloc(relevant_count, sizeof(result->bm25_data[0]));
            if (result->bm25_data) {
                // Copy relevant BM25 data
                size_t idx = 0;
                for (size_t j = 0; j < cdoc->bm25_count; j++) {
                    int token_line = cdoc->bm25_indexes[j].line;
                    int token_col = cdoc->bm25_indexes[j].col;

                    if (token_line < chunk_end_line ||
                        (token_line == chunk_end_line && token_col <= chunk_end_col)) {
                        result->bm25_data[idx].token = safe_strdup(cdoc->bm25_indexes[j].token);
                        result->bm25_data[idx].line = token_line;
                        result->bm25_data[idx].col = token_col;
                        idx++;
                    }
                }
                result->bm25_data_count = relevant_count;
            }
        }

        // Fill in result data
        result->text = safe_strdup(line_numbered_text ? line_numbered_text : chunks.texts[i]);
        result->original_text = safe_strdup(chunks.texts[i]);
        result->bytes_processed = strlen(chunks.texts[i]);
        result->start_line = chunks.start_lines[i];
        result->chunk_idx = i;
        result->relevant_count = relevant_count;
        result->doc_id = safe_strdup(cdoc->id);
        result->parent_doc_id = safe_strdup(cdoc->parent_doc_id);

        // Cleanup
        if (line_numbered_text) free(line_numbered_text);
    }

    *result_count = chunks.count;

    // Cleanup temporary data
    free_text_units(&units);
    free_chunks(&chunks);

    return results;
}

// Note: process_chunk function has been removed in favor of process_document
// which provides proper separator-based splitting that matches TextSplitter behavior

// Helper function to get separator based on split_by type
static const char* get_separator(const char* split_by) {
    if (!split_by) return " "; // default to word

    if (strcmp(split_by, "word") == 0) return " ";
    if (strcmp(split_by, "sentence") == 0) return ".";
    if (strcmp(split_by, "passage") == 0) return "\n\n";
    if (strcmp(split_by, "page") == 0) return "\f";
    if (strcmp(split_by, "token") == 0) return ""; // token splitting handled differently

    return " "; // default to word
}

static PyObject* process_document(PyObject* self, PyObject* args) {
    PyObject* doc_dict;
    int chunk_size;
    int chunk_overlap;
    const char* split_by = "word"; // default
    PyObject* separators_dict = NULL;

    // Parse arguments with GIL held - now supports optional split_by and separators
    if (!PyArg_ParseTuple(args, "Oii|sO", &doc_dict, &chunk_size, &chunk_overlap, &split_by, &separators_dict)) {
        return NULL;
    }

    if (!PyDict_Check(doc_dict)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a dictionary");
        return NULL;
    }

    // Determine separator to use
    const char* separator = get_separator(split_by);

    // If custom separators provided, try to get separator from dict
    if (separators_dict && PyDict_Check(separators_dict)) {
        PyObject* sep_obj = PyDict_GetItemString(separators_dict, split_by);
        if (sep_obj && PyUnicode_Check(sep_obj)) {
            separator = PyUnicode_AsUTF8(sep_obj);
        }
    }
    debug_printf("Using separator: %s\n", separator);

    // Parse document with GIL held
    CDocument cdoc = {0};
    if (!parse_python_document(doc_dict, &cdoc)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse document");
        return NULL;
    }

    if (!cdoc.text || chunk_size <= 0) {
        free_cdocument(&cdoc);
        PyErr_SetString(PyExc_ValueError, "Invalid document or chunk size");
        return NULL;
    }

    debug_printf("chunk_size: %d, chunk_overlap: %d\n", chunk_size, chunk_overlap);

    // Create result list with GIL held
    PyObject* result_list = PyList_New(0);
    if (!result_list) {
        free_cdocument(&cdoc);
        return NULL;
    }

    // Release GIL for the heavy CPU-bound processing
    int result_count = 0;
    chunk_result_t* results = NULL;

    Py_BEGIN_ALLOW_THREADS

    // Pure C processing - no Python API calls
    results = _process_document_impl(&cdoc, chunk_size, chunk_overlap, separator, &result_count);

    Py_END_ALLOW_THREADS

    // Convert results to Python objects (with GIL held)
    if (results) {
        for (int i = 0; i < result_count; i++) {
            PyObject* chunk_dict = convert_chunk_result(&results[i]);
            if (chunk_dict) {
                PyList_Append(result_list, chunk_dict);
                Py_DECREF(chunk_dict);
            }
        }
        free_chunk_results(results, result_count);
    }

    free_cdocument(&cdoc);
    return result_list;
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

// Python module definition
static PyMethodDef FastChunkProcessorMethods[] = {
    {"process_document", process_document, METH_VARARGS,
     "Process an entire document with separator-based splitting and GIL release"},
    {"set_debug_mode", set_debug_mode, METH_VARARGS,
     "Enable or disable debug mode"},
    {"get_debug_mode", get_debug_mode, METH_VARARGS,
     "Get current debug mode"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastchunkprocessormodule = {
    PyModuleDef_HEAD_INIT,
    "fast_chunk_processor",
    "Fast chunk processing with GIL release for parallel execution",
    -1,
    FastChunkProcessorMethods
};

PyMODINIT_FUNC PyInit_fast_chunk_processor(void) {
    return PyModule_Create(&fastchunkprocessormodule);
}