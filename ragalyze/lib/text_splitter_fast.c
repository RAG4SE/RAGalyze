#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>
#include "ragalyze_common.h"

// 1. 先定义辅助结构体：处理复杂字段（元数据字典、向量列表）
// 元数据字典（简化版：key-value 字符串对，动态数组）
typedef struct {
    char** keys;       // 所有 key 的数组（每个 key 是 char*）
    char** values;     // 对应 value 的数组（每个 value 是 char*）
    size_t count;      // key-value 对的数量
} CMetadata;

// BM25 索引结构
typedef struct {
    char* token;       // BM25 token
    int line;          // 行号 (1-based)
    int col;           // 列号 (1-based)
} BM25Index;

// 2. 核心 CDocument 结构体
typedef struct {
    // 1. 基础文本（Python str → C char*，需用 malloc 分配内存）
    char* text;
    size_t text_len;   // 文本长度（可选，方便内存管理）

    // 2. 元数据（Python Dict → 自定义 CMetadata 结构体）
    CMetadata* meta_data;  // 可为 NULL（对应 Python None）

    // 3. 向量（Python List[float] → C float 数组）
    float* vector;
    size_t vector_len;     // 向量元素个数（0 表示空列表）

    // 4. UUID 相关（Python str/UUID → C uuid_t 或 char*）
    // 推荐用 char* 存储 UUID 字符串（如 "550e8400-e29b-41d4-a716-446655440000"），更易处理
    char* id;
    char* parent_doc_id;   // 可为 NULL（对应 Python None）

    // 5. 整数/浮点数字段（Python int/float → C 基础类型）
    int order;             // 可为 -1（对应 Python None）
    float score;           // 可为 NAN（对应 Python None）
    int estimated_num_tokens;  // 可为 -1（对应 Python None）

    // 6. BM25 索引（用于文本分割时的智能边界处理）
    BM25Index* bm25_indexes;
    size_t bm25_count;

} CDocument;

// 文本分割结果结构
typedef struct {
    char* text;                    // 分割后的文本
    char* line_numbered_text;      // 带行号的文本
    int start_line;                // 起始行号
    int start_pos;                 // 起始位置
    int length;                    // 长度
    BM25Index* bm25_indexes;       // 属于此分片的 BM25 索引
    size_t bm25_count;             // BM25 索引数量
} TextSplit;

// ===== 辅助函数 =====

// Use KMP algorithm from common header

// 创建带行号的文本
static char* create_line_numbered_text(const char* text, int start_line) {
    if (!text) return NULL;

    int line_count = 1;
    for (int i = 0; i < strlen(text); i++) {
        if (text[i] == '\n') line_count++;
    }

    int text_len = strlen(text);
    int buffer_size = text_len + line_count * 20;

    char* result = malloc(buffer_size);
    if (!result) return NULL;

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

// Use UUID generation from common header

// ===== Python-C 数据转换函数 =====

// 解析 Python 字典到 CDocument 结构
static int parse_python_document(PyObject* doc_obj, CDocument* cdoc) {
    memset(cdoc, 0, sizeof(CDocument));

    // 检查是否为字典
    if (!PyDict_Check(doc_obj)) {
        return 0;
    }

    // 获取 text
    PyObject* text_obj = PyDict_GetItemString(doc_obj, "text");
    if (!text_obj || !PyUnicode_Check(text_obj)) {
        return 0;
    }
    cdoc->text = strdup(PyUnicode_AsUTF8(text_obj));
    cdoc->text_len = strlen(cdoc->text);

    // 获取 id
    PyObject* id_obj = PyDict_GetItemString(doc_obj, "id");
    if (id_obj && PyUnicode_Check(id_obj)) {
        cdoc->id = strdup(PyUnicode_AsUTF8(id_obj));
    } else {
        cdoc->id = ragalyze_generate_uuid();
    }

    // 获取 order
    PyObject* order_obj = PyDict_GetItemString(doc_obj, "order");
    if (order_obj && PyLong_Check(order_obj)) {
        cdoc->order = PyLong_AsLong(order_obj);
    } else {
        cdoc->order = -1;
    }

    // 获取 parent_doc_id
    PyObject* parent_obj = PyDict_GetItemString(doc_obj, "parent_doc_id");
    if (parent_obj && PyUnicode_Check(parent_obj)) {
        cdoc->parent_doc_id = strdup(PyUnicode_AsUTF8(parent_obj));
    } else {
        cdoc->parent_doc_id = NULL;
    }

    // 获取 score
    PyObject* score_obj = PyDict_GetItemString(doc_obj, "score");
    if (score_obj && PyFloat_Check(score_obj)) {
        cdoc->score = PyFloat_AsDouble(score_obj);
    } else {
        cdoc->score = NAN;
    }

    // 获取 estimated_num_tokens
    PyObject* tokens_obj = PyDict_GetItemString(doc_obj, "estimated_num_tokens");
    if (tokens_obj && PyLong_Check(tokens_obj)) {
        cdoc->estimated_num_tokens = PyLong_AsLong(tokens_obj);
    } else {
        cdoc->estimated_num_tokens = -1;
    }

    // 获取 vector
    PyObject* vector_obj = PyDict_GetItemString(doc_obj, "vector");
    if (vector_obj && PyList_Check(vector_obj)) {
        cdoc->vector_len = PyList_Size(vector_obj);
        if (cdoc->vector_len > 0) {
            cdoc->vector = malloc(cdoc->vector_len * sizeof(float));
            for (size_t i = 0; i < cdoc->vector_len; i++) {
                PyObject* item = PyList_GetItem(vector_obj, i);
                if (PyFloat_Check(item)) {
                    cdoc->vector[i] = PyFloat_AsDouble(item);
                } else {
                    cdoc->vector[i] = 0.0f;
                }
            }
        }
    }

    // 获取 metadata 和 BM25 indexes
    PyObject* meta_obj = PyDict_GetItemString(doc_obj, "meta_data");
    if (meta_obj && PyDict_Check(meta_obj)) {
        PyObject* bm25_obj = PyDict_GetItemString(meta_obj, "bm25_indexes");
        if (bm25_obj && PyList_Check(bm25_obj)) {
            cdoc->bm25_count = PyList_Size(bm25_obj);
            if (cdoc->bm25_count > 0) {
                cdoc->bm25_indexes = malloc(cdoc->bm25_count * sizeof(BM25Index));
                for (size_t i = 0; i < cdoc->bm25_count; i++) {
                    PyObject* bm25_item = PyList_GetItem(bm25_obj, i);
                    if (PyTuple_Check(bm25_item) && PyTuple_Size(bm25_item) >= 3) {
                        PyObject* token_obj = PyTuple_GetItem(bm25_item, 0);
                        PyObject* line_obj = PyTuple_GetItem(bm25_item, 1);
                        PyObject* col_obj = PyTuple_GetItem(bm25_item, 2);

                        if (PyUnicode_Check(token_obj)) {
                            cdoc->bm25_indexes[i].token = strdup(PyUnicode_AsUTF8(token_obj));
                        }
                        if (PyLong_Check(line_obj)) {
                            cdoc->bm25_indexes[i].line = PyLong_AsLong(line_obj);
                        }
                        if (PyLong_Check(col_obj)) {
                            cdoc->bm25_indexes[i].col = PyLong_AsLong(col_obj);
                        }
                    }
                }
            }
        }
    }

    return 1;
}

// 释放 CDocument 内存
static void free_cdocument(CDocument* cdoc) {
    if (cdoc->text) free(cdoc->text);
    if (cdoc->id) free(cdoc->id);
    if (cdoc->parent_doc_id) free(cdoc->parent_doc_id);
    if (cdoc->vector) free(cdoc->vector);
    if (cdoc->bm25_indexes) {
        for (size_t i = 0; i < cdoc->bm25_count; i++) {
            free(cdoc->bm25_indexes[i].token);
        }
        free(cdoc->bm25_indexes);
    }
}

static int find_relevant_bm25_indexes(TextSplit* split, CDocument* doc, int chunk_end, size_t* prev_bm25_index_idx) {
    // Convert chunk position to line numbers (like Python implementation)
    int current_line_0based = count_newlines_up_to(doc->text, chunk_end);

    // Find the current column position (0-based)
    int current_col_0based = 0;
    int last_newline_pos = -1;
    for (int i = 0; i < chunk_end; i++) {
        if (doc->text[i] == '\n') {
            last_newline_pos = i;
        }
    }
    current_col_0based = chunk_end - last_newline_pos - 1;
    if (current_col_0based < 0) current_col_0based = 0;

    split->bm25_count = 0;
    split->bm25_indexes = NULL;

    if (doc->bm25_count == 0) {
        return 1; // Success - no BM25 indexes to process
    }

    // Count relevant BM25 indexes (like Python implementation)
    size_t relevant_count = 0;
    for (size_t i = *prev_bm25_index_idx; i < doc->bm25_count; i++) {
        int token_line = doc->bm25_indexes[i].line;
        int token_col = doc->bm25_indexes[i].col;

        // Convert 1-based BM25 index to 0-based for comparison (like Python)
        int token_line_0based = token_line - 1;
        int token_col_0based = token_col - 1;

        if (token_line_0based < current_line_0based ||
            (token_line_0based == current_line_0based && token_col_0based <= current_col_0based)) {
            relevant_count++;
        } else {
            // Update prev_bm25_index_idx for next chunk (like Python)
            *prev_bm25_index_idx = i;
            break;
        }
    }

    if (relevant_count > 0) {
        split->bm25_indexes = malloc(relevant_count * sizeof(BM25Index));
        if (!split->bm25_indexes) {
            return 0; // Memory allocation failed
        }

        size_t idx = 0;
        for (size_t i = *prev_bm25_index_idx; i < *prev_bm25_index_idx + relevant_count; i++) {
            if (i >= doc->bm25_count) break;

            split->bm25_indexes[idx].token = strdup(doc->bm25_indexes[i].token);
            if (!split->bm25_indexes[idx].token) {
                // Cleanup allocated memory on failure
                for (size_t j = 0; j < idx; j++) {
                    free(split->bm25_indexes[j].token);
                }
                free(split->bm25_indexes);
                split->bm25_indexes = NULL;
                return 0;
            }

            split->bm25_indexes[idx].line = doc->bm25_indexes[i].line;
            split->bm25_indexes[idx].col = doc->bm25_indexes[i].col;
            idx++;
        }

        split->bm25_count = relevant_count;
        return 1;
    }

    return 1; // Success - no relevant BM25 indexes found
}

// 主要的文档分割函数 - 模拟 MyTextSplitter.call 的逻辑
static TextSplit* split_document_fast(CDocument* doc, int chunk_size, int chunk_overlap, int* split_count) {
    if (!doc->text || doc->text_len == 0) {
        *split_count = 0;
        return NULL;
    }

    int max_splits = (doc->text_len / chunk_size) + 2;
    TextSplit* splits = malloc(max_splits * sizeof(TextSplit));
    if (!splits) {
        *split_count = 0;
        return NULL;
    }

    int split_idx = 0;
    int current_pos = 0;
    int pre_chunk_extension_count = 0;
    size_t prev_bm25_index_idx = 0; // Track BM25 index position like Python implementation

    while (current_pos < doc->text_len && split_idx < max_splits) {
        int chunk_end = current_pos + chunk_size;

        // 调整分片结尾以避免断词
        if (chunk_end < doc->text_len) {
            int space_pos = -1;
            for (int i = chunk_end; i > current_pos; i--) {
                if (doc->text[i] == ' ' || doc->text[i] == '\n' || doc->text[i] == '\t') {
                    space_pos = i;
                    break;
                }
            }
            if (space_pos > current_pos) {
                chunk_end = space_pos;
            }
        }

        if (chunk_end > doc->text_len) chunk_end = doc->text_len;

        // 应用 pre_chunk_extension_count
        if (pre_chunk_extension_count > 0) {
            chunk_end += pre_chunk_extension_count;
            if (chunk_end > doc->text_len) chunk_end = doc->text_len;
            pre_chunk_extension_count = 0;
        }

        // 检查是否需要扩展分片以包含完整的 BM25 tokens
        if (doc->bm25_count > 0) {
            int chunk_line, chunk_col;
            get_line_column(doc->text, chunk_end, &chunk_line, &chunk_col);

            // 查找跨越分片边界的 BM25 tokens
            for (size_t i = 0; i < doc->bm25_count; i++) {
                int token_line = doc->bm25_indexes[i].line;
                int token_col = doc->bm25_indexes[i].col;
                char* token = doc->bm25_indexes[i].token;
                int token_len = strlen(token);

                // 检查 token 是否在分片边界的中间
                if (token_line == chunk_line && token_col < chunk_col &&
                    token_col + token_len > chunk_col) {
                    // 为下一个分片存储扩展
                    pre_chunk_extension_count = (token_col + token_len) - chunk_col;
                    break;
                }
            }
        }
        int chunk_len = chunk_end - current_pos;
        if (chunk_len > 0) {
            splits[split_idx].text = malloc(chunk_len + 1);
            if (splits[split_idx].text) {
                strncpy(splits[split_idx].text, doc->text + current_pos, chunk_len);
                splits[split_idx].text[chunk_len] = '\0';
                splits[split_idx].start_line = count_newlines_up_to(doc->text, current_pos);
                splits[split_idx].start_pos = current_pos;
                splits[split_idx].length = chunk_len;

                splits[split_idx].line_numbered_text = create_line_numbered_text(
                    splits[split_idx].text, splits[split_idx].start_line);

                if (!find_relevant_bm25_indexes(&splits[split_idx], doc, chunk_end)) {
                    // BM25 索引查找失败，清理当前分片并继续
                    free(splits[split_idx].text);
                    if (splits[split_idx].line_numbered_text) {
                        free(splits[split_idx].line_numbered_text);
                    }
                    // 由于 find_relevant_bm25_indexes 内部已经清理了 bm25_indexes，
                    // 这里我们只需要清理当前分片并继续下一个
                    continue;
                }

                split_idx++;
            }
        }

        current_pos = chunk_end - chunk_overlap;
        if (current_pos >= doc->text_len) break;
    }
    *split_count = split_idx;
    return splits;
}

// ===== Python 接口函数 =====

// 主要的 Python 包装函数 - 模拟 MyTextSplitter.call
static PyObject* fast_split(PyObject* self, PyObject* args) {
    PyObject* documents;
    int chunk_size = 1000;
    int chunk_overlap = 200;

    if (!PyArg_ParseTuple(args, "O|ii", &documents, &chunk_size, &chunk_overlap)) {
        return NULL;
    }

    if (!PyList_Check(documents)) {
        PyErr_SetString(PyExc_TypeError, "Input should be a list of Documents");
        return NULL;
    }

    int doc_count = PyList_Size(documents);
    PyObject* result = PyList_New(0);

    for (int doc_idx = 0; doc_idx < doc_count; doc_idx++) {
        PyObject* doc_obj = PyList_GetItem(documents, doc_idx);

        // 解析文档
        CDocument cdoc = {0};
        if (!parse_python_document(doc_obj, &cdoc)) {
            PyErr_SetString(PyExc_ValueError, "Failed to parse document");
            return NULL;
        }

        if (!cdoc.text) {
            PyErr_SetString(PyExc_ValueError, "Document text cannot be None");
            free_cdocument(&cdoc);
            return NULL;
        }

        // 分割文档
        int split_count = 0;
        TextSplit* splits = split_document_fast(&cdoc, chunk_size, chunk_overlap, &split_count);

        // 将分片转换为 Document 对象
        char* prev_doc_id = NULL;
        for (int i = 0; i < split_count; i++) {
            // 创建新的 Document 字典
            PyObject* new_doc = PyDict_New();

            // 生成文档 ID
            char* doc_id = ragalyze_generate_uuid();

            // 设置文本（带行号）
            PyObject* text_obj = PyUnicode_FromString(
                splits[i].line_numbered_text ? splits[i].line_numbered_text : splits[i].text);
            PyDict_SetItemString(new_doc, "text", text_obj);
            Py_DECREF(text_obj);

            // 创建元数据
            PyObject* meta_dict = PyDict_New();
            PyObject* start_line_obj = PyLong_FromLong(splits[i].start_line);
            PyDict_SetItemString(meta_dict, "start_line", start_line_obj);
            Py_DECREF(start_line_obj);

            // 添加 BM25 索引到元数据
            if (splits[i].bm25_count > 0) {
                PyObject* bm25_list = PyList_New(splits[i].bm25_count);
                for (size_t j = 0; j < splits[i].bm25_count; j++) {
                    PyObject* bm25_tuple = PyTuple_New(3);
                    PyTuple_SetItem(bm25_tuple, 0, PyUnicode_FromString(splits[i].bm25_indexes[j].token));
                    PyTuple_SetItem(bm25_tuple, 1, PyLong_FromLong(splits[i].bm25_indexes[j].line));
                    PyTuple_SetItem(bm25_tuple, 2, PyLong_FromLong(splits[i].bm25_indexes[j].col));
                    PyList_SetItem(bm25_list, j, bm25_tuple);
                }
                PyDict_SetItemString(meta_dict, "bm25_indexes", bm25_list);
                Py_DECREF(bm25_list);
            }

            // 添加原始文本到元数据
            PyObject* original_text_obj = PyUnicode_FromString(splits[i].text);
            PyDict_SetItemString(meta_dict, "original_text", original_text_obj);
            Py_DECREF(original_text_obj);

            // 设置 prev/next 关系
            if (prev_doc_id) {
                PyObject* prev_id_obj = PyUnicode_FromString(prev_doc_id);
                PyDict_SetItemString(meta_dict, "prev_doc_id", prev_id_obj);
                Py_DECREF(prev_id_obj);
            }

            if (i < split_count - 1) {
                PyDict_SetItemString(meta_dict, "next_doc_id", Py_None);
                Py_INCREF(Py_None);
            }

            PyDict_SetItemString(new_doc, "meta_data", meta_dict);
            Py_DECREF(meta_dict);

            // 设置其他 Document 属性
            PyDict_SetItemString(new_doc, "parent_doc_id",
                PyUnicode_FromString(cdoc.id ? cdoc.id : ""));
            PyDict_SetItemString(new_doc, "order", PyLong_FromLong(i));
            PyDict_SetItemString(new_doc, "id", PyUnicode_FromString(doc_id));
            PyDict_SetItemString(new_doc, "vector", PyList_New(0));

            PyList_Append(result, new_doc);
            Py_DECREF(new_doc);

            // 清理
            free(prev_doc_id);
            prev_doc_id = strdup(doc_id);
            free(doc_id);

            // 释放分片内存
            free(splits[i].text);
            if (splits[i].line_numbered_text) free(splits[i].line_numbered_text);
            if (splits[i].bm25_indexes) {
                for (size_t j = 0; j < splits[i].bm25_count; j++) {
                    free(splits[i].bm25_indexes[j].token);
                }
                free(splits[i].bm25_indexes);
            }
        }

        free(prev_doc_id);
        free(splits);
        free_cdocument(&cdoc);
    }

    return result;
}

// Python 模块定义
static PyMethodDef TextSplitterMethods[] = {
    {"fast_split", fast_split, METH_VARARGS,
     "Fast document splitting with BM25 support - mimics MyTextSplitter.call"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef textsplittermodule = {
    PyModuleDef_HEAD_INIT,
    "text_splitter_fast",
    "Fast text splitter implementation in C",
    -1,
    TextSplitterMethods
};

PyMODINIT_FUNC PyInit_text_splitter_fast(void) {
    return PyModule_Create(&textsplittermodule);
}