#ifndef RAGALYZE_COMMON_H
#define RAGALYZE_COMMON_H

#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>

// Platform-specific UUID includes
#if defined(__APPLE__)
#include <uuid/uuid.h>
#elif defined(__linux__)
#include <uuid/uuid.h>
#elif defined(_WIN32)
#include <windows.h>
#include <rpc.h>
#pragma comment(lib, "rpcrt4.lib")
#else
#include <time.h>
#endif

// Enhanced assert function with cleanup and debugging
static inline void ragalyze_assert(int condition, const char* message, void* cleanup_ptr, const char* file, int line) {
    if (!condition) {
        printf("ASSERTION FAILED at %s:%d: %s\n", file, line, message);
        if (cleanup_ptr) {
            free(cleanup_ptr);
        }
        exit(1);
    }
}

// Macros for easier usage with automatic file/line info
#define RAGALYZE_ASSERT(condition, message, cleanup_ptr) \
    ragalyze_assert(condition, message, cleanup_ptr, __FILE__, __LINE__)

#define ASSERT_ALLOC(ptr, cleanup_ptr) \
    RAGALYZE_ASSERT((ptr) != NULL, "Memory allocation failed", cleanup_ptr)

// Safe string duplication with memory checking
static inline char* ragalyze_strdup(const char* str) {
    if (!str) return NULL;

    size_t len = strlen(str);
    char* result = malloc(len + 1);
    if (!result) return NULL;

    memcpy(result, str, len);
    result[len] = '\0';
    return result;
}

// Safe string concatenation with memory checking
static inline char* ragalyze_strconcat(const char* str1, const char* str2) {
    if (!str1 || !str2) {
        return NULL;
    }

    size_t len1 = strlen(str1);
    size_t len2 = strlen(str2);
    size_t total_len = len1 + len2;

    char* result = malloc(total_len + 1);
    if (!result) {
        return NULL;
    }

    memcpy(result, str1, len1);
    memcpy(result + len1, str2, len2);
    result[total_len] = '\0';

    return result;
}

// Function to check if a string contains only whitespace
static inline int ragalyze_is_whitespace(const char* str) {
    if (!str) return 1;
    while (*str) {
        if (!isspace((unsigned char)*str)) {
            return 0;
        }
        str++;
    }
    return 1;
}

// Cross-platform UUID generation
static inline char* ragalyze_generate_uuid() {
    char* uuid_str = malloc(37);
    if (!uuid_str) return NULL;

#if defined(__APPLE__) || defined(__linux__)
    // macOS 和 Linux 使用 libuuid
    uuid_t uuid;
    uuid_generate(uuid);
    uuid_unparse(uuid, uuid_str);
#elif defined(_WIN32)
    // Windows 使用 RPC UUID
    UUID uuid;
    UuidCreate(&uuid);
    unsigned char* str;
    UuidToStringA(&uuid, &str);
    strcpy(uuid_str, (char*)str);
    RpcStringFreeA(&str);
#else
    // 其他平台使用时间戳+随机数
    time_t now = time(NULL);
    unsigned int random_num = rand();
    snprintf(uuid_str, 37, "%08lx-%04x-%04x-%04x-%012x",
             (long)now,
             (unsigned short)(random_num & 0xFFFF),
             (unsigned short)((random_num >> 16) & 0xFFFF),
             (unsigned short)(random_num & 0xFFFF),
             (unsigned long)random_num);
#endif

    return uuid_str;
}

// KMP algorithm for fast substring search
static inline void kmp_compute_lps(const char* pattern, int* lps) {
    int len = strlen(pattern);
    int i = 1, j = 0;
    lps[0] = 0;

    while (i < len) {
        if (pattern[i] == pattern[j]) {
            lps[i++] = ++j;
        } else {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                lps[i++] = 0;
            }
        }
    }
}

static inline int kmp_search(const char* text, const char* pattern, int start_pos) {
    if (!text || !pattern || strlen(pattern) == 0) return -1;

    int text_len = strlen(text);
    int pattern_len = strlen(pattern);

    if (text_len < pattern_len) return -1;

    int* lps = malloc(pattern_len * sizeof(int));
    kmp_compute_lps(pattern, lps);

    int i = start_pos, j = 0;
    while (i < text_len) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }

        if (j == pattern_len) {
            free(lps);
            return i - j;
        } else if (i < text_len && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }

    free(lps);
    return -1;
}

// Helper functions for text processing
static inline int count_newlines_up_to(const char* text, int pos) {
    if (!text) {
        return 0;
    }

    int text_len = strlen(text);
    int count = 0;
    for (int i = 0; i < pos && i < text_len; i++) {
        if (text[i] == '\n') count++;
    }

    return count;
}

static inline void get_line_column(const char* text, int pos, int* line, int* col) {
    if (!text) {
        *line = 1;
        *col = 1;
        return;
    }

    *line = 1;
    *col = 1;

    int text_len = strlen(text);
    for (int i = 0; i < pos && i < text_len; i++) {
        if (text[i] == '\n') {
            (*line)++;
            *col = 1;
        } else {
            (*col)++;
        }
    }
}

// Find safe UTF-8 boundary (equivalent to Python version)
static inline int find_safe_utf8_boundary(const char* data, int position, int data_len) {
    if (position >= data_len) {
        return data_len;
    }

    if (position <= 0) {
        return 0;
    }

    // Look backward up to 4 bytes to find a valid UTF-8 start
    for (int i = 0; i < ((position + 1) < 4 ? (position + 1) : 4); i++) {
        int check_pos = position - i;
        if (check_pos < 0) {
            break;
        }

        unsigned char byte = (unsigned char)data[check_pos];

        // Check if this is a valid UTF-8 start byte
        if (byte <= 0x7F) {  // ASCII (single byte)
            return check_pos;
        }
        else if ((byte >= 0xC2 && byte <= 0xDF) || (byte >= 0xE0 && byte <= 0xEF) || (byte >= 0xF0 && byte <= 0xF7)) {
            // Try to validate the complete UTF-8 sequence
            int end;
            if (byte >= 0xC2 && byte <= 0xDF) {
                end = check_pos + 2;
            }
            else if (byte >= 0xE0 && byte <= 0xEF) {
                end = check_pos + 3;
            }
            else {  // 0xF0 <= byte <= 0xF7
                end = check_pos + 4;
            }

            // Ensure we don't go beyond data length
            if (end > data_len) {
                end = data_len;
            }

            // Simple validation: check if continuation bytes follow
            int valid = 1;
            for (int j = check_pos + 1; j < end; j++) {
                if (j >= data_len || ((unsigned char)data[j] & 0xC0) != 0x80) {
                    valid = 0;
                    break;
                }
            }

            if (valid) {
                return check_pos;
            }
        }
    }

    return 0;
}

// Safe UTF-8 string copy that respects character boundaries
static inline char* safe_utf8_strncpy(const char* src, int src_len, int copy_len) {

    ragalyze_assert(src_len >= 0, "Invalid source length", NULL, __FILE__, __LINE__);
    ragalyze_assert(copy_len >= 0, "Invalid copy length", NULL, __FILE__, __LINE__);
    ragalyze_assert(src_len >= copy_len, "Copy length exceeds source length", NULL, __FILE__, __LINE__);
    ragalyze_assert(src != NULL, "NULL source pointer", NULL, __FILE__, __LINE__);
    // Find safe boundary
    int safe_len = find_safe_utf8_boundary(src, copy_len, src_len);
    char* result = malloc(safe_len + 1);
    ragalyze_assert(result != NULL, "Memory allocation failed", NULL, __FILE__, __LINE__);

    strncpy(result, src, safe_len);
    result[safe_len] = '\0';
    return result;
}

// Safe memory allocation with debug info
static inline void* ragalyze_malloc(size_t size, const char* var_name) {
    void* ptr = malloc(size);
    ragalyze_assert(ptr != NULL, "Memory allocation failed", NULL, __FILE__, __LINE__);
    return ptr;
}

#define RAGALYZE_MALLOC(size) ragalyze_malloc(size, #size)

// Structure to hold chunk data for GIL-safe processing
typedef struct {
    char* text;
    int bytes_processed;
    int start_line;
    int chunk_idx;
    size_t relevant_count;
    void* cdoc;  // CDocument pointer (opaque)
} chunk_data_t;

// Safe string duplication with NULL check
static inline char* safe_strdup(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    char* result = malloc(len + 1);
    if (result) {
        memcpy(result, str, len + 1);
    }
    return result;
}

#endif // RAGALYZE_COMMON_H