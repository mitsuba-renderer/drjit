#include "common.h"
#include <algorithm>
#include <stdio.h>
#include <stdarg.h>

LogLevel log_level = LogLevel::Trace;

void ad_fail(const char *fmt, ...) {
    fprintf(stderr, "\n\nCritical failure in Enoki AD backend: ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

void ad_log(LogLevel level, const char *fmt, ...) {
    if (unlikely(level <= log_level)) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fputc('\n', stderr);
    }
}

void* malloc_check(size_t size) {
    void *ptr = malloc(size);
    if (unlikely(!ptr))
        ad_fail("malloc_check(): failed to allocate %zu bytes!", size);
    return ptr;
}

Buffer::Buffer(size_t size)
    : m_start(nullptr), m_cur(nullptr), m_end(nullptr) {
    m_start = (char *) malloc_check(size);
    m_end = m_start + size;
    clear();
}

Buffer::~Buffer() {
    free(m_start);
}

size_t Buffer::fmt(const char *format, ...) {
    size_t written;
    do {
        size_t size = m_end - m_cur;
        va_list args;
        va_start(args, format);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);

    return written;
}

void Buffer::expand(size_t minval) {
    size_t old_alloc_size = m_end - m_start,
           new_alloc_size = 2 * old_alloc_size + minval,
           used_size      = m_cur - m_start,
           copy_size      = std::min(used_size + 1, old_alloc_size);

    char *tmp = (char *) malloc_check(new_alloc_size);
    memcpy(tmp, m_start, copy_size);
    free(m_start);

    m_start = tmp;
    m_end = m_start + new_alloc_size;
    m_cur = m_start + used_size;
}
