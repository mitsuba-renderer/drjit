#include <enoki/fwd.h>
#include "common.h"
#include <algorithm>
#include <stdio.h>
#include <stdarg.h>
#include <stdexcept>

LogLevel log_level = LogLevel::Info;
Buffer buffer{0};

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

namespace enoki {
    namespace detail {
        extern void ad_whos_scalar_f32();
        extern void ad_whos_scalar_f64();
#if defined(ENOKI_ENABLE_JIT)
        extern void ad_whos_cuda_f32();
        extern void ad_whos_cuda_f64();
        extern void ad_whos_llvm_f32();
        extern void ad_whos_llvm_f64();
#endif
    }

    struct PrefixEntry {
        PrefixEntry *prev;
        char *value = nullptr;
    };

#if !defined(_MSC_VER)
    static __thread PrefixEntry *prefix = nullptr;
#else
    static __declspec(thread) PrefixEntry *prefix = nullptr;
#endif

    ENOKI_EXPORT void ad_prefix_push(const char *value) {
        if (strchr(value, '/'))
            throw std::runtime_error(
                "ad_prefix_push(): may not contain a '/' character!");
        const char *prev = prefix ? prefix->value : "";
        size_t size = strlen(prev) + strlen(value) + 2;
        char *out = (char *) malloc(size);
        snprintf(out, size, "%s/%s", prev, value);
        prefix = new PrefixEntry{ prefix, out };
    }

    ENOKI_EXPORT void ad_prefix_pop() {
        PrefixEntry *p = prefix;
        if (p) {
            prefix = p->prev;
            free(p->value);
            delete p;
        }
    }

    const char *ad_prefix() {
        PrefixEntry *p = prefix;
        return p ? p->value : nullptr;
    }

    /// Check edge weights for NaNs/infinities?
    bool check_weights = false;

    ENOKI_EXPORT void ad_check_weights(bool value) { check_weights = value; }
    ENOKI_EXPORT void ad_check_weights_cb() { }

    ENOKI_EXPORT const char *ad_whos() {
        buffer.clear();
        buffer.put("\n");
        buffer.put("  ID      E/I Refs   Size        Label\n");
        buffer.put("  =========================================\n");
        detail::ad_whos_scalar_f32();
        detail::ad_whos_scalar_f64();
        #if defined(ENOKI_ENABLE_JIT)
            detail::ad_whos_cuda_f32();
            detail::ad_whos_cuda_f64();
            detail::ad_whos_llvm_f32();
            detail::ad_whos_llvm_f64();
        #endif
        buffer.put("  =========================================\n");
        return buffer.get();
    }

    namespace detail {
        /// Custom graph edge for implementing custom differentiable operations
        struct ENOKI_EXPORT DiffCallback {
            virtual void forward() = 0;
            virtual void backward() = 0;
            virtual ~DiffCallback();
        };

        DiffCallback::~DiffCallback() { }
    }
}
