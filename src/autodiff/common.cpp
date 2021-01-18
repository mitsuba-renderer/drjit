#include "common.h"
#include <tsl/robin_set.h>
#include <algorithm>
#include <stdio.h>
#include <stdarg.h>
#include <stdexcept>

Buffer buffer{0};

void ad_fail(const char *fmt, ...) {
    fprintf(stderr, "\n\nCritical failure in Enoki AD backend: ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}

void ad_raise(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    buffer.clear();
    buffer.put("enoki-autodiff: ");
    buffer.vfmt(fmt, args);
    va_end(args);

    throw std::runtime_error(buffer.get());
}

void ad_log(LogLevel level, const char *fmt, ...) {
    if (likely(level > log_level))
        return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
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
    if (size)
        clear();
}

Buffer::~Buffer() {
    free(m_start);
}

size_t Buffer::fmt(const char *format, ...) {
    size_t written;
    do {
        size_t size = remain();
        va_list args;
        va_start(args, format);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written + 1 < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);

    return written;
}

size_t Buffer::vfmt(const char *format, va_list args_) {
    size_t written;
    va_list args;
    do {
        size_t size = remain();
        va_copy(args, args_);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written + 1 < size)) {
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
    static __thread uint32_t flags = 0;
    static __thread PrefixEntry *prefix = nullptr;
    static __thread tsl::robin_set<int32_t> *dependencies = nullptr;
#else
    static __declspec(thread) PrefixEntry *prefix = nullptr;
    static __declspec(thread) uint32_t flags = 0;
    static __declspec(thread) tsl::robin_set<int32_t> *dependencies = nullptr;
#endif

    tsl::robin_set<int32_t> *ad_dependencies() {
        return dependencies;
    }

    ENOKI_EXPORT size_t ad_dependency_count() {
        return dependencies ? dependencies->size() : 0;
    }

    ENOKI_EXPORT void ad_add_dependency(int32_t index) {
        if (!dependencies)
            dependencies = new tsl::robin_set<int32_t>();
        ad_trace("ad_add_dependency(a%u)", index);
        dependencies->insert(index);
    }

    ENOKI_EXPORT void ad_write_dependencies(int32_t *out) {
        if (!dependencies)
            return;
        size_t ctr = 0;
        for (int32_t index : *dependencies)
            out[ctr++] = index;
    }

    ENOKI_EXPORT void ad_clear_dependencies() {
        if (dependencies) {
            delete dependencies;
            dependencies = nullptr;
        }
    }

    enum class ADFlag : uint32_t;

    ENOKI_EXPORT int ad_flag(ADFlag flag) {
        return (flags & (uint32_t) flag) ? 1 : 0;
    }

    ENOKI_EXPORT void ad_set_flag(ADFlag flag, int enable) {
        uint32_t value = flags;
        ad_trace("ad_set_flag(flag=%u, value=%i)", (uint32_t) flag, enable);

        if (enable)
            value |= (uint32_t) flag;
        else
            value &= ~(uint32_t) flag;

        flags = value;
    }

    ENOKI_EXPORT void ad_prefix_push(const char *value) {
        if (strchr(value, '/'))
            throw std::runtime_error(
                "ad_prefix_push(): may not contain a '/' character!");
        const char *prev = prefix ? prefix->value : "";
        size_t size = strlen(prev) + strlen(value) + 2;
        char *out = (char *) malloc(size);
        snprintf(out, size, "%s%s%s", prev, prefix ? "/" : "", value);
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
    size_t max_edges_per_kernel = 0;

    ENOKI_EXPORT void ad_set_max_edges_per_kernel(size_t value) {
        max_edges_per_kernel = value;
    }

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
