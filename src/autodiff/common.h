#pragma once

#include <string.h>
#include <cstdlib>
#include <cstdarg>
#include <vector>
#include <drjit/fwd.h>
#include <drjit-core/jit.h>

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
#endif

#if 0 // verbose log messages
#define ad_trace(...) ad_log(Trace, __VA_ARGS__)
constexpr LogLevel log_level = LogLevel::Trace;
#else
#define ad_trace(...)
constexpr LogLevel log_level = LogLevel::Info;
#endif

/// RAII helper for *unlocking* a mutex
template <typename T> class unlock_guard {
public:
    unlock_guard(T &mutex) : m_mutex(mutex) { m_mutex.unlock(); }
    ~unlock_guard() { m_mutex.lock(); }
    unlock_guard(const unlock_guard &) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
private:
    T &m_mutex;
};

struct Buffer {
public:
    Buffer(size_t size);

    // Disable copy/move constructor and assignment
    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;

    ~Buffer();

    const char *get() { return m_start; }

    void clear() {
        m_cur = m_start;
        if (m_start != m_end)
            m_start[0] = '\0';
    }

    template <size_t N> void put(const char (&str)[N]) {
        put(str, N - 1);
    }

    /// Append a string with the specified length
    void put(const char *str, size_t size) {
        if (unlikely(m_cur + size >= m_end))
            expand(size + 1 - remain());

        memcpy(m_cur, str, size);
        m_cur += size;
        *m_cur = '\0';
    }

    /// Append a single character to the buffer
    void putc(char c) {
        if (unlikely(m_cur + 1 >= m_end))
            expand();
        *m_cur++ = c;
        *m_cur = '\0';
    }

    /// Append multiple copies of a single character to the buffer
    void putc(char c, size_t count) {
        if (unlikely(m_cur + count >= m_end))
            expand(count + 1 - remain());
        for (size_t i = 0; i < count; ++i)
            *m_cur++ = c;
        *m_cur = '\0';
    }

    /// Append an unsigned 32 bit integer
    void put_uint32(uint32_t value) {
        const int digits = 10;
        const char *num = "0123456789";
        char buf[digits];
        int i = digits;

        do {
            buf[--i] = num[value % 10];
            value /= 10;
        } while (value);

        return put(buf + i, digits - i);
    }

    /// Append a formatted (printf-style) string to the buffer
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
    size_t fmt(const char *format, ...);
    /// Like \ref fmt, but specify arguments through a va_list.
    size_t vfmt(const char *format, va_list args_);

    size_t size() const { return m_cur - m_start; }
    size_t remain() const { return m_end - m_cur; }

private:
    void expand(size_t amount = 2);

private:
    char *m_start, *m_cur, *m_end;
};

struct UInt32Hasher {
    size_t operator()(uint32_t v) const {
        // fmix32 from MurmurHash by Austin Appleby (public domain)
        v ^= v >> 16;
        v *= 0x85ebca6b;
        v ^= v >> 13;
        v *= 0xc2b2ae35;
        v ^= v >> 16;
        return (size_t) v;
    }
};

extern Buffer buffer;
static constexpr LogLevel Disable = LogLevel::Disable;
static constexpr LogLevel Error   = LogLevel::Error;
static constexpr LogLevel Warn    = LogLevel::Warn;
static constexpr LogLevel Info    = LogLevel::Info;
static constexpr LogLevel Debug   = LogLevel::Debug;
static constexpr LogLevel Trace   = LogLevel::Trace;

#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 1, 2)))
#endif
extern void ad_fail(const char *fmt, ...);

#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 1, 2)))
#endif
extern void ad_raise(const char *fmt, ...);

#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
extern void ad_log(LogLevel level, const char *fmt, ...);

namespace drjit {
    extern const char *ad_prefix();
    DRJIT_EXPORT bool ad_enabled() noexcept;
}
