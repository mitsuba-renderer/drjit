#pragma once

#include <string.h>
#include <stdlib.h>
#include <enoki/fwd.h>
#include <enoki-jit/jit.h>

#if !defined(likely)
#  if !defined(_MSC_VER)
#    define likely(x)   __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#  else
#    define unlikely(x) x
#    define likely(x) x
#  endif
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
        m_start[0] = '\0';
    }

    /// Append a string to the buffer
    void put(const char *str) {
        do {
            char* cur = (char*) memccpy(m_cur, str, '\0', m_end - m_cur);

            if (likely(cur)) {
                m_cur = cur - 1;
                break;
            }

            expand();
        } while (true);
    }

    /// Append a formatted (printf-style) string to the buffer
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
    size_t fmt(const char *format, ...);

private:
    void expand(size_t amount = 2);

private:
    char *m_start, *m_cur, *m_end;
};

extern Buffer buffer;
static constexpr LogLevel Disable = LogLevel::Disable;
static constexpr LogLevel Error   = LogLevel::Error;
static constexpr LogLevel Warn    = LogLevel::Warn;
static constexpr LogLevel Info    = LogLevel::Info;
static constexpr LogLevel Debug   = LogLevel::Debug;
static constexpr LogLevel Trace   = LogLevel::Trace;

extern LogLevel log_level;

extern void ad_fail(const char *fmt, ...);
extern void ad_log(LogLevel level, const char *fmt, ...);

namespace enoki {
    extern const char *ad_prefix();
    ENOKI_EXPORT void ad_check_weights_cb();
    ENOKI_EXPORT void ad_check_weights(bool value);
}
