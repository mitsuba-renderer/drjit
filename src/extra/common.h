#include <drjit-core/jit.h>
#include <drjit/array.h>

#define likely(x)   DRJIT_LIKELY(x)
#define unlikely(x) DRJIT_UNLIKELY(x)

#define ad_warn(fmt, ...) jit_log(LogLevel::Warn,   fmt, ## __VA_ARGS__)
#define ad_log(fmt, ...) jit_log(LogLevel::Debug,   fmt, ## __VA_ARGS__)
#define ad_fail(fmt, ...) jit_fail(fmt, ## __VA_ARGS__)
#define ad_raise(fmt, ...) jit_raise(fmt, ## __VA_ARGS__)

#if defined(NDEBUG)
#  define ad_assert(cond, fmt, ...) do { } while (0)
#  define ad_trace(fmt, ...) do { } while (0)
#else
#  define ad_assert(cond, fmt, ...)                                            \
      if (unlikely(!(cond)))                                                   \
          jit_fail("drjit-autodiff: assertion failure (\"%s\") in line %i: "   \
                   fmt, #cond, __LINE__, ##__VA_ARGS__);
#  define ad_trace(fmt, ...) jit_log(LogLevel::Trace, fmt, ## __VA_ARGS__)
#endif

template <typename Value>
using GenericArray = drjit::JitArray<JitBackend::None, Value>;

/// RAII helper for *unlocking* a mutex
template <typename T> class unlock_guard {
public:
    unlock_guard(T &mutex) : m_mutex(mutex) { m_mutex.unlock(); }
    ~unlock_guard() { m_mutex.lock(); }
    unlock_guard(const unlock_guard &) = delete;
    unlock_guard(unlock_guard &&) = delete;
    unlock_guard &operator=(const unlock_guard &) = delete;
    unlock_guard &operator=(unlock_guard &&) = delete;
private:
    T &m_mutex;
};
