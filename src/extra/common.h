#include <drjit-core/jit.h>
#include <drjit/array.h>
#include <drjit/autodiff.h>

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

/// RAII AD Isolation helper
struct scoped_isolation_guard {
    scoped_isolation_guard(int symbolic = -1) : symbolic(symbolic) {
        ad_scope_enter(drjit::ADScope::Isolate, 0, nullptr, symbolic);
    }

    ~scoped_isolation_guard() {
        ad_scope_leave(success);
    }

    void reset() {
        ad_scope_leave(false);
        ad_scope_enter(drjit::ADScope::Isolate, 0, nullptr, symbolic);
    }

    void disarm() { success = true; }

    int symbolic = -1;
    bool success = false;
};

struct scoped_force_grad_guard {
    scoped_force_grad_guard() { value = ad_set_force_grad(1); }
    ~scoped_force_grad_guard() { ad_set_force_grad(value); }
    bool value;
};


/// RAII helper to temporarily push a mask onto the Dr.Jit mask stack
struct scoped_push_mask {
    scoped_push_mask(JitBackend backend, uint32_t index) : backend(backend) {
        jit_var_mask_push(backend, index);
    }

    ~scoped_push_mask() { jit_var_mask_pop(backend); }

    JitBackend backend;
};

/// RAII helper to temporarily record symbolic computation
struct scoped_record {
    scoped_record(JitBackend backend, const char *name = nullptr,
                  bool new_scope = false)
        : backend(backend) {
        checkpoint = jit_record_begin(backend, name);
        if (new_scope)
            scope = jit_new_scope(backend);
    }

    ~scoped_record() {
        if (is_valid())
            jit_record_end(backend, checkpoint, cleanup);
    }

    uint32_t checkpoint_and_rewind() {
        jit_set_scope(backend, scope);
        return jit_record_checkpoint(backend);
    }

    void disarm() { cleanup = false; }

    bool is_valid() const { return checkpoint != (uint32_t)-1; }

    JitBackend backend;
    uint32_t checkpoint, scope;
    bool cleanup = true;
};

using index32_vector = drjit::detail::index32_vector;
using index64_vector = drjit::detail::index64_vector;
