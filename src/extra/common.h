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

/// Index vector that decreases JIT refcounts when destructed
struct dr_index32_vector : drjit::dr_vector<uint32_t> {
    using Base = drjit::dr_vector<uint32_t>;
    using Base::Base;

    dr_index32_vector(dr_index32_vector &&a) : Base(std::move(a)) { }
    ~dr_index32_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            jit_var_dec_ref(operator[](i));
        Base::clear();
    }

    void push_back_steal(uint32_t index) { push_back(index); }
    void push_back_borrow(uint32_t index) {
        jit_var_inc_ref(index);
        push_back(index);
    }
};

/// Index vector that decreases JIT + AD refcounts when destructed
struct dr_index64_vector : drjit::dr_vector<uint64_t> {
    using Base = drjit::dr_vector<uint64_t>;
    using Base::Base;
    dr_index64_vector(dr_index64_vector &&a) : Base(std::move(a)) { }

    ~dr_index64_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
        Base::clear();
    }

    void push_back_steal(uint64_t index) { push_back(index); }
    void push_back_borrow(uint64_t index) {
        push_back(ad_var_inc_ref(index));
    }
};

/// RAII AD Isolation helper
struct scoped_isolation_boundary {
    scoped_isolation_boundary() {
        ad_scope_enter(drjit::ADScope::Isolate, 0, nullptr);
    }

    ~scoped_isolation_boundary() {
        ad_scope_leave(success);
    }

    void defuse() { success = true; }

    bool success = false;
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
        jit_record_end(backend, checkpoint, cleanup);
    }

    uint32_t checkpoint_and_rewind() {
        jit_set_scope(backend, scope);
        return jit_record_checkpoint(backend);
    }

    void disarm() { cleanup = false; }

    JitBackend backend;
    uint32_t checkpoint, scope;
    bool cleanup = true;
};
