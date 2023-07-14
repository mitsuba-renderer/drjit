#include <drjit-core/jit.h>
#include <drjit/fwd.h>

#define likely(x)   DRJIT_LIKELY(x)
#define unlikely(x) DRJIT_UNLIKELY(x)

#define ad_warn(fmt, ...) jit_log(LogLevel::Warn, "drjit-autodiff: " fmt, ## __VA_ARGS__)
#define ad_log(fmt, ...) jit_log(LogLevel::Debug, "drjit-autodiff: " fmt, ## __VA_ARGS__)
#define ad_fail(fmt, ...) jit_fail("drjit-autodiff: " fmt, ## __VA_ARGS__)
#define ad_raise(fmt, ...) jit_raise("drjit-autodiff: " fmt, ## __VA_ARGS__)
#define ad_assert(cond) if (unlikely(!(cond))) jit_fail("drjit-autodiff: assertion failure (%s) at line %i", #cond, __LINE__);

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

template <typename Value>
using GenericArray = dr::JitArray<JitBackend::None, Value>;
