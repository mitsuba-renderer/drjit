#include <drjit/jit.h>
#include <drjit/math.h>
#include "common.h"

#if !defined(_MSC_VER)
#  undef DRJIT_EXTRA_EXPORT
#  define DRJIT_EXTRA_EXPORT __attribute__((flatten, visibility("default")))
#endif

#define EXPORT_MATH_OP(name)                                                   \
    template DRJIT_EXTRA_EXPORT uint32_t jit_var_##name<float>(uint32_t);      \
    template DRJIT_EXTRA_EXPORT uint32_t jit_var_##name<double>(uint32_t);

#define EXPORT_MATH_OP_2(name)                                                 \
    template DRJIT_EXTRA_EXPORT uint32_t jit_var_##name<float>(uint32_t,       \
                                                               uint32_t);      \
    template DRJIT_EXTRA_EXPORT uint32_t jit_var_##name<double>(uint32_t,      \
                                                                uint32_t);

#define EXPORT_MATH_OP_PAIR(name)                                              \
    template DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t>                  \
        jit_var_##name<float>(uint32_t);                                       \
    template DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t>                  \
        jit_var_##name<double>(uint32_t);

#define DEFINE_MATH_OP(name)                                                   \
    template <typename Scalar> uint32_t jit_var_##name(uint32_t i0) {          \
        using T = GenericArray<Scalar>;                                        \
        jit_set_default_backend_from(i0);                                      \
        return dr::name<T, false>(T::borrow(i0)).release();                    \
    }                                                                          \
    EXPORT_MATH_OP(name)

#define DEFINE_MATH_OP_2(name)                                                 \
    template <typename Scalar>                                                 \
    uint32_t jit_var_##name(uint32_t i0, uint32_t i1) {                        \
        using T = GenericArray<Scalar>;                                        \
        jit_set_default_backend_from(i0);                                      \
        return dr::name<T, T, false>(T::borrow(i0), T::borrow(i1)).release();  \
    }                                                                          \
    EXPORT_MATH_OP_2(name)

#define DEFINE_MATH_OP_PAIR(name)                                              \
    template <typename Scalar>                                                 \
    std::pair<uint32_t, uint32_t> jit_var_##name(uint32_t i0) {                \
        using T = GenericArray<Scalar>;                                        \
        jit_set_default_backend_from(i0);                                      \
        auto [a, b] = dr::name<T, false>(T::borrow(i0));                       \
        return { a.release(), b.release() };                                   \
    }                                                                          \
    EXPORT_MATH_OP_PAIR(name)

namespace dr = drjit;

template <typename Scalar> uint32_t jit_var_exp(uint32_t i0) {
    using T = GenericArray<Scalar>;
    JitBackend backend = jit_set_default_backend_from(i0);

    if (std::is_same_v<Scalar, float> && backend == JitBackend::CUDA) {
        T value = T::borrow(i0) * dr::InvLogTwo<Scalar>;
        return jit_var_exp2_intrinsic(value.index());
    }

    return dr::exp<T, false>(T::borrow(i0)).release();
}

template <typename Scalar> uint32_t jit_var_exp2(uint32_t i0) {
    using T = GenericArray<Scalar>;
    JitBackend backend = jit_set_default_backend_from(i0);

    if (std::is_same_v<Scalar, float> && backend == JitBackend::CUDA)
        return jit_var_exp2_intrinsic(i0);

    return dr::exp2<T, false>(T::borrow(i0)).release();
}

template <typename Scalar> uint32_t jit_var_log(uint32_t i0) {
    using T = GenericArray<Scalar>;
    JitBackend backend = jit_set_default_backend_from(i0);

    if (std::is_same_v<Scalar, float> && backend == JitBackend::CUDA)
        return (T::steal(jit_var_log2_intrinsic(i0)) * dr::LogTwo<float>)
            .release();

    return dr::log<T, false>(T::borrow(i0)).release();
}

template <typename Scalar> uint32_t jit_var_log2(uint32_t i0) {
    using T = GenericArray<Scalar>;
    JitBackend backend = jit_set_default_backend_from(i0);

    if (std::is_same_v<Scalar, float> && backend == JitBackend::CUDA)
        return jit_var_log2_intrinsic(i0);

    return dr::log2<T, false>(T::borrow(i0)).release();
}

template <typename Scalar> uint32_t jit_var_sin(uint32_t i0) {
    using T = GenericArray<Scalar>;
    JitBackend backend = jit_set_default_backend_from(i0);

    if (std::is_same_v<Scalar, float> && backend == JitBackend::CUDA)
        return jit_var_sin_intrinsic(i0);

    return dr::sin<T, false>(T::borrow(i0)).release();
}

template <typename Scalar> uint32_t jit_var_cos(uint32_t i0) {
    using T = GenericArray<Scalar>;
    JitBackend backend = jit_set_default_backend_from(i0);

    if (std::is_same_v<Scalar, float> && backend == JitBackend::CUDA)
        return jit_var_cos_intrinsic(i0);

    return dr::cos<T, false>(T::borrow(i0)).release();
}

EXPORT_MATH_OP(exp2)
EXPORT_MATH_OP(exp)
EXPORT_MATH_OP(log2)
EXPORT_MATH_OP(log)
EXPORT_MATH_OP(sin)
EXPORT_MATH_OP(cos)

DEFINE_MATH_OP(tan)
DEFINE_MATH_OP(cot)
DEFINE_MATH_OP(asin)
DEFINE_MATH_OP(acos)
DEFINE_MATH_OP(sinh)
DEFINE_MATH_OP(cosh)
DEFINE_MATH_OP(tanh)
DEFINE_MATH_OP(asinh)
DEFINE_MATH_OP(acosh)
DEFINE_MATH_OP(atanh)
DEFINE_MATH_OP(cbrt)
DEFINE_MATH_OP(erf)

DEFINE_MATH_OP_2(atan2)
DEFINE_MATH_OP_2(ldexp)
DEFINE_MATH_OP_PAIR(frexp)
DEFINE_MATH_OP_PAIR(sincos)
DEFINE_MATH_OP_PAIR(sincosh)
