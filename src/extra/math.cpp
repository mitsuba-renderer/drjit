#include <drjit/jit.h>
#include <drjit/math.h>


#if !defined(_MSC_VER)
#  undef DRJIT_EXTRA_EXPORT
# define DRJIT_EXTRA_EXPORT __attribute__ ((flatten, visibility("default")))
#endif


#undef DRJIT_EXPORT

namespace dr = drjit;

template <typename Value>
struct GenericArray : dr::JitArray<JitBackend::None, Value, GenericArray<Value>> {
    using Base = dr::JitArray<JitBackend::None, Value, GenericArray<Value>>;
    using MaskType = GenericArray<bool>;
    using ArrayType = GenericArray;
    template <typename T> using ReplaceValue = GenericArray<T>;
    DRJIT_ARRAY_IMPORT(GenericArray, Base)
};

using Float  = GenericArray<float>;
using Double = GenericArray<double>;

DRJIT_EXTRA_EXPORT uint32_t jit_var_exp_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA) {
        Float value = Float::borrow(i0) * dr::InvLogTwo<float>;
        return jit_var_exp2_intrinsic(value.index());
    } else {
        return dr::exp<Float, false>(Float::borrow(i0)).release();
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_exp2_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA) {
        return jit_var_exp2_intrinsic(i0);
    } else {
        return dr::exp2<Float, false>(Float::borrow(i0)).release();
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_log_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA)
        return (Float::steal(jit_var_log2_intrinsic(i0)) * dr::LogTwo<float>).release();
    else
        return dr::log<Float, false>(Float::borrow(i0)).release();
}
DRJIT_EXTRA_EXPORT uint32_t jit_var_log2_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA)
        return jit_var_log2_intrinsic(i0);
    else
        return dr::log2<Float, false>(Float::borrow(i0)).release();
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_sin_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA)
        return jit_var_sin_intrinsic(i0);
    else
        return dr::sin<Float, false>(Float::borrow(i0)).release();
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_cos_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA)
        return jit_var_cos_intrinsic(i0);
    else
        return dr::cos<Float, false>(Float::borrow(i0)).release();
}

#define WRAP_F32(name)                                                         \
    DRJIT_EXTRA_EXPORT uint32_t jit_var_##name##_f32(uint32_t i0) {            \
        jit_set_default_backend_from(i0);                                      \
        return dr::name<Float, false>(Float::borrow(i0)).release();            \
    }

#define WRAP_F64(name)                                                         \
    DRJIT_EXTRA_EXPORT uint32_t jit_var_##name##_f64(uint32_t i0) {            \
        jit_set_default_backend_from(i0);                                      \
        return dr::name<Double, false>(Double::borrow(i0)).release();          \
    }

#define WRAP(name)                                                             \
    WRAP_F32(name)                                                             \
    WRAP_F64(name)

WRAP_F64(exp2)
WRAP_F64(exp)
WRAP_F64(log2)
WRAP_F64(log)
WRAP_F64(sin)
WRAP_F64(cos)
WRAP(tan)
WRAP(cot)
WRAP(asin)
WRAP(acos)

WRAP(sinh)
WRAP(cosh)
WRAP(tanh)
WRAP(asinh)
WRAP(acosh)
WRAP(atanh)

WRAP(cbrt)
WRAP(erf)

DRJIT_EXTRA_EXPORT uint32_t jit_var_atan2_f32(uint32_t i0, uint32_t i1) {
    return dr::atan2<Float, Float, false>(Float::borrow(i0), Float::borrow(i1))
        .release();
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_atan2_f64(uint32_t i0, uint32_t i1) {
    return dr::atan2<Double, Double, false>(Double::borrow(i0),
                                            Double::borrow(i1)).release();
}

DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_sincos_f32(uint32_t i0) {
    if (jit_set_default_backend_from(i0) == JitBackend::CUDA) {
        return { jit_var_sin_intrinsic(i0), jit_var_cos_intrinsic(i0) };
    } else {
        auto [s, c] = dr::sincos<Float, false>(Float::borrow(i0));
        return { s.release(), c.release() };
    }
}

DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_sincos_f64(uint32_t i0) {
    auto [s, c] = dr::sincos<Double, false>(Double::borrow(i0));
    return { s.release(), c.release() };
}

DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_sincosh_f32(uint32_t i0) {
    auto [s, c] = dr::sincosh<Float, false>(Float::borrow(i0));
    return { s.release(), c.release() };
}

DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_sincosh_f64(uint32_t i0) {
    auto [s, c] = dr::sincosh<Double, false>(Double::borrow(i0));
    return { s.release(), c.release() };
}

DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_frexp_f32(uint32_t i0) {
    auto [r0, r1] = dr::frexp<Float, false>(Float::borrow(i0));
    return { r0.release(), r1.release() };
}

DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_frexp_f64(uint32_t i0) {
    auto [r0, r1] = dr::frexp<Double, false>(Double::borrow(i0));
    return { r0.release(), r1.release() };
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_ldexp_f32(uint32_t i0, uint32_t i1) {
    return dr::ldexp<Float, Float, false>(Float::borrow(i0), Float::borrow(i1))
        .release();
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_ldexp_f64(uint32_t i0, uint32_t i1) {
    return dr::ldexp<Double, Double, false>(Double::borrow(i0),
                                            Double::borrow(i1)).release();
}
