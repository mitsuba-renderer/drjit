/*
    drjit/extra.h -- Forward/reverse-mode automatic differentiation wrapper

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  if defined(DRJIT_EXTRA_BUILD)
#    define DRJIT_EXTRA_EXPORT    __declspec(dllexport)
#  else
#    define DRJIT_EXTRA_EXPORT    __declspec(dllimport)
#  endif
#else
#  define DRJIT_EXTRA_EXPORT __attribute__ ((visibility("default")))
#endif

#define WRAP_MATH_OP(x)                                                        \
    template <typename T> uint32_t jit_var_##x(uint32_t);                      \
    template <typename T> uint64_t ad_var_##x(uint64_t);                       \
    extern template DRJIT_EXTRA_EXPORT uint32_t jit_var_##x<float>(uint32_t);  \
    extern template DRJIT_EXTRA_EXPORT uint32_t jit_var_##x<double>(uint32_t); \
    extern template DRJIT_EXTRA_EXPORT uint64_t ad_var_##x<float>(uint64_t);   \
    extern template DRJIT_EXTRA_EXPORT uint64_t ad_var_##x<double>(uint64_t);

#define WRAP_MATH_OP_2(x)                                                      \
    template <typename T> uint32_t jit_var_##x(uint32_t, uint32_t);            \
    template <typename T> uint64_t ad_var_##x(uint64_t, uint64_t);             \
    extern template DRJIT_EXTRA_EXPORT uint32_t jit_var_##x<float>(uint32_t,   \
                                                                   uint32_t);  \
    extern template DRJIT_EXTRA_EXPORT uint32_t jit_var_##x<double>(uint32_t,  \
                                                                    uint32_t); \
    extern template DRJIT_EXTRA_EXPORT uint64_t ad_var_##x<float>(uint64_t,    \
                                                                  uint64_t);   \
    extern template DRJIT_EXTRA_EXPORT uint64_t ad_var_##x<double>(uint64_t,   \
                                                                   uint64_t);

#define WRAP_MATH_OP_PAIR(x)                                                   \
    template <typename T> std::pair<uint32_t, uint32_t> jit_var_##x(uint32_t); \
    template <typename T> std::pair<uint64_t, uint64_t> ad_var_##x(uint64_t);  \
    extern template DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t>           \
        jit_var_##x<float>(uint32_t);                                          \
    extern template DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t>           \
        jit_var_##x<double>(uint32_t);                                         \
    extern template DRJIT_EXTRA_EXPORT std::pair<uint64_t, uint64_t>           \
        ad_var_##x<float>(uint64_t);                                           \
    extern template DRJIT_EXTRA_EXPORT std::pair<uint64_t, uint64_t>           \
        ad_var_##x<double>(uint64_t);

WRAP_MATH_OP(exp2)
WRAP_MATH_OP(exp)
WRAP_MATH_OP(log2)
WRAP_MATH_OP(log)
WRAP_MATH_OP(sin)
WRAP_MATH_OP(cos)
WRAP_MATH_OP(tan)
WRAP_MATH_OP(cot)
WRAP_MATH_OP(asin)
WRAP_MATH_OP(acos)
WRAP_MATH_OP(sinh)
WRAP_MATH_OP(cosh)
WRAP_MATH_OP(tanh)
WRAP_MATH_OP(asinh)
WRAP_MATH_OP(acosh)
WRAP_MATH_OP(atanh)
WRAP_MATH_OP(cbrt)
WRAP_MATH_OP(erf)
WRAP_MATH_OP_2(atan2)
WRAP_MATH_OP_2(ldexp)
WRAP_MATH_OP_PAIR(sincos)
WRAP_MATH_OP_PAIR(sincosh)
WRAP_MATH_OP_PAIR(frexp)

#undef WRAP_MATH_OP
#undef WRAP_MATH_OP_2
#undef WRAP_MATH_OP_PAIR

extern DRJIT_EXTRA_EXPORT void ad_var_inc_ref_impl(uint64_t) noexcept (true);
extern DRJIT_EXTRA_EXPORT void ad_var_dec_ref_impl(uint64_t) noexcept (true);

#if defined(__GNUC__)
inline void ad_var_inc_ref(uint64_t index) noexcept(true) {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to ad_var_dec_ref */
    if (!__builtin_constant_p(index) || index != 0)
        ad_var_inc_ref_impl(index);
}
inline void ad_var_dec_ref(uint64_t index) noexcept(true) {
    if (!__builtin_constant_p(index) || index != 0)
        ad_var_dec_ref_impl(index);
}
#else
#define ad_var_dec_ref ad_var_dec_ref_impl
#define ad_var_inc_ref ad_var_inc_ref_impl
#endif

