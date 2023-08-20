/*
    drjit/extra.h -- Forward/reverse-mode automatic differentiation wrapper

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_MSC_VER)
#  if defined(DRJIT_EXTRA_BUILD)
#    define DRJIT_EXTRA_EXPORT    __declspec(dllexport)
#  else
#    define DRJIT_EXTRA_EXPORT    __declspec(dllimport)
#  endif
#else
#  define DRJIT_EXTRA_EXPORT __attribute__ ((visibility("default")))
#endif

struct UInt32Pair {
    uint32_t first;
    uint32_t second;
};

struct UInt64Pair {
    uint64_t first;
    uint64_t second;
};

#define DR_EXPORT(x)                                                           \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x(uint32_t);                  \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t);

#define DR_EXPORT_2(x)                                                         \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x(uint32_t, uint32_t);        \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t, uint64_t);

#define DR_EXPORT_PAIR(x)                                                      \
    extern DRJIT_EXTRA_EXPORT struct UInt32Pair jit_var_##x(uint32_t);         \
    extern DRJIT_EXTRA_EXPORT struct UInt64Pair ad_var_##x(uint64_t);

#define DR_EXPORT_AD(x)                                                        \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t);

#define DR_EXPORT_AD_2(x)                                                      \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t, uint64_t);

#define DR_EXPORT_AD_3(x)                                                      \
    extern DRJIT_EXTRA_EXPORT uint64_t ad_var_##x(uint64_t, uint64_t, uint64_t);

DR_EXPORT(exp2)
DR_EXPORT(exp)
DR_EXPORT(log2)
DR_EXPORT(log)
DR_EXPORT(sin)
DR_EXPORT(cos)
DR_EXPORT(tan)
DR_EXPORT(cot)
DR_EXPORT(asin)
DR_EXPORT(acos)
DR_EXPORT(atan)
DR_EXPORT(sinh)
DR_EXPORT(cosh)
DR_EXPORT(tanh)
DR_EXPORT(asinh)
DR_EXPORT(acosh)
DR_EXPORT(atanh)
DR_EXPORT(cbrt)
DR_EXPORT(erf)
DR_EXPORT_2(atan2)
DR_EXPORT_2(ldexp)
DR_EXPORT_PAIR(sincos)
DR_EXPORT_PAIR(sincosh)
DR_EXPORT_PAIR(frexp)

DR_EXPORT_AD(neg)
DR_EXPORT_AD(abs)
DR_EXPORT_AD(sqrt)
DR_EXPORT_AD(rcp)
DR_EXPORT_AD(rsqrt)

DR_EXPORT_AD_2(add)
DR_EXPORT_AD_2(sub)
DR_EXPORT_AD_2(mul)
DR_EXPORT_AD_2(div)

DR_EXPORT_AD_2(min)
DR_EXPORT_AD_2(max)

DR_EXPORT_AD_3(fma)
DR_EXPORT_AD_3(select)

#undef DR_EXPORT
#undef DR_EXPORT_2
#undef DR_EXPORT_PAIR
#undef DR_EXPORT_AD
#undef DR_EXPORT_AD_2

extern DRJIT_EXTRA_EXPORT uint64_t ad_var_inc_ref_impl(uint64_t) JIT_NOEXCEPT;
extern DRJIT_EXTRA_EXPORT void ad_var_dec_ref_impl(uint64_t) JIT_NOEXCEPT;

extern DRJIT_EXTRA_EXPORT uint64_t ad_var_reduce(JitBackend, VarType, uint64_t,
                                                 JIT_ENUM ReduceOp);

extern DRJIT_EXTRA_EXPORT uint64_t ad_var_cast(uint64_t, VarType);

#if defined(__cplusplus)
}
#endif

#if defined(__GNUC__)
DRJIT_INLINE uint64_t ad_var_inc_ref(uint64_t index) JIT_NOEXCEPT {
    /* If 'index' is known at compile time, it can only be zero, in
       which case we can skip the redundant call to ad_var_dec_ref */
    if (__builtin_constant_p(index))
        return 0;
    else
        return ad_var_inc_ref_impl(index);
}

DRJIT_INLINE void ad_var_dec_ref(uint64_t index) JIT_NOEXCEPT {
    if (!__builtin_constant_p(index))
        ad_var_dec_ref_impl(index);
}
#else
#define ad_var_dec_ref ad_var_dec_ref_impl
#define ad_var_inc_ref ad_var_inc_ref_impl
#endif
