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

#define WRAP_OP(x)                                                             \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x##_f32(uint32_t);            \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x##_f64(uint32_t);

#define WRAP_OP_2(x)                                                           \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x##_f32(uint32_t, uint32_t);  \
    extern DRJIT_EXTRA_EXPORT uint32_t jit_var_##x##_f64(uint32_t, uint32_t);

#define WRAP_OP_PAIR(x)                                                        \
    extern DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_##x##_f32( \
        uint32_t);                                                             \
    extern DRJIT_EXTRA_EXPORT std::pair<uint32_t, uint32_t> jit_var_##x##_f64( \
        uint32_t);

WRAP_OP(exp2)
WRAP_OP(exp)
WRAP_OP(log2)
WRAP_OP(log)
WRAP_OP(sin)
WRAP_OP(cos)
WRAP_OP(tan)
WRAP_OP(cot)
WRAP_OP(asin)
WRAP_OP(acos)
WRAP_OP(sinh)
WRAP_OP(cosh)
WRAP_OP(tanh)
WRAP_OP(asinh)
WRAP_OP(acosh)
WRAP_OP(atanh)
WRAP_OP(cbrt)
WRAP_OP(erf)
WRAP_OP_2(atan2)
WRAP_OP_2(ldexp)
WRAP_OP_PAIR(sincos)
WRAP_OP_PAIR(sincosh)
WRAP_OP_PAIR(frexp)

#undef WRAP_OP
#undef WRAP_OP_2
#undef WRAP_OP_PAIR

extern DRJIT_EXTRA_EXPORT void ad_var_inc_ref_impl(uint64_t) noexcept (true);
extern DRJIT_EXTRA_EXPORT void ad_var_dec_ref_impl(uint64_t) noexcept (true);

extern DRJIT_EXTRA_EXPORT uint64_t ad_var_add(uint64_t, uint64_t);
