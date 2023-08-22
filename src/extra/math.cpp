/*
    extra/math.cpp -- Transcendental functions exported by the drjit-extra library

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/


#include <drjit/jit.h>
#include <drjit/math.h>
#include "common.h"

namespace dr = drjit;

#if !defined(_MSC_VER)
#  undef DRJIT_EXTRA_EXPORT
#  define DRJIT_EXTRA_EXPORT __attribute__((flatten, visibility("default")))
#endif

using Float32 = GenericArray<float>;
using Float64 = GenericArray<double>;

#define DEFINE_MATH_OP(name)                                                   \
    DRJIT_EXTRA_EXPORT uint32_t jit_var_##name(uint32_t i0) {                  \
        VarInfo info = jit_set_backend(i0);                                    \
        switch (info.type) {                                                   \
            case VarType::Float32:                                             \
                return dr::name<Float32, false>(Float32::borrow(i0))           \
                    .release();                                                \
            case VarType::Float64:                                             \
                return dr::name<Float64, false>(Float64::borrow(i0))           \
                    .release();                                                \
            default:                                                           \
                jit_fail("jit_var_" #name "(): invalid operand!");             \
        }                                                                      \
    }

#define DEFINE_MATH_OP_2(name)                                                 \
    DRJIT_EXTRA_EXPORT uint32_t jit_var_##name(uint32_t i0, uint32_t i1) {     \
        VarInfo info = jit_set_backend(i0);                                    \
        switch (info.type) {                                                   \
            case VarType::Float32:                                             \
                return dr::name<Float32, Float32, false>(Float32::borrow(i0),  \
                                                         Float32::borrow(i1))  \
                    .release();                                                \
            case VarType::Float64:                                             \
                return dr::name<Float64, Float64, false>(Float64::borrow(i0),  \
                                                         Float64::borrow(i1))  \
                    .release();                                                \
                                                                               \
            default:                                                           \
                jit_fail("jit_var_" #name "(): invalid operand!");             \
        }                                                                      \
    }

#define DEFINE_MATH_OP_PAIR(name)                                              \
    DRJIT_EXTRA_EXPORT UInt32Pair jit_var_##name(uint32_t i0) {                \
        VarInfo info = jit_set_backend(i0);                                    \
        switch (info.type) {                                                   \
            case VarType::Float32: {                                           \
                auto [a, b] = dr::name<Float32, false>(Float32::borrow(i0));   \
                return { a.release(), b.release() }; }                         \
            case VarType::Float64: {                                           \
                auto [a, b] = dr::name<Float64, false>(Float64::borrow(i0));   \
                return { a.release(), b.release() }; }                         \
            default:                                                           \
                jit_fail("jit_var_" #name "(): invalid operand!");             \
        }                                                                      \
    }

DEFINE_MATH_OP(tan)
DEFINE_MATH_OP(cot)
DEFINE_MATH_OP(asin)
DEFINE_MATH_OP(acos)
DEFINE_MATH_OP(atan)
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

// The operations below need special casing to use intrinsics on CUDA hardware

DRJIT_EXTRA_EXPORT uint32_t jit_var_exp(uint32_t i0) {
    VarInfo info = jit_set_backend(i0);

    switch (info.type) {
        case VarType::Float32:
            if (info.backend == JitBackend::CUDA) {
                Float32 value = Float32::borrow(i0) * dr::InvLogTwo<float>;
                return jit_var_exp2_intrinsic(value.index());
            }

            return dr::exp<Float32, false>(Float32::borrow(i0)).release();

        case VarType::Float64:
            return dr::exp<Float64, false>(Float64::borrow(i0)).release();

        default: jit_fail("jit_var_exp(): invalid operand!");
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_exp2(uint32_t i0) {
    VarInfo info = jit_set_backend(i0);

    switch (info.type) {
        case VarType::Float32:
            if (info.backend == JitBackend::CUDA)
                return jit_var_exp2_intrinsic(i0);
            return dr::exp2<Float32, false>(Float32::borrow(i0)).release();

        case VarType::Float64:
            return dr::exp2<Float64, false>(Float64::borrow(i0)).release();

        default: jit_fail("jit_var_exp2(): invalid operand!");
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_log(uint32_t i0) {
    VarInfo info = jit_set_backend(i0);

    switch (info.type) {
        case VarType::Float32:
            if (info.backend == JitBackend::CUDA)
                return (Float32::steal(jit_var_log2_intrinsic(i0)) *
                        dr::LogTwo<float>).release();
            return dr::log<Float32, false>(Float32::borrow(i0)).release();

        case VarType::Float64:
            return dr::log<Float64, false>(Float64::borrow(i0)).release();

        default: jit_fail("jit_var_log(): invalid operand!");
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_log2(uint32_t i0) {
    VarInfo info = jit_set_backend(i0);

    switch (info.type) {
        case VarType::Float32:
            if (info.backend == JitBackend::CUDA)
                return jit_var_log2_intrinsic(i0);
            return dr::log2<Float32, false>(Float32::borrow(i0)).release();

        case VarType::Float64:
            return dr::log2<Float64, false>(Float64::borrow(i0)).release();

        default: jit_fail("jit_var_log2(): invalid operand!");
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_sin(uint32_t i0) {
    VarInfo info = jit_set_backend(i0);

    switch (info.type) {
        case VarType::Float32:
            if (info.backend == JitBackend::CUDA)
                return jit_var_sin_intrinsic(i0);
            return dr::sin<Float32, false>(Float32::borrow(i0)).release();

        case VarType::Float64:
            return dr::sin<Float64, false>(Float64::borrow(i0)).release();

        default: jit_fail("jit_var_sin(): invalid operand!");
    }
}

DRJIT_EXTRA_EXPORT uint32_t jit_var_cos(uint32_t i0) {
    VarInfo info = jit_set_backend(i0);

    switch (info.type) {
        case VarType::Float32:
            if (info.backend == JitBackend::CUDA)
                return jit_var_cos_intrinsic(i0);
            return dr::cos<Float32, false>(Float32::borrow(i0)).release();

        case VarType::Float64:
            return dr::cos<Float64, false>(Float64::borrow(i0)).release();

        default: jit_fail("jit_var_cos(): invalid operand!");
    }
}
