/*
    drjit/texture_impl.h -- Shared N-dimensional texture interpolation math

    This file provides templated interpolation routines that can be instantiated
    with an operations object ``Ops``. This centralizes code generation for both
    inlined scalar evaluation and outlined JIT-compiled evaluation in
    ``drjit-extra.so``.

    An ``Ops`` object must provide the following components:

    Types: ``Float``, ``Int``, ``UInt``, ``Mask``, where ``Float`` is the query
    precision.

    Fields: ``uint32_t dim, channels_out``, ``FilterMode filter_mode``, and
    ``WrapMode wrap_mode``.

    Methods:

      ``Float lit(double v)``
          Create a query-precision literal.

      ``Float res_f(uint32_t k)``
          Resolution along axis ``k`` as a float.

      ``Int res_i(uint32_t k)``
          Resolution along axis ``k`` as an integer.

      ``Float to_float(const Int &i)``
          Value-preserving numeric cast from ``Int`` to the query ``Float``
          (no rounding).

      ``Int idiv(const Int &a, uint32_t k)``
          Integer division of ``a`` by ``res[k]``.

      ``void gather(const UInt &idx, Float *out)``
          Gather the ``channels_out`` texels at ``idx`` in query precision.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

/// Largest supported texture dimension
static constexpr uint32_t MaxDim = 3;

/// Holds an ``Ops`` channel count as a compile-time constant (so loops unroll
/// and scratch can be fixed stack arrays), or a runtime member when ``C == 0``.
template <uint32_t C> struct ChannelCount { static constexpr uint32_t channels_out = C; };
template <> struct ChannelCount<0> { uint32_t channels_out; };

/// RAII helper to initialize/destruct an array over caller-supplied storage
template <typename T> struct tex_scratch {
    DRJIT_NON_COPYABLE(tex_scratch)

    tex_scratch(T *ptr, size_t n) : m_ptr(ptr), m_size(n) {
        if constexpr (is_jit_v<T>)
            for (size_t i = 0; i < n; ++i)
                new (m_ptr + i) T();
    }

    ~tex_scratch() {
        if constexpr (is_jit_v<T>)
            for (size_t i = m_size; i > 0; --i)
                m_ptr[i - 1].~T();
    }

    T *data() const { return m_ptr; }
    T &operator[](size_t i) const { return m_ptr[i]; }

private:
    T *m_ptr;
    size_t m_size;
};

/// Apply the wrapping mode to one integer coordinate along dimension \c k
template <typename Ops>
typename Ops::Int tex_wrap(const Ops &ops, const typename Ops::Int &pos, uint32_t k) {
    using Int = typename Ops::Int;
    Int res = ops.res_i(k);
    if (ops.wrap_mode == WrapMode::Clamp)
        return clip(pos, 0, res - 1);

    // Repeat/Mirror: reduce into [0, res) via a Euclidean modulo
    Int value_shift_neg = select(pos < 0, pos + 1, pos),
        div = ops.idiv(value_shift_neg, k),
        mod = pos - div * res;
    mod = select(mod < 0, mod + res, mod);

    // Mirror additionally reflects every other period
    if (ops.wrap_mode == WrapMode::Mirror)
        mod = select(((div & 1) == 0) ^ (pos < 0), mod, res - 1 - mod);

    // Repeat returns the bare modulo; Mirror the reflected one
    return mod;
}

/// Linear texel index for a set of wrapped coordinates
template <typename Ops>
typename Ops::UInt tex_index(const Ops &ops, const typename Ops::Int *coord) {
    using UInt = typename Ops::UInt;
    uint32_t dim = ops.dim;
    UInt idx(coord[dim - 1]);
    for (uint32_t k = dim - 1; k-- > 0;)
        idx = fmadd(idx, UInt(ops.res_i(k)), UInt(coord[k]));
    return idx;
}

/// Cubic B-spline basis weights for the 4 taps along one dimension, as a
/// function of the fractional coordinate \c alpha
template <typename Ops>
void tex_cubic_weights(const Ops &ops, const typename Ops::Float &alpha,
                       typename Ops::Float *w) {
    using Float = typename Ops::Float;
    Float a2 = alpha * alpha, a3 = a2 * alpha;
    w[0] = fmadd(ops.lit(-1.0 / 6.0), a3, fmadd(ops.lit(0.5), a2, fmadd(ops.lit(-0.5), alpha, ops.lit(1.0 / 6.0))));
    w[1] = fmadd(ops.lit(0.5), a3, ops.lit(2.0 / 3.0) - a2);
    w[2] = fmadd(ops.lit(-0.5), a3, fmadd(ops.lit(0.5), a2, fmadd(ops.lit(0.5), alpha, ops.lit(1.0 / 6.0))));
    w[3] = ops.lit(1.0 / 6.0) * a3;
}

/// First derivative of \ref tex_cubic_weights() w.r.t. \c alpha
template <typename Ops>
void tex_cubic_weights_grad(const Ops &ops, const typename Ops::Float &alpha,
                            typename Ops::Float *w) {
    using Float = typename Ops::Float;
    Float a2 = alpha * alpha;
    w[0] = fmadd(ops.lit(-0.5), a2, alpha - ops.lit(0.5));
    w[1] = fmadd(ops.lit(1.5), a2, ops.lit(-2.0) * alpha);
    w[2] = fmadd(ops.lit(-1.5), a2, alpha + ops.lit(0.5));
    w[3] = ops.lit(0.5) * a2;
}

/// Second derivative of \ref tex_cubic_weights() w.r.t. \c alpha
template <typename Ops>
void tex_cubic_weights_hessian(const Ops &ops, const typename Ops::Float &alpha,
                               typename Ops::Float *w) {
    w[0] = ops.lit(1.0) - alpha;
    w[1] = fmadd(ops.lit(3.0), alpha, ops.lit(-2.0));
    w[2] = fnmadd(ops.lit(3.0), alpha, ops.lit(1.0));
    w[3] = alpha;
}

/// Nearest / (multi-)linear interpolation at the \c dim query coordinates.
/// ``scratch`` is scratch space for \c channels_out values (unused when nearest).
template <typename Ops>
void tex_eval(const Ops &ops, const typename Ops::Float *pos,
              typename Ops::Float *out,
              typename Ops::Float *scratch) {
    using Float = typename Ops::Float;
    using Int   = typename Ops::Int;
    uint32_t dim = ops.dim, channels_out = ops.channels_out;
    bool nearest = (ops.filter_mode == FilterMode::Nearest);

    Float pos_f[MaxDim];
    Int pos_i[MaxDim];
    for (uint32_t k = 0; k < dim; ++k) {
        pos_f[k] = nearest ? pos[k] * ops.res_f(k)
                           : fmadd(pos[k], ops.res_f(k), ops.lit(-0.5));
        pos_i[k] = floor2int<Int>(pos_f[k]);
    }

    if (nearest) {
        Int coord[MaxDim];
        for (uint32_t k = 0; k < dim; ++k)
            coord[k] = tex_wrap(ops, pos_i[k], k);
        ops.gather(tex_index(ops, coord), out);
        return;
    }

    Float w[MaxDim][2];
    for (uint32_t k = 0; k < dim; ++k) {
        w[k][1] = pos_f[k] - ops.to_float(pos_i[k]);
        w[k][0] = ops.lit(1.0) - w[k][1];
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out[ch] = ops.lit(0.0);

    for (uint32_t corner = 0; corner < (1u << dim); ++corner) {
        Int coord[MaxDim];
        Float weight = ops.lit(1.0);
        for (uint32_t k = 0; k < dim; ++k) {
            uint32_t bit = (corner >> k) & 1;
            coord[k] = tex_wrap(ops, pos_i[k] + (int32_t) bit, k);
            weight = weight * w[k][bit];
        }
        ops.gather(tex_index(ops, coord), scratch);
        for (uint32_t ch = 0; ch < channels_out; ++ch)
            out[ch] = fmadd(scratch[ch], weight, out[ch]);
    }
}

/// Cubic B-spline interpolation by direct evaluation of the basis functions.
/// ``scratch`` is scratch space for \c channels_out values.
template <typename Ops>
void tex_eval_cubic(const Ops &ops, const typename Ops::Float *pos,
                    typename Ops::Float *out, typename Ops::Float *scratch) {
    using Float = typename Ops::Float;
    using Int   = typename Ops::Int;
    uint32_t dim = ops.dim, channels_out = ops.channels_out;
    int32_t offset[4] = { -1, 0, 1, 2 };

    Int pos_i[MaxDim];
    Float w[MaxDim][4];
    for (uint32_t k = 0; k < dim; ++k) {
        Float pos_f = fmadd(pos[k], ops.res_f(k), ops.lit(-0.5));
        pos_i[k] = floor2int<Int>(pos_f);
        tex_cubic_weights(ops, pos_f - ops.to_float(pos_i[k]), w[k]);
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out[ch] = ops.lit(0.0);

    for (uint32_t corner = 0; corner < (1u << (2 * dim)); ++corner) {
        Int coord[MaxDim];
        Float weight = ops.lit(1.0);
        for (uint32_t k = 0, rem = corner; k < dim; ++k, rem /= 4) {
            uint32_t t = rem % 4;
            coord[k] = tex_wrap(ops, pos_i[k] + offset[t], k);
            weight = weight * w[k][t];
        }
        ops.gather(tex_index(ops, coord), scratch);
        for (uint32_t ch = 0; ch < channels_out; ++ch)
            out[ch] = fmadd(scratch[ch], weight, out[ch]);
    }
}

/// Cubic B-spline value, positional gradient, and (when \c hess is non-null)
/// Hessian, by direct evaluation of the differentiated basis functions. The
/// outputs are flat arrays: ``value[ch]``, ``grad[ch*dim+m]``,
/// ``hess[(ch*dim+m)*dim+n]``; both derivatives are scaled to the resolution.
/// ``scratch`` is scratch space for \c channels_out values.
template <typename Ops>
void tex_eval_cubic_deriv(const Ops &ops, const typename Ops::Float *pos,
                          typename Ops::Float *value, typename Ops::Float *grad,
                          typename Ops::Float *hess, typename Ops::Float *scratch) {
    using Float = typename Ops::Float;
    using Int   = typename Ops::Int;
    uint32_t dim = ops.dim, channels_out = ops.channels_out;
    bool want_hess = (hess != nullptr);
    int32_t offset[4] = { -1, 0, 1, 2 };

    Int pos_i[MaxDim];
    Float wv[MaxDim][4], wg[MaxDim][4], wh[MaxDim][4];
    for (uint32_t k = 0; k < dim; ++k) {
        Float pos_f = fmadd(pos[k], ops.res_f(k), ops.lit(-0.5));
        pos_i[k] = floor2int<Int>(pos_f);
        Float alpha = pos_f - ops.to_float(pos_i[k]);
        tex_cubic_weights(ops, alpha, wv[k]);
        tex_cubic_weights_grad(ops, alpha, wg[k]);
        if (want_hess)
            tex_cubic_weights_hessian(ops, alpha, wh[k]);
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch) value[ch] = ops.lit(0.0);
    for (uint32_t i = 0; i < channels_out * dim; ++i) grad[i] = ops.lit(0.0);
    if (want_hess)
        for (uint32_t i = 0; i < channels_out * dim * dim; ++i) hess[i] = ops.lit(0.0);

    for (uint32_t corner = 0; corner < (1u << (2 * dim)); ++corner) {
        Int coord[MaxDim];
        uint32_t t[MaxDim];
        for (uint32_t k = 0, rem = corner; k < dim; ++k, rem /= 4) {
            t[k] = rem % 4;
            coord[k] = tex_wrap(ops, pos_i[k] + offset[t[k]], k);
        }
        ops.gather(tex_index(ops, coord), scratch);

        // Separable weights: a gradient/hessian component replaces one/two of
        // the value bases ``wv`` with their derivatives (``wh`` if coinciding)
        Float w_value = ops.lit(1.0);
        for (uint32_t k = 0; k < dim; ++k)
            w_value = w_value * wv[k][t[k]];

        Float w_grad[MaxDim];
        for (uint32_t m = 0; m < dim; ++m) {
            Float g = ops.lit(1.0);
            for (uint32_t k = 0; k < dim; ++k)
                g = g * (k == m ? wg[k][t[k]] : wv[k][t[k]]);
            w_grad[m] = g;
        }

        Float w_hess[MaxDim][MaxDim];
        if (want_hess)
            for (uint32_t m = 0; m < dim; ++m)
                for (uint32_t n = m; n < dim; ++n) {
                    Float h = ops.lit(1.0);
                    for (uint32_t k = 0; k < dim; ++k) {
                        const Float &wk = (k == m && k == n) ? wh[k][t[k]]
                                        : (k == m || k == n) ? wg[k][t[k]]
                                                             : wv[k][t[k]];
                        h = h * wk;
                    }
                    w_hess[m][n] = h;
                }

        for (uint32_t ch = 0; ch < channels_out; ++ch) {
            value[ch] = fmadd(scratch[ch], w_value, value[ch]);
            for (uint32_t m = 0; m < dim; ++m) {
                grad[ch * dim + m] = fmadd(scratch[ch], w_grad[m], grad[ch * dim + m]);
                if (want_hess)
                    for (uint32_t n = m; n < dim; ++n) {
                        uint32_t e = (ch * dim + m) * dim + n;
                        hess[e] = fmadd(scratch[ch], w_hess[m][n], hess[e]);
                    }
            }
        }
    }

    // Mirror the upper triangle, then map unit-volume derivatives to resolution
    for (uint32_t ch = 0; ch < channels_out; ++ch) {
        for (uint32_t m = 0; m < dim; ++m) {
            grad[ch * dim + m] = grad[ch * dim + m] * ops.res_f(m);
            if (want_hess)
                for (uint32_t n = m + 1; n < dim; ++n)
                    hess[(ch * dim + n) * dim + m] = hess[(ch * dim + m) * dim + n];
        }
        if (want_hess)
            for (uint32_t m = 0; m < dim; ++m)
                for (uint32_t n = 0; n < dim; ++n)
                    hess[(ch * dim + m) * dim + n] =
                        hess[(ch * dim + m) * dim + n] * ops.res_f(m) * ops.res_f(n);
    }
}

/// Fetch the ``2^dim`` corner texels of a linear lookup without interpolation.
/// ``out[corner]`` points to the \c channels_out values of corner \c corner, where
/// bit \c k indicates the offset along dimension \c k.
template <typename Ops>
void tex_fetch(const Ops &ops,
               const typename Ops::Float *pos,
               typename Ops::Float **out) {
    using Float = typename Ops::Float;
    using Int   = typename Ops::Int;
    uint32_t dim = ops.dim;

    Int pos_i[MaxDim];
    for (uint32_t k = 0; k < dim; ++k) {
        Float pos_f = fmadd(pos[k], ops.res_f(k), ops.lit(-0.5));
        pos_i[k] = floor2int<Int>(pos_f);
    }

    for (uint32_t corner = 0; corner < (1u << dim); ++corner) {
        Int coord[MaxDim];
        for (uint32_t k = 0; k < dim; ++k)
            coord[k] = tex_wrap(ops, pos_i[k] + (int32_t) ((corner >> k) & 1), k);
        ops.gather(tex_index(ops, coord), out[corner]);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
