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

      ``Int lit_i(int32_t v)``
          Create an integer literal.

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

    The MIP-mapped lookup routines \ref tex_eval_lod() and \ref
    tex_eval_filtered() additionally require the following members.

    Fields:

      ``Mask active``
          The query mask.

      ``TexLevel<Int, UInt> lvl``
          Pyramid level binding, written by \ref tex_at_level(). When bound,
          ``res_f``/``res_i``/``idiv`` must describe the bound level, and
          ``gather`` must read its texels.

    Methods:

      ``void mip_record(const Int &level, Int *rec)``
          Load the level's constant record from the MIP table into ``rec``
          (8 entries, see \ref tex_mip_table()).

      ``void sum_loop(const Int &n, Float *state, uint32_t n_state,
                      uint32_t n_scratch, Body body)``
          Run ``body(i, m, state, scratch)`` for ``i = 0, ..., n - 1``,
          where ``n`` may vary per lane and ``m`` masks finished lanes. The
          body may only add to the ``n_state`` entries of ``state``.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>
#include <drjit/color.h>
#include <drjit/idiv.h>
#include <memory>

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

/// Reduce a texel-space coordinate to the range that \c Int can represent, so
/// that ``pos_f - to_float(floor2int(pos_f))`` cannot blow up to inf or NaN
template <typename Ops>
typename Ops::Float tex_clamp_pos(const Ops &ops, const typename Ops::Float &pos_f) {
    return clip(pos_f, ops.lit(-0x1p30), ops.lit(0x1p30));
}

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

/// Truncating division by a positive divisor (see ``drjit::divisor``)
template <typename Ops>
typename Ops::Int tex_idiv_dynamic(const Ops &ops, const typename Ops::Int &m,
                                   const typename Ops::Int &s,
                                   const typename Ops::Int &value) {
    using Int = typename Ops::Int;
    Int one = ops.lit_i(1),
        q   = mul_hi(m, value) + value;
    q = q + (sr<31>(q) & select(m == 0, (one << s) - 1, one << s));
    return q >> s;
}

/// Compute the depth and per-level constant table of a texture's MIP pyramid.
/// The table is an ``int32`` buffer holding one record per level, laid out as
///
///     struct Record {
///         int32_t offset;          // Texel offset of the level within the
///                                  // pyramid buffer
///         struct {
///             int32_t multiplier;  // drjit::divisor<int32_t> magic constants
///             int32_t shift;       // of the level's resolution
///         } div[dim];              // One entry per dimension, width first
///         int32_t pad[];           // Zero padding up to ``stride`` entries
///     };
///
/// where ``stride`` (4 for 1D textures, 8 otherwise) rounds the record up to a
/// power-of-two size so that a level's constants load as a single packet. Level
/// zero refers to the base texture. first). The function returns the number of
/// levels including the base. When this exceeds 1, it also allocates and fills
/// ``table`` with ``n_levels * stride`` entries and stores the total texel
/// count of the pyramid levels >= 1 into ``texels``.
inline uint32_t tex_mip_table(std::unique_ptr<int32_t[]> &table,
                              uint32_t &texels, const size_t *shape,
                              uint32_t dim, uint32_t stride) {
    uint32_t n_levels = 1;
    size_t max_res = 0;
    for (uint32_t i = 0; i < dim; ++i)
        max_res = shape[i] > max_res ? shape[i] : max_res;
    while ((max_res >> (n_levels - 1)) > 1)
        n_levels++;

    texels = 0;
    if (n_levels == 1)
        return 1;

    table.reset(new int32_t[(size_t) n_levels * stride]());
    uint32_t accum = 0;
    for (uint32_t l = 0; l < n_levels; ++l) {
        int32_t *rec = table.get() + (size_t) l * stride;
        rec[0] = (int32_t) accum;
        if (l >= 1) {
            uint32_t n = 1;
            for (uint32_t i = 0; i < dim; ++i) {
                size_t r = shape[i] >> l;
                n *= (uint32_t) (r > 0 ? r : 1);
            }
            accum += n;
        }
        for (uint32_t k = 0; k < dim; ++k) {
            size_t r = shape[dim - 1 - k] >> l;
            divisor<int32_t> d((int32_t) (r > 0 ? r : 1));
            rec[1 + 2 * k] = d.multiplier;
            rec[2 + 2 * k] = (int32_t) d.shift;
        }
    }
    texels = accum;
    return n_levels;
}

/// Raw-memory MIP pyramid construction used by the scalar texture modes. The
/// semantics match \ref ad_tex_mipmap_from_base().
template <typename T>
void tex_mipmap_from_base(const T *base, T *mip, const size_t *res_in,
                      uint32_t dim, uint32_t channels, uint32_t n_levels,
                      bool srgb) {
    // Box-filter accumulator: single precision, except for f64 storage
    using Accum = std::conditional_t<std::is_same_v<T, double>, double, float>;
    constexpr bool IsUInt8 = std::is_same_v<T, uint8_t>;

    size_t res[3] = { 1, 1, 1 };
    for (uint32_t k = 0; k < dim; ++k)
        res[k] = res_in[k];
    size_t n = res[0] * res[1] * res[2];

    std::unique_ptr<Accum[]> lin;
    if constexpr (IsUInt8) {
        lin.reset(new Accum[n * channels]);
        for (size_t i = 0; i < n * channels; ++i) {
            Accum v = Accum(base[i]) * Accum(1.0 / 255.0);
            if (srgb && (i % channels) % 4 != 3)
                v = srgb_to_linear(v);
            lin[i] = v;
        }
    }

    uint32_t n_corners = 1u << dim;
    const T *prev = base;
    T *dst = mip;

    for (uint32_t l = 1; l < n_levels; ++l) {
        size_t prev_res[3] = { res[0], res[1], res[2] };
        for (int k = 0; k < 3; ++k)
            res[k] = res[k] > 1 ? res[k] >> 1 : 1;
        n = res[0] * res[1] * res[2];

        // In-place downsampling is safe: output texel 'o' only reads input
        // texels at indices >= o, which have not been overwritten yet
        for (size_t o = 0; o < n; ++o) {
            size_t x = o % res[0], t = o / res[0],
                   y = t % res[1], z = t / res[1];

            for (uint32_t ch = 0; ch < channels; ++ch) {
                Accum acc = 0;
                for (uint32_t corner = 0; corner < n_corners; ++corner) {
                    size_t sx = minimum(2 * x + (corner & 1), prev_res[0] - 1),
                           sy = minimum(2 * y + ((corner >> 1) & 1), prev_res[1] - 1),
                           sz = minimum(2 * z + ((corner >> 2) & 1), prev_res[2] - 1),
                           idx = (sz * prev_res[1] + sy) * prev_res[0] + sx;
                    if constexpr (IsUInt8)
                        acc += lin[idx * channels + ch];
                    else
                        acc += Accum(prev[idx * channels + ch]);
                }
                acc *= Accum(1) / Accum(n_corners);

                if constexpr (IsUInt8) {
                    lin[o * channels + ch] = acc;
                    acc = clip(acc, Accum(0), Accum(1));
                    if (srgb && (ch % 4) != 3)
                        acc = linear_to_srgb(acc);
                    dst[o * channels + ch] = T(fmadd(acc, Accum(255), Accum(0.5)));
                } else {
                    dst[o * channels + ch] = T(acc);
                }
            }
        }

        prev = dst;
        dst += n * channels;
    }
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
        pos_f[k] = tex_clamp_pos(ops, nearest
                                     ? pos[k] * ops.res_f(k)
                                     : fmadd(pos[k], ops.res_f(k), ops.lit(-0.5)));
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

/// MIP level binding state embedded in an ``Ops`` object as a field named
/// ``lvl`` and established by \ref tex_at_level()
template <typename Int, typename UInt> struct TexLevel {
    /// Set when the operations object is bound to a level
    bool bound = false;

    /// Whether ``level`` may be zero, in which case ``gather`` must blend in
    /// the base storage
    bool includes_base = false;

    Int level{};           ///< Bound pyramid level
    UInt offset{};         ///< Texel offset of the level within the pyramid buffer
    Int div[MaxDim][2]{};  ///< Magic division constants of the level's resolution
};

/// Return a copy of ``ops`` bound to MIP level ``l``. This loads the level's
/// record from the constant table (see \ref tex_mip_table()) and fills the
/// ``lvl`` field of the copy.
template <typename Ops>
Ops tex_at_level(const Ops &ops, const typename Ops::Int &l, bool includes_base) {
    using Int  = typename Ops::Int;
    using UInt = typename Ops::UInt;

    Int rec[8];
    ops.mip_record(l, rec);

    Ops o = ops;
    o.lvl.bound = true;
    o.lvl.includes_base = includes_base;
    o.lvl.level = l;
    o.lvl.offset = UInt(rec[0]);
    for (uint32_t k = 0; k < ops.dim; ++k) {
        o.lvl.div[k][0] = rec[1 + 2 * k];
        o.lvl.div[k][1] = rec[2 + 2 * k];
    }
    return o;
}

/// Implementation of the ``sum_loop`` method of the ``Ops`` contract for
/// backends without derivative tracking.
template <typename Ops, typename Body>
void tex_sum_loop(const Ops &ops, const typename Ops::Int &n,
                  typename Ops::Float *state, uint32_t n_scratch, Body body) {
    using Float = typename Ops::Float;
    using Int   = typename Ops::Int;
    using Mask  = typename Ops::Mask;

    Float *scratch_mem = (Float *) alloca(sizeof(Float) * n_scratch);
    tex_scratch<Float> scratch(scratch_mem, n_scratch);

    uint32_t iters;
    if constexpr (is_array_v<Int>)
        iters = (uint32_t) max(n);
    else
        iters = (uint32_t) n;

    for (uint32_t j = 0; j < iters; ++j) {
        Int i = Int((int32_t) j);
        Mask m = ops.active && (i < n);
        body(i, m, state, scratch.data());
    }
}

/// Per-lookup constants of a MIP-mapped texture sample: the level-bound
/// operation objects and the level blend weight, hoisted here so that an
/// anisotropic lookup pays for the level binding once rather than per tap.
template <typename Ops> struct TexMipSample {
    Ops o0, o1;
    typename Ops::Float frac;
    bool linear;

    /// Bind the levels enclosing ``lod``, which must lie in
    /// ``[0, n_levels - 1]``
    TexMipSample(const Ops &ops, const typename Ops::Float &lod,
                 uint32_t n_levels, MipFilter mip_filter) {
        using Int = typename Ops::Int;

        linear = mip_filter == MipFilter::Linear;
        if (linear) {
            Int l0 = floor2int<Int>(lod),
                l1 = minimum(l0 + 1, ops.lit_i((int32_t) n_levels - 1));
            frac = lod - ops.to_float(l0);
            o0 = tex_at_level(ops, l0, true);
            o1 = tex_at_level(ops, l1, false);
        } else {
            o0 = tex_at_level(ops, floor2int<Int>(lod + ops.lit(0.5)), true);
        }
    }

    /// Sample the bound level blend at ``pos``. ``scratch`` must hold
    /// ``2 * channels_out`` values.
    void eval(const typename Ops::Float *pos, typename Ops::Float *out,
              typename Ops::Float *scratch) const {
        uint32_t ch = o0.channels_out;
        tex_eval(o0, pos, out, scratch + ch);
        if (linear) {
            tex_eval(o1, pos, scratch, scratch + ch);
            for (uint32_t c = 0; c < ch; ++c)
                out[c] = fmadd(scratch[c] - out[c], frac, out[c]);
        }
    }
};

/// Sample the texture at an explicit level of detail. A fractional ``lod``
/// blends the two enclosing pyramid levels under ``MipFilter::Linear`` and
/// rounds to the nearest one under ``MipFilter::Nearest``. ``scratch`` must
/// hold ``2 * channels_out`` values.
template <typename Ops>
void tex_eval_lod(const Ops &ops, const typename Ops::Float *pos,
                  const typename Ops::Float &lod, uint32_t n_levels,
                  MipFilter mip_filter, typename Ops::Float *out,
                  typename Ops::Float *scratch) {
    using Float = typename Ops::Float;
    Float l = clip(lod, ops.lit(0.0), ops.lit((double) (n_levels - 1)));
    TexMipSample<Ops> sample(ops, l, n_levels, mip_filter);
    sample.eval(pos, out, scratch);
}

/// Anisotropically filtered texture lookup driven by a screen-space footprint.
///
/// ``ddx`` and ``ddy`` are texture-space differentials spanning an an
/// elliptical footprint, whose semi-axes ``P_major``/``P_minor`` determine the
/// filter.
///
///     lod = clip(log2(max(P_minor, P_major / max_aniso)), 0, levels - 1)
///     N   = min(ceil(P_major / 2^lod), max_aniso)
///
/// The result averages ``N`` MIP-mapped taps distributed along the major
/// ellipse axis. The footprint, LOD, and tap count follow the reference
/// algorithm of the Direct3D 11.3 (section 7.18.11). ``max_aniso == 1``
/// collapses the scheme to an ordinary trilinear lookup.
template <typename Ops>
void tex_eval_filtered(const Ops &ops, const typename Ops::Float *pos,
                       const typename Ops::Float *ddx,
                       const typename Ops::Float *ddy, uint32_t n_levels,
                       MipFilter mip_filter, uint32_t max_aniso,
                       typename Ops::Float *out) {
    using Float = typename Ops::Float;
    using Int   = typename Ops::Int;
    using Mask  = typename Ops::Mask;

    uint32_t dim = ops.dim, ch = ops.channels_out;

    // Texel-scaled footprint axes and their Gram matrix. The footprint is the
    // image of the unit circle under the matrix ``[dx dy]``.
    Float dx[MaxDim], dy[MaxDim];
    Float px2 = ops.lit(0.0), py2 = ops.lit(0.0), pxy = ops.lit(0.0);
    for (uint32_t k = 0; k < dim; ++k) {
        Float r = ops.res_f(k);
        dx[k] = ddx[k] * r;
        dy[k] = ddy[k] * r;
        px2 = fmadd(dx[k], dx[k], px2);
        py2 = fmadd(dy[k], dy[k], py2);
        pxy = fmadd(dx[k], dy[k], pxy);
    }
    Float mid   = ops.lit(0.5) * (px2 + py2),
          half  = ops.lit(0.5) * (px2 - py2),
          disc  = sqrt(fmadd(half, half, pxy * pxy)),
          p_max = sqrt(mid + disc),
          p_min = sqrt(maximum(mid - disc, ops.lit(0.0)));

    // The clamped minor axis sets a continuous LOD, and the tap count is
    // however many taps of extent ``2^lod`` cover the major axis.
    Float lod = log2(maximum(maximum(p_min, p_max * ops.lit(1.0 / max_aniso)),
                             ops.lit(1e-30)));
    lod = clip(lod, ops.lit(0.0), ops.lit((double) (n_levels - 1)));

    // ``cover`` is an exact integer for isotropic footprints and integer
    // anisotropy ratios. Nudge it down by 1e-5 before ``ceil()`` so that
    // rounding error (e.g. from the GPU's approximate log2/exp2) cannot
    // introduce a spurious extra tap.
    Float cover = p_max * exp2(-lod);
    Float n_f = clip(ceil(fmadd(cover, ops.lit(-1e-5), cover)),
                     ops.lit(1.0), ops.lit((double) max_aniso));

    TexMipSample<Ops> sample(ops, lod, n_levels, mip_filter);

    // Taps step along the major ellipse axis, the image of the dominant right
    // singular vector (two algebraically equivalent expressions; select the
    // better-conditioned one).
    Mask x_major = px2 >= py2;
    Float ca = select(x_major, disc + half, pxy),
          cb = select(x_major, pxy, disc - half);
    Float vnorm2 = ops.lit(0.0);
    for (uint32_t k = 0; k < dim; ++k) {
        Float vk = fmadd(ca, dx[k], cb * dy[k]);
        vnorm2 = fmadd(vk, vk, vnorm2);
    }
    Float scale = p_max * rcp(maximum(sqrt(vnorm2), ops.lit(1e-30)));

    Float pos_v[MaxDim], step_uv[MaxDim], inv_n = rcp(n_f);
    for (uint32_t k = 0; k < dim; ++k) {
        pos_v[k] = pos[k];
        step_uv[k] = fmadd(ca, ddx[k], cb * ddy[k]) * scale;
    }

    for (uint32_t c = 0; c < ch; ++c)
        out[c] = ops.lit(0.0);

    Int n = select(ops.active, floor2int<Int>(n_f), ops.lit_i(0));
    ops.sum_loop(n, out, ch, 3 * ch,
        [ops, sample, pos_v, step_uv, inv_n, dim, ch]
        (const Int &i, const Mask &m, Float *state, Float *scratch) {
            Float t = fmadd(ops.to_float(i) + ops.lit(0.5), inv_n, ops.lit(-0.5));
            Float p[MaxDim];
            for (uint32_t k = 0; k < dim; ++k)
                p[k] = fmadd(step_uv[k], t, pos_v[k]);

            Float *tap = scratch + 2 * ch;
            sample.eval(p, tap, scratch);
            for (uint32_t c = 0; c < ch; ++c)
                state[c] = state[c] + select(m, tap[c], ops.lit(0.0));
        });

    for (uint32_t c = 0; c < ch; ++c)
        out[c] = out[c] * inv_n;
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
        Float pos_f = tex_clamp_pos(
            ops, fmadd(pos[k], ops.res_f(k), ops.lit(-0.5)));
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
        Float pos_f = tex_clamp_pos(
            ops, fmadd(pos[k], ops.res_f(k), ops.lit(-0.5)));
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
        Float pos_f = tex_clamp_pos(
            ops, fmadd(pos[k], ops.res_f(k), ops.lit(-0.5)));
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
