/*
    extra/texture.cpp -- Type-erased N-dimensional texture interpolation

    The ``ad_tex_*`` functions implemented below provide a type and
    dimension-erased differentiable interface to texture interpolation. This is
    mainly to avoid binary bloat in users of the ``drjit::Texture<..>`` API.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <drjit/extra.h>
#include <drjit/texture_impl.h>
#include <drjit/idiv.h>
#include <drjit-core/half.h>
#include <drjit-core/texture.h>

namespace dr = drjit;

NAMESPACE_BEGIN()

/// Largest supported texture dimension
static constexpr uint32_t MaxDim = 3;

/// Backend- and type-erased differentiable float type.
using Float = dr::DiffArray<JitBackend::None, void>;

/// Integer siblings and masks stay typed
using Int   = dr::int32_array_t<Float>;
using UInt  = dr::uint32_array_t<Float>;
using Mask  = dr::mask_t<Float>;

using dr::detail::tex_scratch;

/// Create a floating-point literal of the runtime query precision
static Float query_scalar(JitBackend backend, VarType type, double value) {
    switch (type) {
        case VarType::Float16: return Float::steal(jit_var_f16(backend, dr::half(value)));
        case VarType::Float32: return Float::steal(jit_var_f32(backend, (float) value));
        case VarType::Float64: return Float::steal(jit_var_f64(backend, value));
        default: jit_raise("ad_tex_eval(): unsupported query type!");
    }
    return Float();
}

/// Cast an AD variable index to the runtime query precision
static Float to_query(uint64_t index, VarType type) {
    return Float::steal(ad_var_cast(index, type));
}

/// Cast ``dim`` query coordinates to single precision (the hardware texture
/// units always sample in F32), returning the resulting arrays and JIT indices.
static void pos_to_f32(const Float *pos, uint32_t dim, GenericArray<float> *out,
                       uint32_t *out_idx) {
    for (uint32_t k = 0; k < dim; ++k) {
        out[k] = GenericArray<float>::steal(
            jit_var_cast(pos[k].index(), VarType::Float32, 0));
        out_idx[k] = out[k].index();
    }
}

/// Take ownernship of ``channels_stored`` evaluations and return the first
/// ``channels_out`` of them in the requested query precision
template <typename T>
void finalize_lookup(const T *tmp, uint32_t channels_stored, uint32_t channels_out,
                     VarType query_type, Float *out) {
    for (uint32_t ch = 0; ch < channels_stored; ++ch) {
        Float texel = Float::steal((uint64_t) tmp[ch]);
        if (ch < channels_out)
            out[ch] = to_query(texel.index_combined(), query_type);
    }
}

/// Operations object for generating differentiable texture evaluation code
struct JitOps {
    using Float = ::Float;
    using Int   = ::Int;
    using UInt  = ::UInt;
    using Mask  = ::Mask;

    JitBackend backend;
    VarType query_type;
    uint32_t dim, channels_stored, channels_out;
    dr::FilterMode filter_mode;
    dr::WrapMode wrap_mode;
    uint64_t value;
    Mask active;
    Float res_f_[MaxDim];
    Int res_i_[MaxDim];
    dr::divisor<Int> inv_res_[MaxDim];

    Float lit(double v) const { return query_scalar(backend, query_type, v); }
    Float res_f(uint32_t k) const { return res_f_[k]; }
    Int res_i(uint32_t k) const { return res_i_[k]; }
    Float to_float(const Int &i) const { return to_query(i.index_combined(), query_type); }

    /// Floor-divide by the (opaque) resolution via the magic constants (see idiv.h)
    Int idiv(const Int &a, uint32_t k) const { return inv_res_[k](a); }

    /// Gather the ``channels_out`` texels at ``idx`` and cast them to query precision
    void gather(const UInt &idx, Float *out) const {
        uint64_t *tmp = (uint64_t *) alloca(sizeof(uint64_t) * channels_stored);

        // ``gather_packet`` requires a packet size of at least two
        if (channels_stored > 1)
            ad_var_gather_packet(channels_stored, value, idx.index(),
                                 active.index(), tmp, ReduceMode::Auto);
        else
            tmp[0] = ad_var_gather(value, idx.index(), active.index(),
                                   ReduceMode::Auto);

        finalize_lookup(tmp, channels_stored, channels_out, query_type, out);
    }
};

/// Create the ``Ops`` object for ``ad_tex_*`` functions.
static JitOps tex_setup(VarType query_type, uint32_t dim, uint32_t channels_stored,
                        uint32_t channels_out, int filter_mode,
                        int wrap_mode, uint64_t value,
                        const uint32_t *res_idx, const uint32_t *idiv_idx,
                        const uint64_t *pos_idx, uint32_t active_idx,
                        Float *pos) {
    JitOps ops;
    ops.backend = jit_set_backend(res_idx[0]).backend;
    ops.query_type = query_type;
    ops.dim = dim;
    ops.channels_out = channels_out;
    ops.channels_stored = channels_stored;
    ops.filter_mode = (dr::FilterMode) filter_mode;
    ops.wrap_mode = (dr::WrapMode) wrap_mode;
    ops.value = value;
    bool divides = ops.wrap_mode != dr::WrapMode::Clamp;
    for (uint32_t k = 0; k < dim; ++k) {
        UInt res = UInt::borrow(res_idx[k]);
        pos[k] = Float::borrow(pos_idx[k]);
        ops.res_f_[k] = to_query(res.index_combined(), query_type);
        ops.res_i_[k] = Int(res);
        if (divides) {
            ops.inv_res_[k].multiplier = Int::borrow(idiv_idx[2 * k + 0]);
            ops.inv_res_[k].shift      = Int::borrow(idiv_idx[2 * k + 1]);
        }
    }
    ops.active = Mask::borrow(active_idx);
    return ops;
}

/// Could gradient tracking be active on the texture data or a query coordinate?
static bool any_grad(uint64_t value, const uint64_t *pos_idx, uint32_t dim) {
    // Conservative: true if any operand carries an AD index and AD is not
    // globally suspended (avoids per-index ad_grad_enabled() queries).
    uint64_t combined = value;
    for (uint32_t k = 0; k < dim; ++k)
        combined |= pos_idx[k];
    return (combined >> 32) != 0 && !ad_grad_suspended();
}

/// Type-erased ``replace_grad``: splice ``a``'s primal onto ``b``'s gradient
static Float reattach(const Float &a, const Float &b) {
    return Float::borrow(((uint64_t) a.index()) | (((uint64_t) b.index_ad()) << 32));
}

/**
 * \brief Sample the hardware texture at the ``dim`` query coordinates ``pos``.
 *
 * Casts ``pos`` to float32 (the sampling units require single precision),
 * fetches ``channels_stored`` texels per query, and returns the leading
 * ``channels_out`` of them via ``out`` in the query precision, masked by ``active``.
 */
static void tex_eval_accel(void *handle,
                           uint32_t channels_stored,
                           uint32_t channels_out,
                           VarType query_type,
                           uint32_t dim,
                           const Float *pos,
                           const Mask &active,
                           Float *out) {
    GenericArray<float> pos_f32[MaxDim];
    uint32_t pos_idx32[MaxDim];
    pos_to_f32(pos, dim, pos_f32, pos_idx32);

    uint32_t *tmp = (uint32_t *) alloca(sizeof(uint32_t) * channels_stored);
    jit_tex_lookup(handle, pos_idx32, active.index(), tmp);

    finalize_lookup(tmp, channels_stored, channels_out, query_type, out);
}

NAMESPACE_END()

void ad_tex_eval(VarType query_type, uint32_t dim, uint32_t channels_stored,
                 uint32_t channels_out, int filter_mode, int wrap_mode,
                 void *handle, int use_accel, uint64_t value,
                 const uint32_t *res_idx, const uint32_t *idiv_idx,
                 const uint64_t *pos_idx, uint32_t active_idx,
                 uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             filter_mode, wrap_mode, value, res_idx, idiv_idx,
                             pos_idx, active_idx, pos);

    bool accel = handle != nullptr && use_accel;

    Float *result_mem = (Float *) alloca(sizeof(Float) * channels_out);
    tex_scratch<Float> result(result_mem, channels_out);

    if (accel && !any_grad(value, pos_idx, dim)) {
        tex_eval_accel(handle, channels_stored, channels_out, query_type, dim, pos,
                     ops.active, result.data());
    } else {
        // AD case: perform a non-accelerated lookup with gradient tracking and
        // splice the accelerated result into the primal component if possible.
        Float *scratch_mem = (Float *) alloca(sizeof(Float) * channels_out);
        tex_scratch<Float> scratch(scratch_mem, channels_out);
        dr::detail::tex_eval(ops, pos, result.data(), scratch.data());

        if (accel) {
            tex_eval_accel(handle, channels_stored, channels_out, query_type, dim,
                           pos, ops.active, scratch.data());
            for (uint32_t ch = 0; ch < channels_out; ++ch)
                result[ch] = reattach(scratch[ch], result[ch]);
        }
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out_idx[ch] = result[ch].release();
}

void ad_tex_fetch(VarType query_type, uint32_t dim, uint32_t channels_stored,
                  uint32_t channels_out, int wrap_mode, void *handle,
                  int use_accel, uint64_t value, const uint32_t *res_idx,
                  const uint32_t *idiv_idx, const uint64_t *pos_idx,
                  uint32_t active_idx, uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             (int) dr::FilterMode::Linear, wrap_mode, value, res_idx,
                             idiv_idx, pos_idx, active_idx, pos);

    uint32_t ncorner = 1u << dim;

    // Hardware-accelerated corner fetch (always single precision)
    auto fetch_accel = [&](Float *out) {
        if (dim == 2) {
            // A single bilinear-fetch instruction returns all four corners
            GenericArray<float> pos_f32[2];
            uint32_t pos_idx32[2];
            pos_to_f32(pos, 2, pos_f32, pos_idx32);
            uint32_t *tmp = (uint32_t *) alloca(4 * channels_stored * sizeof(uint32_t));
            jit_tex_bilerp_fetch(handle, pos_idx32, active_idx, tmp);

            for (uint32_t ch = 0; ch < channels_stored; ++ch) {
                Float v[4];
                for (uint32_t j = 0; j < 4; ++j)
                    v[j] = Float::steal((uint64_t) tmp[ch * 4 + j]);
                if (ch < channels_out) {
                    out[2 * channels_out + ch] = to_query(v[0].index_combined(), query_type);
                    out[3 * channels_out + ch] = to_query(v[1].index_combined(), query_type);
                    out[1 * channels_out + ch] = to_query(v[2].index_combined(), query_type);
                    out[0 * channels_out + ch] = to_query(v[3].index_combined(), query_type);
                }
            }
        } else {
            // 1D/3D fallback: one hardware lookup per corner at its sample position
            Float pos_f[MaxDim], inv[MaxDim];
            for (uint32_t k = 0; k < dim; ++k) {
                pos_f[k] = dr::floor(dr::fmadd(pos[k], ops.res_f(k), ops.lit(-0.5))) + ops.lit(0.5);
                inv[k] = dr::rcp(ops.res_f(k));
            }
            for (uint32_t corner = 0; corner < ncorner; ++corner) {
                Float cp[MaxDim];
                for (uint32_t k = 0; k < dim; ++k)
                    cp[k] = (pos_f[k] + ops.lit((double) ((corner >> k) & 1))) * inv[k];
                tex_eval_accel(handle, channels_stored, channels_out, query_type, dim,
                             cp, ops.active, out + corner * channels_out);
            }
        }
    };

    bool accel = (handle != nullptr && use_accel);

    Float *result_mem = (Float *) alloca(sizeof(Float) * ncorner * channels_out);
    tex_scratch<Float> result(result_mem, ncorner * channels_out);
    Float *ptrs[1u << MaxDim];
    if (accel && !any_grad(value, pos_idx, dim)) {
        fetch_accel(result.data());
    } else {
        // Arithmetic corner fetch. When also accelerated, its gradient is
        // spliced onto the hardware corners (sampled into ``diff``).
        for (uint32_t corner = 0; corner < ncorner; ++corner)
            ptrs[corner] = result.data() + corner * channels_out;
        dr::detail::tex_fetch(ops, pos, ptrs);
        if (accel) {
            Float *diff_mem = (Float *) alloca(sizeof(Float) * ncorner * channels_out);
            tex_scratch<Float> diff(diff_mem, ncorner * channels_out);
            fetch_accel(diff.data());
            for (uint32_t i = 0; i < ncorner * channels_out; ++i)
                result[i] = reattach(diff[i], result[i]);
        }
    }

    for (uint32_t i = 0; i < ncorner * channels_out; ++i)
        out_idx[i] = result[i].release();
}

void ad_tex_wrap(uint32_t dim, int wrap_mode, const uint32_t *res_idx,
                 const uint32_t *idiv_idx, const uint32_t *pos_idx,
                 uint32_t *out_idx) {
    // Only the integer-coordinate parts of ``JitOps`` are needed here (res_i,
    // idiv, wrap_mode); the float/gather machinery is left default-initialized.
    JitOps ops;
    ops.backend = jit_set_backend(res_idx[0]).backend;
    ops.dim = dim;
    ops.wrap_mode = (dr::WrapMode) wrap_mode;
    bool divides = ops.wrap_mode != dr::WrapMode::Clamp;

    Int pos[MaxDim];
    for (uint32_t k = 0; k < dim; ++k) {
        ops.res_i_[k] = Int(UInt::borrow(res_idx[k]));
        pos[k] = Int::borrow(pos_idx[k]);
        if (divides) {
            ops.inv_res_[k].multiplier = Int::borrow(idiv_idx[2 * k + 0]);
            ops.inv_res_[k].shift      = Int::borrow(idiv_idx[2 * k + 1]);
        }
    }

    // The result is a pure integer variable (no AD component); hand back the
    // owning JIT index.
    for (uint32_t k = 0; k < dim; ++k)
        out_idx[k] = (uint32_t) dr::detail::tex_wrap(ops, pos[k], k).release();
}

void ad_tex_cubic(VarType query_type, uint32_t dim, uint32_t channels_stored,
                  uint32_t channels_out, int wrap_mode, void *handle,
                  int use_accel, uint64_t value, const uint32_t *res_idx,
                  const uint32_t *idiv_idx, const uint64_t *pos_idx,
                  uint32_t active_idx, uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             (int) dr::FilterMode::Linear, wrap_mode, value, res_idx,
                             idiv_idx, pos_idx, active_idx, pos);

    bool accel = (handle != nullptr && use_accel);

    Float *result_mem = (Float *) alloca(sizeof(Float) * channels_out);
    tex_scratch<Float> result(result_mem, channels_out);

    if (!accel) {
        // Without hardware texture units, regular B-spline interpolation is
        // faster than the trick below.
        Float *scratch_mem = (Float *) alloca(sizeof(Float) * channels_out);
        tex_scratch<Float> scratch(scratch_mem, channels_out);
        dr::detail::tex_eval_cubic(ops, pos, result.data(), scratch.data());
    } else {
        // GPU Gems 2, Ch. 20: collapse the 4 cubic taps per dimension into two
        // hardware bilinear lookups with weight ``w01`` and sample coordinates
        // ``coord_{lo,hi}``
        Float w01[MaxDim], coord_lo[MaxDim], coord_hi[MaxDim];
        for (uint32_t k = 0; k < dim; ++k) {
            Float pos_f = dr::fmadd(pos[k], ops.res_f(k), ops.lit(-0.5));
            Int pos_i = dr::floor2int<Int>(pos_f);
            Float integ = ops.to_float(pos_i), inv = dr::rcp(ops.res_f(k));
            Float w[4];
            dr::detail::tex_cubic_weights(ops, pos_f - integ, w);
            Float w_lo = w[0] + w[1], w_hi = ops.lit(1.0) - w_lo;
            w01[k]      = w_lo;
            coord_lo[k] = (integ - ops.lit(0.5) + w[1] / w_lo) * inv;
            coord_hi[k] = (integ + ops.lit(1.5) + w[3] / w_hi) * inv;
        }

        // Evaluate the 2^dim hardware bilinear lookups at the transformed coords
        uint32_t ncorner = 1u << dim;
        Float *f_mem = (Float *) alloca(sizeof(Float) * ncorner * channels_out);
        tex_scratch<Float> f(f_mem, ncorner * channels_out);
        for (uint32_t corner = 0; corner < ncorner; ++corner) {
            Float cp[MaxDim];
            for (uint32_t k = 0; k < dim; ++k)
                cp[k] = ((corner >> k) & 1) ? coord_hi[k] : coord_lo[k];
            tex_eval_accel(handle, channels_stored, channels_out, query_type, dim, cp,
                           ops.active, f.data() + corner * channels_out);
        }

        // Separable lerp reduction over dimensions (lower corner weighted by w01)
        for (uint32_t d = 0; d < dim; ++d) {
            uint32_t bit = 1u << d, mask = (bit << 1) - 1;
            for (uint32_t corner = 0; corner < ncorner; ++corner) {
                if (corner & mask)
                    continue;
                for (uint32_t ch = 0; ch < channels_out; ++ch) {
                    Float &lo = f[corner * channels_out + ch], &hi = f[(corner | bit) * channels_out + ch];
                    lo = dr::fmadd(lo - hi, w01[d], hi);
                }
            }
        }

        for (uint32_t ch = 0; ch < channels_out; ++ch)
            result[ch] = f[ch];

        // The transform is non-linear in `pos`, so replace its AD graph with a
        // direct B-spline evaluation when gradient tracking is on (``f`` is free
        // after the reduction, so reuse it for the recompute).
        if (any_grad(value, pos_idx, dim)) {
            Float *scratch_mem = (Float *) alloca(sizeof(Float) * channels_out);
            tex_scratch<Float> scratch(scratch_mem, channels_out);
            dr::detail::tex_eval_cubic(ops, pos, f.data(), scratch.data());
            for (uint32_t ch = 0; ch < channels_out; ++ch)
                result[ch] = reattach(result[ch], f[ch]);
        }
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out_idx[ch] = result[ch].release();
}

void ad_tex_cubic_deriv(VarType query_type, uint32_t dim, uint32_t channels_stored,
                        uint32_t channels_out, int wrap_mode,
                        uint64_t value, const uint32_t *res_idx,
                        const uint32_t *idiv_idx, const uint64_t *pos_idx,
                        uint32_t active_idx, uint64_t *out_value,
                        uint64_t *out_grad, uint64_t *out_hess) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             (int) dr::FilterMode::Linear, wrap_mode, value, res_idx,
                             idiv_idx, pos_idx, active_idx, pos);

    bool want_hess = out_hess != nullptr;
    Float *value_mem = (Float *) alloca(sizeof(Float) * channels_out);
    Float *grad_mem  = (Float *) alloca(sizeof(Float) * channels_out * dim);
    Float *hess_mem  = want_hess ? (Float *) alloca(sizeof(Float) * channels_out * dim * dim)
                                 : nullptr;
    Float *scratch_mem = (Float *) alloca(sizeof(Float) * channels_out);
    tex_scratch<Float> value_out(value_mem, channels_out),
        grad_out(grad_mem, channels_out * dim),
        hess_out(hess_mem, want_hess ? channels_out * dim * dim : 0),
        scratch(scratch_mem, channels_out);
    dr::detail::tex_eval_cubic_deriv(ops, pos, value_out.data(), grad_out.data(),
                                     want_hess ? hess_out.data() : nullptr,
                                     scratch.data());

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out_value[ch] = value_out[ch].release();
    for (uint32_t i = 0; i < channels_out * dim; ++i)
        out_grad[i] = grad_out[i].release();
    if (want_hess)
        for (uint32_t i = 0; i < channels_out * dim * dim; ++i)
            out_hess[i] = hess_out[i].release();
}

void ad_tex_write(uint32_t channels_stored, uint32_t channels_out, void *handle,
                  const uint32_t *pos_idx, const uint64_t *value,
                  uint32_t active_idx) {
    using F32 = GenericArray<float>;
    jit_set_backend(pos_idx[0]);

    F32 *vals_mem = (F32 *) alloca(sizeof(F32) * channels_stored);
    tex_scratch<F32> vals(vals_mem, channels_stored);
    uint32_t *val_idx = (uint32_t *) alloca(channels_stored * sizeof(uint32_t));
    for (uint32_t ch = 0; ch < channels_stored; ++ch) {
        vals[ch] = ch < channels_out
                       ? F32::steal(jit_var_cast((uint32_t) value[ch],
                                                 VarType::Float32, 0))
                       : dr::zeros<F32>();
        val_idx[ch] = vals[ch].index();
    }

    jit_tex_write(handle, pos_idx, val_idx, active_idx);
}

uint64_t ad_tex_repack(uint64_t source, uint32_t n_pixels, uint32_t dst_channels,
                       uint32_t src_channels) {
    JitBackend backend = jit_set_backend((uint32_t) source).backend;

    UInt idx = UInt::steal(
        jit_var_counter(backend, (size_t) n_pixels * dst_channels));
    UInt pixel = idx / dst_channels, channel = idx % dst_channels;
    Mask active = channel < src_channels; // zero-fill padding lanes
    UInt src_idx = dr::fmadd(pixel, src_channels, channel);

    return ad_var_gather(source, src_idx.index(), active.index(),
                         ReduceMode::Auto);
}
