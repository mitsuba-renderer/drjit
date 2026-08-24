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
#include <drjit/color.h>
#include <drjit/math.h>
#include <drjit-core/half.h>
#include <drjit-core/texture.h>
#include <memory>

namespace dr = drjit;

namespace {

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

/// The type-erased ``Float`` re-tagged with a concrete precision ``T``
template <typename T> using TypedFloat = dr::DiffArray<JitBackend::None, T>;

/// sRGB -> linear, dispatched to the typed ``dr::srgb_to_linear`` since
/// ``Float`` is type-erased
static Float srgb_to_linear(const Float &x, VarType query_type) {
    auto decode = [](const auto &v) {
        return Float::steal(dr::srgb_to_linear(v).release());
    };
    switch (query_type) {
        case VarType::Float16: {
            // Decode in single precision, then cast back
            Float xf = to_query(x.index_combined(), VarType::Float32);
            Float yf = decode(TypedFloat<float>::borrow(xf.index_combined()));
            return to_query(yf.index_combined(), VarType::Float16);
        }
        case VarType::Float32: return decode(TypedFloat<float>::borrow(x.index_combined()));
        case VarType::Float64: return decode(TypedFloat<double>::borrow(x.index_combined()));
        default: jit_raise("ad_tex_eval(): unsupported query type!");
    }
    return Float();
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

/// Operations object for generating differentiable texture evaluation code.
/// It owns references to every variable it mentions, so copies (e.g. those
/// captured by a retained ``sum_loop`` body) remain valid after the
/// originating ``ad_tex_*`` call returns.
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
    bool unorm8 = false;
    bool srgb = false;
    Float value;
    Mask active;
    Float res_f_[MaxDim];
    Int res_i_[MaxDim];
    dr::divisor<Int> inv_res_[MaxDim];

    // MIP pyramid state (see the ``mip_*`` parameters of ``ad_tex_eval_lod``)
    Float mip_value;
    Int mip_table;
    dr::detail::TexLevel<Int, UInt> lvl;

    Float lit(double v) const { return query_scalar(backend, query_type, v); }
    Int lit_i(int32_t v) const { return Int::steal(jit_var_i32(backend, v)); }
    Float res_f(uint32_t k) const {
        return lvl.bound ? to_float(res_i(k)) : res_f_[k];
    }
    Int res_i(uint32_t k) const {
        Int r = res_i_[k];
        if (lvl.bound)
            r = dr::maximum(r >> lvl.level, lit_i(1));
        return r;
    }
    Float to_float(const Int &i) const { return to_query(i.index_combined(), query_type); }

    /// Floor-divide by the (opaque) resolution via the magic constants (see idiv.h)
    Int idiv(const Int &a, uint32_t k) const {
        if (lvl.bound)
            return dr::detail::tex_idiv_dynamic(*this, lvl.div[k][0],
                                                lvl.div[k][1], a);
        return inv_res_[k](a);
    }

    /// Gather the ``channels_out`` texels at ``idx`` and cast them to query precision
    void gather(const UInt &idx, Float *out) const {
        if (!lvl.bound) {
            gather_from(value, idx, active, out);
        } else if (lvl.includes_base) {
            // The bound level may be the base level, whose texels live in the
            // regular texture storage rather than the pyramid buffer
            Mask is_base = lvl.level == 0;
            gather_from(value, idx, active && is_base, out);

            Float *tmp_mem = (Float *) alloca(sizeof(Float) * channels_out);
            tex_scratch<Float> tmp(tmp_mem, channels_out);
            gather_from(mip_value, idx + lvl.offset, active && !is_base,
                        tmp.data());
            for (uint32_t ch = 0; ch < channels_out; ++ch)
                out[ch] = dr::select(is_base, out[ch], tmp[ch]);
        } else {
            gather_from(mip_value, idx + lvl.offset, active, out);
        }

        // Map 8-bit values to [0, 1]. For sRGB textures, apply the transfer
        // curve to linearize. To replicate how GPUs do this, skip every 4th
        // channel (A in RGBA).
        if (unorm8) {
            for (uint32_t ch = 0; ch < channels_out; ++ch) {
                out[ch] = out[ch] * lit(1.0 / 255.0);
                if (srgb && (ch % 4) != 3)
                    out[ch] = srgb_to_linear(out[ch], query_type);
            }
        }
    }

    /// Load the configuration of MIP level ``l`` (see the ``Ops`` contract)
    void mip_record(const Int &l, Int *rec) const {
        uint32_t stride = dim == 1 ? 4 : 8, tmp[8];
        jit_var_gather_packet(stride, mip_table.index(), l.index(),
                              active.index(), tmp);
        for (uint32_t j = 0; j < stride; ++j)
            rec[j] = Int::steal(tmp[j]);
    }

    /// Sum ``body(i, m, state, scratch)`` for ``i = 0, ..., n - 1``
    template <typename Body>
    void sum_loop(const Int &n, Float *state, uint32_t n_state,
                  uint32_t n_scratch, Body body) const {
        struct Payload {
            JitBackend backend;
            Int i, n;
            uint32_t n_state, n_scratch;
            std::unique_ptr<Float[]> state, scratch;
            Body body;
            Mask cond;
        };

        Payload *p = new Payload{ backend, lit_i(0), n, n_state, n_scratch,
                                  std::unique_ptr<Float[]>(new Float[n_state]),
                                  std::unique_ptr<Float[]>(new Float[n_scratch]),
                                  std::move(body), Mask() };
        for (uint32_t j = 0; j < n_state; ++j)
            p->state[j] = state[j];

        ad_loop_read read_cb = [](void *q, dr::vector<uint64_t> &indices) {
            Payload *pl = (Payload *) q;
            indices.push_back(ad_var_inc_ref(pl->i.index_combined()));
            for (uint32_t j = 0; j < pl->n_state; ++j)
                indices.push_back(ad_var_inc_ref(pl->state[j].index_combined()));
        };

        ad_loop_write write_cb = [](void *q, const dr::vector<uint64_t> &indices,
                                    bool) {
            Payload *pl = (Payload *) q;
            pl->i = Int::borrow((uint32_t) indices[0]);
            for (uint32_t j = 0; j < pl->n_state; ++j)
                pl->state[j] = Float::borrow(indices[j + 1]);
        };

        ad_loop_cond cond_cb = [](void *q) -> uint32_t {
            Payload *pl = (Payload *) q;
            pl->cond = pl->i < pl->n;
            return pl->cond.index();
        };

        ad_loop_body body_cb = [](void *q) {
            Payload *pl = (Payload *) q;
            Mask m = Mask::steal(jit_var_bool(pl->backend, true));
            pl->body(pl->i, m, pl->state.get(), pl->scratch.get());
            pl->i = pl->i + 1;

            // Drop temporaries so that no reference to a variable of the
            // recorded loop body outlives the recording
            for (uint32_t j = 0; j < pl->n_scratch; ++j)
                pl->scratch[j] = Float();
        };

        ad_loop_delete delete_cb = [](void *q) { delete (Payload *) q; };

        bool all_done = ad_loop(backend, -1, 0, /* max_iterations */ -1,
                                "dr::Texture::eval_filtered()", p, read_cb,
                                write_cb, cond_cb, body_cb, delete_cb, true);

        for (uint32_t j = 0; j < n_state; ++j)
            state[j] = p->state[j];

        if (all_done) {
            delete p;
        } else {
            // Drop the state references. The loop will repopulate them later.
            for (uint32_t j = 0; j < n_state; ++j)
                p->state[j] = Float();
            p->i = Int();
            p->cond = Mask();
        }
    }

private:
    /// Raw packet gather of the stored channels at ``idx`` from the buffer
    /// ``source``, cast to the query precision
    void gather_from(const Float &source, const UInt &idx, const Mask &m,
                     Float *out) const {
        uint64_t *tmp = (uint64_t *) alloca(sizeof(uint64_t) * channels_stored);

        // ``gather_packet`` requires a packet size of at least two
        if (channels_stored > 1)
            ad_var_gather_packet(channels_stored, source.index_combined(),
                                 idx.index(), m.index(), tmp, ReduceMode::Auto);
        else
            tmp[0] = ad_var_gather(source.index_combined(), idx.index(),
                                   m.index(), ReduceMode::Auto);

        finalize_lookup(tmp, channels_stored, channels_out, query_type, out);
    }
};

/// Create the ``Ops`` object for ``ad_tex_*`` functions.
static JitOps tex_setup(VarType query_type, uint32_t dim, uint32_t channels_stored,
                        uint32_t channels_out, int filter_mode,
                        int wrap_mode, int srgb, uint64_t value,
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
    ops.unorm8 = jit_var_type((uint32_t) value) == VarType::UInt8;
    ops.srgb = srgb != 0;
    ops.value = Float::borrow(value);
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

/// Variant of \ref tex_eval_accel() sampling at an explicit level of detail
static void tex_eval_lod_accel(void *handle,
                               uint32_t channels_stored,
                               uint32_t channels_out,
                               VarType query_type,
                               uint32_t dim,
                               const Float *pos,
                               const Float &lod,
                               const Mask &active,
                               Float *out) {
    GenericArray<float> pos_f32[MaxDim];
    uint32_t pos_idx32[MaxDim];
    pos_to_f32(pos, dim, pos_f32, pos_idx32);
    GenericArray<float> lod_f32 = GenericArray<float>::steal(
        jit_var_cast(lod.index(), VarType::Float32, 0));

    uint32_t *tmp = (uint32_t *) alloca(sizeof(uint32_t) * channels_stored);
    jit_tex_lookup_lod(handle, pos_idx32, lod_f32.index(), active.index(), tmp);

    finalize_lookup(tmp, channels_stored, channels_out, query_type, out);
}

/// Variant of \ref tex_eval_accel() driven by screen-space derivatives
static void tex_eval_grad_accel(void *handle, uint32_t channels_stored,
                                uint32_t channels_out, VarType query_type,
                                uint32_t dim, const Float *pos,
                                const Float *ddx, const Float *ddy,
                                const Mask &active, Float *out) {
    GenericArray<float> f32[3 * MaxDim];
    uint32_t pos_idx32[MaxDim], ddx_idx32[MaxDim], ddy_idx32[MaxDim];
    pos_to_f32(pos, dim, f32, pos_idx32);
    pos_to_f32(ddx, dim, f32 + MaxDim, ddx_idx32);
    pos_to_f32(ddy, dim, f32 + 2 * MaxDim, ddy_idx32);

    uint32_t *tmp = (uint32_t *) alloca(sizeof(uint32_t) * channels_stored);
    jit_tex_lookup_grad(handle, pos_idx32, ddx_idx32, ddy_idx32,
                        active.index(), tmp);

    finalize_lookup(tmp, channels_stored, channels_out, query_type, out);
}

} // anonymous namespace

void ad_tex_eval(VarType query_type, uint32_t dim, uint32_t channels_stored,
                 uint32_t channels_out, int filter_mode, int wrap_mode, int srgb,
                 void *handle, int use_accel, uint64_t value,
                 const uint32_t *res_idx, const uint32_t *idiv_idx,
                 const uint64_t *pos_idx, uint32_t active_idx,
                 uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             filter_mode, wrap_mode, srgb, value, res_idx, idiv_idx,
                             pos_idx, active_idx, pos);

    bool accel = handle != nullptr && use_accel;

    Float *result_mem = (Float *) alloca(sizeof(Float) * channels_out);
    tex_scratch<Float> result(result_mem, channels_out);

    if (accel && !any_grad(value, pos_idx, dim)) {
        tex_eval_accel(handle, channels_stored, channels_out, query_type, dim, pos,
                     ops.active, result.data());
    } else {
        // AD case: perform a non-accelerated lookup with gradient tracking and
        // splice the accelerated result into the primal component when
        // hardware texture lookups are used.
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

namespace {

using U32 = GenericArray<uint32_t>;
using I32 = GenericArray<int32_t>;

/// Decompose the flat texel index ``o`` into per-dimension coordinates
/// (fastest-varying axis first)
static void tex_unpack_coords(const U32 &o, const size_t *res, uint32_t dim,
                              U32 *out) {
    if (dim == 1) {
        out[0] = o;
    } else if (dim == 2) {
        out[1] = o / (uint32_t) res[0];
        out[0] = o - out[1] * (uint32_t) res[0];
    } else {
        uint32_t slice = (uint32_t) (res[0] * res[1]);
        out[2] = o / slice;
        U32 r = o - out[2] * slice;
        out[1] = r / (uint32_t) res[0];
        out[0] = r - out[1] * (uint32_t) res[0];
    }
}

/**
 * \brief Magnify a MIP level by a factor of 2 using bilinear interpolation
 *
 * This function is needed to both create and decompose the Laplacian MIP map
 * representation. It provides the operator ``U`` of the analysis pass ``B_l =
 * G_l - U(G_{l+1})`` and the synthesis pass ``G_l = B_l + U(G_{l+1})``.
 *
 * The implementation gathers directly from the texel array. Output texel ``f``
 * maps to the continuous coarse-level position ``(f + 0.5) * r_coarse / r_fine
 * - 0.5`` along each dimension, and the ``2^dim`` enclosing texels are combined
 * with the corresponding bilinear weights. The mapping aligns the two grids at
 * their texel centers, which produces weights of 3/4 and 1/4 per dimension when
 * the resolution halves exactly. Taps that fall outside the coarse level are
 * clamped to its bounds.
 *
 * The input ``coarse`` holds ``stride`` interleaved channels at resolution
 * ``cres``. The return value is an unevaluated ``accum_type`` expression over
 * ``n_texels(fres) * stride`` elements, where each element recovers its texel
 * and channel from its own index.
 */
static Float tex_upsample_bilinear(JitBackend backend, VarType accum_type,
                                   const Float &coarse, const size_t *cres,
                                   const size_t *fres, uint32_t dim,
                                   uint32_t stride) {
    size_t n = fres[0] * fres[1] * fres[2];
    Mask all = Mask::steal(jit_var_bool(backend, true));

    U32 e = U32::steal(jit_var_counter(backend, n * stride)),
        texel = e / stride, ch = e - texel * stride, coords[3];
    tex_unpack_coords(texel, fres, dim, coords);

    // Per-dimension tap indices and interpolation weights
    U32 lo[3], hi[3];
    Float t[3], u[3];
    for (uint32_t k = 0; k < dim; ++k) {
        double ratio = (double) cres[k] / (double) fres[k];
        Float c = dr::fmadd(
            to_query(coords[k].index(), accum_type),
            query_scalar(backend, accum_type, ratio),
            query_scalar(backend, accum_type, 0.5 * ratio - 0.5));
        Int i0e = dr::floor2int<Int>(c);
        t[k] = c - to_query(i0e.index_combined(), accum_type);
        u[k] = query_scalar(backend, accum_type, 1.0) - t[k];

        I32 i0 = I32::borrow(i0e.index());
        int32_t top = (int32_t) cres[k] - 1;
        lo[k] = U32(dr::clip(i0, 0, top));
        hi[k] = U32(dr::clip(i0 + 1, 0, top));
    }

    Float acc = query_scalar(backend, accum_type, 0.0);
    for (uint32_t corner = 0; corner < (1u << dim); ++corner) {
        U32 idx = (corner & 1) ? hi[0] : lo[0];
        Float w = (corner & 1) ? t[0] : u[0];
        if (dim >= 2) {
            idx = dr::fmadd(((corner >> 1) & 1) ? hi[1] : lo[1],
                            (uint32_t) cres[0], idx);
            w = w * (((corner >> 1) & 1) ? t[1] : u[1]);
        }
        if (dim == 3) {
            idx = dr::fmadd(((corner >> 2) & 1) ? hi[2] : lo[2],
                            (uint32_t) (cres[0] * cres[1]), idx);
            w = w * (((corner >> 2) & 1) ? t[2] : u[2]);
        }

        U32 src = dr::fmadd(idx, stride, ch);
        Float tap = Float::steal(ad_var_gather(coarse.index_combined(),
                                               src.index(), all.index(),
                                               ReduceMode::Auto));

        acc = dr::fmadd(to_query(tap.index_combined(), accum_type), w, acc);
    }

    return acc;
}

/**
 * \brief Shrink a MIP level by a factor of 2 using a box filter
 *
 * This is the reduction step of the pyramid construction. It generates the
 * levels of an ordinary MIP map from a given base level, and it also builds
 * the Gaussian pyramid that the Laplacian analysis pass differences.
 *
 * Each output texel averages the ``2^dim`` texels that cover it in ``prev``.
 * When an input resolution is odd, the last tap along that axis is clamped
 * onto the boundary texel.
 *
 * The average is accumulated in ``accum_type``. If ``unorm8`` is set, the
 * texels are first decoded from their 8-bit normalized representation (and
 * from sRGB when ``srgb`` is set), and the result is re-encoded afterwards.
 * Every fourth channel holds alpha and bypasses the sRGB transfer function.
 *
 * The input ``prev`` holds ``stride`` interleaved channels at resolution
 * ``prev_res``. The return value is an unevaluated ``out_type`` expression
 * holding the ``res`` level in the same layout.
 */
static Float tex_downsample_box(JitBackend backend, VarType accum_type,
                                VarType out_type, const Float &prev,
                                const size_t *prev_res, const size_t *res,
                                uint32_t dim, uint32_t stride, bool unorm8,
                                bool srgb) {
    size_t n = res[0] * res[1] * res[2];
    uint32_t n_corners = 1u << dim;
    Mask all = Mask::steal(jit_var_bool(backend, true));

    U32 e = U32::steal(jit_var_counter(backend, n * stride)),
        texel = e / stride, ch = e - texel * stride, coords[3];
    tex_unpack_coords(texel, res, dim, coords);

    Mask is_alpha;
    if (srgb)
        is_alpha = (ch & 3u) == 3u;

    Float acc = query_scalar(backend, accum_type, 0.0);
    for (uint32_t corner = 0; corner < n_corners; ++corner) {
        U32 idx = dr::minimum(2 * coords[0] + (corner & 1),
                              (uint32_t) (prev_res[0] - 1));
        if (dim >= 2) {
            U32 sy = dr::minimum(2 * coords[1] + ((corner >> 1) & 1),
                                 (uint32_t) (prev_res[1] - 1));
            idx = dr::fmadd(sy, (uint32_t) prev_res[0], idx);
        }
        if (dim == 3) {
            U32 sz = dr::minimum(2 * coords[2] + ((corner >> 2) & 1),
                                 (uint32_t) (prev_res[2] - 1));
            idx = dr::fmadd(sz, (uint32_t) (prev_res[0] * prev_res[1]), idx);
        }

        U32 src = dr::fmadd(idx, stride, ch);
        Float tap = Float::steal(ad_var_gather(prev.index_combined(),
                                               src.index(), all.index(),
                                               ReduceMode::Auto)),
              v = to_query(tap.index_combined(), accum_type);
        if (unorm8) {
            v = v * query_scalar(backend, accum_type, 1.0 / 255.0);
            if (srgb)
                v = dr::select(is_alpha, v, srgb_to_linear(v, accum_type));
        }
        acc = acc + v;
    }

    acc = acc * query_scalar(backend, accum_type, 1.0 / n_corners);
    if (unorm8) {
        acc = dr::clip(acc, query_scalar(backend, accum_type, 0.0),
                       query_scalar(backend, accum_type, 1.0));
        if (srgb) {
            Float enc = Float::steal(
                dr::linear_to_srgb(
                    TypedFloat<float>::borrow(acc.index_combined()))
                    .release());
            acc = dr::select(is_alpha, acc, enc);
        }
        acc = dr::fmadd(acc, query_scalar(backend, accum_type, 255.0),
                        query_scalar(backend, accum_type, 0.5));
    }

    return Float::steal(ad_var_cast(acc.index_combined(), out_type));
}

/// Fill ``lres`` with the ``n_levels`` per-level resolutions (3 entries per
/// level, fastest axis first) implied by the base resolution ``res_in``
static void tex_level_res(const size_t *res_in, uint32_t dim,
                          uint32_t n_levels, size_t *lres) {
    size_t r[3] = { 1, 1, 1 };
    for (uint32_t k = 0; k < dim; ++k)
        r[k] = res_in[k];
    for (uint32_t l = 0; l < n_levels; ++l) {
        for (int k = 0; k < 3; ++k) {
            lres[3 * l + k] = r[k];
            r[k] = r[k] > 1 ? r[k] >> 1 : 1;
        }
    }
}

} // anonymous namespace

uint64_t ad_tex_mipmap_from_base(uint32_t dim, uint32_t channels_stored, int srgb,
                                 uint64_t value, const size_t *res_in,
                                 uint32_t n_levels) {
    if (n_levels <= 1)
        return 0;

    JitBackend backend = jit_set_backend((uint32_t) value).backend;
    VarType storage_type = jit_var_type((uint32_t) value);
    bool unorm8 = storage_type == VarType::UInt8;

    // Box-filter accumulator: single precision, except for f64 storage
    VarType accum_type = storage_type == VarType::Float64 ? VarType::Float64
                                                          : VarType::Float32;

    uint32_t C = channels_stored;

    size_t *lres = (size_t *) alloca(sizeof(size_t) * 3 * n_levels);
    tex_level_res(res_in, dim, n_levels, lres);

    // Total texel count of the pyramid levels >= 1
    size_t total = 0;
    for (uint32_t l = 1; l < n_levels; ++l)
        total += lres[3 * l] * lres[3 * l + 1] * lres[3 * l + 2];

    Float mip = Float::steal(jit_var_undefined(backend, storage_type, total * C));
    Mask all = Mask::steal(jit_var_bool(backend, true));

    Float prev = Float::borrow(value);
    uint32_t offset = 0;
    for (uint32_t l = 1; l < n_levels; ++l) {
        uint32_t n = (uint32_t) (lres[3 * l] * lres[3 * l + 1] * lres[3 * l + 2]);

        Float level = tex_downsample_box(backend, accum_type, storage_type,
                                         prev, lres + 3 * (l - 1), lres + 3 * l,
                                         dim, C, unorm8, srgb != 0);

        // Append the level to the pyramid buffer
        U32 dst = U32::steal(jit_var_counter(backend, (size_t) n * C)) +
                  offset * C;

        mip = Float::steal(ad_var_scatter(mip.index_combined(),
                                          level.index_combined(), dst.index(),
                                          all.index(), ReduceOp::Identity,
                                          ReduceMode::Permute));

        // Materialize the level
        jit_var_eval(level.index());

        offset += n;
        prev = level;
    }

    return mip.release();
}

void ad_tex_mipmap_from_laplacian(uint32_t dim, uint32_t channels,
                                  uint32_t channels_stored,
                                  const uint64_t *coef, uint32_t n_levels,
                                  const size_t *res_in, uint64_t *out_base,
                                  uint64_t *out_mip) {
    JitBackend backend = jit_set_backend((uint32_t) coef[0]).backend;
    VarType storage_type = jit_var_type((uint32_t) coef[0]);
    VarType accum_type = storage_type == VarType::Float64 ? VarType::Float64
                                                          : VarType::Float32;
    uint32_t C = channels, Cs = channels_stored;

    size_t *lres = (size_t *) alloca(sizeof(size_t) * 3 * n_levels);
    tex_level_res(res_in, dim, n_levels, lres);

    // Total texel count of the pyramid levels >= 1
    size_t total = 0;
    for (uint32_t l = 1; l < n_levels; ++l)
        total += lres[3 * l] * lres[3 * l + 1] * lres[3 * l + 2];

    Mask all = Mask::steal(jit_var_bool(backend, true));
    Float mip;
    if (n_levels > 1)
        mip = Float::steal(
            jit_var_undefined(backend, storage_type, total * Cs));

    // Synthesize the Gaussian pyramid coarse to fine: G_{L-1} = B_{L-1},
    // G_l = B_l + U(G_{l+1}), in storage precision with padded channels
    Float prev;
    size_t offset = total;
    for (uint32_t l = n_levels; l-- > 0; ) {
        size_t n_l = lres[3 * l] * lres[3 * l + 1] * lres[3 * l + 2];

        // Coefficient level, padded to the storage layout
        Float b = Float::borrow(coef[l]);
        if (C != Cs)
            b = Float::steal(ad_tex_repack(coef[l], (uint32_t) n_l, Cs, C));
        b = to_query(b.index_combined(), accum_type);

        Float g = b;
        if (l != n_levels - 1)
            g = b + tex_upsample_bilinear(backend, accum_type, prev,
                                          lres + 3 * (l + 1), lres + 3 * l,
                                          dim, Cs);

        Float st = Float::steal(ad_var_cast(g.index_combined(), storage_type));
        if (l > 0) {
            // Append the level to the pyramid buffer
            offset -= n_l;
            U32 dst = U32::steal(jit_var_counter(backend, n_l * Cs)) +
                      (uint32_t) (offset * Cs);
            mip = Float::steal(ad_var_scatter(
                mip.index_combined(), st.index_combined(), dst.index(),
                all.index(), ReduceOp::Identity, ReduceMode::Permute));
        }

        // Materialize the level
        jit_var_eval(st.index());
        prev = st;
    }

    *out_base = prev.release();
    *out_mip = n_levels > 1 ? mip.release() : 0;
}

void ad_tex_laplacian_from_base(uint32_t dim, uint32_t channels,
                                uint64_t value, uint32_t n_levels,
                                const size_t *res_in, uint64_t *out) {
    JitBackend backend = jit_set_backend((uint32_t) value).backend;
    VarType storage_type = jit_var_type((uint32_t) value);
    VarType accum_type = storage_type == VarType::Float64 ? VarType::Float64
                                                          : VarType::Float32;
    uint32_t C = channels;

    size_t *lres = (size_t *) alloca(sizeof(size_t) * 3 * n_levels);
    tex_level_res(res_in, dim, n_levels, lres);

    // Gaussian chain, ignoring any AD component of ``value``
    std::unique_ptr<Float[]> g(new Float[n_levels]);
    g[0] = Float::steal(
        ad_var_cast((uint64_t) (uint32_t) value, accum_type));
    for (uint32_t l = 1; l < n_levels; ++l) {
        g[l] = tex_downsample_box(backend, accum_type, accum_type, g[l - 1],
                                  lres + 3 * (l - 1), lres + 3 * l, dim, C,
                                  /* unorm8 = */ false, /* srgb = */ false);
        jit_var_eval(g[l].index());
    }

    // Coefficients: B_l = G_l - U(G_{l+1}), B_{L-1} = G_{L-1}
    for (uint32_t l = 0; l < n_levels; ++l) {
        Float b = g[l];
        if (l + 1 < n_levels)
            b = b - tex_upsample_bilinear(backend, accum_type, g[l + 1],
                                          lres + 3 * (l + 1), lres + 3 * l,
                                          dim, C);
        out[l] = ad_var_cast(b.index_combined(), storage_type);
        jit_var_schedule((uint32_t) out[l]);
    }
}

void ad_tex_eval_lod(VarType query_type, uint32_t dim, uint32_t channels_stored,
                     uint32_t channels_out, int filter_mode, int wrap_mode,
                     int srgb, void *handle, int use_accel, uint64_t value,
                     uint64_t mip_value, uint32_t mip_table, uint32_t n_levels,
                     int mip_filter, const uint32_t *res_idx,
                     const uint32_t *idiv_idx, const uint64_t *pos_idx,
                     uint32_t lod_idx, uint32_t active_idx, uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                           filter_mode, wrap_mode, srgb, value, res_idx,
                           idiv_idx, pos_idx, active_idx, pos);
    ops.mip_value = Float::borrow(mip_value);
    ops.mip_table = Int::borrow(mip_table);

    Float *result_mem  = (Float *) alloca(sizeof(Float) * channels_out);
    Float *scratch_mem = (Float *) alloca(sizeof(Float) * 2 * channels_out);
    tex_scratch<Float> result(result_mem, channels_out),
                       scratch(scratch_mem, 2 * channels_out);

    if (n_levels > 1) {
        Float lod = Float::borrow(lod_idx);
        bool accel = handle != nullptr && use_accel,
             grad  = any_grad(value, pos_idx, dim) ||
                     any_grad(mip_value, nullptr, 0);

        if (accel && !grad) {
            tex_eval_lod_accel(handle, channels_stored, channels_out,
                               query_type, dim, pos, lod, ops.active,
                               result.data());
        } else {
            // AD case: perform a non-accelerated lookup with gradient tracking
            // and splice the accelerated result into the primal component when
            // hardware texture lookups are used.
            dr::detail::tex_eval_lod(ops, pos, lod, n_levels,
                                     (dr::MipFilter) mip_filter,
                                     result.data(), scratch.data());
            if (accel) {
                tex_eval_lod_accel(handle, channels_stored, channels_out,
                                   query_type, dim, pos, lod, ops.active,
                                   scratch.data());
                for (uint32_t ch = 0; ch < channels_out; ++ch)
                    result[ch] = reattach(scratch[ch], result[ch]);
            }
        }
    } else {
        dr::detail::tex_eval(ops, pos, result.data(), scratch.data());
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out_idx[ch] = result[ch].release();
}

void ad_tex_eval_filtered(VarType query_type, uint32_t dim,
                          uint32_t channels_stored, uint32_t channels_out,
                          int filter_mode, int wrap_mode, int srgb,
                          void *handle, int use_accel, uint64_t value,
                          uint64_t mip_value, uint32_t mip_table,
                          uint32_t n_levels, int mip_filter, uint32_t max_aniso,
                          const uint32_t *res_idx, const uint32_t *idiv_idx,
                          const uint64_t *pos_idx, const uint32_t *ddx_idx,
                          const uint32_t *ddy_idx, uint32_t active_idx,
                          uint64_t *out_idx) {
    Float pos[MaxDim], ddx[MaxDim], ddy[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                           filter_mode, wrap_mode, srgb, value, res_idx,
                           idiv_idx, pos_idx, active_idx, pos);
    ops.mip_value = Float::borrow(mip_value);
    ops.mip_table = Int::borrow(mip_table);

    for (uint32_t k = 0; k < dim; ++k) {
        ddx[k] = Float::borrow(ddx_idx[k]);
        ddy[k] = Float::borrow(ddy_idx[k]);
    }

    Float *result_mem  = (Float *) alloca(sizeof(Float) * channels_out);
    Float *scratch_mem = (Float *) alloca(sizeof(Float) * channels_out);
    tex_scratch<Float> result(result_mem, channels_out),
                       scratch(scratch_mem, channels_out);

    if (n_levels > 1) {
        bool accel = handle != nullptr && use_accel,
             grad  = any_grad(value, pos_idx, dim) ||
                     any_grad(mip_value, nullptr, 0);

        if (accel && !grad) {
            tex_eval_grad_accel(handle, channels_stored, channels_out,
                                query_type, dim, pos, ddx, ddy, ops.active,
                                result.data());
        } else {
            // AD case: perform a non-accelerated lookup with gradient tracking
            // and splice the accelerated result into the primal component when
            // hardware texture lookups are used.
            dr::detail::tex_eval_filtered(ops, pos, ddx, ddy, n_levels,
                                          (dr::MipFilter) mip_filter,
                                          max_aniso, result.data());
            if (accel) {
                tex_eval_grad_accel(handle, channels_stored, channels_out,
                                    query_type, dim, pos, ddx, ddy, ops.active,
                                    scratch.data());
                for (uint32_t ch = 0; ch < channels_out; ++ch)
                    result[ch] = reattach(scratch[ch], result[ch]);
            }
        }
    } else {
        dr::detail::tex_eval(ops, pos, result.data(), scratch.data());
    }

    for (uint32_t ch = 0; ch < channels_out; ++ch)
        out_idx[ch] = result[ch].release();
}

void ad_tex_fetch(VarType query_type, uint32_t dim, uint32_t channels_stored,
                  uint32_t channels_out, int wrap_mode, int srgb, void *handle,
                  int use_accel, uint64_t value, const uint32_t *res_idx,
                  const uint32_t *idiv_idx, const uint64_t *pos_idx,
                  uint32_t active_idx, uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             (int) dr::FilterMode::Linear, wrap_mode, srgb, value,
                             res_idx, idiv_idx, pos_idx, active_idx, pos);

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
    // idiv, wrap_mode). Leave the float/gather machinery default-initialized.
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

    for (uint32_t k = 0; k < dim; ++k)
        out_idx[k] = (uint32_t) dr::detail::tex_wrap(ops, pos[k], k).release();
}

void ad_tex_cubic(VarType query_type, uint32_t dim, uint32_t channels_stored,
                  uint32_t channels_out, int wrap_mode, int srgb, void *handle,
                  int use_accel, uint64_t value, const uint32_t *res_idx,
                  const uint32_t *idiv_idx, const uint64_t *pos_idx,
                  uint32_t active_idx, uint64_t *out_idx) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             (int) dr::FilterMode::Linear, wrap_mode, srgb, value,
                             res_idx, idiv_idx, pos_idx, active_idx, pos);

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

        // The accelerated lookups above aren't AD-attached. The code below
        // splices in the right derivatives by performing a regular cubic
        // lookup (without the GPU gems trick) and then capturing the AD
        // graph of that. (reusing the now-dead `f` as scratch).
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
                        uint32_t channels_out, int wrap_mode, int srgb,
                        uint64_t value, const uint32_t *res_idx,
                        const uint32_t *idiv_idx, const uint64_t *pos_idx,
                        uint32_t active_idx, uint64_t *out_value,
                        uint64_t *out_grad, uint64_t *out_hess) {
    Float pos[MaxDim];
    JitOps ops = tex_setup(query_type, dim, channels_stored, channels_out,
                             (int) dr::FilterMode::Linear, wrap_mode, srgb, value,
                             res_idx, idiv_idx, pos_idx, active_idx, pos);

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

void ad_tex_write(uint32_t channels_stored, uint32_t channels_out,
                  VarType storage_type, int srgb, void *handle,
                  const uint32_t *pos_idx, const uint64_t *value,
                  uint32_t active_idx) {
    using F32 = GenericArray<float>;
    JitBackend backend = jit_set_backend(pos_idx[0]).backend;

    // CUDA stores 8-bit textures as raw unorm8 bytes. We must clip and
    // sRGB-encode in software. Scaling to 0..255 is done by the
    // jit_text_write() step. Metal does this all in hardware.
    bool quantize = backend == JitBackend::CUDA &&
                    storage_type == VarType::UInt8;

    F32 *vals_mem = (F32 *) alloca(sizeof(F32) * channels_stored);
    tex_scratch<F32> vals(vals_mem, channels_stored);
    uint32_t *val_idx = (uint32_t *) alloca(channels_stored * sizeof(uint32_t));
    for (uint32_t ch = 0; ch < channels_stored; ++ch) {
        if (ch < channels_out) {
            F32 v = F32::steal(jit_var_cast((uint32_t) value[ch],
                                            VarType::Float32, 0));
            if (quantize) {
                v = dr::clip(v, 0.f, 1.f);
                if (srgb && (ch % 4) != 3)
                    v = dr::linear_to_srgb(v);
            }
            vals[ch] = v;
        } else {
            vals[ch] = dr::zeros<F32>();
        }
        val_idx[ch] = vals[ch].index();
    }

    jit_tex_write(handle, pos_idx, val_idx, active_idx);
}

uint32_t ad_tex_readback(VarType storage_type, uint32_t dim,
                         uint32_t channels_stored, uint32_t channels_out,
                         int srgb, void *handle, const uint32_t *res_idx,
                         const uint32_t *idiv_idx, size_t n_texels) {
    using F32 = GenericArray<float>;
    using I32 = GenericArray<int32_t>;
    using U32 = GenericArray<uint32_t>;

    JitBackend backend = jit_set_backend(res_idx[0]).backend;
    uint32_t C  = channels_out,
             Cs = channels_stored;

    dr::divisor<uint32_t> div_c(C);
    U32 i     = U32::steal(jit_var_counter(backend, n_texels * C)),
        texel = div_c(i),
        ch    = i - texel * C;

    // Texel-center coordinates, fastest-varying dimension first
    F32 pos[MaxDim];
    uint32_t pos_idx[MaxDim];
    I32 rem = I32(texel);
    for (uint32_t k = 0; k < dim; ++k) {
        dr::divisor<I32> div_r;
        div_r.multiplier = I32::borrow(idiv_idx[2 * k + 0]);
        div_r.shift      = I32::borrow(idiv_idx[2 * k + 1]);

        I32 res   = I32(U32::borrow(res_idx[k])),
            next  = div_r(rem),
            coord = rem - next * res;
        rem = next;
        F32 inv = dr::rcp(F32(res));
        pos[k] = (F32(coord) + 0.5f) * inv;
        pos_idx[k] = pos[k].index();
    }

    dr::mask_t<F32> active = true;
    uint32_t *tex_out = (uint32_t *) alloca(sizeof(uint32_t) * Cs);
    jit_tex_lookup(handle, pos_idx, active.index(), tex_out);

    F32 *val_mem = (F32 *) alloca(sizeof(F32) * Cs);
    tex_scratch<F32> values(val_mem, Cs);
    for (uint32_t k = 0; k < Cs; ++k) {
        F32 v = F32::steal(tex_out[k]);
        if (storage_type == VarType::UInt8) {
            // Invert the hardware decoding (see Texture::store_texel())
            v = dr::clip(v, 0.f, 1.f);
            if (srgb && (k % 4) != 3)
                v = dr::linear_to_srgb(v);
            v = dr::fmadd(v, F32(255.f), F32(0.5f));
        }
        values[k] = v;
    }

    F32 result = values[C - 1];
    for (uint32_t k = C - 1; k-- > 0; )
        result = dr::select(ch == k, values[k], result);

    return jit_var_cast(result.index(), storage_type, 0);
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
