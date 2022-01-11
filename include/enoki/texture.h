/*
    enoki/texture.h -- N-D Texture interpolation with GPU acceleration

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <array>
#include <utility>
#include <enoki-jit/texture.h>
#include <enoki/dynamic.h>
#include <enoki/idiv.h>
#include <enoki/jit.h>
#include <enoki/tensor.h>

#pragma once

NAMESPACE_BEGIN(enoki)

/// Texture interpolation methods
enum class FilterMode : uint32_t {
    Nearest = 0, /// Nearest-neighbor interpolation
    Linear = 1 /// Linear interpolation
};

/// Texture wrapping methods
enum class WrapMode : uint32_t {
    Repeat = 0, /// Repeats the texture
    Clamp = 1, /// Replicates the edge color
    Mirror = 2 /// Mirrors the texture wrt. each edge
};

template <typename Value, size_t Dimension> class Texture {
public:
    static constexpr bool IsCUDA = is_cuda_array_v<Value>;
    static constexpr bool IsDiff = is_diff_array_v<Value>;
    static constexpr bool IsDynamic = is_dynamic_v<Value>;
    // Only single-precision floating-point CUDA textures are supported
    static constexpr bool HasCudaTexture =
        std::is_same_v<scalar_t<Value>, float>;

    using Int32 = int32_array_t<Value>;
    using UInt32 = uint32_array_t<Value>;
    using Mask = mask_t<Value>;
    using PosF = Array<Value, Dimension>;
    using PosI = int32_array_t<PosF>;
    using ArrayX = DynamicArray<Value>;
    using Storage = std::conditional_t<IsDynamic, Value, DynamicArray<Value>>;
    using TensorXf = Tensor<Storage>;

    /// Default constructor: create an invalid texture object
    Texture() = default;

    /**
     * \brief Create a new texture with the specified size and channel count
     *
     * On CUDA, this is a slow operation that synchronizes the GPU pipeline, so
     * texture objects should be reused/updated via \ref set_value() and \ref
     * set_tensor() as much as possible.
     *
     * When \c migrate is set to \c true on CUDA mode, the texture information
     * is *fully* migrated to GPU texture memory to avoid redundant storage. In
     * this case, the fallback evaluation routine \ref eval_enoki() is not
     * usable anymore (it will return zero) and only \ref eval() or \ref
     * eval_cuda() should be used. Note that the texture is still
     * differentiable even when migrated. The \ref value() and \ref tensor()
     * operations will perform a reverse migration in this caes.
     *
     * The \c filter_mode parameter defines the interpolation method to be used
     * in all evaluation routines. By default, the texture is linearly
     * interpolated. Besides nearest/linear filtering, the implementation also
     * provides a clamped cubic B-spline interpolation scheme in case a
     * higher-order interpolation is needed. In CUDA mode, this is done using a
     * series of linear lookups to optimally use the hardwrae (hence, linear
     * filtering must be enabled to use this feature).
     *
     * When evaluating the texture outside of its boundaries, the \c wrap_mode
     * defines the wrapping method. The default behavior is \ref WrapMode::Clamp,
     * which indefinitely extends the colors on the boundary along each dimension.
     */
    Texture(const size_t shape[Dimension], size_t channels, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp) {
        init(shape, channels, migrate, filter_mode, wrap_mode);
        m_value.array() = Storage(0);
    }

    /**
     * \brief Construct a new texture from a given tensor
     *
     * This constructor allocates texture memory just like the previous
     * constructor, though shape information is instead extracted from \c
     * tensor. It then also invokes <tt>set_tensor(tensor)</tt> to fill
     * the texture memory with the provided tensor.
     *
     * Both the \c filter_mode and \c wrap_mode have the same defaults and
     * behaviors as for the previous constructor.
     */
    Texture(const TensorXf &tensor, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp) {
        if (tensor.ndim() != Dimension + 1)
            enoki_raise("Texture::Texture(): tensor dimension must equal "
                        "texture dimension plus one.");
        init(tensor.shape().data(), tensor.shape(Dimension), migrate,
             filter_mode, wrap_mode);
        set_tensor(tensor);
    }

    Texture(Texture &&other) {
        m_handle = other.m_handle;
        other.m_handle = nullptr;
        m_size = other.m_size;
        m_shape_opaque = std::move(other.m_shape_opaque);
        m_value = std::move(other.m_value);
        for (size_t i = 0; i < Dimension; ++i)
            m_inv_resolution[i] = std::move(other.m_inv_resolution[i]);
        m_filter_mode = other.m_filter_mode;
        m_wrap_mode = other.m_wrap_mode;
        m_migrate = other.m_migrate;
        m_migrated = other.m_migrated;
    }

    Texture &operator=(Texture &&other) {
        if constexpr (IsCUDA)
            jit_cuda_tex_destroy(m_handle);
        m_handle = other.m_handle;
        other.m_handle = nullptr;
        m_size = other.m_size;
        m_shape_opaque = std::move(other.m_shape_opaque);
        m_value = std::move(other.m_value);
        for (size_t i = 0; i < Dimension; ++i)
            m_inv_resolution[i] = std::move(other.m_inv_resolution[i]);
        m_filter_mode = other.m_filter_mode;
        m_wrap_mode = other.m_wrap_mode;
        m_migrate = other.m_migrate;
        m_migrated = other.m_migrated;
        return *this;
    }

    Texture(const Texture &) = delete;
    Texture &operator=(const Texture &) = delete;

    ~Texture() {
        if constexpr (IsCUDA)
            jit_cuda_tex_destroy(m_handle);
    }

    /// Return the CUDA handle (EnokiCudaTexture*). NULL on all other backends
    const void *handle() const { return m_handle; }

    /// Return the texture dimension plus one (for the "channel dimension")
    size_t ndim() const { return Dimension + 1; }

    const size_t *shape() const { return m_value.shape().data(); }
    FilterMode filter_mode() const { return m_filter_mode; }
    WrapMode wrap_mode() const { return m_wrap_mode; }

    /// Override the texture contents with the provided linearized 1D array
    void set_value(const Storage &value) {
        if (value.size() != m_size)
            enoki_raise("Texture::set_value(): unexpected array size!");

        if constexpr (IsCUDA && HasCudaTexture) {
            value.eval_(); // Sync the value before copying to texture memory
            jit_cuda_tex_memcpy_d2t(Dimension, m_value.shape().data(),
                                    value.data(), m_handle);

            if (m_migrate) {
                Storage dummy = zero<Storage>(m_size);

                // Fully migrate to texture memory, set m_value to zero
                if constexpr (IsDiff)
                    m_value.array() = replace_grad(dummy, value);
                else
                    m_value.array() = dummy;

                m_migrated = true;
                return;
            }
        }

        m_value.array() = value;
    }

    /// Override the texture contents with the provided tensor
    void set_tensor(const TensorXf &tensor) {
        if (tensor.ndim() != Dimension + 1)
            enoki_raise("Texture::set_tensor(): tensor dimension must equal "
                        "texture dimension plus one (channels).");
        for (size_t i = 0; i < Dimension + 1; ++i) {
            if (tensor.shape(i) != m_value.shape(i))
                enoki_raise("Texture::set_tensor(): tensor shape mismatch!");
        }
        set_value(tensor.array());
    }

    const Storage &value() const { return tensor().array(); }

    const TensorXf &tensor() const {
        if constexpr (IsCUDA && HasCudaTexture) {
            if (m_migrated) {
                Storage primal = empty<Storage>(m_size);
                jit_cuda_tex_memcpy_t2d(Dimension, m_value.shape().data(),
                                        m_handle, primal.data());

                if constexpr (IsDiff)
                    m_value.array() = replace_grad(primal, m_value.array());
                else
                    m_value.array() = primal;

                m_migrated = false;
            }
        }

        return m_value;
    }

    /**
     * \brief Evaluate linear interpolant using a CUDA texture lookup
     *
     * This is an implementation detail, please use \ref eval() that may
     * dispatch to this function depending on its inputs.
     */
    void eval_cuda(const Array<Value, Dimension> &pos, Value *out,
                   Mask active = true) const {
        const size_t channels = m_value.shape(Dimension);

        if constexpr (IsCUDA && HasCudaTexture) {
            uint32_t pos_idx[Dimension], out_idx[channels];
            for (size_t i = 0; i < Dimension; ++i)
                pos_idx[i] = pos[i].index();

            jit_cuda_tex_lookup(Dimension, m_handle, pos_idx, active.index(),
                                out_idx);

            for (size_t ch = 0; ch < channels; ++ch)
                out[ch] = Value::steal(out_idx[ch]);
        } else {
            (void) pos; (void) active;
            for (size_t ch = 0; ch < channels; ++ch)
                out[ch] = zero<Value>();
        }
    }

    /**
     * \brief Evaluate linear interpolant using explicit arithmetic
     *
     * This is an implementation detail, please use \ref eval() that may
     * dispatch to this function depending on its inputs.
     */
    void eval_enoki(const Array<Value, Dimension> &pos, Value *out,
                    Mask active = true) const {
        const uint32_t channels = (uint32_t) m_value.shape(Dimension);

        if (ENOKI_UNLIKELY(m_filter_mode == FilterMode::Nearest)) {
            const PosF pos_f = pos * PosF(m_shape_opaque);
            const PosI pos_i = floor2int<PosI>(pos_f);
            const PosI pos_i_w = wrap(pos_i);

            UInt32 idx = index(pos_i_w);

            for (uint32_t ch = 0; ch < channels; ++ch)
                out[ch] = gather<Value>(m_value.array(), idx + ch, active);
        } else {
            using InterpOffset = Array<Int32, ipow(2, Dimension)>;
            using InterpPosI = Array<InterpOffset, Dimension>;
            using InterpIdx = uint32_array_t<InterpOffset>;

            const PosF pos_f = fmadd(pos, PosF(m_shape_opaque), -.5f);
            const PosI pos_i = floor2int<PosI>(pos_f);

            int32_t offset[2] = { 0, 1 };

            InterpPosI pos_i_w = interp_positions<PosI, 2>(offset, pos_i);
            pos_i_w = wrap(pos_i_w);
            InterpIdx idx = index(pos_i_w);

            for (uint32_t ch = 0; ch < channels; ++ch)
                out[ch] = zero<Value>();

            #define EK_TEX_ACCUM(index, weight)                                        \
                {                                                                      \
                    UInt32 index_ = index;                                             \
                    Value weight_ = weight;                                            \
                    for (uint32_t ch = 0; ch < channels; ++ch)                         \
                        out[ch] =                                                      \
                            fmadd(gather<Value>(m_value.array(), index_ + ch, active), \
                                weight_, out[ch]);                                     \
                }

            const PosF w1 = pos_f - floor2int<PosI>(pos_f),
                       w0 = 1.f - w1;

            if constexpr (Dimension == 1) {
                EK_TEX_ACCUM(idx.x(), w0.x());
                EK_TEX_ACCUM(idx.y(), w1.x());
            } else if constexpr (Dimension == 2) {
                EK_TEX_ACCUM(idx.x(), w0.x() * w0.y());
                EK_TEX_ACCUM(idx.y(), w0.x() * w1.y());
                EK_TEX_ACCUM(idx.z(), w1.x() * w0.y());
                EK_TEX_ACCUM(idx.w(), w1.x() * w1.y());
            } else if constexpr (Dimension == 3) {
                EK_TEX_ACCUM(idx[0], w0.x() * w0.y() * w0.z());
                EK_TEX_ACCUM(idx[1], w0.x() * w0.y() * w1.z());
                EK_TEX_ACCUM(idx[2], w0.x() * w1.y() * w0.z());
                EK_TEX_ACCUM(idx[3], w0.x() * w1.y() * w1.z());
                EK_TEX_ACCUM(idx[4], w1.x() * w0.y() * w0.z());
                EK_TEX_ACCUM(idx[5], w1.x() * w0.y() * w1.z());
                EK_TEX_ACCUM(idx[6], w1.x() * w1.y() * w0.z());
                EK_TEX_ACCUM(idx[7], w1.x() * w1.y() * w1.z());
            }

            #undef EK_TEX_ACCUM
        }
    }

    /**
     * \brief Evaluate the linear interpolant represented by this texture
     *
     * This function dispatches to \ref eval_enoki() or \ref eval_cuda()
     * depending on whether or not CUDA is available. If invoked with CUDA
     * arrays that track derivative information, the function records the AD
     * graph of \ref eval_enoki() and combines it with the primal result of
     * \ref eval_cuda().
     */
    void eval(const Array<Value, Dimension> &pos, Value *out,
              Mask active = true) const {
        if constexpr (IsCUDA && HasCudaTexture) {
            eval_cuda(pos, out, active);

            if constexpr (IsDiff) {
                if (grad_enabled(m_value, pos)) {
                    const size_t channels = m_value.shape(Dimension);

                    ArrayX out_enoki = empty<ArrayX>(channels);
                    eval_enoki(pos, out_enoki.data(), active);

                    for (size_t ch = 0; ch < channels; ++ch)
                        out[ch] = replace_grad(out[ch], out_enoki[ch]);
                }
            }
        } else {
            eval_enoki(pos, out, active);
        }
    }

    /**
     * \brief Helper function to evaluate a clamped cubic B-Spline interpolant
     *
     * This is an implementation detail and should only be called by the \ref
     * eval_cubic() function to construct an AD graph. When only the cubic
     * evaluation result is desired, the \ref eval_cubic() function is faster
     * than this simple implementation
     */
    void eval_cubic_helper(const Array<Value, Dimension> &pos, Value *out,
                           Mask active = true) const {
        using Array4 = Array<Value, 4>;
        using InterpOffset = Array<Int32, ipow(4, Dimension)>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        using InterpIdx = uint32_array_t<InterpOffset>;

        PosF pos_(pos);
        PosF pos_f = fmadd(pos_, PosF(m_shape_opaque), -.5f);
        PosI pos_i = floor2int<PosI>(pos_f);

        // `offset[k]` controls the k-th offset for any dimension.
        // With cubic B-Spline, it is by default [-1, 0, 1, 2].
        int32_t offset[4] = {-1, 0, 1, 2};

        InterpPosI pos_i_w = interp_positions<PosI, 4>(offset, pos_i);
        pos_i_w = wrap(pos_i_w);
        InterpIdx idx = index(pos_i_w);

        PosF pos_a = pos_f - PosF(pos_i);

        const auto compute_weight = [&pos_a](uint32_t dim) -> Array4 {
            const Value &alpha = pos_a[dim];
            Value alpha2 = alpha * alpha,
                  alpha3 = alpha2 * alpha;
            Value multiplier = 1.f / 6.f;
            return multiplier *
                   Array4(-alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f,
                           3.f * alpha3 - 6.f * alpha2 + 4.f,
                          -3.f * alpha3 + 3.f * alpha2 + 3.f * alpha + 1.f,
                           alpha3);
        };

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);
        for (uint32_t ch = 0; ch < channels; ++ch)
            out[ch] = zero<Value>();

        #define EK_TEX_CUBIC_ACCUM(index, weight)                                  \
            {                                                                      \
                UInt32 index_ = index;                                             \
                Value weight_ = weight;                                            \
                for (uint32_t ch = 0; ch < channels; ++ch)                         \
                    out[ch] =                                                      \
                        fmadd(gather<Value>(m_value.array(), index_ + ch, active), \
                              weight_, out[ch]);                                   \
            }

        if constexpr (Dimension == 1) {
            Array4 wx = compute_weight(0);
            for (uint32_t ix = 0; ix < 4; ix++)
                EK_TEX_CUBIC_ACCUM(idx[ix], wx[ix]);
        } else if constexpr (Dimension == 2) {
            Array4 wx = compute_weight(0), wy = compute_weight(1);
            for (uint32_t ix = 0; ix < 4; ix++)
                for (uint32_t iy = 0; iy < 4; iy++)
                    EK_TEX_CUBIC_ACCUM(idx[ix * 4 + iy], wx[ix] * wy[iy]);
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0),
                   wy = compute_weight(1),
                   wz = compute_weight(2);
            for (uint32_t ix = 0; ix < 4; ix++)
                for (uint32_t iy = 0; iy < 4; iy++)
                    for (uint32_t iz = 0; iz < 4; iz++)
                        EK_TEX_CUBIC_ACCUM(idx[ix * 16 + iy * 4 + iz],
                                           wx[ix] * wy[iy] * wz[iz]);
        }

        #undef EK_TEX_CUBIC_ACCUM
    }

    /**
     * \brief Evaluate a clamped cubic B-Spline interpolant represented by this
     * texture
     *
     * Intead of interpolating the texture via B-Spline basis functions, the
     * implementation transforms this calculation into an equivalent weighted
     * sum of several linear interpolant evaluations. In CUDA mode, this can
     * then be accelerated by hardware texture units, which runs faster than
     * a naive implementation. More information can be found in
     *
     *   GPU Gems 2, Chapter 20, "Fast Third-Order Texture Filtering"
     *   by Christian Sigg.
     *
     * When the underlying grid data and the query `pos` are differentiable,
     * this transformation cannot be used as it is not linear w.r.t. `pos`
     * (thus the default AD graph gives incorrect results). The implementation
     * calls \ref eval_cubic_helper() function to replace the AD graph with a
     * direct evaluation of the B-Spline basis functions in that case.
     */
    void eval_cubic(const Array<Value, Dimension> &pos, Value *out,
                    Mask active = true, bool force_enoki = false) const {
        using Array3 = Array<Value, 3>;

        if (m_migrate && force_enoki)
            jit_log(::LogLevel::Warn,
                    "\"force_enoki\" is used while the data has been fully "
                    "migrated to CUDA texture memory");

        PosF res_f = PosF(m_shape_opaque);
        PosF pos_f = fmadd(pos, res_f, -.5f);
        PosI pos_i = floor2int<PosI>(pos_f);
        PosF pos_a = pos_f - PosF(pos_i);
        PosF inv_shape = rcp(res_f);

        /* With cubic B-Spline, normally we have 4 query points and 4 weights
           for each dimension. After the linear interpolation transformation,
           they are reduced to 2 query points and 2 weights. This function
           returns the two weights and query coordinates.
           Note: the two weights sum to be 1.0 so only `w01` is returned. */
        auto compute_weight_coord = [&](uint32_t dim) -> Array3 {
            const Value integ = (Value) pos_i[dim];
            const Value alpha = pos_a[dim];
            Value alpha2 = sqr(alpha),
                  alpha3 = alpha2 * alpha;
            Value multiplier = 1.f / 6.f;
            // four basis functions, transformed to take as input the fractional part
            Value w0 =
                      (-alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f) * multiplier,
                  w1 = (3.f * alpha3 - 6.f * alpha2 + 4.f) * multiplier,
                  w3 = alpha3 * multiplier;
            Value w01 = w0 + w1,
                  w23 = 1.f - w01;
            return Array3(
                w01,
                (integ - 0.5f + w1 / w01) * inv_shape[dim],
                (integ + 1.5f + w3 / w23) * inv_shape[dim]); // (integ + 0.5) +- 1 + weight
        };

        const size_t channels = m_value.shape(Dimension);

        auto eval_helper = [&](const PosF &pos,
                               const Mask &active) -> ArrayX {
            ArrayX out = empty<ArrayX>(channels);
            if constexpr (IsCUDA && HasCudaTexture) {
                if (!force_enoki) {
                    eval_cuda(pos, out.data(), active);
                    return out;
                }
            }
            ENOKI_MARK_USED(force_enoki);
            eval_enoki(pos, out.data(), active);
            return out;
        };


        ArrayX result = empty<ArrayX>(channels);

        if constexpr (Dimension == 1) {
            Array3 cx = compute_weight_coord(0);
            ArrayX f0 = eval_helper(PosF(cx[1]), active),
                   f1 = eval_helper(PosF(cx[2]), active);
            result = lerp(f1, f0, cx[0]);
        } else if constexpr (Dimension == 2) {
            Array3 cx = compute_weight_coord(0),
                   cy = compute_weight_coord(1);
            ArrayX f00 = eval_helper(PosF(cx[1], cy[1]), active),
                   f01 = eval_helper(PosF(cx[1], cy[2]), active),
                   f10 = eval_helper(PosF(cx[2], cy[1]), active),
                   f11 = eval_helper(PosF(cx[2], cy[2]), active);
            ArrayX f0 = lerp(f01, f00, cy[0]),
                   f1 = lerp(f11, f10, cy[0]);
            result = lerp(f1, f0, cx[0]);
        } else if constexpr (Dimension == 3) {
            Array3 cx = compute_weight_coord(0),
                   cy = compute_weight_coord(1),
                   cz = compute_weight_coord(2);
            ArrayX f000 = eval_helper(PosF(cx[1], cy[1], cz[1]), active),
                   f001 = eval_helper(PosF(cx[1], cy[1], cz[2]), active),
                   f010 = eval_helper(PosF(cx[1], cy[2], cz[1]), active),
                   f011 = eval_helper(PosF(cx[1], cy[2], cz[2]), active),
                   f100 = eval_helper(PosF(cx[2], cy[1], cz[1]), active),
                   f101 = eval_helper(PosF(cx[2], cy[1], cz[2]), active),
                   f110 = eval_helper(PosF(cx[2], cy[2], cz[1]), active),
                   f111 = eval_helper(PosF(cx[2], cy[2], cz[2]), active);
            ArrayX f00 = lerp(f001, f000, cz[0]),
                   f01 = lerp(f011, f010, cz[0]),
                   f10 = lerp(f101, f100, cz[0]),
                   f11 = lerp(f111, f110, cz[0]);
            ArrayX f0 = lerp(f01, f00, cy[0]),
                   f1 = lerp(f11, f10, cy[0]);
            result = lerp(f1, f0, cx[0]);
        }

        for (size_t ch = 0; ch < channels; ++ch)
            out[ch] = std::move(result[ch]);

        if constexpr (IsDiff) {
            /* When `pos` has gradient enabled, call a helper function to
               replace the AD graph. The result is unused (and never computed)
               and only the AD graph is replaced. */
            if (grad_enabled(m_value, pos)) {
                ArrayX result_diff = empty<ArrayX>(channels);
                eval_cubic_helper(pos, result_diff.data(), active); // AD graph only
                for (size_t ch = 0; ch < channels; ++ch)
                    out[ch] = replace_grad(out[ch], result_diff[ch]);
            }
        }
    }

    /**
     * \brief Evaluate the positional gradient of a cubic B-Spline
     *
     * This implementation computes the result directly from explicit
     * differentiated basis functions. It has no autodiff support.
     *
     * The resulting gradient and hessian have been multiplied by the spatial extents
     * to count for the transformation from the unit size volume to the size of its
     * shape.
     */
    void eval_cubic_grad(const Array<Value, Dimension> &pos,
                         Array<Value, Dimension> *out,
                         Mask active = true) const {
        using Array4 = Array<Value, 4>;
        using InterpOffset = Array<Int32, ipow(4, Dimension)>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        using InterpIdx = uint32_array_t<InterpOffset>;

        PosF res_f = PosF(m_shape_opaque);
        PosF pos_f = fmadd(pos, res_f, -.5f);
        PosI pos_i = floor2int<PosI>(pos_f);

        int32_t offset[4] = {-1, 0, 1, 2};

        InterpPosI pos_i_w = interp_positions<PosI, 4>(offset, pos_i);
        pos_i_w = wrap(pos_i_w);
        InterpIdx idx = index(pos_i_w);

        PosF pos_a = pos_f - PosF(pos_i);

        const auto compute_weight = [&pos_a](uint32_t dim,
                                             bool is_grad) -> Array4 {
            const Value &alpha = pos_a[dim];
            Value alpha2 = sqr(alpha);
            Value multiplier = 1.f / 6.f;
            if (!is_grad) {
                Value alpha3 = alpha2 * alpha;
                return multiplier * Array4(
                    -alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f,
                     3.f * alpha3 - 6.f * alpha2 + 4.f,
                    -3.f * alpha3 + 3.f * alpha2 + 3.f * alpha + 1.f,
                    alpha3);
            } else {
                return multiplier * Array4(
                    -3.f * alpha2 + 6.f * alpha - 3.f,
                     9.f * alpha2 - 12.f * alpha,
                    -9.f * alpha2 + 6.f * alpha + 3.f,
                     3.f * alpha2);
            }
        };

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);

        for (uint32_t ch = 0; ch < channels; ++ch)
            out[ch] = zero<PosF>();
        ArrayX values = empty<ArrayX>(channels);

        #define EK_TEX_CUBIC_GATHER(index)                                            \
            {                                                                         \
                UInt32 index_ = index;                                                \
                for (uint32_t ch = 0; ch < channels; ++ch)                            \
                    values[ch] = gather<Value>(m_value.array(), index_ + ch, active); \
            }
        #define EK_TEX_CUBIC_ACCUM(dim, weight)                                \
            {                                                                  \
                uint32_t dim_ = dim;                                           \
                Value weight_ = weight;                                        \
                for (uint32_t ch = 0; ch < channels; ++ch)                     \
                    out[ch][dim_] = fmadd(values[ch], weight_, out[ch][dim_]); \
            }

        if constexpr (Dimension == 1) {
            Array4 gx = compute_weight(0, true);
            for (uint32_t ix = 0; ix < 4; ++ix) {
                EK_TEX_CUBIC_GATHER(idx[ix]);
                EK_TEX_CUBIC_ACCUM(0, gx[ix]);
            }
        } else if constexpr (Dimension == 2) {
            Array4 wx = compute_weight(0, false),
                   wy = compute_weight(1, false),
                   gx = compute_weight(0, true),
                   gy = compute_weight(1, true);
            for (uint32_t ix = 0; ix < 4; ++ix)
                for (uint32_t iy = 0; iy < 4; ++iy) {
                    EK_TEX_CUBIC_GATHER(idx[ix * 4 + iy]);
                    EK_TEX_CUBIC_ACCUM(0, gx[ix] * wy[iy]);
                    EK_TEX_CUBIC_ACCUM(1, wx[ix] * gy[iy]);
                }
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0, false),
                   wy = compute_weight(1, false),
                   wz = compute_weight(2, false),
                   gx = compute_weight(0, true),
                   gy = compute_weight(1, true),
                   gz = compute_weight(2, true);
            for (uint32_t ix = 0; ix < 4; ++ix)
                for (uint32_t iy = 0; iy < 4; ++iy)
                    for (uint32_t iz = 0; iz < 4; ++iz) {
                        EK_TEX_CUBIC_GATHER(idx[ix * 16 + iy * 4 + iz]);
                        EK_TEX_CUBIC_ACCUM(0, gx[ix] * wy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM(1, wx[ix] * gy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM(2, wx[ix] * wy[iy] * gz[iz]);
                    }
        }

        #undef EK_TEX_CUBIC_GATHER
        #undef EK_TEX_CUBIC_ACCUM

        // transform volume from unit size to its resolution
        for (uint32_t ch = 0; ch < channels; ++ch)
            for (uint32_t dim = 0; dim < Dimension; ++dim)
                out[ch][dim] *= res_f[dim];
    }

    /**
     * \brief Evaluate the positional gradient and hessian matrix of a cubic B-Spline
     *
     * This implementation computes the result directly from explicit
     * differentiated basis functions. It has no autodiff support.
     *
     * The resulting gradient and hessian have been multiplied by the spatial extents
     * to count for the transformation from the unit size volume to the size of its
     * shape.
     */
    void eval_cubic_hessian(const Array<Value, Dimension> &pos,
                            Array<Value, Dimension> *out_gradient,
                            Matrix<Value, Dimension> *out_hessian,
                            Mask active = true) const {
        using Array4 = Array<Value, 4>;
        using InterpOffset = Array<Int32, ipow(4, Dimension)>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        using InterpIdx = uint32_array_t<InterpOffset>;

        PosF res_f = PosF(m_shape_opaque);
        PosF pos_f = fmadd(pos, res_f, -.5f);
        PosI pos_i = floor2int<PosI>(pos_f);

        int32_t offset[4] = {-1, 0, 1, 2};

        InterpPosI pos_i_w = interp_positions<PosI, 4>(offset, pos_i);
        pos_i_w = wrap(pos_i_w);
        InterpIdx idx = index(pos_i_w);

        PosF pos_a = pos_f - PosF(pos_i);

        const auto compute_weight = [&pos_a](uint32_t dim) -> Array4 {
            const Value &alpha = pos_a[dim];
            Value alpha2 = sqr(alpha),
                  alpha3 = alpha2 * alpha;
            Value multiplier = 1.f / 6.f;
            return multiplier * Array4(
                -alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f,
                    3.f * alpha3 - 6.f * alpha2 + 4.f,
                -3.f * alpha3 + 3.f * alpha2 + 3.f * alpha + 1.f,
                alpha3);
        };
        const auto compute_weight_gradient = [&pos_a](uint32_t dim) -> Array4 {
            const Value &alpha = pos_a[dim];
            Value alpha2 = sqr(alpha);
            Value multiplier = 1.f / 6.f;
            return multiplier * Array4(
                    -3.f * alpha2 + 6.f * alpha - 3.f,
                     9.f * alpha2 - 12.f * alpha,
                    -9.f * alpha2 + 6.f * alpha + 3.f,
                     3.f * alpha2);
        };
        const auto compute_weight_heissian = [&pos_a](uint32_t dim) -> Array4 {
            const Value &alpha = pos_a[dim];
            return Array4(
                - alpha + 1.f,
                3.f * alpha - 2.f,
                -3.f * alpha + 1.f,
                alpha
            );
        };

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);
        for (uint32_t ch = 0; ch < channels; ++ch) {
            out_gradient[ch] = zero<PosF>();
            for (uint32_t dim1 = 0; dim1 < Dimension; ++dim1)
                out_hessian[ch][dim1] = zero<PosF>();
        }
        ArrayX values = empty<ArrayX>(channels);

        // Make sure channel related operations are executed together
        #define EK_TEX_CUBIC_GATHER(index)                                            \
            {                                                                         \
                UInt32 index_ = index;                                                \
                for (uint32_t ch = 0; ch < channels; ++ch)                            \
                    values[ch] = gather<Value>(m_value.array(), index_ + ch, active); \
            }
        #define EK_TEX_CUBIC_ACCUM_GRAD(dim, weight_grad)                                             \
            {                                                                                         \
                uint32_t dim_ = dim;                                                                  \
                Value weight_grad_ = weight_grad;                                                     \
                for (uint32_t ch = 0; ch < channels; ++ch)                                            \
                    out_gradient[ch][dim_] = fmadd(values[ch], weight_grad_, out_gradient[ch][dim_]); \
            }
        #define EK_TEX_CUBIC_ACCUM_HESSIAN(dim1, dim2, weight_hessian)                                                 \
            {                                                                                                          \
                uint32_t dim1_ = dim1,                                                                                 \
                         dim2_ = dim2;                                                                                 \
                Value weight_hessian_ = weight_hessian;                                                                \
                for (uint32_t ch = 0; ch < channels; ++ch)                                                             \
                    out_hessian[ch][dim1_][dim2_] = fmadd(values[ch], weight_hessian_, out_hessian[ch][dim1_][dim2_]); \
            }
        #define EK_TEX_CUBIC_HESSIAN_SYMM(dim1, dim2)                              \
            {                                                                      \
                uint32_t dim1_ = dim1,                                             \
                         dim2_ = dim2;                                             \
                for (uint32_t ch = 0; ch < channels; ++ch)                         \
                    out_hessian[ch][dim2_][dim1_] = out_hessian[ch][dim1_][dim2_]; \
            }

        if constexpr (Dimension == 1) {
            Array4 gx  = compute_weight_gradient(0),
                   ggx = compute_weight_heissian(0);
            for (uint32_t ix = 0; ix < 4; ++ix) {
                EK_TEX_CUBIC_GATHER(idx[ix]);
                EK_TEX_CUBIC_ACCUM_GRAD(0, gx[ix]);
                EK_TEX_CUBIC_ACCUM_HESSIAN(0, 0, ggx[ix]);
            }
        } else if constexpr (Dimension == 2) {
            Array4 wx  = compute_weight(0),
                   wy  = compute_weight(1),
                   gx  = compute_weight_gradient(0),
                   gy  = compute_weight_gradient(1),
                   ggx = compute_weight_heissian(0),
                   ggy = compute_weight_heissian(1);
            for (uint32_t ix = 0; ix < 4; ++ix)
                for (uint32_t iy = 0; iy < 4; ++iy) {
                    EK_TEX_CUBIC_GATHER(idx[ix * 4 + iy]);
                    EK_TEX_CUBIC_ACCUM_GRAD(0, gx[ix] * wy[iy]);
                    EK_TEX_CUBIC_ACCUM_GRAD(1, wx[ix] * gy[iy]);
                    EK_TEX_CUBIC_ACCUM_HESSIAN(0, 0, ggx[ix] * wy[iy]);
                    EK_TEX_CUBIC_ACCUM_HESSIAN(0, 1, gx[ix] * gy[iy]);
                    EK_TEX_CUBIC_ACCUM_HESSIAN(1, 1, wx[ix] * ggy[iy]);
                }
            EK_TEX_CUBIC_HESSIAN_SYMM(0, 1);
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0),
                   wy = compute_weight(1),
                   wz = compute_weight(2),
                   gx = compute_weight_gradient(0),
                   gy = compute_weight_gradient(1),
                   gz = compute_weight_gradient(2),
                   ggx = compute_weight_heissian(0),
                   ggy = compute_weight_heissian(1),
                   ggz = compute_weight_heissian(2);
            for (uint32_t ix = 0; ix < 4; ++ix)
                for (uint32_t iy = 0; iy < 4; ++iy)
                    for (uint32_t iz = 0; iz < 4; ++iz) {
                        EK_TEX_CUBIC_GATHER(idx[ix * 16 + iy * 4 + iz]);
                        EK_TEX_CUBIC_ACCUM_GRAD(0, gx[ix] * wy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM_GRAD(1, wx[ix] * gy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM_GRAD(2, wx[ix] * wy[iy] * gz[iz]);
                        EK_TEX_CUBIC_ACCUM_HESSIAN(0, 0, ggx[ix] * wy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM_HESSIAN(1, 1, wx[ix] * ggy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM_HESSIAN(2, 2, wx[ix] * wy[iy] * ggz[iz]);
                        EK_TEX_CUBIC_ACCUM_HESSIAN(0, 1, gx[ix] * gy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM_HESSIAN(0, 2, gx[ix] * wy[iy] * gz[iz]);
                        EK_TEX_CUBIC_ACCUM_HESSIAN(1, 2, wx[ix] * gy[iy] * gz[iz]);
                    }
            EK_TEX_CUBIC_HESSIAN_SYMM(0, 1);
            EK_TEX_CUBIC_HESSIAN_SYMM(0, 2);
            EK_TEX_CUBIC_HESSIAN_SYMM(1, 2);
        }

        #undef EK_TEX_CUBIC_GATHER
        #undef EK_TEX_CUBIC_ACCUM_GRAD
        #undef EK_TEX_CUBIC_ACCUM_HESSIAN
        #undef EK_TEX_CUBIC_HESSIAN_SYMM

        // transform volume from unit size to its resolution
        for (uint32_t ch = 0; ch < channels; ++ch)
            for (uint32_t dim1 = 0; dim1 < Dimension; ++dim1) {
                out_gradient[ch][dim1] *= res_f[dim1];
                for (uint32_t dim2 = 0; dim2 < Dimension; ++dim2)
                    out_hessian[ch][dim1][dim2] *= res_f[dim1] * res_f[dim2];
            }
    }

protected:
    void init(const size_t *shape, size_t channels, bool migrate,
              FilterMode filter_mode, WrapMode wrap_mode) {
        if (channels == 0)
            enoki_raise("Texture::Texture(): must have at least 1 channel!");

        m_size = channels;
        size_t tensor_shape[Dimension + 1]{};

        for (size_t i = 0; i < Dimension; ++i) {
            tensor_shape[i] = shape[i];
            m_shape_opaque[i] = opaque<UInt32>((uint32_t) shape[i]);
            m_inv_resolution[i] = divisor<int32_t>((int32_t) shape[i]);
            m_size *= shape[i];
        }

        tensor_shape[Dimension] = channels;
        m_value = TensorXf(zero<Storage>(m_size), Dimension + 1, tensor_shape);
        m_value.array() = Storage(0);
        m_migrate = migrate;
        m_filter_mode = filter_mode;
        m_wrap_mode = wrap_mode;

        if constexpr (IsCUDA && HasCudaTexture)
            m_handle = jit_cuda_tex_create(Dimension, shape, channels,
                                           (int) filter_mode, (int) wrap_mode);
    }

private:
    /// Helper function to compmute integer powers of numbers
    template <typename T>
    constexpr static T ipow(T num, unsigned int pow) {
        return pow == 0 ? 1 : num * ipow(num, pow - 1);
    }

    /// Builds the set of integer positions necessary for the interpolation
    template <typename T, size_t Length>
    static Array<Array<Int32, ipow(Length, Dimension)>, Dimension>
    interp_positions(const int *offset, const T &pos) {
        using Scalar = scalar_t<T>;
        using InterpOffset = Array<Int32, ipow(Length, Dimension)>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        static_assert(
            array_size_v<T> == Dimension &&
            std::is_integral_v<Scalar> &&
            std::is_signed_v<Scalar>
        );

        InterpPosI pos_i;
        if constexpr (Dimension == 1) {
            for (uint32_t ix = 0; ix < Length; ix++) {
                pos_i[0][ix] = offset[ix] + pos.x();
            }
        } else if constexpr (Dimension == 2) {
            for (uint32_t ix = 0; ix < Length; ix++) {
                for (uint32_t iy = 0; iy < Length; iy++) {
                    pos_i[0][iy * Length + ix] = offset[iy] + pos.x();
                    pos_i[1][ix * Length + iy] = offset[iy] + pos.y();
                }
            }
        } else if constexpr (Dimension == 3) {
            constexpr size_t LengthSqr = Length * Length;
            for (uint32_t ix = 0; ix < Length; ix++) {
                for (uint32_t iy = 0; iy < Length; iy++) {
                    for (uint32_t iz = 0; iz < Length; iz++) {
                        pos_i[0][iz * LengthSqr + iy * Length + ix] =
                            offset[iz] + pos.x();
                        pos_i[1][ix * LengthSqr + iz * Length + iy] =
                            offset[iz] + pos.y();
                        pos_i[2][iy * LengthSqr + ix * Length + iz] =
                            offset[iz] + pos.z();
                    }
                }
            }
        }

        return pos_i;
    }

    /// Apply the configured texture wrapping mode to an integer position
    template <typename T> T wrap(const T &pos) const {
        using Scalar = scalar_t<T>;
        static_assert(
            array_size_v<T> == Dimension &&
            std::is_integral_v<Scalar> &&
            std::is_signed_v<Scalar>
        );

        const Array<Int32, Dimension> shape = m_shape_opaque;
        if (m_wrap_mode == WrapMode::Clamp) {
            return clamp(pos, 0, shape - 1);
        } else {
            const T value_shift_neg = select(pos < 0, pos + 1, pos);

            T div;
            for (size_t i = 0; i < Dimension; ++i)
                div[i] = m_inv_resolution[i](value_shift_neg[i]);

            T mod = pos - div * shape;
            mod[mod < 0] += T(shape);

            if (m_wrap_mode == WrapMode::Mirror)
                // Starting at 0, flip the texture every other repetition
                // (flip when: even number of repetitions in negative direction,
                // or odd number of repetions in positive direction)
                mod =
                    select(eq(div & 1, 0) ^ (pos < 0), mod, shape - 1 - mod);

            return mod;
        }
    }

    /// Helper function to compute the array index for a given N-D position
    template <typename T>
    uint32_array_t<value_t<T>> index(const T &pos) const {
        using Scalar = scalar_t<T>;
        using Index = uint32_array_t<value_t<T>>;
        static_assert(
            array_size_v<T> == Dimension &&
            std::is_integral_v<Scalar> &&
            std::is_signed_v<Scalar>
        );

        Index index;
        if constexpr (Dimension == 1) {
            index = Index(pos.x());
        } else if constexpr (Dimension == 2) {
            index = Index(
                fmadd(Index(pos.y()), m_shape_opaque.x(), Index(pos.x())));
        } else if constexpr (Dimension == 3) {
            index = Index(fmadd(
                fmadd(Index(pos.z()), m_shape_opaque.y(), Index(pos.y())),
                m_shape_opaque.x(), Index(pos.x())));
        }

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);

        return index * channels;
    }

private:
    void *m_handle = nullptr;
    size_t m_size = 0;
    Array<UInt32, Dimension> m_shape_opaque;
    mutable TensorXf m_value;
    divisor<int32_t> m_inv_resolution[Dimension] { };
    FilterMode m_filter_mode;
    WrapMode m_wrap_mode;
    bool m_migrate = false;
    mutable bool m_migrated = false;
};

NAMESPACE_END(enoki)
