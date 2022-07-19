/*
    drjit/texture.h -- N-D Texture interpolation with GPU acceleration

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <array>
#include <utility>
#include <drjit-core/texture.h>
#include <drjit/dynamic.h>
#include <drjit/idiv.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>

#pragma once

NAMESPACE_BEGIN(drjit)

/// Texture interpolation methods
enum class FilterMode : uint32_t {
    Nearest = 0, /// Nearest-neighbor interpolation
    Linear = 1   /// Linear interpolation
};

/// Texture wrapping methods
enum class WrapMode : uint32_t {
    Repeat = 0, /// Repeats the texture
    Clamp = 1,  /// Replicates the edge color
    Mirror = 2  /// Mirrors the texture wrt. each edge
};

template <typename Value, size_t Dimension> class Texture {
public:
    static constexpr bool IsCUDA = is_cuda_v<Value>;
    static constexpr bool IsDiff = is_diff_v<Value>;
    static constexpr bool IsDynamic = is_dynamic_v<Value>;
    // Only single-precision floating-point CUDA textures are supported
    static constexpr bool HasCudaTexture =
        std::is_same_v<scalar_t<Value>, float> && IsCUDA;

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
     * When \c use_accel is set to \c false on CUDA mode, the texture will not
     * use the hardware acceleration (allocation and evaluation). In other modes
     * this argument has no effect.
     *
     * The \c filter_mode parameter defines the interpolation method to be used
     * in all evaluation routines. By default, the texture is linearly
     * interpolated. Besides nearest/linear filtering, the implementation also
     * provides a clamped cubic B-spline interpolation scheme in case a
     * higher-order interpolation is needed. In CUDA mode, this is done using a
     * series of linear lookups to optimally use the hardware (hence, linear
     * filtering must be enabled to use this feature).
     *
     * When evaluating the texture outside of its boundaries, the \c wrap_mode
     * defines the wrapping method. The default behavior is \ref WrapMode::Clamp,
     * which indefinitely extends the colors on the boundary along each dimension.
     */
    Texture(const size_t shape[Dimension], size_t channels,
            bool use_accel = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp) {
        init(shape, channels, use_accel, filter_mode, wrap_mode);
    }

    /**
     * \brief Construct a new texture from a given tensor
     *
     * This constructor allocates texture memory just like the previous
     * constructor, though shape information is instead extracted from \c
     * tensor. It then also invokes <tt>set_tensor(tensor)</tt> to fill
     * the texture memory with the provided tensor.
     *
     * When \c migrate is set to \c true on CUDA mode, the texture information
     * is *fully* migrated to GPU texture memory to avoid redundant storage. In
     * this case, the fallback evaluation routine \ref eval_nonaccel() is not
     * usable anymore (it will return zero) and only \ref eval() or \ref
     * eval_cuda() should be used. Note that the texture is still
     * differentiable even when migrated. The \ref value() and \ref tensor()
     * operations will perform a reverse migration in this case.
     *
     * Both the \c filter_mode and \c wrap_mode have the same defaults and
     * behaviors as for the previous constructor.
     */
    Texture(const TensorXf &tensor, bool use_accel = true, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp) {
        if (tensor.ndim() != Dimension + 1)
            drjit_raise("Texture::Texture(): tensor dimension must equal "
                        "texture dimension plus one.");
        init(tensor.shape().data(), tensor.shape(Dimension), use_accel,
             filter_mode, wrap_mode);
        set_tensor(tensor, migrate);
    }

    Texture(Texture &&other) noexcept {
        m_handle = other.m_handle;
        other.m_handle = nullptr;
        m_size = other.m_size;
        m_shape_opaque = std::move(other.m_shape_opaque);
        m_value = std::move(other.m_value);
        for (size_t i = 0; i < Dimension; ++i)
            m_inv_resolution[i] = std::move(other.m_inv_resolution[i]);
        m_filter_mode = other.m_filter_mode;
        m_wrap_mode = other.m_wrap_mode;
        m_use_accel = other.m_use_accel;
        m_migrated = other.m_migrated;
    }

    Texture &operator=(Texture &&other) noexcept {
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
        m_use_accel = other.m_use_accel;
        m_migrated = other.m_migrated;
        return *this;
    }

    Texture(const Texture &) = delete;
    Texture &operator=(const Texture &) = delete;

    ~Texture() {
        if constexpr (HasCudaTexture) {
            if (m_use_accel)
                jit_cuda_tex_destroy(m_handle);
        }
    }

    /// Return the CUDA handle (Dr.JitCudaTexture*). NULL on all other backends
    const void *handle() const { return m_handle; }

    /// Return the texture dimension plus one (for the "channel dimension")
    size_t ndim() const { return Dimension + 1; }

    // Return the texture shape
    const size_t *shape() const { return m_value.shape().data(); }

    FilterMode filter_mode() const { return m_filter_mode; }
    WrapMode wrap_mode() const { return m_wrap_mode; }
    bool migrated() const { return m_migrated; }
    bool use_accel() const { return m_use_accel; }

    /**
     * \brief Override the texture contents with the provided linearized 1D array
     *
     * When \c migrate is set to \c true on CUDA mode, the texture information
     * is *fully* migrated to GPU texture memory to avoid redundant storage.
     */
    void set_value(const Storage &value, bool migrate=false) {
        if (value.size() != m_size)
            drjit_raise("Texture::set_value(): unexpected array size!");

        drjit::eval(value);

        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                value.eval_(); // Sync the value before copying to texture memory

                size_t tex_shape[Dimension + 1];
                reverse_tensor_shape(tex_shape, true);
                jit_cuda_tex_memcpy_d2t(Dimension, tex_shape, value.data(), m_handle);

                if (migrate) {
                    Storage dummy = zeros<Storage>(m_size);

                    // Fully migrate to texture memory, set m_value to zero
                    if constexpr (IsDiff)
                        m_value.array() = replace_grad(dummy, value);
                    else
                        m_value.array() = dummy;

                    m_migrated = true;

                    return;
                }
            }
        }

        m_value.array() = value;
    }

    /**
     * \brief Override the texture contents with the provided tensor
     *
     * This method updates the values of all texels. Changing the texture
     * resolution or its number of channels is also supported. However, on CUDA,
     * such operations have a significantly larger overhead (the GPU pipeline
     * needs to be synchronized for new texture objects to be created).
     *
     * When \c migrate is set to \c true on CUDA mode, the texture information
     * is *fully* migrated to GPU texture memory to avoid redundant storage.
     */
    void set_tensor(const TensorXf &tensor, bool migrate=false) {
        if (tensor.ndim() != Dimension + 1)
            drjit_raise("Texture::set_tensor(): tensor dimension must equal "
                        "texture dimension plus one (channels).");

        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                bool is_inplace_update = (&tensor == &m_value);
                size_t current_shape[Dimension + 1];

                if (is_inplace_update) {
                    jit_cuda_tex_get_shape(Dimension, m_handle, current_shape);
                } else {
                    for (size_t i = 0; i < Dimension + 1; ++i)
                        current_shape[i] = m_value.shape(i);
                }

                bool shape_changed = false;
                for (size_t i = 0; i < Dimension + 1; ++i) {
                    if (current_shape[i] != tensor.shape(i)) {
                        shape_changed = true;
                        break;
                    }
                }

                if (shape_changed) {
                    jit_cuda_tex_destroy(m_handle);
                    init(tensor.shape().data(), tensor.shape(Dimension), m_use_accel,
                         m_filter_mode, m_wrap_mode, !is_inplace_update);
                }
            } else {
                init(tensor.shape().data(), tensor.shape(Dimension),
                     m_use_accel, m_filter_mode, m_wrap_mode, false);
            }
        } else {
            init(tensor.shape().data(), tensor.shape(Dimension),
                 m_use_accel, m_filter_mode, m_wrap_mode, false);
        }

        set_value(tensor.array(), migrate);
    }

    const Storage &value() const { return tensor().array(); }

    /**
     * \brief Return the texture data as a tensor object
     */
    const TensorXf &tensor() const {
        if constexpr (HasCudaTexture) {
            if (m_use_accel && m_migrated) {
                Storage primal = empty<Storage>(m_size);

                size_t tex_shape[Dimension + 1];
                reverse_tensor_shape(tex_shape, true);
                jit_cuda_tex_memcpy_t2d(Dimension, tex_shape, m_handle,
                                        primal.data());

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
     * \brief Return the texture data as a tensor object
     *
     * Although the returned object is not const, changes to it are only fully
     * propagated to the Texture instance when a subsequent call to
     * \ref set_tensor() is made.
     */
    TensorXf &tensor() {
        return const_cast<TensorXf &>(
            const_cast<const Texture<Value, Dimension> *>(this)->tensor());
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

        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                uint32_t pos_idx[Dimension];
                uint32_t *out_idx =
                    (uint32_t *) alloca(channels * sizeof(uint32_t));
                for (size_t i = 0; i < Dimension; ++i)
                    pos_idx[i] = pos[i].index();

                jit_cuda_tex_lookup(Dimension, m_handle, pos_idx, active.index(),
                                    out_idx);

                for (size_t ch = 0; ch < channels; ++ch)
                    out[ch] = Value::steal(out_idx[ch]);
                return;
            }
        }
        DRJIT_MARK_USED(pos); DRJIT_MARK_USED(active);
        for (size_t ch = 0; ch < channels; ++ch)
            out[ch] = zeros<Value>();
    }

    /**
     * \brief Evaluate linear interpolant using explicit arithmetic
     *
     * This is an implementation detail, please use \ref eval() that may
     * dispatch to this function depending on its inputs.
     */
    void eval_nonaccel(const Array<Value, Dimension> &pos, Value *out,
                       Mask active = true) const {
        if constexpr (!is_array_v<Mask>)
            active = true;

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);
        if (DRJIT_UNLIKELY(m_filter_mode == FilterMode::Nearest)) {
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
                out[ch] = zeros<Value>();

            #define DR_TEX_ACCUM(index, weight)                                        \
                {                                                                      \
                    UInt32 index_ = index;                                             \
                    Value weight_ = weight;                                            \
                    for (uint32_t ch = 0; ch < channels; ++ch)                         \
                        out[ch] =                                                      \
                            fmadd(gather<Value>(m_value.array(), index_ + ch, active), \
                                weight_, out[ch]);                                     \
                }

            const PosF w1 = pos_f - pos_i,
                       w0 = 1.f - w1;

            if constexpr (Dimension == 1) {
                DR_TEX_ACCUM(idx.x(), w0.x());
                DR_TEX_ACCUM(idx.y(), w1.x());
            } else if constexpr (Dimension == 2) {
                DR_TEX_ACCUM(idx.x(), w0.x() * w0.y());
                DR_TEX_ACCUM(idx.y(), w1.x() * w0.y());
                DR_TEX_ACCUM(idx.z(), w0.x() * w1.y());
                DR_TEX_ACCUM(idx.w(), w1.x() * w1.y());
            } else if constexpr (Dimension == 3) {
                DR_TEX_ACCUM(idx[0], w0.x() * w0.y() * w0.z());
                DR_TEX_ACCUM(idx[1], w1.x() * w0.y() * w0.z());
                DR_TEX_ACCUM(idx[2], w0.x() * w1.y() * w0.z());
                DR_TEX_ACCUM(idx[3], w1.x() * w1.y() * w0.z());
                DR_TEX_ACCUM(idx[4], w0.x() * w0.y() * w1.z());
                DR_TEX_ACCUM(idx[5], w1.x() * w0.y() * w1.z());
                DR_TEX_ACCUM(idx[6], w0.x() * w1.y() * w1.z());
                DR_TEX_ACCUM(idx[7], w1.x() * w1.y() * w1.z());
            }

            #undef DR_TEX_ACCUM
        }
    }

    /**
     * \brief Evaluate the linear interpolant represented by this texture
     *
     * This function dispatches to \ref eval_nonaccel() or \ref eval_cuda()
     * depending on whether or not CUDA is available. If invoked with CUDA
     * arrays that track derivative information, the function records the AD
     * graph of \ref eval_nonaccel() and combines it with the primal result of
     * \ref eval_cuda().
     */
    void eval(const Array<Value, Dimension> &pos, Value *out,
              Mask active = true) const {
        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                eval_cuda(pos, out, active);

                if constexpr (IsDiff) {
                    if (grad_enabled(m_value, pos)) {
                        const size_t channels = m_value.shape(Dimension);

                        ArrayX out_nonaccel = empty<ArrayX>(channels);
                        eval_nonaccel(pos, out_nonaccel.data(), active);

                        for (size_t ch = 0; ch < channels; ++ch)
                            out[ch] = replace_grad(out[ch], out_nonaccel[ch]);
                    }
                }

                return;
            }
        }

        eval_nonaccel(pos, out, active);
    }

    /**
     * \brief Fetch the texels that would be referenced in a CUDA texture lookup
     * with linear interpolation without actually performing this interpolation.
     *
     * This is an implementation detail, please use \ref eval_fetch() that may
     * dispatch to this function depending on its inputs.
     */
    void eval_fetch_cuda(const Array<Value, Dimension> &pos,
                         Array<Value *, 1 << Dimension> &out,
                         Mask active = true) const {
        const size_t channels = m_value.shape(Dimension);

        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                if constexpr (Dimension == 1) {
                    const PosF res_f = PosF(m_shape_opaque);
                    const PosF pos_f = floor(fmadd(pos, res_f, -.5f)) + .5f;

                    PosF fetch_pos;
                    PosF inv_shape = rcp(res_f);
                    for (size_t ix = 0; ix < 2; ix++) {
                        fetch_pos = pos_f + PosF(ix);
                        fetch_pos *= inv_shape;

                        eval_cuda(fetch_pos, out[ix], active);
                    }
                } else if constexpr (Dimension == 2) {
                    uint32_t pos_idx[Dimension];
                    uint32_t *out_idx =
                        (uint32_t *) alloca(4 * channels * sizeof(uint32_t));
                    for (size_t i = 0; i < Dimension; ++i)
                        pos_idx[i] = pos[i].index();

                    jit_cuda_tex_bilerp_fetch(Dimension, m_handle, pos_idx,
                                            active.index(), out_idx);

                    for (size_t ch = 0; ch < channels; ++ch) {
                        out[2][ch] = Value::steal(out_idx[ch]);
                        out[3][ch] = Value::steal(out_idx[channels + ch]);
                        out[1][ch] = Value::steal(out_idx[2 * channels + ch]);
                        out[0][ch] = Value::steal(out_idx[3 * channels + ch]);
                    }
                } else if constexpr (Dimension == 3) {
                    const PosF res_f = PosF(m_shape_opaque);
                    const PosF pos_f = floor(fmadd(pos, res_f, -.5f)) + .5f;

                    PosF fetch_pos;
                    PosF inv_shape = rcp(res_f);
                    for (size_t iz = 0; iz < 2; iz++) {
                        for (size_t iy = 0; iy < 2; iy++) {
                            for (size_t ix = 0; ix < 2; ix++) {
                                size_t index = iz * 4 + iy * 2 + ix;
                                fetch_pos = pos_f + PosF(ix, iy, iz);
                                fetch_pos *= inv_shape;

                                eval_cuda(fetch_pos, out[index], active);
                            }
                        }
                    }
                }
                return;
            }
        }

        DRJIT_MARK_USED(pos); DRJIT_MARK_USED(active);
        for (size_t i = 0; i < ipow(2ul, Dimension); ++i)
            for (size_t ch = 0; ch < channels; ++ch)
                out[i][ch] = zeros<Value>();
    }

    /**
     * \brief Fetch the texels that would be referenced in a texture lookup with
     * linear interpolation without actually performing this interpolation.
     *
     * If the texture data is fully migrated to the GPU, this method will return
     * zeroes.
     *
     * This is an implementation detail, please use \ref eval_fetch() that may
     * dispatch to this function depending on its inputs.
     */
    void eval_fetch_nonaccel(const Array<Value, Dimension> &pos,
                             Array<Value *, 1 << Dimension> &out,
                             Mask active = true) const {
        using InterpOffset = Array<Int32, 1 << Dimension>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        using InterpIdx = uint32_array_t<InterpOffset>;

        if constexpr (!is_array_v<Mask>)
            active = true;

        const PosF pos_f = fmadd(pos, PosF(m_shape_opaque), -.5f);
        const PosI pos_i = floor2int<PosI>(pos_f);

        int32_t offset[2] = { 0, 1 };

        InterpPosI pos_i_w = interp_positions<PosI, 2>(offset, pos_i);
        pos_i_w = wrap(pos_i_w);
        InterpIdx idx = index(pos_i_w);

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);
        for (size_t i = 0; i < InterpOffset::Size; ++i)
            for (uint32_t ch = 0; ch < channels; ++ch)
                out[i][ch] =
                    gather<Value>(m_value.array(), idx[i] + ch, active);
    }

    /**
     * \brief Fetch the texels that would be referenced in a texture lookup with
     * linear interpolation without actually performing this interpolation.
     *
     * This function dispatches to \ref eval_fetch_nonaccel() or \ref
     * eval_fetch_cuda() depending on whether or not CUDA is available. If
     * invoked with CUDA arrays that track derivative information, the function
     * records the AD graph of \ref eval_fetch_nonaccel() and combines it with
     * the primal result of \ref eval_fetch_cuda().
     */
    void eval_fetch(const Array<Value, Dimension> &pos,
                    Array<Value *, 1 << Dimension> &out,
                    Mask active = true) const {
        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                eval_fetch_cuda(pos, out, active);

                if constexpr (IsDiff) {
                    if (grad_enabled(m_value, pos)) {
                        const size_t channels = m_value.shape(Dimension);
                        constexpr size_t out_size = 1 << Dimension;

                        Array<Value *, out_size> out_nonaccel;
                        ArrayX out_nonaccel_values =
                            empty<ArrayX>(out_size * channels);
                        for (size_t i = 0; i < out_size; ++i)
                            out_nonaccel[i] =
                                out_nonaccel_values.data() + i * channels;
                        eval_fetch_nonaccel(pos, out_nonaccel, active);

                        for (size_t i = 0; i < out_size; ++i)
                            for (size_t ch = 0; ch < channels; ++ch)
                                out[i][ch] =
                                    replace_grad(out[i][ch], out_nonaccel[i][ch]);
                    }
                }
                return;
            }
        }

        eval_fetch_nonaccel(pos, out, active);
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

        if constexpr (!is_array_v<Mask>)
            active = true;

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
            out[ch] = zeros<Value>();

        #define DR_TEX_CUBIC_ACCUM(index, weight)                                  \
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
                DR_TEX_CUBIC_ACCUM(idx[ix], wx[ix]);
        } else if constexpr (Dimension == 2) {
            Array4 wx = compute_weight(0),
                   wy = compute_weight(1);
            for (uint32_t iy = 0; iy < 4; iy++)
                for (uint32_t ix = 0; ix < 4; ix++)
                    DR_TEX_CUBIC_ACCUM(idx[iy * 4 + ix], wx[ix] * wy[iy]);
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0),
                   wy = compute_weight(1),
                   wz = compute_weight(2);
            for (uint32_t iz = 0; iz < 4; iz++)
                for (uint32_t iy = 0; iy < 4; iy++)
                    for (uint32_t ix = 0; ix < 4; ix++)
                        DR_TEX_CUBIC_ACCUM(idx[iz * 16 + iy * 4 + ix],
                                           wx[ix] * wy[iy] * wz[iz]);
        }

        #undef DR_TEX_CUBIC_ACCUM
    }

    /**
     * \brief Evaluate a clamped cubic B-Spline interpolant represented by this
     * texture
     *
     * Instead of interpolating the texture via B-Spline basis functions, the
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
                    Mask active = true, bool force_nonaccel = false) const {
        using Array3 = Array<Value, 3>;

        if constexpr (!is_array_v<Mask>)
            active = true;

        if (HasCudaTexture && m_migrate && force_nonaccel)
            jit_log(::LogLevel::Warn,
                    "\"force_nonaccel\" is used while the data has been fully "
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
            if constexpr (HasCudaTexture) {
                if (m_use_accel && !force_nonaccel) {
                    eval_cuda(pos, out.data(), active);
                    return out;
                }
            }
            DRJIT_MARK_USED(force_nonaccel);
            eval_nonaccel(pos, out.data(), active);
            return out;
        };

        ArrayX result = empty<ArrayX>(channels);

        if constexpr (Dimension == 1) {
            Array3 cx = compute_weight_coord(0);
            ArrayX f0 = eval_helper(PosF(cx[1]), active),
                   f1 = eval_helper(PosF(cx[2]), active);

            for (size_t ch = 0; ch < channels; ++ch)
                out[ch] = lerp(f1[ch], f0[ch], cx[0]);
        } else if constexpr (Dimension == 2) {
            Array3 cx = compute_weight_coord(0),
                   cy = compute_weight_coord(1);
            ArrayX f00 = eval_helper(PosF(cx[1], cy[1]), active),
                   f01 = eval_helper(PosF(cx[1], cy[2]), active),
                   f10 = eval_helper(PosF(cx[2], cy[1]), active),
                   f11 = eval_helper(PosF(cx[2], cy[2]), active);

            Value f0, f1;
            for (size_t ch = 0; ch < channels; ++ch) {
                f0 = lerp(f01[ch], f00[ch], cy[0]);
                f1 = lerp(f11[ch], f10[ch], cy[0]);

                out[ch] = lerp(f1, f0, cx[0]);
            }
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

            Value f00, f01, f10, f11, f0, f1;
            for (size_t ch = 0; ch < channels; ++ch) {
                f00 = lerp(f001[ch], f000[ch], cz[0]);
                f01 = lerp(f011[ch], f010[ch], cz[0]);
                f10 = lerp(f101[ch], f100[ch], cz[0]);
                f11 = lerp(f111[ch], f110[ch], cz[0]);
                f0 = lerp(f01, f00, cy[0]);
                f1 = lerp(f11, f10, cy[0]);

                out[ch] = lerp(f1, f0, cx[0]);
            }
        }

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
                         Value *out_value, Array<Value, Dimension> *out_gradient,
                         Mask active = true) const {
        using Array4 = Array<Value, 4>;
        using InterpOffset = Array<Int32, ipow(4, Dimension)>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        using InterpIdx = uint32_array_t<InterpOffset>;

        if constexpr (!is_array_v<Mask>)
            active = true;

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

        for (uint32_t ch = 0; ch < channels; ++ch) {
            out_value[ch] = zeros<Value>();
            out_gradient[ch] = zeros<PosF>();
        }
        ArrayX values = empty<ArrayX>(channels);

        #define DR_TEX_CUBIC_GATHER(index)                                            \
            {                                                                         \
                UInt32 index_ = index;                                                \
                for (uint32_t ch = 0; ch < channels; ++ch)                            \
                    values[ch] = gather<Value>(m_value.array(), index_ + ch, active); \
            }
        #define DR_TEX_CUBIC_ACCUM_VALUE(weight)                               \
            {                                                                  \
                Value weight_ = weight;                                        \
                for (uint32_t ch = 0; ch < channels; ++ch)                     \
                    out_value[ch] = fmadd(values[ch], weight_, out_value[ch]); \
            }

        #define DR_TEX_CUBIC_ACCUM_GRAD(dim, weight)                                             \
            {                                                                                    \
                uint32_t dim_ = dim;                                                             \
                Value weight_ = weight;                                                          \
                for (uint32_t ch = 0; ch < channels; ++ch)                                       \
                    out_gradient[ch][dim_] = fmadd(values[ch], weight_, out_gradient[ch][dim_]); \
            }

        if constexpr (Dimension == 1) {
            Array4 wx = compute_weight(0, false),
                   gx = compute_weight(0, true);
            for (uint32_t ix = 0; ix < 4; ++ix) {
                DR_TEX_CUBIC_GATHER(idx[ix]);
                DR_TEX_CUBIC_ACCUM_VALUE(wx[ix]);
                DR_TEX_CUBIC_ACCUM_GRAD(0, gx[ix]);
            }
        } else if constexpr (Dimension == 2) {
            Array4 wx = compute_weight(0, false),
                   wy = compute_weight(1, false),
                   gx = compute_weight(0, true),
                   gy = compute_weight(1, true);
            for (uint32_t iy = 0; iy < 4; ++iy)
                for (uint32_t ix = 0; ix < 4; ++ix) {
                    DR_TEX_CUBIC_GATHER(idx[iy * 4 + ix]);
                    DR_TEX_CUBIC_ACCUM_VALUE(wx[ix] * wy[iy]);
                    DR_TEX_CUBIC_ACCUM_GRAD(0, gx[ix] * wy[iy]);
                    DR_TEX_CUBIC_ACCUM_GRAD(1, wx[ix] * gy[iy]);
                }
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0, false),
                   wy = compute_weight(1, false),
                   wz = compute_weight(2, false),
                   gx = compute_weight(0, true),
                   gy = compute_weight(1, true),
                   gz = compute_weight(2, true);
            for (uint32_t iz = 0; iz < 4; ++iz)
                for (uint32_t iy = 0; iy < 4; ++iy)
                    for (uint32_t ix = 0; ix < 4; ++ix) {
                        DR_TEX_CUBIC_GATHER(idx[iz * 16 + iy * 4 + ix]);
                        DR_TEX_CUBIC_ACCUM_VALUE(wx[ix] * wy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_GRAD(0, gx[ix] * wy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_GRAD(1, wx[ix] * gy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_GRAD(2, wx[ix] * wy[iy] * gz[iz]);
                    }
        }

        #undef DR_TEX_CUBIC_GATHER
        #undef DR_TEX_CUBIC_ACCUM_VALUE
        #undef DR_TEX_CUBIC_ACCUM_GRAD

        // transform volume from unit size to its resolution
        for (uint32_t ch = 0; ch < channels; ++ch)
            for (uint32_t dim = 0; dim < Dimension; ++dim)
                out_gradient[ch][dim] *= res_f[dim];
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
                            Value *out_value,
                            Array<Value, Dimension> *out_gradient,
                            Matrix<Value, Dimension> *out_hessian,
                            Mask active = true) const {
        using Array4 = Array<Value, 4>;
        using InterpOffset = Array<Int32, ipow(4, Dimension)>;
        using InterpPosI = Array<InterpOffset, Dimension>;
        using InterpIdx = uint32_array_t<InterpOffset>;

        if constexpr (!is_array_v<Mask>)
            active = true;

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

        const auto compute_weight_hessian = [&pos_a](uint32_t dim) -> Array4 {
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
            out_value[ch] = zeros<Value>();
            out_gradient[ch] = zeros<PosF>();
            for (uint32_t dim1 = 0; dim1 < Dimension; ++dim1)
                out_hessian[ch][dim1] = zeros<PosF>();
        }
        ArrayX values = empty<ArrayX>(channels);

        // Make sure channel related operations are executed together
        #define DR_TEX_CUBIC_GATHER(index)                                            \
            {                                                                         \
                UInt32 index_ = index;                                                \
                for (uint32_t ch = 0; ch < channels; ++ch)                            \
                    values[ch] = gather<Value>(m_value.array(), index_ + ch, active); \
            }
        #define DR_TEX_CUBIC_ACCUM_VALUE(weight_value)                                \
            {                                                                         \
                Value weight_value_ = weight_value;                                   \
                for (uint32_t ch = 0; ch < channels; ++ch)                            \
                    out_value[ch] = fmadd(values[ch], weight_value_, out_value[ch]);  \
            }
        #define DR_TEX_CUBIC_ACCUM_GRAD(dim, weight_grad)                                             \
            {                                                                                         \
                uint32_t dim_ = dim;                                                                  \
                Value weight_grad_ = weight_grad;                                                     \
                for (uint32_t ch = 0; ch < channels; ++ch)                                            \
                    out_gradient[ch][dim_] = fmadd(values[ch], weight_grad_, out_gradient[ch][dim_]); \
            }
        #define DR_TEX_CUBIC_ACCUM_HESSIAN(dim1, dim2, weight_hessian)                                                 \
            {                                                                                                          \
                uint32_t dim1_ = dim1,                                                                                 \
                         dim2_ = dim2;                                                                                 \
                Value weight_hessian_ = weight_hessian;                                                                \
                for (uint32_t ch = 0; ch < channels; ++ch)                                                             \
                    out_hessian[ch][dim1_][dim2_] = fmadd(values[ch], weight_hessian_, out_hessian[ch][dim1_][dim2_]); \
            }
        #define DR_TEX_CUBIC_HESSIAN_SYMM(dim1, dim2)                              \
            {                                                                      \
                uint32_t dim1_ = dim1,                                             \
                         dim2_ = dim2;                                             \
                for (uint32_t ch = 0; ch < channels; ++ch)                         \
                    out_hessian[ch][dim2_][dim1_] = out_hessian[ch][dim1_][dim2_]; \
            }

        if constexpr (Dimension == 1) {
            Array4 wx  = compute_weight(0),
                   gx  = compute_weight_gradient(0),
                   ggx = compute_weight_hessian(0);
            for (uint32_t ix = 0; ix < 4; ++ix) {
                DR_TEX_CUBIC_GATHER(idx[ix]);
                DR_TEX_CUBIC_ACCUM_VALUE(wx[ix]);
                DR_TEX_CUBIC_ACCUM_GRAD(0, gx[ix]);
                DR_TEX_CUBIC_ACCUM_HESSIAN(0, 0, ggx[ix]);
            }
            DRJIT_MARK_USED(compute_weight);
        } else if constexpr (Dimension == 2) {
            Array4 wx  = compute_weight(0),
                   wy  = compute_weight(1),
                   gx  = compute_weight_gradient(0),
                   gy  = compute_weight_gradient(1),
                   ggx = compute_weight_hessian(0),
                   ggy = compute_weight_hessian(1);
            for (uint32_t iy = 0; iy < 4; ++iy)
                for (uint32_t ix = 0; ix < 4; ++ix) {
                    DR_TEX_CUBIC_GATHER(idx[iy * 4 + ix]);
                    DR_TEX_CUBIC_ACCUM_VALUE(wx[ix] * wy[iy]);
                    DR_TEX_CUBIC_ACCUM_GRAD(0, gx[ix] * wy[iy]);
                    DR_TEX_CUBIC_ACCUM_GRAD(1, wx[ix] * gy[iy]);
                    DR_TEX_CUBIC_ACCUM_HESSIAN(0, 0, ggx[ix] * wy[iy]);
                    DR_TEX_CUBIC_ACCUM_HESSIAN(0, 1, gx[ix] * gy[iy]);
                    DR_TEX_CUBIC_ACCUM_HESSIAN(1, 1, wx[ix] * ggy[iy]);
                }
            DR_TEX_CUBIC_HESSIAN_SYMM(0, 1);
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0),
                   wy = compute_weight(1),
                   wz = compute_weight(2),
                   gx = compute_weight_gradient(0),
                   gy = compute_weight_gradient(1),
                   gz = compute_weight_gradient(2),
                   ggx = compute_weight_hessian(0),
                   ggy = compute_weight_hessian(1),
                   ggz = compute_weight_hessian(2);
            for (uint32_t iz = 0; iz < 4; ++iz)
                for (uint32_t iy = 0; iy < 4; ++iy)
                    for (uint32_t ix = 0; ix < 4; ++ix) {
                        DR_TEX_CUBIC_GATHER(idx[iz * 16 + iy * 4 + ix]);
                        DR_TEX_CUBIC_ACCUM_VALUE(wx[ix] * wy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_GRAD(0, gx[ix] * wy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_GRAD(1, wx[ix] * gy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_GRAD(2, wx[ix] * wy[iy] * gz[iz]);
                        DR_TEX_CUBIC_ACCUM_HESSIAN(0, 0, ggx[ix] * wy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_HESSIAN(1, 1, wx[ix] * ggy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_HESSIAN(2, 2, wx[ix] * wy[iy] * ggz[iz]);
                        DR_TEX_CUBIC_ACCUM_HESSIAN(0, 1, gx[ix] * gy[iy] * wz[iz]);
                        DR_TEX_CUBIC_ACCUM_HESSIAN(0, 2, gx[ix] * wy[iy] * gz[iz]);
                        DR_TEX_CUBIC_ACCUM_HESSIAN(1, 2, wx[ix] * gy[iy] * gz[iz]);
                    }
            DR_TEX_CUBIC_HESSIAN_SYMM(0, 1);
            DR_TEX_CUBIC_HESSIAN_SYMM(0, 2);
            DR_TEX_CUBIC_HESSIAN_SYMM(1, 2);
        }

        #undef DR_TEX_CUBIC_GATHER
        #undef DR_TEX_CUBIC_ACCUM_VALUE
        #undef DR_TEX_CUBIC_ACCUM_GRAD
        #undef DR_TEX_CUBIC_ACCUM_HESSIAN
        #undef DR_TEX_CUBIC_HESSIAN_SYMM

        // transform volume from unit size to its resolution
        for (uint32_t ch = 0; ch < channels; ++ch)
            for (uint32_t dim1 = 0; dim1 < Dimension; ++dim1) {
                out_gradient[ch][dim1] *= res_f[dim1];
                for (uint32_t dim2 = 0; dim2 < Dimension; ++dim2)
                    out_hessian[ch][dim1][dim2] *= res_f[dim1] * res_f[dim2];
            }
    }

    /**
     * \brief Applies the configured texture wrapping mode to an integer
     * position
     */
    template <typename T> T wrap(const T &pos) const {
        using Scalar = scalar_t<T>;
        static_assert(
            array_size_v<T> == Dimension &&
            std::is_integral_v<Scalar> &&
            std::is_signed_v<Scalar>
        );

        Array<Int32, Dimension> shape = m_shape_opaque;
        if (m_wrap_mode == WrapMode::Clamp) {
            return clamp(pos, 0, shape - 1);
        } else {
            T value_shift_neg = select(pos < 0, pos + 1, pos);

            T div;
            for (size_t i = 0; i < Dimension; ++i)
                div[i] = m_inv_resolution[i](value_shift_neg[i]);

            T mod = pos - div * shape;
            mod[mod < 0] += T(shape);

            if (m_wrap_mode == WrapMode::Mirror)
                // Starting at 0, flip the texture every other repetition
                // (flip when: even number of repetitions in negative direction,
                // or odd number of repetitions in positive direction)
                mod = select(eq(div & 1, 0) ^ (pos < 0), mod, shape - 1 - mod);

            return mod;
        }
    }

protected:
    void init(const size_t *shape, size_t channels, bool use_accel,
              FilterMode filter_mode, WrapMode wrap_mode,
              bool init_tensor = true) {
        if (channels == 0)
            drjit_raise("Texture::Texture(): must have at least 1 channel!");

        m_size = channels;
        size_t tensor_shape[Dimension + 1]{};

        for (size_t i = 0; i < Dimension; ++i) {
            tensor_shape[i] = shape[i];
            m_shape_opaque[Dimension - 1 - i] = opaque<UInt32>((uint32_t) shape[i]);
            m_inv_resolution[Dimension - 1 - i] = divisor<int32_t>((int32_t) shape[i]);
            m_size *= shape[i];
        }
        tensor_shape[Dimension] = channels;

        if (init_tensor)
            m_value = TensorXf(zeros<Storage>(m_size), Dimension + 1, tensor_shape);

        m_use_accel = use_accel;
        m_filter_mode = filter_mode;
        m_wrap_mode = wrap_mode;

        if constexpr (HasCudaTexture) {
            if (m_use_accel) {
                size_t tex_shape[Dimension];
                reverse_tensor_shape(tex_shape, false);
                m_handle = jit_cuda_tex_create(Dimension, tex_shape, channels,
                                               (int) filter_mode, (int) wrap_mode);
            }
        }
    }

private:
    /// Helper function to reverse the tensor (\ref Texture.m_value) shape
    void reverse_tensor_shape(size_t *output, bool include_channels) const {
        for (size_t i = 0; i < Dimension; ++i)
            output[i] = m_value.shape(Dimension - 1 - i);
        if (include_channels)
            output[Dimension] = m_value.shape(Dimension);
    }

    /// Helper function to compute integer powers of numbers
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
            for (uint32_t iy = 0; iy < Length; iy++) {
                for (uint32_t ix = 0; ix < Length; ix++) {
                    pos_i[0][iy * Length + ix] = offset[ix] + pos.x();
                    pos_i[1][ix * Length + iy] = offset[ix] + pos.y();
                }
            }
        } else if constexpr (Dimension == 3) {
            constexpr size_t LengthSqr = Length * Length;
            for (uint32_t iz = 0; iz < Length; iz++) {
                for (uint32_t iy = 0; iy < Length; iy++) {
                    for (uint32_t ix = 0; ix < Length; ix++) {
                        pos_i[0][iz * LengthSqr + iy * Length + ix] =
                            offset[ix] + pos.x();
                        pos_i[1][iz * LengthSqr + ix * Length + iy] =
                            offset[ix] + pos.y();
                        pos_i[2][ix * LengthSqr + iy * Length + iz] =
                            offset[ix] + pos.z();
                    }
                }
            }
        }

        return pos_i;
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

        uint32_t channels = (uint32_t) m_value.shape(Dimension);

        return index * channels;
    }

private:
    void *m_handle = nullptr;
    size_t m_size = 0;
    mutable TensorXf m_value;

    // Stored in this order: width, height, depth
    Array<UInt32, Dimension> m_shape_opaque;
    divisor<int32_t> m_inv_resolution[Dimension] { };

    FilterMode m_filter_mode;
    WrapMode m_wrap_mode;
    bool m_use_accel = false;
    bool m_migrate = false;
    mutable bool m_migrated = false;
};

NAMESPACE_END(drjit)
