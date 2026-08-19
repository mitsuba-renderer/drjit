/*
    drjit/texture.h -- N-dimensional Texture interpolation with GPU acceleration

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/array.h>
#include <drjit-core/half.h>
#include <drjit-core/texture.h>
#include <drjit/color.h>
#include <drjit/dynamic.h>
#include <drjit/extra.h>
#include <drjit/texture_impl.h>
#include <drjit/idiv.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>
#include <drjit/util.h>
#include <drjit/traversable_base.h>
#include <array>
#include <cassert>
#include <memory>

#pragma once

NAMESPACE_BEGIN(drjit)

template <typename Storage_, size_t Dimension> class Texture : TraversableBase {
public:
    static constexpr bool IsCUDA = is_cuda_v<Storage_>;
    static constexpr bool IsMetal = is_metal_v<Storage_>;
    static constexpr bool IsDynamic = is_dynamic_v<Storage_>;
    static constexpr bool IsHalf = std::is_same_v<scalar_t<Storage_>, drjit::half>;
    static constexpr bool IsSingle = std::is_same_v<scalar_t<Storage_>, float>;
    static constexpr bool IsUInt8 = std::is_same_v<scalar_t<Storage_>, uint8_t>;
    static constexpr bool IsDiff = is_diff_v<Storage_> && !IsUInt8;

    // Half/single-precision float and normalized 8-bit hardware textures are supported
    static constexpr bool HasGPUTexture =
        (IsHalf || IsSingle || IsUInt8) && (IsCUDA || IsMetal);

    using Int32 = int32_array_t<Storage_>;
    using UInt32 = uint32_array_t<Storage_>;
    using Storage = std::conditional_t<IsDynamic, Storage_, DynamicArray<Storage_>>;
    using TensorXf = Tensor<Storage>;

    // Dynamic integer array holding the per-level MIP constant records
    using Int32Buffer = int32_array_t<Storage>;

    /// Stride of the records in \ref m_mip_table
    static constexpr uint32_t MipStride = Dimension == 1 ? 4 : 8;

    // Precomputed reciprocal for the Repeat/Mirror wrap math.
    using Divisor = std::conditional_t<is_jit_v<Storage_>, divisor<Int32, true>,
                                       divisor<int32_t, true>>;

    /// Query position type for an evaluation returning the array type \c Output
    template <typename Output>
    using position_for = Array<value_t<Output>, Dimension>;

    /// Active mask type for an evaluation returning the array type \c Output
    template <typename Output>
    using mask_for = mask_t<value_t<Output>>;

    // Backend that ``jit_tex_*`` dispatches on (only meaningful if HasGPUTexture)
    static constexpr JitBackend Backend = backend_v<Storage_>;

    /// Default constructor: create an invalid texture object
    Texture() = default;

    /**
     * \brief Create a new texture with the specified size and channel count
     *
     * On GPU backends, this is a slow operation that synchronizes the pipeline
     * to rewrite the device memory map. Therefore, prefer reusing and updating
     * texture objects via \ref set_value() and \ref set_tensor() over creating
     * new ones.
     *
     * When \c use_accel is set to \c false, GPU backends will emulate the
     * texture API instead of using the hardware texture units. In other modes,
     * this argument has no effect.
     *
     * A \c writable texture can be modified using \ref write(). Such a
     * texture cannot be MIP-mapped.
     *
     * The \c filter_mode parameter defines the interpolation method to be used
     * in all evaluation routines. By default, the texture is linearly
     * interpolated. Besides nearest/linear filtering, the implementation also
     * provides a clamped cubic B-spline interpolation scheme in case a
     * higher-order interpolation is needed. On the CUDA and Metal backends,
     * this is done using a series of linear lookups to optimally use the
     * hardware (hence, linear filtering must be enabled to use this feature).
     *
     * When evaluating the texture outside of its boundaries, the \c wrap_mode
     * defines the wrapping method. The default behavior is \ref
     * WrapMode::Clamp, which indefinitely extends the colors on the boundary
     * along each dimension.
     *
     * On the CUDA and Metal backends, hardware texture units resolve the
     * sub-texel position using reduced-precision fixed-point weights (8
     * fractional bits on CUDA, i.e. 256 steps between texels). This does not
     * degrade the stored values or the interpolated quantity, only how finely
     * the fractional position within a texel is resolved. Set \c use_accel to
     * \c false to disable the texture units and avoid this approximation at
     * some cost in performance.
     *
     * For 8-bit textures, setting \c srgb additionally requests that samples
     * be decoded from sRGB to linear. Passing it for a floating-point texture
     * raises an error. Channels are grouped into hardware RGBA quads, so within
     * each group of four the first three are decoded and the fourth (alpha) is
     * left linear (e.g. channel 3 is linear for a 6-channel texture).
     *
     * Specifying a \c mip_filter causes the implementation to create a MIP
     * pyramid for filtered lookups via \ref eval_lod() and \ref
     * eval_filtered(). The functions \ref set_value() / \ref set_tensor()
     * regenerate it from the base level using a box filter that averages two
     * texels per axis (applied in linear space for sRGB textures). MIP-mapped
     * textures cannot be \c writable.
     *
     * The \c max_aniso parameter only applies to MIP-mapped textures and
     * controls the number of taps that anisotropic filtering in \ref
     * eval_filtered() may use. The value 1 selects isotropic filtering, and
     * values above the hardware limit of 16 raise an error.
     *
     * The \c mip_basis parameter determines the internal representation of
     * MIP-mapped textures. The default \ref MipBasis::Standard derives a
     * standard MIP pyramid from the base image.
     *
     * When \c mip_basis is set to \ref MipBasis::Laplacian, the authoritative
     * representation is no longer the base image but a set of per-level
     * coefficient tensors (see \ref tensor(size_t)). The MIP pyramid uploaded
     * to the GPU is then derived from these tensors by repeated upsampling and
     * summation. This choice is mainly useful for workloads that perform
     * gradient-based optimization of textures with filtered texture lookups.
     * See the Dr.Jit documentation on textures for additional detail. Laplacian
     * mode requires a MIP-mapped texture with floating-point storage on a JIT
     * backend and does not support migration.
     */
    Texture(const size_t shape[Dimension], size_t channels,
            bool use_accel = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp,
            bool writable = false, bool srgb = false,
            MipFilter mip_filter = MipFilter::Disabled,
            size_t max_aniso = 8,
            MipBasis mip_basis = MipBasis::Standard)
        : m_srgb(srgb) {
        init(shape, channels, use_accel, filter_mode, wrap_mode,
             /* init_tensor = */ true, writable, /* external = */ nullptr,
             mip_filter, max_aniso, mip_basis);
    }

    /**
     * \brief Construct a new texture from a given tensor
     *
     * This constructor allocates texture memory just like the previous
     * constructor, extracting shape information from \c tensor. It then also
     * invokes <tt>set_tensor(tensor)</tt> to fill the texture memory with the
     * provided tensor.
     *
     * When ``migrate`` is set to ``true`` (the default), Dr.Jit moves the
     * texture to the GPU backends to avoid redundant storage. Values like \ref
     * tensor(), \ref value() then produce a differentiable symbolic view of
     * the migrated memory.
     *
     * The \c use_accel, \c filter_mode, \c wrap_mode, \c srgb, \c mip_filter,
     * \c max_aniso, and \c mip_basis parameters have the same defaults and
     * behaviors as in the shape-based constructor. This overload infers the
     * shape and channel count from the tensor and does not accept \c writable.
     */
    template <typename TensorT>
    Texture(TensorT &&tensor, bool use_accel = true, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp, bool srgb = false,
            MipFilter mip_filter = MipFilter::Disabled, size_t max_aniso = 8,
            MipBasis mip_basis = MipBasis::Standard)
        : m_srgb(srgb) {
        if (tensor.ndim() != Dimension + 1)
            jit_raise("Texture::Texture(): tensor dimension must equal "
                        "texture dimension plus one.");
        if (mip_basis == MipBasis::Laplacian && migrate)
            jit_raise("Texture(): migration is not supported in Laplacian "
                      "mode, please pass migrate=false.");
        init(tensor.shape().data(), tensor.shape(Dimension), use_accel,
             filter_mode, wrap_mode, /* init_tensor = */ true,
             /* writable = */ false, /* external = */ nullptr,
             mip_filter, max_aniso, mip_basis);
        set_tensor(std::forward<TensorT>(tensor), migrate);
    }

    /**
     * \brief Wrap an existing native texture as a Dr.Jit texture
     *
     * Builds a texture that *wraps* an externally-owned native texture rather
     * than allocating its own storage. The \c handle encodes an
     * ``id<MTLTexture>`` pointer on Metal or an OpenGL texture ID on CUDA.
     * Shape and channel count are inferred from the texture; its dimensionality
     * and component type must match this texture type.
     *
     * If \c writable is \c false the texture is wrapped for sampling (\ref
     * eval()). If \c true it is wrapped for *rendering into* via \ref write(),
     * and the native texture must allow shader writes / surface stores. On CUDA
     * such a wrap is bound as a surface and cannot also be sampled.
     *
     * A texture wrapping a cross-API handle (OpenGL on CUDA) requires a \ref
     * map() / \ref unmap() pair around each use; on Metal those are no-ops. The
     * native handle can be recovered with \ref native_handle().
     */
    static Texture from_native_handle(uintptr_t handle, bool writable = false,
                                      FilterMode filter_mode = FilterMode::Linear,
                                      WrapMode wrap_mode = WrapMode::Clamp,
                                      bool srgb = false) {
        return Texture(handle, writable, filter_mode, wrap_mode, srgb);
    }

    Texture(Texture &&other) noexcept { move_from(std::move(other)); }

    Texture &operator=(Texture &&other) noexcept {
        if constexpr (HasGPUTexture)
            jit_tex_destroy(m_handle);
        move_from(std::move(other));
        return *this;
    }

    Texture(const Texture &) = delete;
    Texture &operator=(const Texture &) = delete;

    ~Texture() {
        if constexpr (HasGPUTexture) {
            if (m_use_accel)
                jit_tex_destroy(m_handle);
        }
    }

private:
    /// Private constructor for \ref from_native_handle()
    Texture(uintptr_t handle, bool writable, FilterMode filter_mode,
            WrapMode wrap_mode, bool srgb) : m_srgb(srgb) {
        if constexpr (HasGPUTexture) {
            if (srgb && !IsUInt8)
                jit_raise("Texture(): the 'srgb' flag is only supported for "
                          "8-bit (UInt8) textures.");
            void *h = jit_tex_wrap(Backend, handle, Dimension,
                                   (int) type_v<scalar_t<Storage_>>,
                                   (int) writable, (int) filter_mode,
                                   (int) wrap_mode, (int) m_srgb);

            // The native shape is innermost-first (+channels); reverse it into
            // the tensor order that init() expects.
            size_t shape_tex[Dimension + 1];
            jit_tex_get_shape(h, shape_tex);
            size_t channels = shape_tex[Dimension];
            size_t tensor_shape[Dimension];
            for (size_t i = 0; i < Dimension; ++i)
                tensor_shape[i] = shape_tex[Dimension - 1 - i];

            init(tensor_shape, channels, /* use_accel = */ true, filter_mode,
                 wrap_mode, /* init_tensor = */ true, writable,
                 /* external = */ h);
        } else {
            (void) handle; (void) writable; (void) filter_mode; (void) wrap_mode;
            (void) srgb;
            jit_raise("Texture::from_native_handle() requires the CUDA or Metal "
                      "backend.");
        }
    }

public:
    /// Opaque texture handle on GPU backends, nullptr elsewhere
    const void *handle() const { return m_handle; }

    /// Return the texture dimension plus one (for the "channel dimension")
    size_t ndim() const { return Dimension + 1; }

    /// Return the texture shape
    const size_t *shape() const { return m_shape; }

    /// Return the number of channels (equals ``shape()[ndim()-1]``)
    size_t channel_count() const { return m_channels; }

    /// Return the texture filtering mode (e.g., nearest, bilinear, etc.)
    FilterMode filter_mode() const { return m_filter_mode; }

    /// Return the boundary handling mode for out-of-bounds lookups
    WrapMode wrap_mode() const { return m_wrap_mode; }

    /// Return the MIP level selection mode of filtered lookups
    MipFilter mip_filter() const { return m_mip_filter; }

    /// Return the number of MIP pyramid levels, including the base level
    /// (1 when the texture is not MIP-mapped)
    size_t mip_levels() const { return m_level_count; }

    /// Return the anisotropic tap bound of \ref eval_filtered()
    size_t max_aniso() const { return m_max_aniso; }

    /// Return the MIP basis of the texture
    MipBasis mip_basis() const { return m_mip_basis; }

    /// Is the texture data held exclusively in GPU texture memory? True
    /// after a migration, and for writable/wrapped textures.
    bool migrated() const { return m_migrated || m_hw_mutable; }

    /// Are hardware texture units used for evaluation?
    bool use_accel() const { return m_use_accel; }

    /// Was this texture created so that kernels may store into it via \ref write()?
    bool writable() const { return m_writable; }

    /// Are 8-bit samples decoded from sRGB to linear?
    bool srgb() const { return m_srgb; }

    /// Map an imported texture (\ref from_native_handle()) for use by Dr.Jit
    /// (no-op on Metal, required for CUDA/OpenGL).
    void map() {
        if constexpr (HasGPUTexture)
            jit_tex_map(m_handle);
    }

    /// Release a mapping established by \ref map().
    void unmap() {
        if constexpr (HasGPUTexture)
            jit_tex_unmap(m_handle);
    }

    /**
     * \brief Return the native texture handle (as an integer)
     *
     * On Metal this is the ``id<MTLTexture>`` of sub-texture \c sub_index. On
     * CUDA it is the wrapped OpenGL texture id (\c sub_index is ignored, and the
     * result is 0 unless the texture wraps an OpenGL handle).
     */
    uintptr_t native_handle(size_t sub_index = 0) const {
        if constexpr (HasGPUTexture)
            return jit_tex_native_handle(m_handle, sub_index);
        else
            return 0;
    }

    /**
     * \brief Overwrite the texture contents with the provided linearized 1D
     * array
     *
     * When \c migrate is set, the CUDA and Metal backends migrate the texture
     * data into the GPU's native texture format to avoid redundant storage.
     *
     * With \ref MipBasis::Laplacian, the array is first decomposed into
     * per-level coefficients. A subsequent \ref value() then reproduces it up
     * to floating point rounding. Migration is unavailable in that case and
     * raises an exception.
     */
    template <typename StorageT>
    void set_value(StorageT &&value, bool migrate = false) {
        if constexpr (!is_jit_v<Storage_>) {
            if (value.size() != m_size)
                jit_raise("Texture::set_value(): unexpected array size!");
            m_padded_tensor.array() = std::forward<StorageT>(value);
            build_mipmap(m_padded_tensor.array());
        } else /* JIT variant */ {
            if (m_mip_basis == MipBasis::Laplacian) {
                if (migrate)
                    jit_raise("Texture::set_value(): migration is not "
                              "supported in Laplacian mode.");
                size_t unpadded_size = m_size / m_channels_storage * m_channels;
                if (value.size() != unpadded_size)
                    jit_raise("Texture::set_value(): unexpected array size "
                              "(%zu vs %zu)!", value.size(), unpadded_size);
                decompose(value);
                rebuild();
                return;
            }

            Storage padded_value;

            if (m_channels_storage != m_channels) {
                padded_value = steal_storage(ad_tex_repack(
                    combined_index(value),
                    (uint32_t) (m_size / m_channels_storage),
                    (uint32_t) m_channels_storage, (uint32_t) m_channels));
            } else {
                padded_value = value;
            }

            if (padded_value.size() != m_size)
                jit_raise(
                    "Texture::set_value(): unexpected array size (%zu vs %zu)!",
                    padded_value.size(), m_size);

            // Stash the AD index of the unpadded `value`.
            // The updates to `m_tensor` below re-attach it.
            if constexpr (IsDiff) {
                if (grad_enabled(value))
                    m_tensor.array() =
                        replace_grad(m_tensor.array(), value);
            }

            build_mipmap(padded_value);

            if constexpr (HasGPUTexture) {
                if (m_use_accel) {
                    upload_levels(padded_value);

                    // Hardware-mutable textures (writable/ wrapped) keep their
                    // authoritative copy in the hardware texture and retain no
                    // buffer, regardless of `migrate`. For them, every update
                    // therefore takes the migration path, which replaces the
                    // tensor members with symbolic views of the uploaded data.
                    if (migrate || m_hw_mutable) {
                        m_padded_tensor.array() = padded_value;
                        install_views();
                        m_migrated = migrate;
                        return;
                    }
                }
            }

            m_padded_tensor.array() = padded_value;
            m_migrated = false;
            update_tensor();
        }
    }

    /**
     * \brief Overwrite the texture contents with the provided tensor
     *
     * This method updates the values of all texels. Changing the texture
     * resolution or its number of channels is also supported. However, on the
     * CUDA and Metal backends, such operations have a significantly larger
     * overhead (new hardware texture objects must be created; on CUDA this also
     * synchronizes the GPU pipeline).
     *
     * When \c migrate is set to \c true on the CUDA and Metal backends, the
     * texture information is migrated to GPU texture memory to avoid
     * redundant storage.
     *
     * With \ref MipBasis::Laplacian, the tensor is first decomposed into the
     * per-level coefficient tensors, and the sampled pyramid is then rebuilt
     * from them. A subsequent \ref tensor() reproduces the input up to floating
     * point rounding. Migration is unavailable in that case and raises an
     * exception.
     */
    template <typename TensorT>
    void set_tensor(TensorT &&tensor, bool migrate = false) {
        if (tensor.ndim() != Dimension + 1)
            jit_raise("Texture::set_tensor(): tensor dimension must equal "
                      "texture dimension plus one (channels).");

        if ((void *) &tensor == (void *) &m_tensor) {
            jit_log(::LogLevel::Warn,
                    "Texture::set_tensor(): the `tensor` argument is a "
                    "reference to this texture's own tensor representation "
                    "(obtained through `Texture::tensor()`. Such an update "
                    "must be applied with the `Texture::update_inplace()` "
                    "method.");
            return;
        }

        bool shape_changed = false;
        for (size_t i = 0; i < Dimension + 1; ++i) {
            if (m_shape[i] != tensor.shape(i)) {
                shape_changed = true;
                break;
            }
        }

        // Only update tensors & CUDA texture if shape changed
        init(tensor.shape().data(), tensor.shape(Dimension),
             m_use_accel, m_filter_mode, m_wrap_mode, shape_changed,
             m_writable, nullptr, m_mip_filter, m_max_aniso,
             m_mip_basis);

        if constexpr (std::is_lvalue_reference_v<TensorT>)
            set_value(tensor.array(), migrate);
        else
            set_value(std::move(tensor.array()), migrate);
    }

    /**
     * \brief Update the texture after applying an indirect update to its tensor
     * representation (obtained with \ref tensor()).
     *
     * A tensor representation of this texture object can be retrieved with
     * \ref tensor(). That representation can be modified, but in order to apply
     * it successfully to the texture, this method must also be called. In
     * short, this method will use the tensor representation to update the
     * texture's internal state.
     *
     * When \c migrate is set to \c true on the CUDA and Metal backends, the
     * texture information is migrated to GPU texture memory to avoid redundant
     * storage.
     *
     * With \ref MipBasis::Laplacian, the per-level coefficient tensors (see
     * \ref tensor(size_t)) are the authoritative state instead, and this
     * method rebuilds the sampled pyramid from their current contents. An
     * optimization loop should write the coefficient tensors in place and call
     * this method once per step. Migration is unavailable in that case and
     * raises an exception.
     */
    void update_inplace(bool migrate = false) {
        if (m_mip_basis == MipBasis::Laplacian) {
            // The coefficient tensors are the authoritative state; rebuild
            // the sampled pyramid from their current (possibly externally
            // modified) contents.
            if (migrate)
                jit_raise("Texture::update_inplace(): migration is not "
                          "supported in Laplacian mode.");
            rebuild();
            return;
        }

        if (m_tensor.ndim() != Dimension + 1)
            jit_raise("Texture::update_inplace(): tensor dimension must equal "
                      "texture dimension plus one (channels).");

        bool shape_changed = false;
        for (size_t i = 0; i < Dimension + 1; ++i) {
            if (m_shape[i] != m_tensor.shape(i)) {
                shape_changed = true;
                break;
            }
        }

        if constexpr (!is_jit_v<Storage_>) {
            if (shape_changed) {
                init(m_tensor.shape().data(),
                     m_tensor.shape(Dimension), m_use_accel, m_filter_mode,
                     m_wrap_mode, true, m_writable, nullptr, m_mip_filter,
                     m_max_aniso, m_mip_basis);
            } else {
                // Only the MIP pyramid must be refreshed
                build_mipmap(m_padded_tensor.array());
                return;
            }
        } else {
            // `Texture::init` might overwrite `m_tensor` with a
            // zero-initialized tensor, so let's copy it first
            TensorXf inbound_tensor(m_tensor);

            init(m_tensor.shape().data(),
                 m_tensor.shape(Dimension), m_use_accel, m_filter_mode,
                 m_wrap_mode, shape_changed, m_writable, nullptr, m_mip_filter,
                 m_max_aniso, m_mip_basis);

            m_tensor.array() = inbound_tensor;
        }

        set_value(m_tensor.array(), migrate);
    }

    /// Return the texture data as an array object. See the remark in \ref tensor()
    const Storage &value() const { return tensor().array(); }

    /**
     * \brief Return the texture data as a tensor object
     *
     * \remark
     *    When the texture was migrated to the GPU, this function returns a
     *    symbolic view that occupies no actual storage. Its evaluation will
     *    query the migrated hardware texture. Changing the texture contents
     *    via \ref set_tensor(), \ref write(), etc., will also change this
     *    view, so be sure to evaluate beforehand.
     */
    const TensorXf &tensor() const {
        if constexpr (!is_jit_v<Storage_>) {
            // Scalar storage is always unpadded, and ``m_tensor`` is unused.
            return m_padded_tensor;
        } else {
            refresh();
            return m_tensor;
        }
    }

    /**
     * \brief Return the texture data as a tensor object
     *
     * Although the returned object is not const, changes to it are only fully
     * propagated to the Texture instance when a subsequent call to
     * \ref update_inplace() is made.
     */
    TensorXf &tensor() {
        return const_cast<TensorXf &>(
            const_cast<const Texture<Storage_, Dimension> *>(this)->tensor());
    }

    /**
     * \brief Return the coefficient tensor of one pyramid level (Laplacian
     * basis only)
     *
     * The tensor uses the resolution of pyramid level \c level and the
     * public (unpadded) channel count. Changes to it are only propagated to
     * the texture by a subsequent call to \ref update_inplace().
     */
    const TensorXf &tensor(size_t level) const {
        check_level_access("tensor", level);
        return m_levels[level];
    }

    TensorXf &tensor(size_t level) {
        return const_cast<TensorXf &>(
            const_cast<const Texture<Storage_, Dimension> *>(this)->tensor(level));
    }

    /**
     * \brief Overwrite the coefficient tensor of one pyramid level and
     * rebuild the sampled pyramid (Laplacian basis only)
     *
     * The tensor shape must match the level; resolution changes go through
     * the whole-image \ref set_tensor(). Passing ``rebuild = false`` only
     * rebinds the coefficient tensor, which costs no computation; the
     * sampled pyramid is then refreshed by a later call to \ref
     * update_inplace(). An optimization loop should assign all levels this
     * way and rebuild once per iteration.
     */
    template <typename TensorT>
    void set_tensor(size_t level, TensorT &&tensor, bool rebuild = true) {
        check_level_access("set_tensor", level);
        if (tensor.ndim() != Dimension + 1)
            jit_raise("Texture::set_tensor(): tensor dimension must equal "
                      "texture dimension plus one (channels).");
        for (size_t i = 0; i < Dimension + 1; ++i)
            if (tensor.shape(i) != m_levels[level].shape(i))
                jit_raise("Texture::set_tensor(): tensor shape mismatch at "
                          "level %zu (resolution changes must use the "
                          "whole-image set_tensor()).", level);
        m_levels[level] = std::forward<TensorT>(tensor);
        if (rebuild)
            this->rebuild();
    }

    /// Overwrite the coefficients of one pyramid level with the provided
    /// linearized 1D array (Laplacian basis only). See the tensor variant for
    /// the meaning of ``rebuild``.
    template <typename StorageT>
    void set_value(size_t level, StorageT &&value, bool rebuild = true) {
        check_level_access("set_value", level);
        if (value.size() != m_levels[level].size())
            jit_raise("Texture::set_value(): unexpected array size "
                      "(%zu vs %zu)!", value.size(), m_levels[level].size());
        m_levels[level].array() = std::forward<StorageT>(value);
        if (rebuild)
            this->rebuild();
    }

    /**
     * \brief Allocate an output array sized to the texture's channel count
     *
     * Statically-sized outputs (e.g. ``Array<Value, 3>``) are
     * default-constructed; dynamically-sized outputs (e.g.
     * ``DynamicArray<Value>``) are allocated to hold \ref m_channels entries.
     */
    template <typename Output> Output alloc_output() const {
        if constexpr (is_dynamic_v<Output>)
            return empty<Output>(m_channels);
        else
            return Output();
    }

    /// Allocate the ``2^Dimension`` corner outputs returned by \ref eval_fetch()
    template <typename Output> Array<Output, (1 << Dimension)>
    alloc_fetch_output() const {
        Array<Output, (1 << Dimension)> out;
        if constexpr (is_dynamic_v<Output>)
            for (size_t i = 0; i < (1 << Dimension); ++i)
                out.set_entry(i, empty<Output>(m_channels));
        // The corners are intentionally uninitialized
#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wuninitialized"
#endif
        return out;
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
    }

    /**
     * \brief Evaluate linear interpolant using explicit arithmetic
     *
     * This is an implementation detail, please use \ref eval() that may
     * dispatch to this function depending on its inputs.
     *
     * If the texture was migrated to the native GPU format (see \c migrate in
     * \c Texture constructors), this function evaluates the hardware texture
     * by querying it at pixel centers and then interpolating manually.
     */
    template <typename Output>
    Output eval_nonaccel(const position_for<Output> &pos,
                         mask_for<Output> active = true) const {
        Output out = alloc_output<Output>();
        if constexpr (is_jit_v<Storage_>)
            eval_jit(pos, out, active, /* use_accel = */ false);
        else
            eval_nonaccel_scalar(pos, out, active);
        return out;
    }

    /**
     * \brief Evaluate the linear interpolant represented by this texture
     *
     * When using the non-hardware-accelerated evaluation, the numerical
     * precision of the interpolation is dictated by the floating point
     * precision of the query point type.
     */
    template <typename Output>
    Output eval(const position_for<Output> &pos,
                mask_for<Output> active = true) const {
        if constexpr (is_jit_v<Storage_>) {
            Output out = alloc_output<Output>();
            eval_jit(pos, out, active, m_use_accel);
            return out;
        } else {
            return eval_nonaccel<Output>(pos, active);
        }
    }

    /**
     * \brief Evaluate the texture at an explicit MIP level of detail
     *
     * A fractional ``lod`` blends the two enclosing pyramid levels under
     * \ref MipFilter::Linear and rounds to the nearest level under \ref
     * MipFilter::Nearest. Out-of-range values are clamped.
     *
     * The method is differentiable with respect to the query position and
     * texture data (including derivative propagation through the MIP pyramid
     * construction) but not with respect to the \c lod argument
     *
     * On a texture without a MIP pyramid, the lookup degrades to a regular
     * non-filtered \ref eval().
     */
    template <typename Output>
    Output eval_lod(const position_for<Output> &pos,
                    const value_t<Output> &lod,
                    mask_for<Output> active = true) const {
        if (m_level_count <= 1)
            return eval<Output>(pos, active);

        if constexpr (is_jit_v<Storage_>) {
            using Value = value_t<Output>;

            Output out = alloc_output<Output>();
            uint64_t *o = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);

            ad_tex_eval_lod(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) m_channels,
                (int) m_filter_mode, (int) m_wrap_mode, (int) m_srgb, m_handle,
                (int) m_use_accel, value_index(), combined_index(m_mip),
                m_mip_table.index(), m_level_count, (int) m_mip_filter,
                resolution_indices().data(), idiv_indices().data(),
                pos_indices(pos).data(), lod.index(), active.index(), o);

            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, steal_value<Value>(o[ch]));
            return out;
        } else {
            using Value = value_t<Output>;
            Output out = alloc_output<Output>();
            Value *res_mem     = (Value *) alloca(sizeof(Value) * m_channels),
                  *scratch_mem = (Value *) alloca(sizeof(Value) * 2 * m_channels);
            detail::tex_scratch<Value> res(res_mem, m_channels),
                                       scratch(scratch_mem, 2 * m_channels);
            detail::tex_eval_lod(scalar_ops<Value>(active), pos.data(), lod,
                                 m_level_count, m_mip_filter, res.data(),
                                 scratch.data());
            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, res[ch]);
            return out;
        }
    }

    /**
     * \brief Perform an anisotropically filtered texture lookup
     *
     * Besides the query position, this function additionally takes
     * texture-space differentials ``ddx`` and ``ddy`` that span the pixel's
     * elliptical footprint. The method averages up to \c max_aniso trilinear
     * taps that are distributed along the major ellipse axis. For \c max_aniso
     * equal to 1, it performs an ordinary trilinear lookup.
     *
     * Hardware anisotropic filtering (if enabled via the \c use_accel
     * constructor argument) is approximate and vendor specific. Results may
     * deviate from the software path by several percent for off-axis
     * footprints. Pass ``use_accel=false`` if it is important that the output
     * remains consistent across backends.
     *
     * The method is differentiable with respect to the query position and
     * texture data (including derivative propagation through the MIP pyramid
     * construction) but not with respect to the \c ddx and \c ddy argument.
     *
     * On a texture without a MIP pyramid, the lookup degrades to a regular
     * non-filtered \ref eval().
     */
    template <typename Output>
    Output eval_filtered(const position_for<Output> &pos,
                         const position_for<Output> &ddx,
                         const position_for<Output> &ddy,
                         mask_for<Output> active = true) const {
        if (m_level_count <= 1)
            return eval<Output>(pos, active);

        if constexpr (is_jit_v<Storage_>) {
            using Value = value_t<Output>;

            Output out = alloc_output<Output>();
            uint64_t *o = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);

            // ``ddx``/``ddy`` are detached filtering metadata, hence plain
            // JIT indices suffice
            uint32_t ddx_i[Dimension], ddy_i[Dimension];
            for (size_t k = 0; k < Dimension; ++k) {
                ddx_i[k] = ddx[k].index();
                ddy_i[k] = ddy[k].index();
            }

            ad_tex_eval_filtered(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) m_channels,
                (int) m_filter_mode, (int) m_wrap_mode, (int) m_srgb, m_handle,
                (int) m_use_accel, value_index(), combined_index(m_mip),
                m_mip_table.index(), m_level_count, (int) m_mip_filter, m_max_aniso,
                resolution_indices().data(), idiv_indices().data(),
                pos_indices(pos).data(), ddx_i, ddy_i, active.index(), o);

            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, steal_value<Value>(o[ch]));
            return out;
        } else {
            using Value = value_t<Output>;
            Output out = alloc_output<Output>();
            Value *res_mem = (Value *) alloca(sizeof(Value) * m_channels);
            detail::tex_scratch<Value> res(res_mem, m_channels);
            detail::tex_eval_filtered(scalar_ops<Value>(active), pos.data(),
                                      ddx.data(), ddy.data(), m_level_count,
                                      m_mip_filter, m_max_aniso, res.data());
            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, res[ch]);
            return out;
        }
    }

    /**
     * \brief Store values into a writable texture
     *
     * The per-channel values in \c value are written to the texel addressed by
     * the integer coordinates \c pos. The texture must have been created with
     * <tt>writable = true</tt>.
     *
     * The store is a side effect and not differentiable. Backends providing a
     * hardware texture write into it, and such a texture is meant for display
     * / external sampling rather than \ref eval(). Without one (LLVM, or
     * double precision), the values are scattered into the backing storage.
     *
     * Reading the texture after writing to it (via \ref value(), \ref
     * tensor(), or the ``eval_*()`` methods) requires an intermediate
     * ``drjit.eval()`` call. The write and the read may otherwise end up in
     * the same kernel, where their relative order is undefined.
     */
    template <typename Value>
    void write(const Array<uint32_array_t<Value>, Dimension> &pos,
               const Value *value, mask_t<Value> active = true) {
        static_assert(is_jit_v<Storage_>,
                      "Texture::write() requires a JIT backend");
        if (!m_writable)
            jit_raise("Texture::write(): texture was not created with "
                      "writable=true.");

        if constexpr (HasGPUTexture) {
            uint32_t pos_idx[Dimension];
            for (size_t i = 0; i < Dimension; ++i)
                pos_idx[i] = pos[i].index();

            uint64_t *val_idx = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);
            for (size_t ch = 0; ch < m_channels; ++ch)
                val_idx[ch] = (uint64_t) value[ch].index();

            ad_tex_write((uint32_t) m_channels_storage, (uint32_t) m_channels,
                         type_v<scalar_t<Storage_>>, (int) m_srgb, m_handle,
                         pos_idx, val_idx, active.index());
        } else {
            // No hardware texture (LLVM, or double precision): scatter into the
            // backing storage instead.
            write_nonaccel(pos, value, active);
            m_tensor_dirty = true;
        }
    }

    /**
     * \brief Fetch the texels that would be referenced in a texture lookup with
     * linear interpolation without actually performing this interpolation.
     *
     * This is an implementation detail, please use \ref eval_fetch() that may
     * dispatch to this function depending on its inputs.
     */
    template <typename Output>
    void eval_fetch_nonaccel(const position_for<Output> &pos,
                             Array<Output, (1 << Dimension)> &out,
                             mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        constexpr size_t ncorner = 1 << Dimension;
        if constexpr (!is_jit_v<Storage_> && !is_dynamic_v<Output>) {
            constexpr uint32_t C = (uint32_t) size_v<Output>;
            assert(m_channels == C);
            Value buf[ncorner * C];
            Value *ptrs[ncorner];
            for (uint32_t c = 0; c < ncorner; ++c)
                ptrs[c] = buf + c * C;
            detail::tex_fetch(scalar_ops<Value, C>(active), pos.data(), ptrs);
            for (uint32_t c = 0; c < ncorner; ++c)
                for (uint32_t ch = 0; ch < C; ++ch)
                    out.entry(c).set_entry(ch, ptrs[c][ch]);
        } else {
            Value *buf_mem = (Value *) alloca(sizeof(Value) * ncorner * m_channels);
            detail::tex_scratch<Value> buf(buf_mem, ncorner * m_channels);
            Value *ptrs[ncorner];
            for (size_t c = 0; c < ncorner; ++c)
                ptrs[c] = buf.data() + c * m_channels;
            detail::tex_fetch(scalar_ops<Value>(active), pos.data(), ptrs);
            for (size_t c = 0; c < ncorner; ++c)
                for (size_t ch = 0; ch < m_channels; ++ch)
                    out.entry(c).set_entry(ch, ptrs[c][ch]);
        }
    }

    /**
     * \brief Fetch the texels that would be referenced in a texture lookup with
     * linear interpolation without actually performing this interpolation.
     */
    template <typename Output>
    Array<Output, (1 << Dimension)>
    eval_fetch(const position_for<Output> &pos,
               mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        constexpr size_t ncorner = 1 << Dimension;
        Array<Output, ncorner> out = alloc_fetch_output<Output>();

        if constexpr (is_jit_v<Storage_>) {
            uint64_t *o = (uint64_t *) alloca(sizeof(uint64_t) * ncorner *
                                              m_channels);
            ad_tex_fetch(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) m_channels,
                (int) m_wrap_mode, (int) m_srgb, m_handle, (int) m_use_accel,
                value_index(), resolution_indices().data(),
                idiv_indices().data(), pos_indices(pos).data(), active.index(), o);

            for (size_t c = 0; c < ncorner; ++c)
                for (size_t ch = 0; ch < m_channels; ++ch)
                    out.entry(c).set_entry(
                        ch, steal_value<Value>(o[c * m_channels + ch]));
        } else {
            eval_fetch_nonaccel<Output>(pos, out, active);
        }
        return out;
    }

    /**
     * \brief Helper function to evaluate a clamped cubic B-Spline interpolant
     *
     * This is an implementation detail and should only be called by the \ref
     * eval_cubic() function to construct an AD graph. When only the cubic
     * evaluation result is desired, the \ref eval_cubic() function is faster
     * than this simple implementation
     */
    template <typename Output>
    Output eval_cubic_helper(const position_for<Output> &pos,
                             mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        Output out = alloc_output<Output>();

        // This helper also runs on JIT arrays (to build an AD graph), whose
        // storage is channel-padded; only unpadded scalar storage with a
        // statically-sized output can use the compile-time channel count. The
        // ``else`` keeps the ``alloca`` out of the static path entirely (so it
        // stays inlinable), rather than relying on dead-code elimination.
        if constexpr (!is_jit_v<Storage_> && !is_dynamic_v<Output>) {
            constexpr uint32_t C = (uint32_t) size_v<Output>;
            assert(m_channels == C);
            Value res[C], scratch[C];
            detail::tex_eval_cubic(scalar_ops<Value, C>(active), pos.data(),
                                   res, scratch);
            for (uint32_t ch = 0; ch < C; ++ch)
                out.set_entry(ch, res[ch]);
        } else {
            Value *res_mem     = (Value *) alloca(sizeof(Value) * m_channels);
            Value *scratch_mem = (Value *) alloca(sizeof(Value) * m_channels);
            detail::tex_scratch<Value> res(res_mem, m_channels),
                                       scratch(scratch_mem, m_channels);

            detail::tex_eval_cubic(scalar_ops<Value>(active), pos.data(),
                                   res.data(), scratch.data());
            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, res[ch]);
        }
        return out;
    }

    /**
     * \brief Evaluate a clamped cubic B-Spline interpolant represented by this
     * texture
     *
     * Instead of interpolating the texture via B-Spline basis functions, the
     * implementation transforms this calculation into an equivalent weighted
     * sum of several linear interpolant evaluations. On the CUDA and Metal
     * backends, these steps can then be accelerated by hardware texture units,
     * which runs faster than a naive implementation. More information can be
     * found in
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
    template <typename Output>
    Output eval_cubic(const position_for<Output> &pos,
                      mask_for<Output> active = true,
                      bool force_nonaccel  = false) const {
        using Value = value_t<Output>;
        if constexpr (is_jit_v<Storage_>) {
            bool use_accel = m_use_accel && !force_nonaccel;
            Output out = alloc_output<Output>();
            uint64_t *o = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);
            ad_tex_cubic(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) m_channels,
                (int) m_wrap_mode, (int) m_srgb, m_handle, (int) use_accel,
                value_index(), resolution_indices().data(),
                idiv_indices().data(), pos_indices(pos).data(), active.index(), o);

            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, steal_value<Value>(o[ch]));
            return out;
        } else {
            // Direct B-spline evaluation (faster than the linear-lookup
            // transform without hardware bilinear units).
            DRJIT_MARK_USED(force_nonaccel);
            return eval_cubic_helper<Output>(pos, active);
        }
    }

    /// Per-channel value and positional gradient returned by \ref eval_cubic_grad()
    template <typename Output> struct CubicGrad {
        using Value = value_t<Output>;
        Output value;
        replace_value_t<Output, Array<Value, Dimension>> gradient;
    };

    /**
     * \brief Evaluate the positional gradient of a cubic B-Spline
     *
     * This implementation computes the result directly from explicit
     * differentiated basis functions. It has no autodiff support.
     *
     * The resulting gradient has been multiplied by the spatial extents to
     * count for the transformation from the unit size volume to the size of
     * its shape.
     */
    template <typename Output>
    CubicGrad<Output> eval_cubic_grad(const position_for<Output> &pos,
                                      mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        using Gradient = replace_value_t<Output, Array<Value, Dimension>>;
        using Hessian = replace_value_t<Output, Matrix<Value, Dimension>>;
        Output out_value = alloc_output<Output>();
        Gradient out_gradient = alloc_output<Gradient>();
        eval_cubic_deriv(pos, active, out_value, out_gradient, (Hessian *) nullptr);
        return { out_value, out_gradient };
    }

    /// Per-channel value, gradient, and hessian returned by \ref eval_cubic_hessian()
    template <typename Output> struct CubicHessian {
        using Value = value_t<Output>;
        Output value;
        replace_value_t<Output, Array<Value, Dimension>> gradient;
        replace_value_t<Output, Matrix<Value, Dimension>> hessian;
    };

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
    template <typename Output>
    CubicHessian<Output> eval_cubic_hessian(const position_for<Output> &pos,
                                            mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        using Gradient = replace_value_t<Output, Array<Value, Dimension>>;
        using Hessian = replace_value_t<Output, Matrix<Value, Dimension>>;
        Output out_value = alloc_output<Output>();
        Gradient out_gradient = alloc_output<Gradient>();
        Hessian out_hessian = alloc_output<Hessian>();
        eval_cubic_deriv(pos, active, out_value, out_gradient, &out_hessian);
        return { out_value, out_gradient, out_hessian };
    }

    /// Apply the configured texture wrapping mode to an integer position
    template <typename T> T wrap(const T &pos) const {
        using Int = value_t<T>;
        static_assert(size_v<T> == Dimension &&
                          std::is_integral_v<scalar_t<T>> &&
                          std::is_signed_v<scalar_t<T>>,
                      "Texture::wrap(): expected a signed integer position with "
                      "one component per texture dimension.");

        if constexpr (is_jit_v<Storage_>) {
            std::array<uint32_t, Dimension> pos_idx;
            for (size_t k = 0; k < Dimension; ++k)
                pos_idx[k] = pos[k].index();

            std::array<uint32_t, Dimension> out_idx;
            ad_tex_wrap((uint32_t) Dimension, (int) m_wrap_mode,
                        resolution_indices().data(), idiv_indices().data(),
                        pos_idx.data(), out_idx.data());

            T result;
            for (size_t k = 0; k < Dimension; ++k)
                result[k] = Int::steal(out_idx[k]);
            return result;
        } else {
            auto ops = scalar_ops<float32_array_t<Int>>(true);
            T result;
            for (uint32_t k = 0; k < Dimension; ++k)
                result[k] = detail::tex_wrap(ops, pos[k], k);
            return result;
        }
    }

    /// Cast a raw stored texel to the query precision. For 8-bit storage this
    /// normalizes 0..255 to [0, 1] and optionally decodes sRGB (matching the
    /// hardware read; the sRGB formats leave each RGBA group's 4th channel
    /// linear). A plain cast for floating-point storage.
    template <typename Value>
    Value convert_texel(const Storage_ &s, uint32_t ch) const {
        Value v = Value(s);
        if constexpr (IsUInt8) {
            v *= Value(1.f / 255.f);
            if (m_srgb && (ch % 4) != 3)
                v = srgb_to_linear(v);
        }
        return v;
    }

    /// Gather the channels at \c idx from \c src and cast them to the query
    /// precision. ``CChannels`` is the channel count when statically known.
    template <uint32_t CChannels = 0, typename Value>
    void gather_texel(const Storage &src, const uint32_array_t<Value> &idx,
                      const mask_t<Value> &active, Value *out) const {
        if constexpr (CChannels != 0) {
            // Scalar storage is unpadded, so m_channels_storage == CChannels.
            Storage_ packet[CChannels];
            gather_packet_dynamic(CChannels, src, idx, packet, active);
            for (uint32_t ch = 0; ch < CChannels; ++ch)
                out[ch] = convert_texel<Value>(packet[ch], ch);
        } else {
            // Per-channel packet scratch on the stack
            Storage_ *packet_mem = (Storage_ *) alloca(sizeof(Storage_) * m_channels_storage);
            detail::tex_scratch<Storage_> packet(packet_mem, m_channels_storage);
            gather_packet_dynamic(m_channels_storage, src, idx,
                                  packet.data(), active);
            for (uint32_t ch = 0; ch < m_channels; ++ch)
                out[ch] = convert_texel<Value>(packet[ch], ch);
        }
    }

    /// Overload of the above that reads the base texture storage
    template <uint32_t CChannels = 0, typename Value>
    void gather_texel(const uint32_array_t<Value> &idx,
                      const mask_t<Value> &active, Value *out) const {
        gather_texel<CChannels>(m_padded_tensor.array(), idx, active, out);
    }

    /// Convert a query-precision value to stored form (the inverse of \ref
    /// convert_texel): 8-bit storage is sRGB-encoded and quantized to 0..255,
    /// floating-point storage is a plain cast.
    template <typename Value>
    Storage_ store_texel(const Value &v, uint32_t ch) const {
        if constexpr (IsUInt8) {
            Value w = clip(v, Value(0), Value(1));
            if (m_srgb && (ch % 4) != 3)
                w = linear_to_srgb(w);
            return Storage_(fmadd(w, Value(255), Value(0.5)));
        } else {
            return Storage_(v);
        }
    }

    /// Software fallback for \ref write(): scatter the per-channel values into
    /// the channel-padded backing storage. Used whenever the texture has no
    /// hardware representation (the LLVM backend, or double precision).
    template <typename Value>
    void write_nonaccel(const Array<uint32_array_t<Value>, Dimension> &pos,
                        const Value *value, mask_t<Value> active) {
        using UInt = uint32_array_t<Value>;

        // Row-major flat texel index into the [shape..., channels] storage
        UInt pixel = pos[Dimension - 1];
        for (size_t i = 1; i < Dimension; ++i)
            pixel = fmadd(pixel, (uint32_t) m_shape[i], pos[Dimension - 1 - i]);
        UInt base = pixel * (uint32_t) m_channels_storage;

        for (uint32_t ch = 0; ch < m_channels; ++ch)
            scatter(m_padded_tensor.array(), store_texel(value[ch], ch), base + ch, active);
    }

protected:
    void init(const size_t *shape, size_t channels, bool use_accel,
              FilterMode filter_mode, WrapMode wrap_mode,
              bool init_tensor = true, bool writable = false,
              void *external = nullptr,
              MipFilter mip_filter = MipFilter::Disabled, size_t max_aniso = 8,
              MipBasis mip_basis = MipBasis::Standard) {
        if (channels == 0)
            jit_raise("Texture::Texture(): must have at least 1 channel!");

        if (m_srgb && !IsUInt8)
            jit_raise("Texture(): the 'srgb' flag is only supported for 8-bit "
                      "(UInt8) textures.");

        if (mip_filter != MipFilter::Disabled) {
            if (writable)
                jit_raise("Texture(): MIP-mapped textures cannot be writable!");
            if (max_aniso == 0 || max_aniso > 16)
                jit_raise("Texture(): 'max_aniso' must be between 1 and 16 "
                          "(the hardware limit), got %zu!", max_aniso);
        }

        if (mip_basis == MipBasis::Laplacian) {
            if (!is_jit_v<Storage_>)
                jit_raise("Texture(): the Laplacian basis requires "
                          "a JIT backend (CUDA, Metal, or LLVM).");
            if (IsUInt8)
                jit_raise("Texture(): the Laplacian basis requires "
                          "floating-point storage.");
            if (mip_filter == MipFilter::Disabled)
                jit_raise("Texture(): the Laplacian basis requires "
                          "a MIP-mapped texture (mip_filter must not be "
                          "MipFilter::Disabled).");
        }

        m_writable = writable;
        m_channels = channels;
        m_mip_filter = mip_filter;
        m_max_aniso = (uint32_t) max_aniso;
        m_mip_basis = mip_basis;

        // Determine padding used for channels depending on backend
        if constexpr (is_jit_v<Storage_>) {
            m_channels_storage = 1;
            while (m_channels_storage < m_channels)
                m_channels_storage <<= 1;
        } else {
            m_channels_storage = channels;
        }

        m_size = m_channels_storage;
        size_t unpadded_size = m_channels;
        size_t tensor_shape[Dimension + 1]{};
        for (size_t i = 0; i < Dimension; ++i) {
            tensor_shape[i] = shape[i];
            m_shape[i] = shape[i];
            m_resolution_opaque[Dimension - 1 - i] = opaque<UInt32>((uint32_t) shape[i]);
            m_inv_resolution[Dimension - 1 - i] = Divisor((int32_t) shape[i]);
            m_size *= shape[i];
            unpadded_size *= shape[i];
        }

        // Only make the divisor opaque when it is actually used
        if constexpr (is_jit_v<Storage_>) {
            if (wrap_mode != WrapMode::Clamp)
                for (size_t i = 0; i < Dimension; ++i)
                    make_opaque(m_inv_resolution[i]);
        }
        tensor_shape[Dimension] = m_channels_storage;
        m_shape[Dimension] = channels;

        m_use_accel = use_accel;
        m_filter_mode = filter_mode;
        m_wrap_mode = wrap_mode;

        init_mip_table();

        if (init_tensor) {
            if constexpr (is_jit_v<Storage_>) {
                m_padded_tensor =
                    TensorXf(empty<Storage>(m_size), Dimension + 1, tensor_shape);
                m_tensor =
                    TensorXf(empty<Storage>(unpadded_size), Dimension + 1, m_shape);

                // Zero-initialized coefficient tensors, one per pyramid level
                m_levels.clear();
                if (m_mip_basis == MipBasis::Laplacian) {
                    for (uint32_t l = 0; l < m_level_count; ++l) {
                        size_t level_shape[Dimension + 1], n = m_channels;
                        for (size_t i = 0; i < Dimension; ++i) {
                            size_t r = m_shape[i] >> l;
                            level_shape[i] = r > 0 ? r : 1;
                            n *= level_shape[i];
                        }
                        level_shape[Dimension] = m_channels;
                        m_levels.push_back(TensorXf(zeros<Storage>(n),
                                                    Dimension + 1, level_shape));
                    }
                }
            } else {
                // Don't allocate memory in scalar modes
                m_padded_tensor =
                    TensorXf(Storage::map_(nullptr, m_size), Dimension + 1, tensor_shape);
                m_tensor =
                    TensorXf(Storage::map_(nullptr, unpadded_size), Dimension + 1, m_shape);
            }
        }

        if constexpr (HasGPUTexture) {
            if (m_use_accel && init_tensor) {
                if (m_handle)
                    jit_tex_destroy(m_handle);

                bool external_wrap = external != nullptr;
                if (external_wrap) {
                    // Wrap an externally-owned native texture (\ref from_native_handle).
                    m_handle = external;
                } else {
                    size_t tex_shape[Dimension];
                    reverse_tensor_shape(tex_shape, false);
                    m_handle = jit_tex_create(
                        Backend, Dimension, tex_shape, m_channels_storage,
                        (int) type_v<scalar_t<Storage_>>, (int) filter_mode,
                        (int) wrap_mode, (int) m_writable, (int) m_srgb,
                        m_level_count, m_mip_filter == MipFilter::Nearest ? 0 : 1,
                        m_max_aniso);
                }
                m_hw_mutable = (m_writable || external_wrap) &&
                               !(IsCUDA && external_wrap && m_writable);
                if (m_hw_mutable)
                    install_views();
            }
        }
    }

private:
    /// Steal all members from \c other (shared by the move constructor and the
    /// move-assignment operator)
    void move_from(Texture &&other) noexcept {
        m_handle = other.m_handle;
        other.m_handle = nullptr;
        m_size = other.m_size;
        m_channels = other.m_channels;
        m_channels_storage = other.m_channels_storage;
        for (size_t i = 0; i < Dimension + 1; ++i)
            m_shape[i] = std::move(other.m_shape[i]);
        m_padded_tensor = std::move(other.m_padded_tensor);
        m_tensor = std::move(other.m_tensor);
        m_resolution_opaque = std::move(other.m_resolution_opaque);
        for (size_t i = 0; i < Dimension; ++i)
            m_inv_resolution[i] = std::move(other.m_inv_resolution[i]);
        m_filter_mode = other.m_filter_mode;
        m_wrap_mode = other.m_wrap_mode;
        m_use_accel = other.m_use_accel;
        m_writable = other.m_writable;
        m_srgb = other.m_srgb;
        m_migrated = other.m_migrated;
        m_hw_mutable = other.m_hw_mutable;
        m_tensor_dirty = other.m_tensor_dirty;
        m_mip_filter = other.m_mip_filter;
        m_max_aniso = other.m_max_aniso;
        m_level_count = other.m_level_count;
        m_mip_texels = other.m_mip_texels;
        m_mip = std::move(other.m_mip);
        m_mip_table = std::move(other.m_mip_table);
        m_mip_basis = other.m_mip_basis;
        m_levels = std::move(other.m_levels);
    }

    /// Rebind the tensor members to fresh unevaluated readback expressions,
    /// carrying over their AD identity (see \ref readback_view()).
    void install_views() const {
        if constexpr (HasGPUTexture) {
            Storage view = readback_view(m_channels_storage);
            if constexpr (IsDiff)
                view = replace_grad(view, m_padded_tensor.array());
            m_padded_tensor.array() = std::move(view);

            if (m_channels_storage != m_channels) {
                Storage uview = readback_view(m_channels);
                if constexpr (IsDiff)
                    uview = replace_grad(uview, m_tensor.array());
                m_tensor.array() = std::move(uview);
            } else {
                m_tensor.array() = m_padded_tensor.array();
            }
        }
    }

    /// Recompute the public tensor from the padded storage, re-attaching
    /// the AD identity of \ref m_tensor (see \ref set_value())
    void update_tensor() const {
        if (m_channels == m_channels_storage) {
            m_tensor.array() = m_padded_tensor.array();
        } else {
            Storage u = steal_storage(ad_tex_repack(
                value_index(), (uint32_t) (m_size / m_channels_storage),
                (uint32_t) m_channels, (uint32_t) m_channels_storage));
            if constexpr (IsDiff)
                u = replace_grad(u, m_tensor.array());
            m_tensor.array() = std::move(u);
        }
    }

    /// Bring the tensor members up to date before handing them out
    void refresh() const {
        if constexpr (HasGPUTexture) {
            if (m_hw_mutable) {
                // The hardware texture contents can change behind our back. A
                // readback view that was already materialized pins the old
                // contents, so swap in a fresh one. Unevaluated views need no
                // refresh: they read the current contents once evaluated.
                auto materialized = [](uint32_t index) {
                    VarState s = jit_var_state(index);
                    return s == VarState::Evaluated || s == VarState::Dirty;
                };

                if (materialized((uint32_t) m_padded_tensor.array().index()) ||
                    materialized((uint32_t) m_tensor.array().index()))
                    install_views();
                return;
            }
        }

        // Deferred unpadded update after a write() into the storage buffer
        if (m_tensor_dirty) {
            m_tensor_dirty = false;
            update_tensor();
        }
    }

    /// Build an unevaluated expression that reads the texel data back from
    /// GPU texture memory. Each texel is sampled at its center, which
    /// reproduces the stored value exactly. The result interleaves
    /// ``channels_out`` channels per texel in the row-major tensor layout
    /// (``m_channels_storage`` yields the padded storage layout,
    /// ``m_channels`` the public one).
    Storage readback_view(size_t channels_out) const {
        if constexpr (HasGPUTexture) {
            using Plain = detached_t<Storage>;
            return Storage(Plain::steal(ad_tex_readback(
                type_v<scalar_t<Storage_>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) channels_out,
                (int) m_srgb, m_handle, resolution_indices().data(),
                idiv_indices().data(), m_size / m_channels_storage)));
        } else {
            (void) channels_out;
            return Storage();
        }
    }

    /// Texel count of MIP level ``level``
    size_t level_texels(uint32_t level) const {
        size_t texels = 1;
        for (size_t i = 0; i < Dimension; ++i) {
            size_t r = m_shape[i] >> level;
            texels *= r > 0 ? r : 1;
        }
        return texels;
    }

    /// Compute the MIP pyramid depth for the current shape and upload the
    /// per-level constant table. Invoked by \ref init().
    void init_mip_table() {
        m_level_count = 1;
        m_mip_texels = 0;

        if (m_mip_filter == MipFilter::Disabled)
            return;

        std::unique_ptr<int32_t[]> table;
        m_level_count = detail::tex_mip_table(table, m_mip_texels, m_shape,
                                              (uint32_t) Dimension, MipStride);
        if (m_level_count > 1)
            m_mip_table = load<Int32Buffer>(table.get(),
                                            (size_t) m_level_count * MipStride);
    }

    /// Upload the base texels and the MIP pyramid into the hardware texture
    void upload_levels(const Storage &padded_value) {
        jit_tex_memcpy_d2t(padded_value.data(), m_handle);

        if (m_level_count > 1) {
            const uint8_t *ptr = (const uint8_t *) m_mip.data();
            size_t stride = m_channels_storage * sizeof(scalar_t<Storage_>);
            for (uint32_t l = 1; l < m_level_count; ++l) {
                jit_tex_memcpy_d2t(ptr, m_handle, l);
                ptr += level_texels(l) * stride;
            }
        }
    }

    /// Regenerate the MIP pyramid
    void build_mipmap(const Storage &base) {
        if (m_level_count <= 1)
            return;

        // Resolutions with the width along ``x`` (fastest axis)
        size_t res[Dimension];
        for (size_t i = 0; i < Dimension; ++i)
            res[Dimension - 1 - i] = m_shape[i];

        if constexpr (is_jit_v<Storage_>) {
            m_mip = steal_storage(ad_tex_mipmap_from_base(
                (uint32_t) Dimension, (uint32_t) m_channels_storage,
                (int) m_srgb, combined_index(base), res, m_level_count));
        } else {
            using T = scalar_t<Storage_>;
            m_mip = empty<Storage>((size_t) m_mip_texels * m_channels_storage);
            detail::tex_mipmap_from_base((const T *) base.data(), (T *) m_mip.data(),
                                         res, (uint32_t) Dimension,
                                         (uint32_t) m_channels_storage,
                                         m_level_count, m_srgb);
        }
    }

    /// Validate a level-indexed accessor call (Laplacian mode only)
    void check_level_access(const char *name, size_t level) const {
        if (m_mip_basis != MipBasis::Laplacian)
            jit_raise("Texture::%s(): level-indexed access requires the "
                      "Laplacian basis.", name);
        if (level >= m_level_count)
            jit_raise("Texture::%s(): level %zu is out of bounds (the "
                      "pyramid has %u levels).", name, level, m_level_count);
    }

    /// Laplacian mode: initialize the coefficient tensors from a physical
    /// image
    void decompose(const Storage &value) {
        DRJIT_MARK_USED(value);
        if constexpr (is_jit_v<Storage_>) {
            size_t res[Dimension];
            for (size_t i = 0; i < Dimension; ++i)
                res[Dimension - 1 - i] = m_shape[i];

            uint64_t *out =
                (uint64_t *) alloca(sizeof(uint64_t) * m_level_count);
            ad_tex_laplacian_from_base((uint32_t) Dimension,
                                       (uint32_t) m_channels,
                                       combined_index(value), m_level_count,
                                       res, out);
            for (uint32_t l = 0; l < m_level_count; ++l)
                m_levels[l].array() = steal_storage(out[l]);
        }
    }

    /// Laplacian mode: run the differentiable synthesis that reconstructs the
    /// sampled pyramid from the coefficient tensors, refresh the tensor
    /// members, and upload the result to the hardware texture.
    void rebuild() {
        if constexpr (is_jit_v<Storage_>) {
            size_t res[Dimension];
            for (size_t i = 0; i < Dimension; ++i)
                res[Dimension - 1 - i] = m_shape[i];

            uint64_t *coef =
                (uint64_t *) alloca(sizeof(uint64_t) * m_level_count);
            for (uint32_t l = 0; l < m_level_count; ++l)
                coef[l] = combined_index(m_levels[l].array());

            uint64_t out_base = 0, out_mip = 0;
            ad_tex_mipmap_from_laplacian(
                (uint32_t) Dimension, (uint32_t) m_channels,
                (uint32_t) m_channels_storage, coef, m_level_count, res,
                &out_base, &out_mip);

            m_padded_tensor.array() = steal_storage(out_base);
            if (m_level_count > 1)
                m_mip = steal_storage(out_mip);

            if constexpr (HasGPUTexture) {
                if (m_use_accel)
                    upload_levels(m_padded_tensor.array());
            }

            // The public tensor is attached through the (differentiable)
            // repacking of the synthesized base level rather than a stashed
            // AD identity as in update_tensor()
            if (m_channels == m_channels_storage)
                m_tensor.array() = m_padded_tensor.array();
            else
                m_tensor.array() = steal_storage(ad_tex_repack(
                    value_index(), (uint32_t) (m_size / m_channels_storage),
                    (uint32_t) m_channels, (uint32_t) m_channels_storage));
        }
    }

    /// Helper function to reverse the tensor (\ref Texture.m_padded_tensor) shape
    void reverse_tensor_shape(size_t *output, bool include_channels) const {
        for (size_t i = 0; i < Dimension; ++i)
            output[i] = m_padded_tensor.shape(Dimension - 1 - i);
        if (include_channels)
            output[Dimension] = m_padded_tensor.shape(Dimension);
    }

    /// Operations object for generating scalar texture evaluation code.
    /// ``CChannels`` is the channel count when statically known (so the loops
    /// unroll), or 0 for a runtime count (see \ref detail::ChannelCount).
    template <typename Value, uint32_t CChannels = 0>
    struct ScalarOps : detail::ChannelCount<CChannels> {
        using Float = Value;
        using Int   = int32_array_t<Value>;
        using UInt  = uint32_array_t<Value>;
        using Mask  = mask_t<Value>;

        // The texture dimension is always a compile-time constant
        static constexpr uint32_t dim = (uint32_t) Dimension;

        const Texture *tex;
        Mask active;
        FilterMode filter_mode;
        WrapMode wrap_mode;

        // Pyramid level binding (see the ``Ops`` contract in texture_impl.h)
        detail::TexLevel<Int, UInt> lvl;

        Float lit(double v) const { return Value(v); }
        Int lit_i(int32_t v) const { return Int(v); }
        Float res_f(uint32_t k) const { return Value(res_i(k)); }
        Int res_i(uint32_t k) const {
            Int r = Int(tex->m_resolution_opaque[k]);
            if (lvl.bound)
                r = maximum(r >> lvl.level, Int(1));
            return r;
        }
        Float to_float(const Int &i) const { return Value(i); }
        Int idiv(const Int &a, uint32_t k) const {
            if (lvl.bound)
                return detail::tex_idiv_dynamic(*this, lvl.div[k][0],
                                                lvl.div[k][1], a);
            return tex->m_inv_resolution[k](a);
        }
        void gather(const UInt &idx, Float *out) const {
            if (!lvl.bound) {
                tex->template gather_texel<CChannels>(idx, active, out);
                return;
            }
            UInt mip_idx = idx + lvl.offset;
            if (lvl.includes_base) {
                // The bound level may be the base level, whose texels live in
                // the regular texture storage rather than the pyramid buffer
                Mask is_base = lvl.level == 0;
                tex->template gather_texel<CChannels>(idx, active && is_base, out);

                uint32_t n = this->channels_out;
                Float *tmp_mem = (Float *) alloca(sizeof(Float) * n);
                detail::tex_scratch<Float> tmp(tmp_mem, n);
                tex->template gather_texel<CChannels>(
                    tex->m_mip, mip_idx, active && !is_base, tmp.data());
                for (uint32_t ch = 0; ch < n; ++ch)
                    out[ch] = select(is_base, out[ch], tmp[ch]);
            } else {
                tex->template gather_texel<CChannels>(tex->m_mip, mip_idx,
                                                      active, out);
            }
        }

        /// Load the constant record of MIP level ``l``
        void mip_record(const Int &l, Int *rec) const {
            auto r = drjit::gather<Array<Int, MipStride>>(tex->m_mip_table, UInt(l));
            for (uint32_t j = 0; j < MipStride; ++j)
                rec[j] = r[j];
        }

        template <typename Body>
        void sum_loop(const Int &n, Float *state, uint32_t /* n_state */,
                      uint32_t n_scratch, Body body) const {
            detail::tex_sum_loop(*this, n, state, n_scratch, body);
        }
    };

    /// Build a \ref ScalarOps bound to this texture and the query mask
    template <typename Value, uint32_t CChannels = 0>
    ScalarOps<Value, CChannels> scalar_ops(mask_t<Value> active) const {
        if constexpr (!is_array_v<mask_t<Value>>)
            active = true;
        ScalarOps<Value, CChannels> ops;
        if constexpr (CChannels == 0)
            ops.channels_out = (uint32_t) m_channels;
        ops.tex = this;
        ops.active = active;
        ops.filter_mode = m_filter_mode;
        ops.wrap_mode = m_wrap_mode;
        return ops;
    }

    // -- Type-erased marshalling helpers backing the JIT ``ad_tex_*`` calls --

    /// Combined AD/JIT index of a storage array
    static uint64_t combined_index(const Storage &v) {
        if constexpr (is_diff_v<Storage_>)
            return v.index_combined();
        else
            return (uint64_t) v.index();
    }

    /// Adopt an owned combined index returned by an ``ad_tex_*`` call as Storage
    static Storage steal_storage(uint64_t index) {
        return Storage::steal((typename Storage::Index) index);
    }

    /// Combined AD/JIT index of the padded texture storage tensor
    uint64_t value_index() const { return combined_index(m_padded_tensor.array()); }

    /// JIT indices of the per-dimension opaque resolution variables
    std::array<uint32_t, Dimension> resolution_indices() const {
        std::array<uint32_t, Dimension> r;
        for (size_t k = 0; k < Dimension; ++k)
            r[k] = m_resolution_opaque[k].index();
        return r;
    }

    /// JIT indices of the opaque magic-division constants (multiplier, shift per
    /// dimension) backing the Repeat/Mirror wrap math (0 for Clamp)
    std::array<uint32_t, 2 * Dimension> idiv_indices() const {
        std::array<uint32_t, 2 * Dimension> r;
        for (size_t k = 0; k < Dimension; ++k) {
            r[2 * k + 0] = m_inv_resolution[k].multiplier.index();
            r[2 * k + 1] = m_inv_resolution[k].shift.index();
        }
        return r;
    }

    /// Combined AD/JIT indices of a query position
    template <typename Value>
    static std::array<uint64_t, Dimension>
    pos_indices(const Array<Value, Dimension> &pos) {
        std::array<uint64_t, Dimension> r;
        for (size_t k = 0; k < Dimension; ++k) {
            if constexpr (is_diff_v<Value>)
                r[k] = pos[k].index_combined();
            else
                r[k] = (uint64_t) pos[k].index();
        }
        return r;
    }

    /// Adopt an owned combined index returned by an ``ad_tex_*`` call
    template <typename Value>
    static Value steal_value(uint64_t index) {
        if constexpr (is_diff_v<Value>)
            return Value::steal(index);
        else
            return Value::steal((uint32_t) index);
    }

    /// Evaluate the texture interpolant symbolically
    template <typename Output, typename Value = value_t<Output>>
    void eval_jit(const position_for<Output> &pos, Output &out,
                  mask_for<Output> active, bool use_accel) const {
        uint64_t *out_idx = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);
        ad_tex_eval(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                    (uint32_t) m_channels_storage, (uint32_t) m_channels,
                    (int) m_filter_mode, (int) m_wrap_mode, (int) m_srgb,
                    m_handle, (int) use_accel, value_index(),
                    resolution_indices().data(), idiv_indices().data(),
                    pos_indices(pos).data(), active.index(), out_idx);
        for (size_t ch = 0; ch < m_channels; ++ch)
            out.set_entry(ch, steal_value<Value>(out_idx[ch]));
    }

    /// Scalar fallback for \ref eval_nonaccel().
    template <typename Output, typename Value = value_t<Output>>
    void eval_nonaccel_scalar(const position_for<Output> &pos, Output &out,
                              mask_for<Output> active = true) const {
        if constexpr (!is_jit_v<Storage_> && !is_dynamic_v<Output>) {
            constexpr uint32_t C = (uint32_t) size_v<Output>;
            assert(m_channels == C);
            Value res[C], scratch[C];
            detail::tex_eval(scalar_ops<Value, C>(active), pos.data(), res, scratch);
            for (uint32_t ch = 0; ch < C; ++ch)
                out.set_entry(ch, res[ch]);
        } else {
            Value *res_mem     = (Value *) alloca(sizeof(Value) * m_channels);
            Value *scratch_mem = (Value *) alloca(sizeof(Value) * m_channels);
            detail::tex_scratch<Value> res(res_mem, m_channels),
                                       scratch(scratch_mem, m_channels);
            detail::tex_eval(scalar_ops<Value>(active), pos.data(), res.data(),
                             scratch.data());
            for (size_t ch = 0; ch < m_channels; ++ch)
                out.set_entry(ch, res[ch]);
        }
    }


    /// Shared marshaller for \ref eval_cubic_grad() / \ref eval_cubic_hessian():
    /// fills value + gradient (and hessian, when \c out_hessian is non-null).
    template <typename Output, typename Gradient, typename Hessian>
    void eval_cubic_deriv(const position_for<Output> &pos, mask_for<Output> active,
                          Output &out_value, Gradient &out_gradient,
                          Hessian *out_hessian) const {
        using Value = value_t<Output>;
        bool want_hess = out_hessian != nullptr;
        size_t n_grad = m_channels * Dimension,
               n_hess = want_hess ? n_grad * Dimension : 0;

        if constexpr (is_jit_v<Storage_>) {
            uint64_t *vp = (uint64_t *) alloca(sizeof(uint64_t) * m_channels),
                     *gp = (uint64_t *) alloca(sizeof(uint64_t) * n_grad),
                     *hp = want_hess ? (uint64_t *) alloca(sizeof(uint64_t) * n_hess)
                                     : nullptr;
            ad_tex_cubic_deriv(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                               (uint32_t) m_channels_storage, (uint32_t) m_channels,
                               (int) m_wrap_mode, (int) m_srgb, value_index(),
                               resolution_indices().data(), idiv_indices().data(),
                               pos_indices(pos).data(), active.index(), vp, gp, hp);
            // The kernel's flat outputs match this iteration order; walk linearly
            for (size_t ch = 0; ch < m_channels; ++ch) {
                out_value.set_entry(ch, steal_value<Value>(*vp++));
                for (size_t m = 0; m < Dimension; ++m) {
                    out_gradient.entry(ch).set_entry(m, steal_value<Value>(*gp++));
                    if (want_hess)
                        for (size_t n = 0; n < Dimension; ++n)
                            out_hessian->entry(ch).entry(m).set_entry(
                                n, steal_value<Value>(*hp++));
                }
            }
        } else {
            Value *vmem       = (Value *) alloca(sizeof(Value) * m_channels),
                  *gmem       = (Value *) alloca(sizeof(Value) * n_grad),
                  *hmem       = want_hess ? (Value *) alloca(sizeof(Value) * n_hess) : nullptr,
                  *scratch_mem = (Value *) alloca(sizeof(Value) * m_channels);
            detail::tex_scratch<Value> vs(vmem, m_channels), gs(gmem, n_grad),
                                       hs(hmem, n_hess), scratch(scratch_mem, m_channels);
            detail::tex_eval_cubic_deriv(scalar_ops<Value>(active), pos.data(),
                                         vs.data(), gs.data(),
                                         want_hess ? hs.data() : nullptr, scratch.data());
            Value *vp = vs.data(), *gp = gs.data(), *hp = hs.data();
            for (size_t ch = 0; ch < m_channels; ++ch) {
                out_value.set_entry(ch, *vp++);
                for (size_t m = 0; m < Dimension; ++m) {
                    out_gradient.entry(ch).set_entry(m, *gp++);
                    if (want_hess)
                        for (size_t n = 0; n < Dimension; ++n)
                            out_hessian->entry(ch).entry(m).set_entry(n, *hp++);
                }
            }
        }
    }

private:
    void *m_handle = nullptr;
    size_t m_size = 0;                       ///< Total size of array
    size_t m_channels = 0;                   ///< Number of channels

    /// Rounded-up number of channels (depends on the backend)
    size_t m_channels_storage = 0;

    /// Unpadded shape of texture
    size_t m_shape[Dimension + 1] = {};

    /* Texel storage model: the two tensor members below always hold valid
       (possibly unevaluated) data, in one of three regimes.

       1. Buffer-backed (LLVM, use_accel=false, or migrate=false):
          m_padded_tensor stores the texels and is authoritative; the hardware
          texture, when present, is a sampling copy of it. The public
          tensor may lag a write() into the buffer (m_tensor_dirty).

       2. Migrated (set_value() with migrate=true): the hardware texture is
          authoritative and no buffer is retained. Both members hold
          symbolic readback expressions (see readback_view()) that fetch the
          texels once evaluated. The contents only change through another
          set_value(), which reinstalls the views.

       3. Hardware-mutable (m_hw_mutable: writable or wrapped textures):
          like (2), but the hardware contents can also change behind our
          back. refresh() swaps out views that were already materialized and
          are therefore pinned to old contents.

       In regimes (2) and (3), a tensor returned by tensor() reflects the
       texture contents at the time it is evaluated. */

    /// Storage tensor with the channel count padded to a power of two. This
    /// backs the sampling code and the AD graph. In scalar modes the padding
    /// is the identity, and this member doubles as the public tensor.
    mutable TensorXf m_padded_tensor;

    /// Public-facing tensor in the unpadded shape, returned by \ref tensor()
    mutable TensorXf m_tensor;

    // Stored in this order: width, height, depth
    Array<UInt32, Dimension> m_resolution_opaque;

    // Reciprocal resolution for the Repeat/Mirror wrap math.
    Divisor m_inv_resolution[Dimension] { };

    /// Texture interpolation mode
    FilterMode m_filter_mode;

    /// Texture wrapping mode
    WrapMode m_wrap_mode;

    /// Use the hardware texture units?
    bool m_use_accel = false;

    /// Texture was created so kernels may store into it via write()
    bool m_writable = false;

    /// 8-bit textures: decode sRGB -> linear on sampling
    bool m_srgb = false;

    /// Hardware-texture flag: is the data held exclusively on the device?
    bool m_migrated = false;

    /// The hardware texture contents can change behind our back (the texture
    /// is writable or externally managed); the tensor members permanently
    /// hold readback views that \ref refresh() keeps current
    bool m_hw_mutable = false;

    /// \ref m_tensor lags a \ref write() into the storage buffer and is
    /// recomputed by \ref refresh()
    mutable bool m_tensor_dirty = false;

    /// MIP level selection mode of filtered lookups
    MipFilter m_mip_filter = MipFilter::Disabled;

    /// Bound on the number of anisotropic taps of \ref eval_filtered()
    uint32_t m_max_aniso = 1;

    /// Number of MIP pyramid levels including the base (1 = no MIP mapping)
    uint32_t m_level_count = 1;

    /// Total texel count of the pyramid levels >= 1
    uint32_t m_mip_texels = 0;

    /// Channel-padded texels of the pyramid levels >= 1, stored back to back
    Storage m_mip;

    /// Per-level constants for the MIP lookup from \ref detail::tex_mip_table()
    Int32Buffer m_mip_table;

    /// Basis in which the texture stores its degrees of freedom
    MipBasis m_mip_basis = MipBasis::Standard;

    /// Laplacian mode: per-level coefficient tensors (finest first, unpadded
    /// channel count). Empty in the standard basis.
    vector<TensorXf> m_levels;

public:
    void
    traverse_1_cb_ro(void *payload,
                     drjit ::detail ::traverse_callback_ro fn) const override {
        // Traverse the function to react to changes when freezing code via
        // @dr.freeze. In all other contexts, the texture is read-only and does
        // not require traversal
        if (!jit_flag(JitFlag::EnableObjectTraversal))
            return;

        DRJIT_MAP(DR_TRAVERSE_MEMBER_RO, m_padded_tensor, m_tensor,
                  m_resolution_opaque, m_inv_resolution, m_mip, m_mip_table,
                  m_levels);
        if constexpr (HasGPUTexture) {
            uint32_t n_indices = tex_n_indices();
            uint32_t *indices = (uint32_t *) alloca(sizeof(uint32_t) * n_indices);
            jit_tex_get_indices(m_handle, indices);
            for (uint32_t i = 0; i < n_indices; i++)
                fn(payload, indices[i], "", "");
        }
    }
    void traverse_1_cb_rw(void *payload,
                          drjit ::detail ::traverse_callback_rw fn) override {
        // Only traverse the scene for frozen functions, since accidentally
        // traversing the scene in loops or vcalls can cause errors with
        // variable size mismatches, and backpropagation of gradients.
        if (!jit_flag(JitFlag::EnableObjectTraversal))
            return;

        DRJIT_MAP(DR_TRAVERSE_MEMBER_RW, m_padded_tensor, m_tensor,
                  m_resolution_opaque, m_inv_resolution, m_mip, m_mip_table,
                  m_levels);

        if constexpr (HasGPUTexture) {
            uint32_t n_indices = tex_n_indices();
            uint32_t *indices = (uint32_t *) alloca(sizeof(uint32_t) * n_indices);
            jit_tex_get_indices(m_handle, indices);
            for (uint32_t i = 0; i < n_indices; i++) {
                uint64_t new_index = fn(payload, indices[i], "", "");
                if (new_index != indices[i])
                    jit_raise("A texture was changed by traversing it. This is "
                              "not supported!");
            }
        }
    }

    // Number of JIT variables backing the texture: the sub-textures, plus the
    // Metal sampler, plus (for writable CUDA textures) surface handles.
    uint32_t tex_n_indices() const {
        uint32_t n_textures = 1 + ((uint32_t(m_channels) - 1) / 4);
        uint32_t extra = IsMetal ? 1u : 0u;
        if (IsCUDA && m_writable)
            extra += n_textures;
        return n_textures + extra;
    }

};

NAMESPACE_END(drjit)
