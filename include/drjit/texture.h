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
#include <drjit/dynamic.h>
#include <drjit/extra.h>
#include <drjit/texture_impl.h>
#include <drjit/idiv.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>
#include <drjit/util.h>
#include <drjit/traversable_base.h>
#include <array>

#pragma once

NAMESPACE_BEGIN(drjit)

template <typename Storage_, size_t Dimension> class Texture : TraversableBase {
public:
    static constexpr bool IsCUDA = is_cuda_v<Storage_>;
    static constexpr bool IsMetal = is_metal_v<Storage_>;
    static constexpr bool IsDiff = is_diff_v<Storage_>;
    static constexpr bool IsDynamic = is_dynamic_v<Storage_>;
    static constexpr bool IsHalf = std::is_same_v<scalar_t<Storage_>, drjit::half>;
    static constexpr bool IsSingle = std::is_same_v<scalar_t<Storage_>, float>;

    // Only half/single-precision floating-point hardware textures are supported
    static constexpr bool HasGPUTexture = (IsHalf || IsSingle) && (IsCUDA || IsMetal);

    using Int32 = int32_array_t<Storage_>;
    using UInt32 = uint32_array_t<Storage_>;
    using Storage = std::conditional_t<IsDynamic, Storage_, DynamicArray<Storage_>>;
    using TensorXf = Tensor<Storage>;

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
     */
    Texture(const size_t shape[Dimension], size_t channels,
            bool use_accel = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp,
            bool writable = false) {
        init(shape, channels, use_accel, filter_mode, wrap_mode,
             /* init_tensor = */ true, writable);
    }

    /**
     * \brief Construct a new texture from a given tensor
     *
     * This constructor allocates texture memory just like the previous
     * constructor, extracting shape information from \c tensor. It then also
     * invokes <tt>set_tensor(tensor)</tt> to fill the texture memory with the
     * provided tensor.
     *
     * When \c migrate is set to \c true on a GPU backend, the texture is
     * *fully* migrated to GPU texture memory to avoid redundant storage. Note
     * that the texture is still differentiable even when migrated. The \ref
     * value() and \ref tensor() operations will perform a reverse migration in
     * this case.
     *
     * Both the \c filter_mode and \c wrap_mode have the same defaults and
     * behaviors as for the previous constructor.
     */
    template <typename TensorT>
    Texture(TensorT &&tensor, bool use_accel = true, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Clamp) {
        if (tensor.ndim() != Dimension + 1)
            jit_raise("Texture::Texture(): tensor dimension must equal "
                        "texture dimension plus one.");
        init(tensor.shape().data(), tensor.shape(Dimension), use_accel,
             filter_mode, wrap_mode);
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
                                      WrapMode wrap_mode = WrapMode::Clamp) {
        return Texture(handle, writable, filter_mode, wrap_mode);
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
            WrapMode wrap_mode) {
        if constexpr (HasGPUTexture) {
            void *h = jit_tex_wrap(Backend, handle, Dimension,
                                   (int) type_v<scalar_t<Storage_>>,
                                   (int) writable, (int) filter_mode,
                                   (int) wrap_mode);

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

    /// Return the texture filtering mode (e.g., nearest, bilinear, etc.)
    FilterMode filter_mode() const { return m_filter_mode; }

    /// Return the boundary handling mode for out-of-bounds lookups
    WrapMode wrap_mode() const { return m_wrap_mode; }

    /// Is the texture data held exclusively in GPU texture memory?
    bool migrated() const { return m_migrated; }

    /// Are hardware texture units used for evaluation?
    bool use_accel() const { return m_use_accel; }

    /// Was this texture created so that kernels may store into it via \ref write()?
    bool writable() const { return m_writable; }

    /// Map an imported texture (\ref from_native_handle()) for use by Dr.Jit
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
     * When \c migrate is set to \c true on the CUDA and Metal backends, the
     * texture information is *fully* migrated to GPU texture memory to avoid
     * redundant storage.
     */
    template <typename StorageT>
    void set_value(StorageT &&value, bool migrate = false) {
        if constexpr (!is_jit_v<Storage_>) {
            if (value.size() != m_size)
                jit_raise("Texture::set_value(): unexpected array size!");
            m_value.array() = std::forward<StorageT>(value);
        } else /* JIT variant */ {
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

            // Stash the AD index of the unpadded `value` and re-attach it in
            // `tensor()` so gradients stay queryable via `tensor().grad`.
            // Recomputing the unpadded values from the padded ones would
            // instead overwrite `m_unpadded_value` (the gradient source) on the
            // next `tensor()` call.
            if constexpr (IsDiff) {
                if (grad_enabled(value))
                    m_unpadded_value.array() =
                        replace_grad(m_unpadded_value.array(), value);
            }

            if constexpr (HasGPUTexture) {
                if (m_use_accel) {
                    jit_tex_memcpy_d2t(padded_value.data(), m_handle);

                    if (migrate) {
                        // Fully migrate to texture memory, set m_value to zero
                        Storage dummy = zeros<Storage>(m_size);

                        if constexpr (IsDiff)
                            m_value.array() = replace_grad(dummy, padded_value);
                        else
                            m_value.array() = dummy;

                        m_migrated = true;
                        m_tensor_dirty = true;

                        return;
                    }
                }
            }

            m_value.array() = padded_value;
            m_tensor_dirty = true;
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
     * texture information is *fully* migrated to GPU texture memory to avoid
     * redundant storage.
     */
    template <typename TensorT>
    void set_tensor(TensorT &&tensor, bool migrate = false) {
        if (tensor.ndim() != Dimension + 1)
            jit_raise("Texture::set_tensor(): tensor dimension must equal "
                      "texture dimension plus one (channels).");

        if ((void *) &tensor == (void *) &m_unpadded_value) {
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
             m_use_accel, m_filter_mode, m_wrap_mode, shape_changed);

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
     * it successfully to the texture, this method must also be called. In short,
     * this method will use the tensor representation to update the texture's
     * internal state.
     *
     * When \c migrate is set to \c true on the CUDA and Metal backends, the
     * texture information is *fully* migrated to GPU texture memory to avoid
     * redundant storage.
     */
    void update_inplace(bool migrate = false) {
        if (m_unpadded_value.ndim() != Dimension + 1)
            jit_raise("Texture::update_inplace(): tensor dimension must equal "
                      "texture dimension plus one (channels).");

        bool shape_changed = false;
        for (size_t i = 0; i < Dimension + 1; ++i) {
            if (m_shape[i] != m_unpadded_value.shape(i)) {
                shape_changed = true;
                break;
            }
        }

        if constexpr (!is_jit_v<Storage_>) {
            if (shape_changed)
                init(m_unpadded_value.shape().data(),
                     m_unpadded_value.shape(Dimension), m_use_accel, m_filter_mode,
                     m_wrap_mode, true);
            else
                // Avoid unnecessary copy when working with `DynamicArray`
                return;
        } else {
            // `Texture::init` might overwrite `m_unpadded_value` with a
            // zero-initialized tensor, so let's copy it first
            TensorXf inbound_tensor(m_unpadded_value);

            init(m_unpadded_value.shape().data(),
                 m_unpadded_value.shape(Dimension), m_use_accel, m_filter_mode,
                 m_wrap_mode, shape_changed);

            m_unpadded_value.array() = inbound_tensor;
        }

        set_value(m_unpadded_value.array(), migrate);
    }

    const Storage &value() const { return tensor().array(); }

    /// Return the texture data as a tensor object
    const TensorXf &tensor() const {
        if constexpr (!is_jit_v<Storage_>) {
            return m_value;
        } else {
            sync_device_data();
            if (m_tensor_dirty) {
                // We need to update the unpadded tensor representation
                // (`m_unpadded_value`). We therefore override any ongoing AD
                // scope, to guarantee that `m_unpadded_value` is always
                // AD-enabled if the original data `m_value` is also AD-enabled.
                resume_grad<Storage> ad_scope_guard;

                if (m_channels != m_channels_storage) {
                    Storage values = steal_storage(ad_tex_repack(
                        value_index(), (uint32_t) (m_size / m_channels_storage),
                        (uint32_t) m_channels, (uint32_t) m_channels_storage));

                    // On the last call to `set_value` we saved the AD index
                    // of the unpadded values. We can re-attach it here.
                    if constexpr (IsDiff)
                        m_unpadded_value.array() =
                            replace_grad(values, m_unpadded_value.array());
                    else
                        m_unpadded_value.array() = values;
                } else {
                    m_unpadded_value.array() = m_value.array();
                }

                m_tensor_dirty = false;
            }

            return m_unpadded_value;
        }
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
            const_cast<const Texture<Storage_, Dimension> *>(this)->tensor());
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
     * This routine returns zero when the texture has been fully migrated to GPU
     * texture memory (see the \c migrate argument of the constructors), as no
     * host-side copy then remains to read from.
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

    /// Scalar fallback for \ref eval_nonaccel(); the JIT path uses the
    /// type-erased kernel in ``libdrjit-extra``
    template <typename Output, typename Value = value_t<Output>>
    void eval_nonaccel_scalar(const position_for<Output> &pos, Output &out,
                              mask_for<Output> active = true) const {
        Value *res_mem     = (Value *) alloca(sizeof(Value) * m_channels);
        Value *scratch_mem = (Value *) alloca(sizeof(Value) * m_channels);
        detail::tex_scratch<Value> res(res_mem, m_channels),
                                   scratch(scratch_mem, m_channels);
        detail::tex_eval(scalar_ops<Value>(active), pos.data(), res.data(),
                         scratch.data());
        for (size_t ch = 0; ch < m_channels; ++ch)
            out.set_entry(ch, res[ch]);
    }

    /**
     * \brief Evaluate the linear interpolant represented by this texture
     *
     * On the JIT backends the evaluation is performed by the type-erased
     * ``ad_tex_eval`` kernel in ``libdrjit-extra``, which uses the hardware
     * texture units when available and re-attaches the differentiable
     * arithmetic for gradient tracking. The scalar backend uses \ref
     * eval_nonaccel().
     *
     * When using the non-hardware-accelerated evaluation, the numerical
     * precision of the interpolation is dictated by the floating point
     * precision of the query point type.
     */
    template <typename Output>
    Output eval(const position_for<Output> &pos,
                mask_for<Output> active = true) const {
        if constexpr (is_jit_v<Storage_>) {
            // Derivatives w.r.t. `pos` require the primal texture data; sync it
            // back if the texture was fully migrated to GPU texture memory.
            if constexpr (HasGPUTexture) {
                if (m_use_accel && grad_enabled(pos))
                    sync_device_data();
            }
            Output out = alloc_output<Output>();
            eval_jit(pos, out, active, m_use_accel);
            return out;
        } else {
            return eval_nonaccel<Output>(pos, active);
        }
    }

    /**
     * \brief Store values into a writable hardware texture
     *
     * The per-channel values in \c value are written to the texel addressed by
     * the integer coordinates \c pos. The texture must have been created with
     * <tt>writable = true</tt>.
     *
     * This is a hardware texture store (a side effect): it is not
     * differentiable, and the written texture is meant for display / external
     * sampling rather than \ref eval().
     */
    template <typename Value>
    void write(const Array<uint32_array_t<Value>, Dimension> &pos,
               const Value *value, mask_t<Value> active = true) {
        static_assert(HasGPUTexture,
                      "Texture::write() requires the CUDA or Metal backend.");
        if (!m_writable)
            jit_raise("Texture::write(): texture was not created with "
                      "writable=true.");

        uint32_t pos_idx[Dimension];
        for (size_t i = 0; i < Dimension; ++i)
            pos_idx[i] = pos[i].index();

        uint64_t *val_idx = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);
        for (size_t ch = 0; ch < m_channels; ++ch)
            val_idx[ch] = (uint64_t) value[ch].index();

        ad_tex_write((uint32_t) m_channels_storage, (uint32_t) m_channels,
                     m_handle, pos_idx, val_idx, active.index());

        // The GPU texture object is now the authoritative copy
        m_migrated = true;
        m_tensor_dirty = true;
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
    template <typename Output>
    void eval_fetch_nonaccel(const position_for<Output> &pos,
                             Array<Output, (1 << Dimension)> &out,
                             mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        constexpr size_t ncorner = 1 << Dimension;
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

    /**
     * \brief Fetch the texels that would be referenced in a texture lookup with
     * linear interpolation without actually performing this interpolation.
     *
     * On the JIT backends this is performed by the type-erased ``ad_tex_fetch``
     * kernel in ``libdrjit-extra`` (hardware-accelerated when available, with
     * the differentiable arithmetic re-attached); the scalar backend uses
     * \ref eval_fetch_nonaccel().
     */
    template <typename Output>
    Array<Output, (1 << Dimension)>
    eval_fetch(const position_for<Output> &pos,
               mask_for<Output> active = true) const {
        using Value = value_t<Output>;
        constexpr size_t ncorner = 1 << Dimension;
        Array<Output, ncorner> out = alloc_fetch_output<Output>();

        if constexpr (is_jit_v<Storage_>) {
            // Derivatives w.r.t. `pos` require the primal texture data; sync it
            // back if the texture was fully migrated to GPU texture memory.
            if constexpr (HasGPUTexture) {
                if (m_use_accel && grad_enabled(pos))
                    sync_device_data();
            }

            uint64_t *o = (uint64_t *) alloca(sizeof(uint64_t) * ncorner *
                                              m_channels);
            ad_tex_fetch(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) m_channels,
                (int) m_wrap_mode, m_handle, (int) m_use_accel, value_index(),
                resolution_indices().data(), idiv_indices().data(),
                pos_indices(pos).data(), active.index(), o);

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

        // Per-channel scratch on the stack (this helper also runs on JIT arrays
        // to build an AD graph, hence ``tex_scratch``).
        Value *res_mem     = (Value *) alloca(sizeof(Value) * m_channels);
        Value *scratch_mem = (Value *) alloca(sizeof(Value) * m_channels);
        detail::tex_scratch<Value> res(res_mem, m_channels),
                                   scratch(scratch_mem, m_channels);

        detail::tex_eval_cubic(scalar_ops<Value>(active), pos.data(),
                               res.data(), scratch.data());
        for (size_t ch = 0; ch < m_channels; ++ch)
            out.set_entry(ch, res[ch]);
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
            if constexpr (HasGPUTexture) {
                if (m_migrated && force_nonaccel)
                    jit_log(::LogLevel::Warn,
                            "\"force_nonaccel\" is used while the data has been "
                            "fully migrated to GPU texture memory");
                // Derivatives w.r.t. `pos` require the primal texture data
                if (use_accel && grad_enabled(pos))
                    sync_device_data();
            }

            Output out = alloc_output<Output>();
            uint64_t *o = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);
            ad_tex_cubic(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                (uint32_t) m_channels_storage, (uint32_t) m_channels,
                (int) m_wrap_mode, m_handle, (int) use_accel, value_index(),
                resolution_indices().data(), idiv_indices().data(),
                pos_indices(pos).data(), active.index(), o);

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
     * The resulting gradient and hessian have been multiplied by the spatial extents
     * to count for the transformation from the unit size volume to the size of its
     * shape.
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

    /// Gather the channels at \c idx and cast them to the query precision
    template <typename Value>
    void gather_texel(const uint32_array_t<Value> &idx,
                      const mask_t<Value> &active, Value *out) const {
        // Per-channel packet scratch on the stack
        Storage_ *packet_mem = (Storage_ *) alloca(sizeof(Storage_) * m_channels_storage);
        detail::tex_scratch<Storage_> packet(packet_mem, m_channels_storage);

        gather_packet_dynamic(m_channels_storage, m_value.array(), idx,
                              packet.data(), active);
        for (uint32_t ch = 0; ch < m_channels; ++ch)
            out[ch] = Value(packet[ch]);
    }

protected:
    void init(const size_t *shape, size_t channels, bool use_accel,
              FilterMode filter_mode, WrapMode wrap_mode,
              bool init_tensor = true, bool writable = false,
              void *external = nullptr) {
        if (channels == 0)
            jit_raise("Texture::Texture(): must have at least 1 channel!");

        m_writable = writable;
        m_channels = channels;

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

        if (init_tensor) {
            if constexpr (is_jit_v<Storage_>) {
                m_value =
                    TensorXf(empty<Storage>(m_size), Dimension + 1, tensor_shape);
                m_unpadded_value =
                    TensorXf(empty<Storage>(unpadded_size), Dimension + 1, m_shape);
            } else {
                // Don't allocate memory in scalar modes
                m_value =
                    TensorXf(Storage::map_(nullptr, m_size), Dimension + 1, tensor_shape);
                m_unpadded_value =
                    TensorXf(Storage::map_(nullptr, unpadded_size), Dimension + 1, m_shape);
            }
        }

        if constexpr (HasGPUTexture) {
            if (m_use_accel && init_tensor) {
                if (m_handle)
                    jit_tex_destroy(m_handle);

                if (external) {
                    // Wrap an externally-owned native texture (\ref from_native_handle).
                    m_handle = external;
                } else {
                    size_t tex_shape[Dimension];
                    reverse_tensor_shape(tex_shape, false);
                    m_handle = jit_tex_create(
                        Backend, Dimension, tex_shape, m_channels_storage,
                        (int) type_v<scalar_t<Storage_>>, (int) filter_mode,
                        (int) wrap_mode, (int) m_writable);
                }
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
        m_value = std::move(other.m_value);
        m_unpadded_value = std::move(other.m_unpadded_value);
        m_resolution_opaque = std::move(other.m_resolution_opaque);
        for (size_t i = 0; i < Dimension; ++i)
            m_inv_resolution[i] = std::move(other.m_inv_resolution[i]);
        m_filter_mode = other.m_filter_mode;
        m_wrap_mode = other.m_wrap_mode;
        m_use_accel = other.m_use_accel;
        m_writable = other.m_writable;
        m_migrated = other.m_migrated;
        m_tensor_dirty = other.m_tensor_dirty;
    }

    /// Updates the device-side padded tensor
    void sync_device_data() const {
        if constexpr (HasGPUTexture) {
            // Writable textures always read back: the hardware holds the
            // authoritative copy. We can't rely on write()'s m_migrated flag
            // alone, as that host-side assignment is skipped when a frozen
            // function is replayed.
            if (m_use_accel && (m_migrated || m_writable)) {
                Storage primal = empty<Storage>(m_size);

                /* The CUDA texture here is already padded with respect to the
                 * m_channels_storage size so we directly copy into device
                 * memory. Note, that for correct gradient tracking during
                 * texture evaluation, we need the tensor to be on the device,
                 * and moreover the padded storage allows us to leverage
                 * PacketOps when performing gathers/scatters.
                 */
                jit_tex_memcpy_t2d(m_handle, primal.data());

                if constexpr (IsDiff)
                    m_value.array() = replace_grad(primal, m_value.array());
                else
                    m_value.array() = primal;

                m_migrated = false;
                m_tensor_dirty = true; // the unpadded view must be refreshed
            }
        }
    }

    /// Helper function to reverse the tensor (\ref Texture.m_value) shape
    void reverse_tensor_shape(size_t *output, bool include_channels) const {
        for (size_t i = 0; i < Dimension; ++i)
            output[i] = m_value.shape(Dimension - 1 - i);
        if (include_channels)
            output[Dimension] = m_value.shape(Dimension);
    }

    /// Operations object for generating scalar texture evaluation code
    template <typename Value> struct ScalarOps {
        using Float = Value;
        using Int   = int32_array_t<Value>;
        using UInt  = uint32_array_t<Value>;
        using Mask  = mask_t<Value>;

        const Texture *tex;
        Mask active;
        uint32_t dim, channels_out;
        FilterMode filter_mode;
        WrapMode wrap_mode;

        Float lit(double v) const { return Value(v); }
        Float res_f(uint32_t k) const { return Value(tex->m_resolution_opaque[k]); }
        Int res_i(uint32_t k) const { return Int(tex->m_resolution_opaque[k]); }
        Float to_float(const Int &i) const { return Value(i); }
        Int idiv(const Int &a, uint32_t k) const { return tex->m_inv_resolution[k](a); }
        void gather(const UInt &idx, Float *out) const {
            tex->gather_texel(idx, active, out);
        }
    };

    /// Build a \ref ScalarOps bound to this texture and the query mask
    template <typename Value>
    ScalarOps<Value> scalar_ops(mask_t<Value> active) const {
        if constexpr (!is_array_v<mask_t<Value>>)
            active = true;
        return ScalarOps<Value>{ this, active, (uint32_t) Dimension,
            (uint32_t) m_channels, m_filter_mode, m_wrap_mode };
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
        if constexpr (is_diff_v<Storage_>)
            return Storage::steal(index);
        else
            return Storage::steal((uint32_t) index);
    }

    /// Combined AD/JIT index of the padded texture storage tensor
    uint64_t value_index() const { return combined_index(m_value.array()); }

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

    /// Evaluate via the type-erased ``ad_tex_eval`` kernel (JIT backends only)
    template <typename Output, typename Value = value_t<Output>>
    void eval_jit(const position_for<Output> &pos, Output &out,
                  mask_for<Output> active, bool use_accel) const {
        uint64_t *out_idx = (uint64_t *) alloca(sizeof(uint64_t) * m_channels);
        ad_tex_eval(type_v<scalar_t<Value>>, (uint32_t) Dimension,
                    (uint32_t) m_channels_storage, (uint32_t) m_channels,
                    (int) m_filter_mode, (int) m_wrap_mode, m_handle,
                    (int) use_accel, value_index(), resolution_indices().data(),
                    idiv_indices().data(), pos_indices(pos).data(),
                    active.index(), out_idx);
        for (size_t ch = 0; ch < m_channels; ++ch)
            out.set_entry(ch, steal_value<Value>(out_idx[ch]));
    }

    /// Shared marshaller for \ref eval_cubic_grad() / \ref eval_cubic_hessian():
    /// fills value + gradient (and hessian, when \c out_hessian is non-null),
    /// dispatching to ``ad_tex_cubic_deriv`` on the JIT backends and the scalar
    /// B-spline math otherwise.
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
                               (int) m_wrap_mode, value_index(),
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
    size_t m_shape[Dimension + 1] = {};      ///< Unpadded shape of texture
    mutable TensorXf m_value;                ///< Tensor padded for packet size
    /// Lazily computed if texture data is updated after initialization
    mutable TensorXf m_unpadded_value;

    // Stored in this order: width, height, depth
    Array<UInt32, Dimension> m_resolution_opaque;

    // Reciprocal resolution for the Repeat/Mirror wrap math. On the JIT backends
    // the magic constants are opaque variables consumed by the type-erased
    // kernel (see ad_tex_eval); populated only when the wrap mode divides.
    Divisor m_inv_resolution[Dimension] { };

    FilterMode m_filter_mode;
    WrapMode m_wrap_mode;
    bool m_use_accel = false;
    /// Texture was created so kernels may store into it via write()
    bool m_writable = false;
    /// Hardware-texture flag: is the data held exclusively on the device?
    mutable bool m_migrated = false;
    /// Does the public-facing unpadded tensor need to be updated?
    mutable bool m_tensor_dirty = false;

public:
    void
    traverse_1_cb_ro(void *payload,
                     drjit ::detail ::traverse_callback_ro fn) const override {
        // Traverse the function to react to changes when freezing code via
        // @dr.freeze. In all other contexts, the texture is read-only and does
        // not require traversal
        if (!jit_flag(JitFlag::EnableObjectTraversal))
            return;

        DRJIT_MAP(DR_TRAVERSE_MEMBER_RO, m_value, m_unpadded_value,
                  m_resolution_opaque, m_inv_resolution);
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

        DRJIT_MAP(DR_TRAVERSE_MEMBER_RW, m_value, m_unpadded_value,
                  m_resolution_opaque, m_inv_resolution);
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
