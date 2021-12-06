/*
    enoki/texture.h -- N-D Texture interpolation with GPU acceleration

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <enoki/jit.h>
#include <enoki/tensor.h>
#include <enoki/dynamic.h>
#include <enoki-jit/texture.h>

#pragma once

NAMESPACE_BEGIN(enoki)

enum class FilterMode : uint32_t {
    Nearest = 0,
    Linear = 1
};

template <typename Value, size_t Dimension> class Texture {
public:
    static constexpr bool IsCUDA = is_cuda_array_v<Value>;
    static constexpr bool IsDiff = is_diff_array_v<Value>;
    static constexpr bool IsDynamic = is_dynamic_v<Value>;

    using UInt32 = uint32_array_t<Value>;
    using UInt64 = uint64_array_t<Value>;
    using Mask = mask_t<Value>;

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
     * usable anymore (it will return zero.) and only \ref eval() or \ref
     * eval_cuda() should be used. Note that the texture is still
     * differentiable even when migrated. The \ref value() and \ref tensor()
     * operations need to perform a reverse migration in this mode.
     */
    Texture(const size_t shape[Dimension], size_t channels, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear) {
        init(shape, channels, migrate, filter_mode);
        m_value.array() = Storage(0);
    }

    /**
     * \brief Construct a new texture from a given tensor
     *
     * This constructor allocates texture memory just like the previous
     * constructor, though shape information is instead extracted from \c
     * tensor. It then also invokes <tt>set_tensor(tensor)</tt> to fill
     * the texture memory with the provided tensor.
     */
    Texture(const TensorXf &tensor, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear) {
        if (tensor.ndim() != Dimension + 1)
            enoki_raise("Texture::Texture(): tensor dimension must equal "
                        "texture dimension plus one.");
        init(tensor.shape().data(), tensor.shape(Dimension), migrate,
             filter_mode);
        set_tensor(tensor);
    }

    Texture(Texture &&other) {
        m_handle = other.handle;
        other.handle = nullptr;
        memcpy(m_shape, other.shape, sizeof(size_t) * (Dimension + 1));
        m_size = other.m_size;
        m_handle_opaque = std::move(other.m_handle_opaque);
        m_shape_opaque = std::move(other.m_shape_opaque);
        m_value = std::move(other.m_value);
        m_migrate = other.m_migrate;
    }

    Texture &operator=(Texture &&other) {
        if constexpr (IsCUDA) {
            jit_cuda_tex_destroy(m_handle);
            m_handle = nullptr;
        }
        m_handle = other.m_handle;
        other.m_handle = nullptr;
        memcpy(m_shape, other.m_shape, sizeof(size_t) * (Dimension + 1));
        m_size = other.m_size;
        m_handle_opaque = std::move(other.m_handle_opaque);
        m_shape_opaque = std::move(other.m_shape_opaque);
        m_value = std::move(other.m_value);
        m_migrate = other.m_migrate;
        return *this;
    }

    Texture(const Texture &) = delete;
    Texture &operator=(const Texture &) = delete;

    ~Texture() {
        if constexpr (IsCUDA)
            jit_cuda_tex_destroy(m_handle);
    }

    /// Return the CUDA handle (cudaTextureObject_t). NULL on all other backends
    const void *handle() const { return m_handle; }

    size_t ndim() const { return Dimension + 1; }
    const size_t *shape() const { return m_shape; }
    FilterMode filter_mode() const { return m_filter_mode; }

    void set_value(const Storage &value) {
        if (value.size() != m_size)
            enoki_raise("Texture::set_value(): unexpected array size!");

        if constexpr (IsCUDA) {
            jit_cuda_tex_memcpy_d2t(Dimension, m_shape, m_shape[Dimension],
                                    value.data(), m_handle);

            if (m_migrate) {
                // Fully migrate to texture memory, set m_value to zero
                if constexpr (IsDiff)
                    m_value.array() = replace_grad(Storage(0), value);
                else
                    m_value.array() = Storage(0);

                return;
            }
        }

        m_value.array() = value;
    }

    void set_tensor(const TensorXf &tensor) {
        if (tensor.ndim() != Dimension + 1)
            enoki_raise("Texture::set_tensor(): tensor dimension must equal "
                        "texture dimension plus one (channels).");
        for (size_t i = 0; i < Dimension + 1; ++i) {
            if (tensor.shape(i) != m_shape[i])
                enoki_raise("Texture::set_tensor(): tensor shape mismatch!");
        }
        set_value(tensor.array());
    }

    Storage &value() const {
        return tensor().array();
    }

    TensorXf &tensor() const {
        if constexpr (IsCUDA) {
            if (m_migrate) {
                if (m_value.array().size() != m_size) {
                    Storage primal = empty<Storage>(m_size);
                    jit_cuda_tex_memcpy_t2d(Dimension, m_shape, m_shape[Dimension],
                                            m_handle, primal.data());
                    if constexpr (IsDiff)
                        m_value.array() = replace_grad(primal, m_value.array());
                    else
                        m_value.array() = primal;
                }
            }
        }
        return m_value;
    }

    Array<Value, 4> eval_cuda(const Array<Value, Dimension> &pos,
                              Mask active = true) const {
        if constexpr (IsCUDA) {
            uint32_t pos_idx[Dimension], out[4];
            for (size_t i = 0; i < Dimension; ++i)
                pos_idx[i] = pos[i].index();

            jit_cuda_tex_lookup(Dimension, m_handle_opaque.index(),
                                pos_idx, active.index(), out);

            return {
                Value::steal(out[0]), Value::steal(out[1]),
                Value::steal(out[2]), Value::steal(out[3])
            };
        } else {
            (void) pos; (void) active;
            return 0;
        }
    }

    Array<Value, 4> eval_enoki(const Array<Value, Dimension> &pos,
                               Mask active = true) const {
        using PosF = Array<Value, Dimension>;
        using PosI = uint32_array_t<PosF>;

        PosF pos_f;
        if (ENOKI_UNLIKELY(m_filter_mode == FilterMode::Nearest))
            pos_f = pos * m_shape_opaque;
        else
            pos_f = fmadd(pos, m_shape_opaque, -.5f);

        PosI pos_i = clamp(PosI(pos_f), 0u, m_shape_opaque - 1u),
             step = select(pos_f >= 0.f && pos_i < m_shape_opaque - 1u, 1u, 0u);

        PosF w1 = pos_f - floor(pos_f),
             w0 = 1.f - w1;

        UInt32 index;
        if constexpr (Dimension == 1) {
            index = pos_i.x();
        } else if constexpr (Dimension == 2) {
            index = fmadd(pos_i.y(), m_shape_opaque.x(), pos_i.x());
            step.y() *= m_shape_opaque.x();
        } else if constexpr (Dimension == 3) {
            index = fmadd(
                fmadd(pos_i.z(), m_shape_opaque.y(), pos_i.y()),
                m_shape_opaque.x(),
                pos_i.x());
            step.y() *= m_shape_opaque.x();
            step.z() *= m_shape_opaque.x() * m_shape_opaque.y();
        }

        const size_t channels = m_shape[Dimension];
        index *= channels;
        step *= channels;

        #define EK_TEX_ACCUM(index, weight)                                        \
            {                                                                      \
                UInt32 index_ = index;                                             \
                Value weight_ = weight;                                            \
                for (uint32_t ch = 0; ch < channels; ++ch)                         \
                    result[ch] =                                                   \
                        fmadd(gather<Value>(m_value.array(), index_ + ch, active), \
                            weight_, result[ch]);                                  \
            }

        Array<Value, 4> result(0);

        if (ENOKI_UNLIKELY(m_filter_mode == FilterMode::Nearest)) {
            for (uint32_t ch = 0; ch < channels; ++ch)
                result[ch] = gather<Value>(m_value.array(), index + ch, active);
            return result;
        }

        if constexpr (Dimension == 1) {
            EK_TEX_ACCUM(index,            w0.x());
            EK_TEX_ACCUM(index + step.x(), w1.x());
        } else if constexpr (Dimension == 2) {
            EK_TEX_ACCUM(index,                       w0.x() * w0.y());
            EK_TEX_ACCUM(index + step.x(),            w1.x() * w0.y());
            EK_TEX_ACCUM(index + step.y(),            w0.x() * w1.y());
            EK_TEX_ACCUM(index + step.x() + step.y(), w1.x() * w1.y());
        } else if constexpr (Dimension == 3) {
            EK_TEX_ACCUM(index,                                  w0.x() * w0.y() * w0.z());
            EK_TEX_ACCUM(index + step.x(),                       w1.x() * w0.y() * w0.z());
            EK_TEX_ACCUM(index + step.y(),                       w0.x() * w1.y() * w0.z());
            EK_TEX_ACCUM(index + step.x() + step.y(),            w1.x() * w1.y() * w0.z());
            EK_TEX_ACCUM(index + step.z(),                       w0.x() * w0.y() * w1.z());
            EK_TEX_ACCUM(index + step.x() + step.z(),            w1.x() * w0.y() * w1.z());
            EK_TEX_ACCUM(index + step.y() + step.z(),            w0.x() * w1.y() * w1.z());
            EK_TEX_ACCUM(index + step.x() + step.y() + step.z(), w1.x() * w1.y() * w1.z());
        }

        #undef EK_TEX_ACCUM

        return result;
    }

    Array<Value, 4> eval(const Array<Value, Dimension> &pos,
                         Mask active = true) const {
        if constexpr (IsCUDA) {
            Array<Value, 4> result = eval_cuda(pos, active);

            if constexpr (IsDiff) {
                if (grad_enabled(m_value, pos)) {
                    Array<Value, 4> result_diff = eval_enoki(pos, active);
                    result = replace_grad(result, result_diff);
                }
            }
            return result;
        }

        return eval_enoki(pos, active);
    }

protected:
    void init(const size_t *shape, size_t channels, bool migrate,
              FilterMode filter_mode) {
        if (channels != 1 && channels != 2 && channels != 4)
            enoki_raise("Texture::Texture(): must have 1, 2, or 4 channels!");

        m_size = channels;
        for (size_t i = 0; i < Dimension; ++i) {
            m_shape[i] = shape[i];
            m_shape_opaque[i] = opaque<Value>(scalar_t<UInt32>(shape[i]));
            m_size *= m_shape[i];
        }
        m_shape[Dimension] = (uint32_t) channels;
        m_value = TensorXf(zero<Storage>(m_size), Dimension + 1, m_shape);
        m_value.array() = Storage(0);
        m_migrate = migrate;
        m_filter_mode = filter_mode;

        if constexpr (IsCUDA) {
            m_handle = jit_cuda_tex_create(Dimension, shape, channels,
                                           (int) filter_mode);
            m_handle_opaque = UInt64::steal(
                jit_var_new_pointer(JitBackend::CUDA, m_handle, 0, 0));
        }
    }

private:
    void *m_handle = nullptr;
    size_t m_shape[Dimension + 1] { };
    size_t m_size = 0;
    UInt64 m_handle_opaque;
    Array<UInt32, Dimension> m_shape_opaque;
    mutable TensorXf m_value;
    FilterMode m_filter_mode;
    bool m_migrate = false;
};

NAMESPACE_END(enoki)
