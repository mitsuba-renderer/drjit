/*
    enoki/texture.h -- N-D Texture interpolation with GPU acceleration

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <enoki/jit.h>
#include <enoki/dynamic.h>
#include <enoki-jit/texture.h>

#pragma once

NAMESPACE_BEGIN(enoki)

template <typename Value, size_t Dimension> class Texture {
public:
    static constexpr bool IsCUDA = is_cuda_array_v<Value>;
    static constexpr bool IsDiff = is_diff_array_v<Value>;
    static constexpr bool IsDynamic = is_dynamic_v<Value>;

    using UInt32 = uint32_array_t<Value>;
    using UInt64 = uint64_array_t<Value>;
    using Mask = mask_t<Value>;

    using Storage = std::conditional_t<IsDynamic, Value, DynamicArray<Value>>;

    Texture(size_t shape[Dimension], size_t channels) : m_handle(nullptr) {
        m_size = channels;
        for (size_t i = 0; i < Dimension; ++i) {
            m_shape[i] = shape[i];
            m_shape_opaque[i] = opaque<Value>(scalar_t<UInt32>(shape[i]));
            m_size *= m_shape[i];
        }
        m_channels = (uint32_t) channels;

        if constexpr (IsCUDA) {
            m_handle = jit_cuda_tex_create(Dimension, shape, channels);
            m_handle_opaque = UInt64::steal(
                jit_var_new_pointer(JitBackend::CUDA, m_handle, 0, 0));
        }
    }

    ~Texture() {
        if constexpr (IsCUDA)
            jit_cuda_tex_destroy(m_handle);
    }

    void set_value(const Storage &value) {
        m_value = value;
        if (value.size() != m_size)
            enoki_raise("Texture::set_value(): unexpected array size!");

        if constexpr (IsCUDA)
            jit_cuda_tex_memcpy(Dimension, m_shape, m_channels,
                                value.data(), m_handle);
    }

    const Storage &value() const { return m_value; }

    Array<Value, 4> eval_cuda(const Array<Value, Dimension> &pos,
                              Mask active = true) const {
        if constexpr (IsCUDA) {
            if (m_value.empty())
                enoki_raise("Texture::eval_cuda(): texture has not been initialized yet!");

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
        if (m_value.empty())
            enoki_raise("Texture::eval_enoki(): texture has not been initialized yet!");

        using PosF = Array<Value, Dimension>;
        using PosI = uint32_array_t<PosF>;

        PosF pos_f = fmadd(pos, m_shape_opaque, -.5f);

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

        index *= m_channels;
        step *= m_channels;

        #define EK_TEX_ACCUM(index, weight) {                                   \
            UInt32 index_ = index;                                              \
            Value weight_ = weight;                                             \
            for (uint32_t ch = 0; ch < m_channels; ++ch)                        \
                result[ch] = fmadd(gather<Value>(m_value, index_ + ch, active), \
                                   weight_, result[ch]);                        \
        }

        Array<Value, 4> result(0);

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

private:
    size_t m_shape[Dimension];
    uint32_t m_channels;
    size_t m_size;
    void *m_handle;
    UInt64 m_handle_opaque;
    Array<UInt32, Dimension> m_shape_opaque;
    Storage m_value;
};

NAMESPACE_END(enoki)
