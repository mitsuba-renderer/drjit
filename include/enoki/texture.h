/*
    enoki/texture.h -- N-D Texture interpolation with GPU acceleration

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <array>
#include <enoki-jit/texture.h>
#include <enoki/dynamic.h>
#include <enoki/idiv.h>
#include <enoki/jit.h>
#include <enoki/tensor.h>

#pragma once

NAMESPACE_BEGIN(enoki)

enum class FilterMode : uint32_t { Nearest = 0, Linear = 1 };

enum class WrapMode : uint32_t { Repeat = 0, Clamp = 1, Mirror = 2 };

template <typename Value, size_t Dimension> class Texture {
public:
    static constexpr bool IsCUDA = is_cuda_array_v<Value>;
    static constexpr bool IsDiff = is_diff_array_v<Value>;
    static constexpr bool IsDynamic = is_dynamic_v<Value>;

    using Int32  = int32_array_t<Value>;
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
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Repeat) {
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
     */
    Texture(const TensorXf &tensor, bool migrate = true,
            FilterMode filter_mode = FilterMode::Linear,
            WrapMode wrap_mode = WrapMode::Repeat) {
        if (tensor.ndim() != Dimension + 1)
            enoki_raise("Texture::Texture(): tensor dimension must equal "
                        "texture dimension plus one.");
        init(tensor.shape().data(), tensor.shape(Dimension), migrate,
             filter_mode, wrap_mode);
        set_tensor(tensor);
    }

    Texture(Texture &&other) {
        m_handle         = other.handle;
        other.handle     = nullptr;
        m_size           = other.m_size;
        m_handle_opaque  = std::move(other.m_handle_opaque);
        m_shape_opaque   = std::move(other.m_shape_opaque);
        m_value          = std::move(other.m_value);
        m_migrate        = other.m_migrate;
        m_inv_resolution = std::move(other.m_inv_resolution);
        m_filter_mode    = other.m_filter_mode;
        m_wrap_mode      = other.m_wrap_mode;
    }

    Texture &operator=(Texture &&other) {
        if constexpr (IsCUDA) {
            jit_cuda_tex_destroy(m_handle);
            m_handle = nullptr;
        }
        m_handle         = other.m_handle;
        other.m_handle   = nullptr;
        m_size           = other.m_size;
        m_handle_opaque  = std::move(other.m_handle_opaque);
        m_shape_opaque   = std::move(other.m_shape_opaque);
        m_value          = std::move(other.m_value);
        m_migrate        = other.m_migrate;
        m_inv_resolution = std::move(other.m_inv_resolution);
        m_filter_mode    = other.m_filter_mode;
        m_wrap_mode      = other.m_wrap_mode;
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
    const size_t *shape() const { return m_value.shape().data(); }
    FilterMode filter_mode() const { return m_filter_mode; }
    WrapMode wrap_mode() const { return m_wrap_mode; }

    void set_value(const Storage &value) {
        if (value.size() != m_size)
            enoki_raise("Texture::set_value(): unexpected array size!");

        if constexpr (IsCUDA) {
            value.eval_(); // Sync the value before copying to texture memory
            jit_cuda_tex_memcpy_d2t(Dimension, m_value.shape().data(),
                                    m_value.shape(Dimension), value.data(),
                                    m_handle);

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
            if (tensor.shape(i) != m_value.shape(i))
                enoki_raise("Texture::set_tensor(): tensor shape mismatch!");
        }
        set_value(tensor.array());
    }

    Storage &value() const { return tensor().array(); }

    TensorXf &tensor() const {
        if constexpr (IsCUDA) {
            if (m_migrate) {
                if (m_value.array().size() != m_size) {
                    Storage primal = empty<Storage>(m_size);
                    jit_cuda_tex_memcpy_t2d(Dimension, m_value.shape().data(),
                                            m_value.shape(Dimension), m_handle,
                                            primal.data());
                    if constexpr (IsDiff)
                        m_value.array() = replace_grad(primal, m_value.array());
                    else
                        m_value.array() = primal;
                }
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
    Array<Value, 4> eval_cuda(const Array<Value, Dimension> &pos,
                              Mask active = true) const {
        if constexpr (IsCUDA) {
            uint32_t pos_idx[Dimension], out[4];
            for (size_t i = 0; i < Dimension; ++i)
                pos_idx[i] = pos[i].index();

            jit_cuda_tex_lookup(Dimension, m_handle_opaque.index(), pos_idx,
                                active.index(), out);

            return { Value::steal(out[0]), Value::steal(out[1]),
                     Value::steal(out[2]), Value::steal(out[3]) };
        } else {
            (void) pos;
            (void) active;
            return 0;
        }
    }

    template <typename T> T wrap(const T &value) const {
        const Array<Int32, Dimension> shape = m_shape_opaque;
        if (m_wrap_mode == WrapMode::Clamp) {
            return clamp(value, 0, shape - 1);
        } else {
            const T value_shift_neg = select(value < 0, value + 1, value);

            T div;
            for (size_t i = 0; i < Dimension; ++i) {
                div[i] = m_inv_resolution[i](value_shift_neg[i]);
            }

            T mod = value - div * shape;
            masked(mod, mod < 0) += T(shape);

            if (m_wrap_mode == WrapMode::Mirror)
                mod =
                    select(eq(div & 1, 0) ^ (value < 0), mod, shape - 1 - mod);

            return mod;
        }
    }

    /**
     * \brief Evaluate linear interpolant using explicit arithmetic
     *
     * This is an implementation detail, please use \ref eval() that may
     * dispatch to this function depending on its inputs.
     */
    Array<Value, 4> eval_enoki(const Array<Value, Dimension> &pos,
                               Mask active = true) const {
        using PosF = Array<Value, Dimension>;
        using PosI = int32_array_t<PosF>;

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);

        if (ENOKI_UNLIKELY(m_filter_mode == FilterMode::Nearest)) {
            const PosF pos_f   = pos * m_shape_opaque;
            const PosI pos_i   = floor2int<PosI>(pos_f);
            const PosI pos_i_w = wrap(pos_i);

            UInt32 index;
            if constexpr (Dimension == 1) {
                index = pos_i_w.x();
            } else if constexpr (Dimension == 2) {
                index = fmadd(pos_i_w.y(), m_shape_opaque.x(), pos_i_w.x());
            } else if constexpr (Dimension == 3) {
                index =
                    fmadd(fmadd(pos_i_w.z(), m_shape_opaque.y(), pos_i_w.y()),
                          m_shape_opaque.x(), pos_i_w.x());
            }
            index *= channels;

            Array<Value, 4> result(0);
            for (uint32_t ch = 0; ch < channels; ++ch)
                result[ch] = gather<Value>(m_value.array(), index + ch, active);

            return result;
        } else {
            using LerpIdx  = Array<Int32, 1 << Dimension>;
            using LerpPosI = Array<LerpIdx, Dimension>;

            const PosF pos_f = fmadd(pos, m_shape_opaque, -.5f);
            const PosI pos_i = floor2int<PosI>(pos_f);

            LerpPosI pos_i_w;
            if constexpr (Dimension == 1) {
                using Int2 = Array<Int32, 2>;
                pos_i_w    = wrap(LerpPosI(Int2(0, 1) + pos_i.x()));
            } else if constexpr (Dimension == 2) {
                using Int4 = Array<Int32, 4>;
                pos_i_w    = wrap(LerpPosI(Int4(0, 1, 0, 1) + pos_i.x(),
                                        Int4(0, 0, 1, 1) + pos_i.y()));
            } else if constexpr (Dimension == 3) {
                using Int8 = Array<Int32, 8>;
                pos_i_w =
                    wrap(LerpPosI(Int8(0, 1, 0, 1, 0, 1, 0, 1) + pos_i.x(),
                                  Int8(0, 0, 0, 0, 1, 1, 1, 1) + pos_i.y(),
                                  Int8(0, 0, 1, 1, 0, 0, 1, 1) + pos_i.z()));
            }

            LerpIdx index;
            if constexpr (Dimension == 1)
                index = pos_i_w.x();
            else if constexpr (Dimension == 2)
                index = fmadd(pos_i_w.y(), m_shape_opaque.x(), pos_i_w.x());
            else if constexpr (Dimension == 3)
                index =
                    fmadd(fmadd(pos_i_w.z(), m_shape_opaque.y(), pos_i_w.y()),
                          m_shape_opaque.x(), pos_i_w.x());
            index *= channels;

            Array<Value, 4> result(0);
            const PosF w1 = pos_f - floor(pos_f), w0 = 1.f - w1;

            if constexpr (Dimension == 1) {
                for (size_t ch = 0; ch < channels; ++ch) {
                    const Value v0 = gather<Value>(m_value.array(),
                                                   index.x() + ch, active),
                                v1 = gather<Value>(m_value.array(),
                                                   index.y() + ch, active);

                    result[ch] = fmadd(w0.x(), v0, w1.x() * v1);
                }
            } else if constexpr (Dimension == 2) {
                for (size_t ch = 0; ch < channels; ++ch) {
                    const Value v00 = gather<Value>(m_value.array(),
                                                    index.x() + ch, active),
                                v10 = gather<Value>(m_value.array(),
                                                    index.y() + ch, active),
                                v01 = gather<Value>(m_value.array(),
                                                    index.z() + ch, active),
                                v11 = gather<Value>(m_value.array(),
                                                    index.w() + ch, active);

                    const Value v0 = fmadd(w0.x(), v00, w1.x() * v10),
                                v1 = fmadd(w0.x(), v01, w1.x() * v11);

                    result[ch] = fmadd(w0.y(), v0, w1.y() * v1);
                }
            } else if constexpr (Dimension == 3) {
                for (size_t ch = 0; ch < channels; ++ch) {
                    const Value v000 = gather<Value>(m_value.array(),
                                                     index[0] + ch, active),
                                v100 = gather<Value>(m_value.array(),
                                                     index[1] + ch, active),
                                v001 = gather<Value>(m_value.array(),
                                                     index[2] + ch, active),
                                v101 = gather<Value>(m_value.array(),
                                                     index[3] + ch, active),
                                v010 = gather<Value>(m_value.array(),
                                                     index[4] + ch, active),
                                v110 = gather<Value>(m_value.array(),
                                                     index[5] + ch, active),
                                v011 = gather<Value>(m_value.array(),
                                                     index[6] + ch, active),
                                v111 = gather<Value>(m_value.array(),
                                                     index[7] + ch, active);

                    const Value v00 = fmadd(w0.x(), v000, w1.x() * v100),
                                v10 = fmadd(w0.x(), v010, w1.x() * v110),
                                v01 = fmadd(w0.x(), v001, w1.x() * v101),
                                v11 = fmadd(w0.x(), v011, w1.x() * v111);

                    const Value v0 = fmadd(w0.y(), v00, w1.y() * v10),
                                v1 = fmadd(w0.y(), v01, w1.y() * v11);

                    result[ch] = fmadd(w0.z(), v0, w1.z() * v1);
                }
            }

            return result;
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
        } else {
            return eval_enoki(pos, active);
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
    Array<Value, 4> eval_cubic_helper(const Array<Value, Dimension> &pos,
                                      Mask active = true) const {
        using PosF = Array<Value, Dimension>;
        using PosI = int32_array_t<PosF>;
        using Array4 = Array<Value, 4>;

        PosF pos_(pos);
        // This multiplication should not be recorded in the AD graph
        PosF pos_f = fmadd(pos_, m_shape_opaque, -.5f);
        if constexpr (IsDiff)
            if (grad_enabled(pos))
                pos_f = replace_grad(pos_f, pos_);
        PosI pos_i = floor2int<PosI>(pos_f);
        // `step[k][d]` controls the k-th offset in the d-th dimension.
        // With cubic B-Spline, it is by default [-1, 0, 1, 2] for all
        // dimensions. When the query point gets too close to (or exceeds) the
        // border, it needs to be cut down to the correct value to achieve the
        // 'clamping' goal
        Array<PosI, 4> step(0);
        step[0] = select(pos_f >= 1.f && pos_i <= m_shape_opaque - 1, -1, 0);
        step[2] = select(pos_f >= 0.f && pos_i < m_shape_opaque - 1, 1, 0);
        step[3] = select(pos_f >= -1.f && pos_i < m_shape_opaque - 2, 2, 0);
        step[3] = select(pos_f >= -1.f && pos_f < 0.f, 1, step[3]);
        step[3] =
            select(pos_f >= m_shape_opaque - 2 && pos_i <= m_shape_opaque - 2,
                   1, step[3]);
        PosF pos_a = pos_f - pos_i;
        pos_i = clamp(pos_i, 0, PosI(m_shape_opaque) - 1);

        const auto compute_weight = [&pos_a](uint32_t dim) -> Array4 {
            const Value &alpha = pos_a[dim];
            Value alpha2 = alpha * alpha, alpha3 = alpha2 * alpha;
            Value multiplier = rcp(6.f);
            return multiplier *
                   Array4(-alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f,
                          3.f * alpha3 - 6.f * alpha2 + 4.f,
                          -3.f * alpha3 + 3.f * alpha2 + 3.f * alpha + 1.f,
                          alpha3);
        };

        UInt32 index;
        if constexpr (Dimension == 1) {
            index = pos_i.x();
        } else if constexpr (Dimension == 2) {
            index = fmadd(pos_i.y(), m_shape_opaque.x(), pos_i.x());
            for (uint32_t idx = 0; idx < 4; ++idx)
                step[idx][1] *= m_shape_opaque.x();
        } else if constexpr (Dimension == 3) {
            index = fmadd(fmadd(pos_i.z(), m_shape_opaque.y(), pos_i.y()),
                          m_shape_opaque.x(), pos_i.x());
            for (uint32_t idx = 0; idx < 4; ++idx) {
                step[idx][1] *= m_shape_opaque.x();
                step[idx][2] *= m_shape_opaque.x() * m_shape_opaque.y();
            }
        }

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);
        index *= channels;
        step *= channels;

#define EK_TEX_CUBIC_ACCUM(index, weight)                                      \
    {                                                                          \
        UInt32 index_ = index;                                                 \
        Value weight_ = weight;                                                \
        for (uint32_t ch = 0; ch < channels; ++ch)                             \
            result[ch] =                                                       \
                fmadd(gather<Value>(m_value.array(), index_ + ch, active),     \
                      weight_, result[ch]);                                    \
    }

        Array<Value, 4> result(0);

        if constexpr (Dimension == 1) {
            Array4 wx = compute_weight(0);
            for (uint32_t ix = 0; ix < 4; ix++)
                EK_TEX_CUBIC_ACCUM(index + step[ix].x(), wx[ix]);
        } else if constexpr (Dimension == 2) {
            Array4 wx = compute_weight(0), wy = compute_weight(1);
            for (uint32_t ix = 0; ix < 4; ix++)
                for (uint32_t iy = 0; iy < 4; iy++)
                    EK_TEX_CUBIC_ACCUM(index + step[ix].x() + step[iy].y(),
                                       wx[ix] * wy[iy]);
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0), wy = compute_weight(1),
                   wz = compute_weight(2);
            for (uint32_t ix = 0; ix < 4; ix++)
                for (uint32_t iy = 0; iy < 4; iy++)
                    for (uint32_t iz = 0; iz < 4; iz++)
                        EK_TEX_CUBIC_ACCUM(index + step[ix].x() + step[iy].y() +
                                               step[iz].z(),
                                           wx[ix] * wy[iy] * wz[iz]);
        }

#undef EK_TEX_CUBIC_ACCUM

        return result;
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
    Array<Value, 4> eval_cubic(const Array<Value, Dimension> &pos,
                               Mask active = true,
                               bool force_enoki = false) const {
        using PosF = Array<Value, Dimension>;
        using PosI = int32_array_t<PosF>;
        using Array3 = Array<Value, 3>;
        using Array4 = Array<Value, 4>;

        if (m_migrate && force_enoki)
            jit_log(::LogLevel::Warn,
                    "\"force_enoki\" is used while the data has been fully "
                    "migrated to CUDA texture memory");

        PosF pos_f = fmadd(pos, m_shape_opaque, -.5f);
        PosI pos_i = floor2int<PosI>(pos_f);
        PosF pos_a = pos_f - pos_i;
        PosF inv_shape = rcp(PosF(m_shape_opaque));

        /* With cubic B-Spline, normally we have 4 query points and 4 weights
           for each dimension. After the linear interpolation transformation,
           they are reduced to 2 query points and 2 weights. This function
           returns the two weights and query coordinates.
           Note: the two weights sum to be 1.0 so only `w01` is returned. */
        auto compute_weight_coord = [&](uint32_t dim) -> Array3 {
            const Value integ = (Value) pos_i[dim];
            const Value alpha = pos_a[dim];
            Value alpha2 = sqr(alpha), alpha3 = alpha2 * alpha;
            Value multiplier = 1.f / 6.f;
            // four basis functions, transformed to take as input the fractional
            // part
            Value w0 =
                      (-alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f) * multiplier,
                  w1 = (3.f * alpha3 - 6.f * alpha2 + 4.f) * multiplier,
                  w3 = (alpha3) *multiplier;
            Value w01 = w0 + w1, w23 = 1.f - w01;
            return Array3(w01, (integ - 0.5f + w1 / w01) * inv_shape[dim],
                          (integ + 1.5f + w3 / w23) *
                              inv_shape[dim]); // (integ + 0.5) +- 1 + weight
        };

        auto eval_helper = [&force_enoki, this](const PosF &pos,
                                                const Mask &active) -> Array4 {
            if constexpr (IsCUDA) {
                if (!force_enoki)
                    return eval_cuda(pos, active);
            }
            ENOKI_MARK_USED(force_enoki);
            return eval_enoki(pos, active);
        };

        Array4 result(0);

        if constexpr (Dimension == 1) {
            Array3 cx = compute_weight_coord(0);
            Array4 f0 = eval_helper(PosF(cx[1]), active),
                   f1 = eval_helper(PosF(cx[2]), active);
            result = lerp(f1, f0, cx[0]);
        } else if constexpr (Dimension == 2) {
            Array3 cx = compute_weight_coord(0), cy = compute_weight_coord(1);
            Array4 f00 = eval_helper(PosF(cx[1], cy[1]), active),
                   f01 = eval_helper(PosF(cx[1], cy[2]), active),
                   f10 = eval_helper(PosF(cx[2], cy[1]), active),
                   f11 = eval_helper(PosF(cx[2], cy[2]), active);
            Array4 f0 = lerp(f01, f00, cy[0]), f1 = lerp(f11, f10, cy[0]);
            result = lerp(f1, f0, cx[0]);
        } else if constexpr (Dimension == 3) {
            Array3 cx = compute_weight_coord(0), cy = compute_weight_coord(1),
                   cz = compute_weight_coord(2);
            Array4 f000 = eval_helper(PosF(cx[1], cy[1], cz[1]), active),
                   f001 = eval_helper(PosF(cx[1], cy[1], cz[2]), active),
                   f010 = eval_helper(PosF(cx[1], cy[2], cz[1]), active),
                   f011 = eval_helper(PosF(cx[1], cy[2], cz[2]), active),
                   f100 = eval_helper(PosF(cx[2], cy[1], cz[1]), active),
                   f101 = eval_helper(PosF(cx[2], cy[1], cz[2]), active),
                   f110 = eval_helper(PosF(cx[2], cy[2], cz[1]), active),
                   f111 = eval_helper(PosF(cx[2], cy[2], cz[2]), active);
            Array4 f00 = lerp(f001, f000, cz[0]), f01 = lerp(f011, f010, cz[0]),
                   f10 = lerp(f101, f100, cz[0]), f11 = lerp(f111, f110, cz[0]);
            Array4 f0 = lerp(f01, f00, cy[0]), f1 = lerp(f11, f10, cy[0]);
            result = lerp(f1, f0, cx[0]);
        }

        if constexpr (IsDiff) {
            /* When `pos` has gradient enabled, call a helper function to
               replace the AD graph. The result is unused (and never computed)
               and only the AD graph is replaced. */
            if (grad_enabled(m_value, pos)) {
                Array4 result_diff =
                    eval_cubic_helper(pos, active); // AD graph only
                result = replace_grad(result, result_diff);
            }
        }

        return result;
    }

    /// Evaluate the positional gradient of clamped cubic B-Spline from the
    /// explicit differentiated basis functions
    std::array<Array<Value, 4>, Dimension>
    eval_cubic_grad(const Array<Value, Dimension> &pos,
                    Mask active = true) const {
        using PosF = Array<Value, Dimension>;
        using PosI = int32_array_t<PosF>;
        using Array4 = Array<Value, 4>;

        PosF pos_f = fmadd(pos, m_shape_opaque, -.5f);
        PosI pos_i = floor2int<PosI>(pos_f);
        Array<PosI, 4> step(0);
        step[0] = select(pos_f >= 1.f && pos_i <= m_shape_opaque - 1, -1, 0);
        step[2] = select(pos_f >= 0.f && pos_i < m_shape_opaque - 1, 1, 0);
        step[3] = select(pos_f >= -1.f && pos_i < m_shape_opaque - 2, 2, 0);
        step[3] = select(pos_f >= -1.f && pos_f < 0.f, 1, step[3]);
        step[3] =
            select(pos_f >= m_shape_opaque - 2 && pos_i <= m_shape_opaque - 2,
                   1, step[3]);
        PosF pos_a = pos_f - pos_i;
        pos_i = clamp(pos_i, 0, PosI(m_shape_opaque) - 1);

        const auto compute_weight = [&pos_a](uint32_t dim,
                                             bool is_grad) -> Array4 {
            const Value &alpha = pos_a[dim];
            Value alpha2 = alpha * alpha;
            Value multiplier = rcp(6.f);
            if (!is_grad) {
                Value alpha3 = alpha2 * alpha;
                return multiplier *
                       Array4(-alpha3 + 3.f * alpha2 - 3.f * alpha + 1.f,
                              3.f * alpha3 - 6.f * alpha2 + 4.f,
                              -3.f * alpha3 + 3.f * alpha2 + 3.f * alpha + 1.f,
                              alpha3);
            } else {
                return multiplier * Array4(-3.f * alpha2 + 6.f * alpha - 3.f,
                                           9.f * alpha2 - 12.f * alpha,
                                           -9.f * alpha2 + 6.f * alpha + 3.f,
                                           3.f * alpha2);
            }
        };

        UInt32 index;
        if constexpr (Dimension == 1) {
            index = pos_i.x();
        } else if constexpr (Dimension == 2) {
            index = fmadd(pos_i.y(), m_shape_opaque.x(), pos_i.x());
            for (uint32_t idx = 0; idx < 4; ++idx)
                step[idx][1] *= m_shape_opaque.x();
        } else if constexpr (Dimension == 3) {
            index = fmadd(fmadd(pos_i.z(), m_shape_opaque.y(), pos_i.y()),
                          m_shape_opaque.x(), pos_i.x());
            for (uint32_t idx = 0; idx < 4; ++idx) {
                step[idx][1] *= m_shape_opaque.x();
                step[idx][2] *= m_shape_opaque.x() * m_shape_opaque.y();
            }
        }

        const uint32_t channels = (uint32_t) m_value.shape(Dimension);
        index *= channels;
        step *= channels;

#define EK_TEX_CUBIC_GATHER(index)                                             \
    {                                                                          \
        UInt32 index_ = index;                                                 \
        for (uint32_t ch = 0; ch < channels; ++ch)                             \
            values[ch] = gather<Value>(m_value.array(), index_ + ch, active);  \
    }
#define EK_TEX_CUBIC_ACCUM(dim, weight)                                        \
    {                                                                          \
        uint32_t dim_ = dim;                                                   \
        Value weight_ = weight;                                                \
        for (uint32_t ch = 0; ch < channels; ++ch)                             \
            result[dim_][ch] = fmadd(values[ch], weight_, result[dim_][ch]);   \
    }

        std::array<Array4, Dimension> result;
        for (uint32_t dim = 0; dim < Dimension; ++dim)
            result[dim] = Array4(0.f);
        Array4 values;

        if constexpr (Dimension == 1) {
            Array4 gx = compute_weight(0, true);
            for (uint32_t ix = 0; ix < 4; ++ix) {
                EK_TEX_CUBIC_GATHER(index + step[ix].x());
                EK_TEX_CUBIC_ACCUM(0, gx[ix]);
            }
        } else if constexpr (Dimension == 2) {
            Array4 wx = compute_weight(0, false), wy = compute_weight(1, false),
                   gx = compute_weight(0, true), gy = compute_weight(1, true);
            for (uint32_t ix = 0; ix < 4; ++ix)
                for (uint32_t iy = 0; iy < 4; ++iy) {
                    EK_TEX_CUBIC_GATHER(index + step[ix].x() + step[iy].y());
                    EK_TEX_CUBIC_ACCUM(0, gx[ix] * wy[iy]);
                    EK_TEX_CUBIC_ACCUM(1, wx[ix] * gy[iy]);
                }
        } else if constexpr (Dimension == 3) {
            Array4 wx = compute_weight(0, false), wy = compute_weight(1, false),
                   wz = compute_weight(2, false), gx = compute_weight(0, true),
                   gy = compute_weight(1, true), gz = compute_weight(2, true);
            for (uint32_t ix = 0; ix < 4; ++ix)
                for (uint32_t iy = 0; iy < 4; ++iy)
                    for (uint32_t iz = 0; iz < 4; ++iz) {
                        EK_TEX_CUBIC_GATHER(index + step[ix].x() +
                                            step[iy].y() + step[iz].z());
                        EK_TEX_CUBIC_ACCUM(0, gx[ix] * wy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM(1, wx[ix] * gy[iy] * wz[iz]);
                        EK_TEX_CUBIC_ACCUM(2, wx[ix] * wy[iy] * gz[iz]);
                    }
        }

#undef EK_TEX_CUBIC_GATHER
#undef EK_TEX_CUBIC_ACCUM

        return result;
    }

protected:
    void init(const size_t *shape, size_t channels, bool migrate,
              FilterMode filter_mode, WrapMode wrap_mode) {
        if (channels != 1 && channels != 2 && channels != 4)
            enoki_raise("Texture::Texture(): must have 1, 2, or 4 channels!");

        m_size = channels;
        size_t tensor_shape[Dimension + 1]{};

        for (size_t i = 0; i < Dimension; ++i) {
            tensor_shape[i]     = shape[i];
            m_shape_opaque[i]   = opaque<UInt32>((uint32_t) shape[i]);
            m_inv_resolution[i] = divisor<int32_t>((int32_t) shape[i]);
            m_size *= shape[i];
        }

        tensor_shape[Dimension] = channels;
        m_value = TensorXf(zero<Storage>(m_size), Dimension + 1, tensor_shape);
        m_value.array() = Storage(0);
        m_migrate = migrate;
        m_filter_mode = filter_mode;
        m_wrap_mode = wrap_mode;

        if constexpr (IsCUDA) {
            m_handle = jit_cuda_tex_create(Dimension, shape, channels,
                                           (int) filter_mode, (int) wrap_mode);
            m_handle_opaque = UInt64::steal(
                jit_var_new_pointer(JitBackend::CUDA, m_handle, 0, 0));
        }
    }

private:
    void *m_handle = nullptr;
    size_t m_size  = 0;
    UInt64 m_handle_opaque;
    Array<UInt32, Dimension> m_shape_opaque;
    mutable TensorXf m_value;
    Array<divisor<int32_t>, Dimension> m_inv_resolution;
    FilterMode m_filter_mode;
    WrapMode m_wrap_mode;
    bool m_migrate = false;
};

NAMESPACE_END(enoki)
