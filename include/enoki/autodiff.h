/*
    enoki/autodiff.h -- Forward/backward-mode automatic differentiation wrapper

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#define ENOKI_AUTODIFF_H

#if defined(ENOKI_BUILD_AUTODIFF)
#  define ENOKI_AUTODIFF_EXPORT
#  define ENOKI_AUTODIFF_EXPORT_TEMPLATE(T)
#else
#  define ENOKI_AUTODIFF_EXPORT ENOKI_IMPORT
#  define ENOKI_AUTODIFF_EXPORT_TEMPLATE(T)                                    \
     extern template ENOKI_AUTODIFF_EXPORT struct DiffArray<T>;
#endif

#include <enoki/array.h>
#include <enoki-jit/jit.h>

NAMESPACE_BEGIN(enoki)

NAMESPACE_BEGIN(detail)

// -----------------------------------------------------------------------
//! @{ \name External API compiled as part of libenoki-ad.so
// -----------------------------------------------------------------------

/// Increase the external reference count of a given variable
template <typename Value> void ad_inc_ref_impl(int32_t index) noexcept (true);

/// Decrease the external reference count of a given variable
template <typename Value> void ad_dec_ref_impl(int32_t index) noexcept (true);

/// Create a new variable with the given number of operands and AD weights
template <typename Value> int32_t ad_new(const char *label, uint32_t size, uint32_t ops,
                                         const int32_t *indices, Value *weights);

/// Query the gradient associated with a variable
template <typename Value> Value ad_grad(int32_t index, bool fail_if_missing);

/// Overwrite the gradient associated with a variable
template <typename Value>
void ad_set_grad(int32_t index, const Value &v, bool fail_if_missing);

/// Accumulate gradients into a variable
template <typename Value>
void ad_accum_grad(int32_t index, const Value &v, bool fail_if_missing);

/// Enqueue a variable for a subsequent ad_traverse() command
template <typename Value> void ad_enqueue(int32_t index);

/// Perform a forward or backward mode traversal of queued variables
template <typename Value> void ad_traverse(bool backward, bool retain_graph);

/// Label a variable (useful for debugging via graphviz etc.)
template <typename Value> void ad_set_label(int32_t index, const char *);

/// Return the label associated with a variable
template <typename Value> const char *ad_label(int32_t index);

/// Generate a graphviz plot of all registered variables
template <typename Value> const char *ad_graphviz();

/// Clear the gradient of variables accessed in ad_traverse()
template <typename Value> void ad_clear();

/// Special case of ad_new: create a node for a select() statement.
template <typename Value, typename Mask>
int32_t ad_new_select(const char *label, uint32_t size, const Mask &m,
                      int32_t t_index, int32_t f_index);

/// Special case of ad_new: create a node for a gather() expression
template <typename Value, typename Mask, typename Index>
int32_t ad_new_gather(const char *label, uint32_t size, int32_t src_index,
                      const Index &offset, const Mask &mask, bool permute);


/// Special case of ad_new: create a node for a scatter[_reduce]() statement.
template <typename Value, typename Mask, typename Index>
int32_t ad_new_scatter(const char *label, uint32_t size, ReduceOp op,
                       int32_t src_index, int32_t dst_index,
                       const Index &offset, const Mask &mask, bool permute);

/// Custom graph edge for implementing custom differentiable operations
struct ENOKI_AUTODIFF_EXPORT DiffCallback {
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual ~DiffCallback();
};

/// Register a custom/user-provided differentiable operation
template <typename Value>
void ad_add_edge(int32_t src_index, int32_t dst_index,
                 DiffCallback *callback = nullptr);

//! @}
// -----------------------------------------------------------------------

#if defined(__GNUC__)
template <typename T> ENOKI_INLINE void ad_inc_ref(int32_t index) noexcept {
    if (!__builtin_constant_p(index) || index != 0)
        ad_inc_ref_impl<T>(index);
}
template <typename T> ENOKI_INLINE void ad_dec_ref(int32_t index) noexcept {
    if (!__builtin_constant_p(index) || index != 0)
        ad_dec_ref_impl<T>(index);
}
#else
#define ad_inc_ref ad_inc_ref_impl
#define ad_dec_ref ad_dec_ref_impl
#endif

NAMESPACE_END(detail)

template <typename Type_>
struct DiffArray : ArrayBase<value_t<Type_>, is_mask_v<Type_>, DiffArray<Type_>> {
    static_assert(std::is_scalar_v<Type_> || (is_dynamic_array_v<Type_> && array_depth_v<Type_> == 1),
                  "DiffArray template parameter must either be a scalar (e.g. "
                  "float) or a dynamic array of depth 1.");
    using Base = ArrayBase<value_t<Type_>, is_mask_v<Type_>, DiffArray<Type_>>;

    template <typename> friend struct DiffArray;

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Type = Type_;
    using MaskType = DiffArray<mask_t<Type_>>;
    using ArrayType = DiffArray;
    using IndexType = DiffArray<uint32_array_t<Type_>>;
    using typename Base::Value;
    using typename Base::Scalar;

    static constexpr size_t Size  = std::is_scalar_v<Type_> ? 1 : array_size_v<Type_>;
    static constexpr size_t Depth = std::is_scalar_v<Type_> ? 1 : array_depth_v<Type_>;

    static constexpr bool IsDiff = true;
    static constexpr bool IsJIT = is_jit_array_v<Type_>;
    static constexpr bool IsCUDA = is_cuda_array_v<Type_>;
    static constexpr bool IsLLVM = is_llvm_array_v<Type_>;
    static constexpr bool IsDynamic = is_dynamic_v<Type_>;
    static constexpr bool IsEnabled = std::is_floating_point_v<scalar_t<Type_>>;

    template <typename T>
    using ReplaceValue = DiffArray<replace_scalar_t<Type_, scalar_t<T>>>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    ENOKI_INLINE DiffArray() = default;

    ENOKI_INLINE ~DiffArray() noexcept {
        if constexpr (IsEnabled)
            detail::ad_dec_ref<Type>(m_index);
    }

    ENOKI_INLINE DiffArray(const DiffArray &a) : m_value(a.m_value) {
        if constexpr (IsEnabled) {
            m_index = a.m_index;
            detail::ad_inc_ref<Type>(m_index);
        }
    }

    ENOKI_INLINE DiffArray(DiffArray &&a) noexcept
        : m_value(std::move(a.m_value)) {
        if constexpr (IsEnabled) {
            m_index = a.m_index;
            a.m_index = 0;
        }
    }

    template <typename T>
    DiffArray(const DiffArray<T> &v) : m_value(v.m_value) { }

    template <typename T>
    DiffArray(const DiffArray<T> &v, detail::reinterpret_flag)
        : m_value(v.m_value, detail::reinterpret_flag()) { }

    DiffArray(const Type &value, detail::reinterpret_flag)
        : m_value(value) { }

    template <typename T = Value, enable_if_t<!std::is_same_v<T, Type>> = 0>
    DiffArray(const Type &value) : m_value(value) { }
    DiffArray(Type &&value) : m_value(std::move(value)) { }

    template <typename T, enable_if_scalar_t<T> = 0>
    DiffArray(T value) : m_value((Value) value) { }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    DiffArray(Ts&&... ts) : m_value(std::forward<Ts>(ts)...) { }

    ENOKI_INLINE DiffArray &operator=(const DiffArray &a) {
        m_value = a.m_value;
        if constexpr (IsEnabled) {
            detail::ad_inc_ref<Type>(a.m_index);
            detail::ad_dec_ref<Type>(m_index);
            m_index = a.m_index;
        }
        return *this;
    }

    ENOKI_INLINE DiffArray &operator=(DiffArray &&a) {
        m_value = std::move(a.m_value);
        if constexpr (IsEnabled)
            std::swap(m_index, a.m_index);
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DiffArray add_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("add_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = m_value + a.m_value;
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0) {
                    int32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { 1, 1 };
                    index_new = detail::ad_new<Type>(
                        "add", (uint32_t) width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sub_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("sub_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = m_value - a.m_value;
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0) {
                    int32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { 1, -1 };
                    index_new = detail::ad_new<Type>(
                        "sub", (uint32_t) width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray mul_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("mul_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = m_value * a.m_value;
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0) {
                    int32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { a.m_value, m_value };
                    index_new = detail::ad_new<Type>(
                        "mul", (uint32_t) width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray div_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("div_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = m_value / a.m_value;
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0) {
                    int32_t indices[2] = { m_index, a.m_index };
                    Type rcp_a = rcp(a.m_value);
                    Type weights[2] = { rcp_a, -m_value * sqr(rcp_a) };
                    index_new = detail::ad_new<Type>(
                        "div", (uint32_t) width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray neg_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("neg_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = -m_value;
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -1 };
                    index_new = detail::ad_new<Type>(
                        "neg", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fmadd_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = fmadd(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0 || b.m_index > 0) {
                    int32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { a.m_value, m_value, 1 };
                    index_new = detail::ad_new<Type>(
                        "fmadd", (uint32_t) width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fmsub_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = fmsub(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0 || b.m_index > 0) {
                    int32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { a.m_value, m_value, -1 };
                    index_new = detail::ad_new<Type>(
                        "fmsub", (uint32_t) width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fnmadd_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = fnmadd(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0 || b.m_index > 0) {
                    int32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { -a.m_value, -m_value, 1 };
                    index_new = detail::ad_new<Type>(
                        "fnmadd", (uint32_t) width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fnmsub_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = fnmsub(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0 || b.m_index > 0) {
                    int32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { -a.m_value, -m_value, -1 };
                    index_new = detail::ad_new<Type>(
                        "fnmsub", (uint32_t) width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray abs_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("abs_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = abs(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { sign(m_value) };
                    index_new = detail::ad_new<Type>(
                        "abs", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sqrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sqrt_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = sqrt(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { .5f * rcp(result) };
                    index_new = detail::ad_new<Type>(
                        "sqrt", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray cbrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("cbrt_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = cbrt(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { (1 / 3.f) * sqr(rcp(result)) };
                    index_new = detail::ad_new<Type>(
                        "cbrt", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray erf_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("erf_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = erf(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { (2.f * InvSqrtPi<Type>) * exp(-sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "erf", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rcp_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("rcp_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = rcp(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -sqr(result) };
                    index_new = detail::ad_new<Type>(
                        "rcp", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rsqrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("rsqrt_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = rsqrt(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    Type rsqrt_2 = sqr(result), rsqrt_3 = result * rsqrt_2;
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -.5f * rsqrt_3 };
                    index_new = detail::ad_new<Type>(
                        "rsqrt", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray min_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("min_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = min(m_value, a.m_value);

            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0) {
                    mask_t<Type> m = m_value <= a.m_value;
                    int32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { select(m, Type(1), Type(0)),
                                        select(m, Type(0), Type(1)) };
                    index_new = detail::ad_new<Type>(
                        "min", (uint32_t) width(result), 2, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray max_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("max_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = max(m_value, a.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0) {
                    int32_t indices[2] = { m_index, a.m_index };
                    mask_t<Type> m = m_value > a.m_value;
                    Type weights[2] = { select(m, Type(1), Type(0)),
                                        select(m, Type(0), Type(1)) };
                    index_new = detail::ad_new<Type>(
                        "max", (uint32_t) width(result), 2, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    static DiffArray select_(const MaskType m,
                             const DiffArray &t,
                             const DiffArray &f) {
        if constexpr (std::is_scalar_v<Type>) {
            return m.m_value ? t : f;
        } else {
            Type result = select(m.m_value, t.m_value, f.m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (t.m_index > 0 || f.m_index > 0) {
                    index_new = detail::ad_new_select<Type>(
                        "select", (int32_t) width(result),
                        m.m_value, t.m_index, f.m_index);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray and_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index > 0)
                return select(mask, *this, DiffArray(0));
        }
        return DiffArray::create(0, detail::and_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray or_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            const Scalar value = memcpy_cast<Scalar>(int_array_t<Scalar>(-1));
            if (m_index > 0)
                return select(mask, DiffArray(value), *this);
        }
        return DiffArray::create(0, detail::or_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray xor_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index > 0)
                enoki_raise("xor_(): operation not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray andnot_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index > 0)
                enoki_raise("andnot_(): operation not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::andnot_(m_value, mask.m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Transcendental functions
    // -----------------------------------------------------------------------

    DiffArray sin_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sin_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    index_new = detail::ad_new<Type>("sin", (uint32_t) width(s),
                                                     1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("cos_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -s };
                    index_new = detail::ad_new<Type>("cos", (uint32_t) width(c),
                                                     1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sincos_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            int32_t index_s = 0, index_c = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights_s[1] = { c }, weights_c[1] = { -s };
                    uint32_t w = (uint32_t) width(s);
                    index_s =
                        detail::ad_new<Type>("sincos[s]", w, 1, indices, weights_s);
                    index_c =
                        detail::ad_new<Type>("sincos[c]", w, 1, indices, weights_c);
                }
            }

            return {
                DiffArray::create(index_s, std::move(s)),
                DiffArray::create(index_c, std::move(c)),
            };
        }
    }

    DiffArray csc_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("csc_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = csc(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -result * cot(m_value) };
                    index_new = detail::ad_new<Type>(
                        "csc", (uint32_t) width(result), 1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sec_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sec_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = sec(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { result * tan(m_value) };
                    index_new = detail::ad_new<Type>(
                        "sec", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray tan_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("tan_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = tan(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { sqr(sec(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "tan", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray cot_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("cot_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = cot(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -sqr(csc(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "cot", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asin_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("asin_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = asin(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt(fnmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "asin", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray acos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("acos_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = acos(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { -rsqrt(fnmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "acos", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atan_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("atan_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = atan(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(fmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "atan", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atan2_(const DiffArray &x) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("atan2_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = atan2(m_value, x.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || x.m_index > 0) {
                    Type il2 = rcp(fmadd(m_value, m_value, sqr(x.m_value)));
                    int32_t indices[2] = { m_index, x.m_index };
                    Type weights[2] = { il2 * x.m_value, -il2 * m_value };
                    index_new = detail::ad_new<Type>(
                        "atan2", (uint32_t) width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray exp_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("exp_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = exp(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { result };
                    index_new = detail::ad_new<Type>(
                        "exp", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray exp2_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("exp2_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = exp2(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { result * LogTwo<Value> };
                    index_new = detail::ad_new<Type>(
                        "exp2", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("log_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = log(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(m_value) };
                    index_new = detail::ad_new<Type>(
                        "log", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log2_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("log2_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = log2(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(m_value) * InvLogTwo<Value> };
                    index_new = detail::ad_new<Type>(
                        "log2", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sinh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sinh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    index_new = detail::ad_new<Type>("sinh", (uint32_t) width(s),
                                                     1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("cosh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { s };
                    index_new = detail::ad_new<Type>("cosh", (uint32_t) width(c),
                                                     1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sincosh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            int32_t index_s = 0, index_c = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights_s[1] = { c }, weights_c[1] = { s };
                    uint32_t w = (uint32_t) width(s);
                    index_s =
                        detail::ad_new<Type>("sincosh[s]", w, 1, indices, weights_s);
                    index_c =
                        detail::ad_new<Type>("sincosh[c]", w, 1, indices, weights_c);
                }
            }

            return {
                DiffArray::create(index_s, std::move(s)),
                DiffArray::create(index_c, std::move(c)),
            };
        }
    }

    DiffArray tanh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("tanh_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = tanh(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { sqr(sech(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "tanh", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asinh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("asinh_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = asinh(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt((Scalar) 1 + sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "asinh", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray acosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("acosh_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = acosh(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt(sqr(m_value) - (Scalar) 1) };
                    index_new = detail::ad_new<Type>(
                        "acosh", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atanh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("atanh_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = atanh(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { rcp((Scalar) 1 - sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "atanh", (uint32_t) width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Operations that don't require derivatives
    // -----------------------------------------------------------------------

    DiffArray mod_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("mod_(): invalid operand type!");
        else
            return m_value % a.m_value;
    }

    DiffArray mulhi_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("mulhi_(): invalid operand type!");
        else
            return mulhi(m_value, a.m_value);
    }

    MaskType eq_ (const DiffArray &d) const { return eq(m_value, d.m_value); }
    MaskType neq_(const DiffArray &d) const { return neq(m_value, d.m_value); }
    MaskType lt_ (const DiffArray &d) const { return m_value < d.m_value; }
    MaskType le_ (const DiffArray &d) const { return m_value <= d.m_value; }
    MaskType gt_ (const DiffArray &d) const { return m_value > d.m_value; }
    MaskType ge_ (const DiffArray &d) const { return m_value >= d.m_value; }

    DiffArray not_() const {
        if constexpr (is_floating_point_v<Scalar>)
            enoki_raise("not_(): invalid operand type!");
        else
            return DiffArray::create(0, ~m_value);
    }

    DiffArray or_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index > 0 || a.m_index > 0)
                enoki_raise("or_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::or_(m_value, a.m_value));
    }

    DiffArray and_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index > 0 || a.m_index > 0)
                enoki_raise("and_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::and_(m_value, a.m_value));
    }

    DiffArray xor_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index > 0 || a.m_index > 0)
                enoki_raise("xor_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, a.m_value));
    }

    DiffArray andnot_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index > 0 || a.m_index > 0)
                enoki_raise("andnot_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::andnot_(m_value, a.m_value));
    }

    DiffArray floor_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("floor_(): invalid operand type!");
        else
            return DiffArray::create(0, floor(m_value));
    }

    DiffArray ceil_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("ceil_(): invalid operand type!");
        else
            return DiffArray::create(0, ceil(m_value));
    }

    DiffArray trunc_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("trunc_(): invalid operand type!");
        else
            return DiffArray::create(0, trunc(m_value));
    }

    DiffArray round_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("round_(): invalid operand type!");
        else
            return DiffArray::create(0, round(m_value));
    }

    template <typename T> T ceil2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("ceil2int_(): invalid operand type!");
        else
            return T::create(0, ceil2int<typename T::Type>(m_value));
    }

    template <typename T> T floor2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("floor2int_(): invalid operand type!");
        else
            return T::create(0, floor2int<typename T::Type>(m_value));
    }

    template <typename T> T round2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("round2int_(): invalid operand type!");
        else
            return T::create(0, round2int<typename T::Type>(m_value));
    }

    template <typename T> T trunc2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("trunc2int_(): invalid operand type!");
        else
            return T::create(0, trunc2int<typename T::Type>(m_value));
    }

    DiffArray sl_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("sl_(): invalid operand type!");
        else
            return DiffArray::create(0, m_value << a.m_value);
    }

    DiffArray sr_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("sr_(): invalid operand type!");
        else
            return DiffArray::create(0, m_value >> a.m_value);
    }

    template <int Imm> DiffArray sl_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("sl_(): invalid operand type!");
        else
            return DiffArray::create(0, sl<Imm>(m_value));
    }

    template <int Imm> DiffArray sr_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("sr_(): invalid operand type!");
        else
            return DiffArray::create(0, sr<Imm>(m_value));
    }

    DiffArray tzcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("tzcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, tzcnt(m_value));
    }

    DiffArray lzcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("lzcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, lzcnt(m_value));
    }

    DiffArray popcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            enoki_raise("popcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, popcnt(m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const {
        if constexpr (!is_mask_v<Type>)
            enoki_raise("all_(): invalid operand type!");
        else
            return all(m_value);
    }

    bool any_() const {
        if constexpr (!is_mask_v<Type>)
            enoki_raise("any_(): invalid operand type!");
        else
            return any(m_value);
    }

    size_t count_() const {
        if constexpr (!is_mask_v<Type>)
            enoki_raise("count_(): invalid operand type!");
        else
            return count(m_value);
    }

    DiffArray hsum_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hsum_async_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { 1 };
                    index_new = detail::ad_new<Type>(
                        "hsum_async", 1, 1, indices, weights);
                }
            }

            return DiffArray::create(index_new, hsum_async(m_value));
        }
    }

    Value hsum_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hsum_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index > 0)
                    enoki_raise("hsum_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hsum_async() instead, which "
                                "returns a differentiable array.");
            }
            return hsum(m_value);
        }
    }

    DiffArray hprod_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hprod_async_(): invalid operand type!");
        } else {
            int32_t index_new = 0;
            Type result = hprod_async(m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    int32_t indices[1] = { m_index };
                    Type weights[1] = { select(eq(m_value, (Scalar) 0),
                                               (Scalar) 0, result / m_value) };
                    index_new = detail::ad_new<Type>(
                        "hprod_async", 1, 1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    Value hprod_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hprod_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index > 0)
                    enoki_raise("hprod_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hprod_async() instead, which "
                                "returns a differentiable array.");
            }
            return hprod(m_value);
        }
    }

    DiffArray hmin_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hmin_async_(): invalid operand type!");
        } else {
            Type result = hmin_async(m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the minimum , which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    int32_t indices[1] = { m_index };
                    Type weights[1] = { select(
                        eq(m_value, result), Type(1), Type(0)) };
                    index_new = detail::ad_new<Type>(
                        "hmin_async", 1, 1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    Value hmin_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hmin_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index > 0)
                    enoki_raise("hmin_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hmin_async() instead, which "
                                "returns a differentiable array.");
            }
            return hmin(m_value);
        }
    }

    DiffArray hmax_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hmax_async_(): invalid operand type!");
        } else {
            Type result = hmax_async(m_value);
            int32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index > 0) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the maximum, which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    int32_t indices[1] = { m_index };
                    Type weights[1] = { select(
                        eq(m_value, result), Type(1), Type(0)) };
                    index_new = detail::ad_new<Type>(
                        "hmax_async", 1, 1, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    Value hmax_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hmax_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index > 0)
                    enoki_raise("hmax_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hmax_async() instead, which "
                                "returns a differentiable array.");
            }
            return hmax(m_value);
        }
    }

    DiffArray dot_async_(const DiffArray &a) const {
        return hsum_async(*this * a);
    }

    Value dot_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("dot_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index > 0 || a.m_index > 0)
                    enoki_raise("dot_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use dot_async() instead, which "
                                "returns a differentiable array.");
            }
            return dot(m_value, a.m_value);
        }
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather operations
    // -----------------------------------------------------------------------

    template <bool Permute>
    static DiffArray gather_(const DiffArray &src, const IndexType &offset,
                             const MaskType &mask = true) {
        if constexpr (std::is_scalar_v<Type>) {
            enoki_raise("Array gather operation not supported for scalar array type.");
        } else {
            Type result = gather<Type>(src.m_value, offset.m_value, mask.m_value);
            int32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (src.m_index > 0)
                    index_new = detail::ad_new_gather<Type>(
                        Permute ? "gather[permute]" : "gather",
                        (uint32_t) width(result), src.m_index, offset.m_value,
                        mask.m_value, Permute);
            }
            return create(index_new, std::move(result));
        }
    }

    template <bool Permute>
    void scatter_(DiffArray &dst, const IndexType &offset,
                  const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            enoki_raise("Array scatter operation not supported for scalar array type.");
        } else {
            scatter(dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0 || (dst.m_index > 0 && !Permute)) {
                    int32_t index = detail::ad_new_scatter<Type>(
                        Permute ? "scatter[permute]" : "scatter", (uint32_t) width(dst),
                        ReduceOp::None, m_index, dst.m_index, offset.m_value,
                        mask.m_value, Permute);
                    detail::ad_dec_ref<Type>(dst.m_index);
                    dst.m_index = index;
                }
            }
        }
    }

    void scatter_reduce_(ReduceOp op, DiffArray &dst, const IndexType &offset,
                         const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            enoki_raise("Array scatter_reduce operation not supported for scalar array type.");
        } else {
            scatter_reduce(op, dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index > 0) { // safe to ignore dst.m_index in the case of scatter_reduce
                    int32_t index = detail::ad_new_scatter<Type>(
                        "scatter_reduce", (uint32_t) width(dst), op, m_index,
                        dst.m_index, offset.m_value, mask.m_value, false);
                    detail::ad_dec_ref<Type>(dst.m_index);
                    dst.m_index = index;
                }
            }
        }
    }

    template <bool>
    static DiffArray gather_(const void *src, const IndexType &offset,
                             const MaskType &mask = true) {
        return create(0, gather<Type>(src, offset.m_value, mask.m_value));
    }

    template <bool>
    void scatter_(void *dst, const IndexType &offset,
                  const MaskType &mask = true) const {
        scatter(dst, m_value, offset.m_value, mask.m_value);
    }

    void scatter_reduce_(ReduceOp op, void *dst, const IndexType &offset,
                         const MaskType &mask = true) const {
        scatter_reduce(op, dst, m_value, offset.m_value, mask.m_value);
    }

    auto compress_() const {
        if constexpr (!is_mask_v<Type>)
            enoki_raise("compress_(): invalid operand type!");
        else
            return uint32_array_t<ArrayType>::create(0, compress(m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Standard initializers
    // -----------------------------------------------------------------------

    DiffArray placeholder_(bool preserve_size, bool propagate_literals) const {
        return placeholder<Type>(m_value, preserve_size, propagate_literals);
    }

    static DiffArray empty_(size_t size) {
        return enoki::empty<Type>(size);
    }

    static DiffArray zero_(size_t size) {
        return zero<Type>(size);
    }

    static DiffArray full_(Value value, size_t size) {
        return full<Type>(value, size);
    }

    static DiffArray opaque_(Value value, size_t size) {
        return opaque<Type>(value, size);
    }

    static DiffArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        return arange<Type>(start, stop, step);
    }

    static DiffArray linspace_(Value min, Value max, size_t size) {
        return linspace<Type>(min, max, size);
    }

    static DiffArray map_(void *ptr, size_t size, bool free = false) {
        if constexpr (is_jit_array_v<Type>)
            return Type::map_(ptr, size, free);
        else
            enoki_raise("map_(): not supported in scalar mode!");
    }

    static DiffArray load_(const void *ptr, size_t size) {
        return load<Type>(ptr, size);
    }

    void store_(void *ptr) const {
        store(ptr, m_value);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    DiffArray copy() const {
        if constexpr (IsEnabled) {
            if (m_index) {
                int32_t indices[1] = { m_index };
                Type weights[1] = { 1 };

                DiffArray result;
                result.m_index = detail::ad_new<Type>(
                    "copy", (uint32_t) width(m_value), 1, indices, weights);
                result.m_value = m_value;
                return result;
            }
        }

        return *this;
    }

    auto vcall_() const {
        if constexpr (is_jit_array_v<Type>)
            return m_value.vcall_();
        else
            enoki_raise("vcall_(): not supported in scalar mode!");
    }

    static DiffArray steal(int32_t index) {
        if constexpr (is_jit_array_v<Type>)
            return Type::steal(index);
        else
            enoki_raise("steal(): not supported in scalar mode!");
    }

    static DiffArray borrow(int32_t index) {
        if constexpr (is_jit_array_v<Type>)
            return Type::borrow(index);
        else
            enoki_raise("borrow(): not supported in scalar mode!");
    }

    void set_grad_enabled_(bool value) {
        if constexpr (IsEnabled) {
            if (value) {
                if (m_index > 0)
                    return;
                m_index = detail::ad_new<Type>(nullptr, (uint32_t) width(m_value),
                                               0, nullptr, (Type *) nullptr);
            } else {
                if (m_index == 0)
                    return;
                detail::ad_dec_ref<Type>(m_index);
                m_index = 0;
            }
        }
    }

    void set_grad_suspended_(bool value) {
        if constexpr (IsEnabled) {
            if (value != (m_index < 0))
                m_index = -m_index;
        }
    }

    DiffArray migrate_(AllocType type) const {
        if constexpr (is_jit_array_v<Type_>)
            return m_value.migrate_(type);
        return *this;
    }

    bool schedule_() const {
        if constexpr (is_jit_array_v<Type_>)
            return m_value.schedule_();
        return false;
    }

    bool eval_() const {
        if constexpr (is_jit_array_v<Type_>)
            return m_value.eval_();
        return false;
    }

    void enqueue_() const {
        if constexpr (IsEnabled)
            detail::ad_enqueue<Type>(m_index);
    }

    static void traverse_(bool backward, bool retain_graph) {
        ENOKI_MARK_USED(backward);
        if constexpr (IsEnabled)
            detail::ad_traverse<Type>(backward, retain_graph);
    }

    void set_label_(const char *label) const {
        set_label(m_value, label);

        if constexpr (IsEnabled) {
            if (m_index)
                detail::ad_set_label<Type>(m_index, label);
        }
    }

    const char *label_() const {
        const char *result = nullptr;
        if constexpr (IsEnabled) {
            if (m_index > 0)
                result = detail::ad_label<Type>(m_index);
        }
        if constexpr (is_jit_array_v<Type>) {
            if (!result)
                result = m_value.label_();
        }
        return result;
    }

    static const char *graphviz_() {
        if constexpr (IsEnabled)
            return detail::ad_graphviz<Type>();
    }

    const Type &detach_() const {
        return m_value;
    }

    Type &detach_() {
        return m_value;
    }

    const Type grad_(bool fail_if_missing = false) const {
        if constexpr (IsEnabled)
            return detail::ad_grad<Type>(m_index, fail_if_missing);
        else
            return zero<Type>();
    }

    void set_grad_(const Type &value, bool fail_if_missing = false) {
        if constexpr (IsEnabled)
            detail::ad_set_grad<Type>(m_index, value, fail_if_missing);
    }

    void accum_grad_(const Type &value, bool fail_if_missing = false) {
        if constexpr (IsEnabled)
            detail::ad_accum_grad<Type>(m_index, value, fail_if_missing);
    }

    size_t size() const {
        if constexpr (std::is_scalar_v<Type>)
            return 1;
        else
            return m_value.size();
    }

    Value entry(size_t offset) const {
        if constexpr (std::is_scalar_v<Type>)
            return m_value;
        else
            return m_value.entry(offset);
    }

    void set_entry(size_t offset, Value value) {
        if (m_index)
            enoki_raise("Attempted to overwrite entries of a variable that is "
                        "attached to the AD graph. This is not allowed.");

        if constexpr (is_dynamic_v<Type_>) {
            m_value.set_entry(offset, value);
        }
        else {
#if !defined(NDEBUG) && !defined(ENOKI_DISABLE_RANGE_CHECK)
            if (offset != 0)
                enoki_raise("Out of range access (tried to access index %u in "
                            "an array of size 1)", offset);
#endif
            m_value = value;
        }
    }

    void resize(size_t size) {
        if constexpr (is_dynamic_v<Type>)
            m_value.resize(size);
    }

    Scalar *data() {
        if constexpr (std::is_scalar_v<Type>)
            return &m_value;
        else
            return m_value.data();
    }

    const Scalar *data() const {
        if constexpr (std::is_scalar_v<Type>)
            return &m_value;
        else
            return m_value.data();
    }

    static DiffArray create(int32_t index, Type&& value) {
        DiffArray result;
        result.m_index = index;
        result.m_value = std::move(value);
        return result;
    }

    static DiffArray create_borrow(int32_t index, const Type &value) {
        DiffArray result;
        result.m_index = index;
        result.m_value = value;
        if constexpr (IsEnabled)
            detail::ad_inc_ref<Type>(index);
        return result;
    }

    void init_(size_t size) {
        if constexpr (is_dynamic_v<Type>)
            m_value.init_(size);
    }

    bool is_placeholder() const {
        if constexpr (is_jit_array_v<Type>)
            return m_value.is_placeholder();
        else
            enoki_raise("is_placeholder(): expected a JIT array type");
    }

    bool is_literal() const {
        if constexpr (is_jit_array_v<Type>)
            return m_value.is_literal();
        else
            enoki_raise("is_literal(): expected a JIT array type");
    }

    uint32_t index() const {
        if constexpr (is_jit_array_v<Type>)
            return m_value.index();
        else
            enoki_raise("index(): expected a JIT array type");
    }

    int32_t index_ad() const { return m_index; }

    /// Change variable index without involving reference counting. Dangerous, used by custom.h
    void set_index_ad(uint32_t value) { m_index = value; }

    //! @}
    // -----------------------------------------------------------------------

protected:
    Type m_value {};
    int32_t m_index = 0;
};

#define ENOKI_DECLARE_EXTERN_TEMPLATE(T, Mask, Index)                          \
    ENOKI_AUTODIFF_EXPORT_TEMPLATE(T)                                          \
    namespace detail {                                                         \
    extern template ENOKI_AUTODIFF_EXPORT void                                 \
        ad_inc_ref_impl<T>(int32_t) noexcept;                                  \
    extern template ENOKI_AUTODIFF_EXPORT void                                 \
        ad_dec_ref_impl<T>(int32_t) noexcept;                                  \
    extern template ENOKI_AUTODIFF_EXPORT int32_t                              \
    ad_new<T>(const char *, uint32_t, uint32_t, const int32_t *, T *);         \
    extern template ENOKI_AUTODIFF_EXPORT T ad_grad<T>(int32_t, bool);         \
    extern template ENOKI_AUTODIFF_EXPORT void                                 \
    ad_set_grad<T>(int32_t, const T &, bool);                                  \
    extern template ENOKI_AUTODIFF_EXPORT void                                 \
    ad_accum_grad<T>(int32_t, const T &, bool);                                \
    extern template ENOKI_AUTODIFF_EXPORT void ad_set_label<T>(int32_t,        \
                                                               const char *);  \
    extern template ENOKI_AUTODIFF_EXPORT const char *ad_label<T>(int32_t);    \
    extern template ENOKI_AUTODIFF_EXPORT const char *ad_graphviz<T>();        \
    extern template ENOKI_AUTODIFF_EXPORT void ad_enqueue<T>(int32_t);         \
    extern template ENOKI_AUTODIFF_EXPORT void ad_traverse<T>(bool, bool);     \
    extern template ENOKI_AUTODIFF_EXPORT int32_t ad_new_select<T, Mask>(      \
        const char *, uint32_t, const Mask &, int32_t, int32_t);               \
    extern template ENOKI_AUTODIFF_EXPORT int32_t                              \
    ad_new_gather<T, Mask, Index>(const char *, uint32_t, int32_t,             \
                                  const Index &, const Mask &, bool);          \
    extern template ENOKI_AUTODIFF_EXPORT int32_t                              \
    ad_new_scatter<T, Mask, Index>(const char *, uint32_t, ReduceOp, int32_t,  \
                                   int32_t, const Index &, const Mask &,       \
                                   bool);                                      \
    extern template ENOKI_AUTODIFF_EXPORT void                                 \
    ad_add_edge<T>(int32_t, int32_t, DiffCallback *);                          \
    extern template ENOKI_AUTODIFF_EXPORT void ad_clear<T>();                  \
    }

ENOKI_DECLARE_EXTERN_TEMPLATE(float,  bool, uint32_t)
ENOKI_DECLARE_EXTERN_TEMPLATE(double, bool, uint32_t)
#if defined(ENOKI_JIT_H)
ENOKI_DECLARE_EXTERN_TEMPLATE(CUDAArray<float>,  CUDAArray<bool>, CUDAArray<uint32_t>)
ENOKI_DECLARE_EXTERN_TEMPLATE(CUDAArray<double>, CUDAArray<bool>, CUDAArray<uint32_t>)
ENOKI_DECLARE_EXTERN_TEMPLATE(LLVMArray<float>,  LLVMArray<bool>, LLVMArray<uint32_t>)
ENOKI_DECLARE_EXTERN_TEMPLATE(LLVMArray<double>, LLVMArray<bool>, LLVMArray<uint32_t>)
#endif

template <typename T> void ad_clear() {
    detail::ad_clear<detached_t<T>>();
}

extern ENOKI_AUTODIFF_EXPORT const char *ad_whos();
extern ENOKI_AUTODIFF_EXPORT void ad_prefix_push(const char *value);
extern ENOKI_AUTODIFF_EXPORT void ad_prefix_pop();
extern ENOKI_AUTODIFF_EXPORT void ad_check_weights(bool value);

extern ENOKI_AUTODIFF_EXPORT size_t ad_dependency_count();
extern ENOKI_AUTODIFF_EXPORT void ad_add_dependency(int32_t index);
extern ENOKI_AUTODIFF_EXPORT void ad_write_dependencies(int32_t *out);
extern ENOKI_AUTODIFF_EXPORT void ad_clear_dependencies();
extern ENOKI_AUTODIFF_EXPORT void ad_set_max_edges_per_kernel(size_t value);

NAMESPACE_END(enoki)

#if defined(ENOKI_VCALL_H)
#  include <enoki/vcall_autodiff.h>
#endif
