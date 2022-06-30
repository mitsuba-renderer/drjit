/*
    drjit/autodiff.h -- Forward/reverse-mode automatic differentiation wrapper

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#define DRJIT_AUTODIFF_H

#if defined(DRJIT_BUILD_AUTODIFF)
#  define DRJIT_AD_EXPORT_TEMPLATE(T)
#else
#  define DRJIT_AD_EXPORT_TEMPLATE(T)                      \
     extern template DRJIT_AD_EXPORT struct DiffArray<T>;
#endif

#include <drjit/array.h>
#include <drjit-core/jit.h>

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)

// -----------------------------------------------------------------------
//! @{ \name External API compiled as part of libdrjit-autodiff.so
// -----------------------------------------------------------------------

/// Increase the external reference count of a given variable
template <typename Value> void ad_inc_ref_impl(uint32_t index) noexcept (true);

/// .. like above, but do nothing and return zero if AD is disabled for 'index'
template <typename Value> uint32_t ad_inc_ref_cond_impl(uint32_t index) noexcept (true);

/// Decrease the external reference count of a given variable
template <typename Value> void ad_dec_ref_impl(uint32_t index) noexcept (true);

/// Create a new variable with the given number of operands and AD weights
template <typename Value>
uint32_t ad_new(const char *label, size_t size, uint32_t ops = 0,
                uint32_t *indices = nullptr, Value *weights = nullptr);

/// Query the gradient associated with a variable
template <typename Value> Value ad_grad(uint32_t index, bool fail_if_missing);

/// Overwrite the gradient associated with a variable
template <typename Value>
void ad_set_grad(uint32_t index, const Value &v, bool fail_if_missing);

/// Accumulate gradients into a variable
template <typename Value>
void ad_accum_grad(uint32_t index, const Value &v, bool fail_if_missing);

/// Enqueue a variable for a subsequent ad_traverse() command
template <typename Value> void ad_enqueue(ADMode mode, uint32_t index);

/// Propagate derivatives through the enqueued set of edges
template <typename Value> void ad_traverse(ADMode mode, uint32_t flags);

/// Number of observed implicit dependencies
template <typename Value> size_t ad_implicit();

/// Extract implicit dependencies since 'snapshot' (obtained via ad_implicit())
template <typename Value> void ad_extract_implicit(size_t snapshot, uint32_t *out);

/// Enqueue implicit dependencies since 'snapshot' (obtained via ad_implicit())
template <typename Value> void ad_enqueue_implicit(size_t snapshot);

/// Dequeue implicit dependencies since 'snapshot' (obtained via ad_implicit())
template <typename Value> void ad_dequeue_implicit(size_t snapshot);

/// Selectively enable/disable differentiation for a subset of the variables
template <typename Value>
void ad_scope_enter(ADScope type, size_t size, const uint32_t *indices);

/// Leave a scope created by ad_scope_enter()
template <typename Value> void ad_scope_leave(bool process_postponed);

/// Check if a variable is suspended from derivative tracking
template <typename Value> bool ad_grad_enabled(uint32_t index);

/// Label a variable (useful for debugging via graphviz etc.)
template <typename Value> void ad_set_label(uint32_t index, const char *);

/// Return the label associated with a variable
template <typename Value> const char *ad_label(uint32_t index);

/// Generate a graphviz plot of all registered variables
template <typename Value> const char *ad_graphviz();

/// Special case of ad_new: create a node for a select() statement.
template <typename Value, typename Mask>
uint32_t ad_new_select(const char *label, size_t size, const Mask &m,
                       uint32_t t_index, uint32_t f_index);

/// Special case of ad_new: create a node for a gather() expression
template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather(const char *label, size_t size, uint32_t src_index,
                       const Index &offset, const Mask &mask, bool permute);

/// Special case of ad_new: create a node for a scatter[_reduce]() statement.
template <typename Value, typename Mask, typename Index>
uint32_t ad_new_scatter(const char *label, size_t size, ReduceOp op,
                        uint32_t src_index, uint32_t dst_index,
                        const Index &offset, const Mask &mask, bool permute);

/// Check if AD is enabled, and whether there are any AD-attaced variables
template <typename Value> bool ad_enabled() noexcept(true);

/// Custom graph edge for implementing custom differentiable operations
struct DRJIT_AD_EXPORT DiffCallback {
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual ~DiffCallback();
};

/// Register a custom/user-provided differentiable operation
template <typename Value>
void ad_add_edge(uint32_t src_index, uint32_t dst_index,
                 DiffCallback *callback = nullptr);

//! @}
// -----------------------------------------------------------------------

#if defined(__GNUC__)
template <typename T> DRJIT_INLINE void ad_inc_ref(uint32_t index) noexcept {
    if (!(__builtin_constant_p(index) && index == 0))
        ad_inc_ref_impl<T>(index);
}

template <typename T> DRJIT_INLINE uint32_t ad_inc_ref_cond(uint32_t index) noexcept {
    if (__builtin_constant_p(index) && index == 0)
        return 0;
    else
        return ad_inc_ref_cond_impl<T>(index);
}

template <typename T> DRJIT_INLINE void ad_dec_ref(uint32_t index) noexcept {
    if (!(__builtin_constant_p(index) && index == 0))
        ad_dec_ref_impl<T>(index);
}
#else
#define ad_inc_ref ad_inc_ref_impl
#define ad_inc_ref_cond ad_inc_ref_cond_impl
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
    static constexpr bool IsJIT = is_jit_v<Type_>;
    static constexpr bool IsCUDA = is_cuda_v<Type_>;
    static constexpr bool IsLLVM = is_llvm_v<Type_>;
    static constexpr bool IsDynamic = is_dynamic_v<Type_>;
    static constexpr bool IsEnabled = std::is_floating_point_v<scalar_t<Type_>>;

    template <typename T>
    using ReplaceValue = DiffArray<replace_scalar_t<Type_, scalar_t<T>>>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    DRJIT_INLINE DiffArray() = default;

    #if defined(_MSC_VER)
    // Static analysis may detect that ad_inc_ref/dec_ref are unnecessary in
    // some cases. Please don't warn about this..
    #  pragma warning(push)
    #  pragma warning(disable: 4702) // unreachable code
    #endif

    DRJIT_INLINE ~DiffArray() noexcept {
        if constexpr (IsEnabled)
            detail::ad_dec_ref<Type>(m_index);
    }

    DRJIT_INLINE DiffArray(const DiffArray &a) : m_value(a.m_value) {
        if constexpr (IsEnabled)
            m_index = detail::ad_inc_ref_cond<Type>(a.m_index);
    }

    DRJIT_INLINE DiffArray(DiffArray &&a) noexcept
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

    DRJIT_INLINE DiffArray &operator=(const DiffArray &a) {
        m_value = a.m_value;
        if constexpr (IsEnabled) {
            uint32_t old_index = m_index;
            m_index = detail::ad_inc_ref_cond<Type>(a.m_index);
            detail::ad_dec_ref<Type>(old_index);
        }
        return *this;
    }

    DRJIT_INLINE DiffArray &operator=(DiffArray &&a) {
        m_value = std::move(a.m_value);
        if constexpr (IsEnabled)
            std::swap(m_index, a.m_index);
        return *this;
    }

    #if defined(_MSC_VER)
    #  pragma warning(pop)
    #endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DiffArray add_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("add_(): invalid operand type!");
        } else {
            Type result = m_value + a.m_value;
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { 1, 1 };
                    index_new = detail::ad_new<Type>(
                        "add", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sub_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("sub_(): invalid operand type!");
        } else {
            Type result = m_value - a.m_value;
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { 1, -1 };
                    index_new = detail::ad_new<Type>(
                        "sub", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray mul_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("mul_(): invalid operand type!");
        } else {
            Type result = m_value * a.m_value;
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { a.m_value, m_value };
                    index_new = detail::ad_new<Type>(
                        "mul", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray div_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("div_(): invalid operand type!");
        } else {
            Type result = m_value / a.m_value;
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type rcp_a = rcp(a.m_value);
                    Type weights[2] = { rcp_a, -m_value * sqr(rcp_a) };
                    index_new = detail::ad_new<Type>(
                        "div", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray neg_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("neg_(): invalid operand type!");
        } else {
            Type result = -m_value;
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -1 };
                    index_new = detail::ad_new<Type>(
                        "neg", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("fmadd_(): invalid operand type!");
        } else {
            Type result = fmadd(m_value, a.m_value, b.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { a.m_value, m_value, 1 };
                    index_new = detail::ad_new<Type>(
                        "fmadd", width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("fmsub_(): invalid operand type!");
        } else {
            Type result = fmsub(m_value, a.m_value, b.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { a.m_value, m_value, -1 };
                    index_new = detail::ad_new<Type>(
                        "fmsub", width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("fnmadd_(): invalid operand type!");
        } else {
            Type result = fnmadd(m_value, a.m_value, b.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { -a.m_value, -m_value, 1 };
                    index_new = detail::ad_new<Type>(
                        "fnmadd", width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("fnmsub_(): invalid operand type!");
        } else {
            Type result = fnmsub(m_value, a.m_value, b.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { -a.m_value, -m_value, -1 };
                    index_new = detail::ad_new<Type>(
                        "fnmsub", width(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray abs_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("abs_(): invalid operand type!");
        } else {
            Type result = abs(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { sign(m_value) };
                    index_new = detail::ad_new<Type>(
                        "abs", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sqrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sqrt_(): invalid operand type!");
        } else {
            Type result = sqrt(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { .5f * rcp(result) };
                    index_new = detail::ad_new<Type>(
                        "sqrt", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray cbrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cbrt_(): invalid operand type!");
        } else {
            Type result = cbrt(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { (1 / 3.f) * sqr(rcp(result)) };
                    index_new = detail::ad_new<Type>(
                        "cbrt", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray erf_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("erf_(): invalid operand type!");
        } else {
            Type result = erf(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { (2.f * InvSqrtPi<Type>) * exp(-sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "erf", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rcp_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("rcp_(): invalid operand type!");
        } else {
            Type result = rcp(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -sqr(result) };
                    index_new = detail::ad_new<Type>(
                        "rcp", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rsqrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("rsqrt_(): invalid operand type!");
        } else {
            Type result = rsqrt(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    Type rsqrt_2 = sqr(result), rsqrt_3 = result * rsqrt_2;
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -.5f * rsqrt_3 };
                    index_new = detail::ad_new<Type>(
                        "rsqrt", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray minimum_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("minimum_(): invalid operand type!");
        } else {
            Type result = minimum(m_value, a.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    mask_t<Type> m = m_value <= a.m_value;
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { select(m, Type(1), Type(0)),
                                        select(m, Type(0), Type(1)) };
                    index_new = detail::ad_new<Type>(
                        "minimum", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray maximum_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("maximum_(): invalid operand type!");
        } else {
            Type result = maximum(m_value, a.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    mask_t<Type> m = m_value > a.m_value;
                    Type weights[2] = { select(m, Type(1), Type(0)),
                                        select(m, Type(0), Type(1)) };
                    index_new = detail::ad_new<Type>(
                        "maximum", width(result), 2, indices, weights);
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
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (t.m_index || f.m_index) {
                    index_new = detail::ad_new_select<Type>(
                        "select", width(result),
                        m.m_value, t.m_index, f.m_index);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray and_(const MaskType &mask) const {
        return select(mask, *this, DiffArray(Scalar(0)));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray or_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            const Scalar value = memcpy_cast<Scalar>(int_array_t<Scalar>(-1));
            if (m_index)
                return select(mask, DiffArray(value), *this);
        }
        return DiffArray::create(0, detail::or_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray xor_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
                drjit_raise("xor_(): operation not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray andnot_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
                drjit_raise("andnot_(): operation not permitted for "
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
            drjit_raise("sin_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    index_new = detail::ad_new<Type>("sin", width(s),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cos_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -s };
                    index_new = detail::ad_new<Type>("cos", width(c),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sincos_(): invalid operand type!");
        } else {
            auto [s, c] = sincos(m_value);
            uint32_t index_s = 0, index_c = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights_s[1] = { c }, weights_c[1] = { -s };
                    uint32_t w = (uint32_t) width(s);
                    index_s = detail::ad_new<Type>("sincos[s]", w, 1, indices,
                                                   weights_s);
                    index_c = detail::ad_new<Type>("sincos[c]", w, 1, indices,
                                                   weights_c);
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
            drjit_raise("csc_(): invalid operand type!");
        } else {
            Type result = csc(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -result * cot(m_value) };
                    index_new = detail::ad_new<Type>(
                        "csc", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sec_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sec_(): invalid operand type!");
        } else {
            Type result = sec(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { result * tan(m_value) };
                    index_new = detail::ad_new<Type>(
                        "sec", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray tan_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("tan_(): invalid operand type!");
        } else {
            Type result = tan(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { sqr(sec(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "tan", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray cot_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cot_(): invalid operand type!");
        } else {
            Type result = cot(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -sqr(csc(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "cot", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asin_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("asin_(): invalid operand type!");
        } else {
            Type result = asin(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt(fnmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "asin", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray acos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("acos_(): invalid operand type!");
        } else {
            Type result = acos(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -rsqrt(fnmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "acos", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atan_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("atan_(): invalid operand type!");
        } else {
            Type result = atan(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(fmadd(m_value, m_value, 1)) };
                    index_new = detail::ad_new<Type>(
                        "atan", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atan2_(const DiffArray &x) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("atan2_(): invalid operand type!");
        } else {
            Type result = atan2(m_value, x.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index || x.m_index) {
                    Type il2 = rcp(fmadd(m_value, m_value, sqr(x.m_value)));
                    uint32_t indices[2] = { m_index, x.m_index };
                    Type weights[2] = { il2 * x.m_value, -il2 * m_value };
                    index_new = detail::ad_new<Type>(
                        "atan2", width(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray exp_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("exp_(): invalid operand type!");
        } else {
            Type result = exp(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { result };
                    index_new = detail::ad_new<Type>(
                        "exp", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray exp2_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("exp2_(): invalid operand type!");
        } else {
            Type result = exp2(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { result * LogTwo<Value> };
                    index_new = detail::ad_new<Type>(
                        "exp2", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("log_(): invalid operand type!");
        } else {
            Type result = log(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(m_value) };
                    index_new = detail::ad_new<Type>(
                        "log", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray log2_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("log2_(): invalid operand type!");
        } else {
            Type result = log2(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp(m_value) * InvLogTwo<Value> };
                    index_new = detail::ad_new<Type>(
                        "log2", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sinh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sinh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    index_new = detail::ad_new<Type>("sinh", width(s),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(s));
        }
    }

    DiffArray cosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("cosh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { s };
                    index_new = detail::ad_new<Type>("cosh", width(c),
                                                     1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(c));
        }
    }

    std::pair<DiffArray, DiffArray> sincosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("sincosh_(): invalid operand type!");
        } else {
            auto [s, c] = sincosh(m_value);
            uint32_t index_s = 0, index_c = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights_s[1] = { c }, weights_c[1] = { s };
                    size_t w = width(s);
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
            drjit_raise("tanh_(): invalid operand type!");
        } else {
            Type result = tanh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { sqr(sech(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "tanh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray asinh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("asinh_(): invalid operand type!");
        } else {
            Type result = asinh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt((Scalar) 1 + sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "asinh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray acosh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("acosh_(): invalid operand type!");
        } else {
            Type result = acosh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rsqrt(sqr(m_value) - (Scalar) 1) };
                    index_new = detail::ad_new<Type>(
                        "acosh", width(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray atanh_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            drjit_raise("atanh_(): invalid operand type!");
        } else {
            Type result = atanh(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { rcp((Scalar) 1 - sqr(m_value)) };
                    index_new = detail::ad_new<Type>(
                        "atanh", width(result), 1, indices, weights);
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
            drjit_raise("mod_(): invalid operand type!");
        else
            return m_value % a.m_value;
    }

    DiffArray mulhi_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("mulhi_(): invalid operand type!");
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
            drjit_raise("not_(): invalid operand type!");
        else
            return DiffArray::create(0, ~m_value);
    }

    DiffArray or_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("or_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::or_(m_value, a.m_value));
    }

    DiffArray and_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("and_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::and_(m_value, a.m_value));
    }

    DiffArray xor_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("xor_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, a.m_value));
    }

    DiffArray andnot_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                drjit_raise("andnot_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::andnot_(m_value, a.m_value));
    }

    DiffArray floor_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("floor_(): invalid operand type!");
        else
            return DiffArray::create(0, floor(m_value));
    }

    DiffArray ceil_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("ceil_(): invalid operand type!");
        else
            return DiffArray::create(0, ceil(m_value));
    }

    DiffArray trunc_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("trunc_(): invalid operand type!");
        else
            return DiffArray::create(0, trunc(m_value));
    }

    DiffArray round_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("round_(): invalid operand type!");
        else
            return DiffArray::create(0, round(m_value));
    }

    template <typename T> T ceil2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("ceil2int_(): invalid operand type!");
        else
            return T::create(0, ceil2int<typename T::Type>(m_value));
    }

    template <typename T> T floor2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("floor2int_(): invalid operand type!");
        else
            return T::create(0, floor2int<typename T::Type>(m_value));
    }

    template <typename T> T round2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("round2int_(): invalid operand type!");
        else
            return T::create(0, round2int<typename T::Type>(m_value));
    }

    template <typename T> T trunc2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            drjit_raise("trunc2int_(): invalid operand type!");
        else
            return T::create(0, trunc2int<typename T::Type>(m_value));
    }

    DiffArray sl_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sl_(): invalid operand type!");
        else
            return DiffArray::create(0, m_value << a.m_value);
    }

    DiffArray sr_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sr_(): invalid operand type!");
        else
            return DiffArray::create(0, m_value >> a.m_value);
    }

    template <int Imm> DiffArray sl_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sl_(): invalid operand type!");
        else
            return DiffArray::create(0, sl<Imm>(m_value));
    }

    template <int Imm> DiffArray sr_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("sr_(): invalid operand type!");
        else
            return DiffArray::create(0, sr<Imm>(m_value));
    }

    DiffArray tzcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("tzcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, tzcnt(m_value));
    }

    DiffArray lzcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("lzcnt_(): invalid operand type!");
        else
            return DiffArray::create(0, lzcnt(m_value));
    }

    DiffArray popcnt_() const {
        if constexpr (!std::is_integral_v<Scalar>)
            drjit_raise("popcnt_(): invalid operand type!");
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
            drjit_raise("all_(): invalid operand type!");
        else
            return all(m_value);
    }

    bool any_() const {
        if constexpr (!is_mask_v<Type>)
            drjit_raise("any_(): invalid operand type!");
        else
            return any(m_value);
    }

    size_t count_() const {
        if constexpr (!is_mask_v<Type>)
            drjit_raise("count_(): invalid operand type!");
        else
            return count(m_value);
    }

    DiffArray sum_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("sum_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { 1 };
                    index_new = detail::ad_new<Type>(
                        "sum", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, sum(m_value));
        }
    }

    DiffArray prod_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("prod_(): invalid operand type!");
        } else {
            Type result = prod(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { select(eq(m_value, (Scalar) 0),
                                               (Scalar) 0, result / m_value) };
                    index_new = detail::ad_new<Type>(
                        "prod", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray min_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("min_(): invalid operand type!");
        } else {
            Type result = min(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the minimum , which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { select(
                        eq(m_value, result), Type(1), Type(0)) };
                    index_new = detail::ad_new<Type>(
                        "min", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray max_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            drjit_raise("max_(): invalid operand type!");
        } else {
            Type result = max(m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the maximum, which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { select(
                        eq(m_value, result), Type(1), Type(0)) };
                    index_new = detail::ad_new<Type>(
                        "max", 1, 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray dot_(const DiffArray &a) const {
        return sum(*this * a);
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
            drjit_raise("Array gather operation not supported for scalar array type.");
        } else {
            Type result = gather<Type>(src.m_value, offset.m_value, mask.m_value);
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (src.m_index)
                    index_new = detail::ad_new_gather<Type>(
                        Permute ? "gather[permute]" : "gather",
                        width(result), src.m_index, offset.m_value,
                        mask.m_value, Permute);
            }
            return create(index_new, std::move(result));
        }
    }

    template <bool Permute>
    void scatter_(DiffArray &dst, const IndexType &offset,
                  const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            (void) dst; (void) offset; (void) mask;
            drjit_raise("Array scatter operation not supported for scalar array type.");
        } else {
            scatter(dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index || (dst.m_index && !Permute)) {
                    uint32_t index = detail::ad_new_scatter<Type>(
                        Permute ? "scatter[permute]" : "scatter", width(dst),
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
            (void) op; (void) dst; (void) offset; (void) mask;
            drjit_raise("Array scatter_reduce operation not supported for scalar array type.");
        } else {
            scatter_reduce(op, dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index) { // safe to ignore dst.m_index in the case of scatter_reduce
                    uint32_t index = detail::ad_new_scatter<Type>(
                        "scatter_reduce", width(dst), op, m_index,
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
            drjit_raise("compress_(): invalid operand type!");
        else
            return uint32_array_t<ArrayType>::create(0, compress(m_value));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Standard initializers
    // -----------------------------------------------------------------------

    static DiffArray empty_(size_t size) {
        return drjit::empty<Type>(size);
    }

    static DiffArray zero_(size_t size) {
        return zeros<Type>(size);
    }

    static DiffArray full_(Value value, size_t size) {
        return full<Type>(value, size);
    }

    static DiffArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        return arange<Type>(start, stop, step);
    }

    static DiffArray linspace_(Value min, Value max, size_t size, bool endpoint) {
        return linspace<Type>(min, max, size, endpoint);
    }

    static DiffArray map_(void *ptr, size_t size, bool free = false) {
        DRJIT_MARK_USED(size);
        DRJIT_MARK_USED(free);
        DRJIT_MARK_USED(ptr);
        if constexpr (is_jit_v<Type>)
            return Type::map_(ptr, size, free);
        else
            drjit_raise("map_(): not supported in scalar mode!");
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
                uint32_t indices[1] = { m_index };
                Type weights[1] = { 1 };

                uint32_t index_new = detail::ad_new<Type>(
                    "copy", width(m_value), 1, indices, weights);

                return DiffArray::create(index_new, std::move(Type(m_value)));
            }
        }

        return *this;
    }

    auto vcall_() const {
        if constexpr (is_jit_v<Type>)
            return m_value.vcall_();
        else
            drjit_raise("vcall_(): not supported in scalar mode!");
    }

    DiffArray block_sum_(size_t block_size) {
        if constexpr (is_jit_v<Type>) {
            if (m_index)
                drjit_raise("block_sum_(): not supported for attached arrays!");
            return m_value.block_sum_(block_size);
        } else {
            DRJIT_MARK_USED(block_size);
            drjit_raise("block_sum_(): not supported in scalar mode!");
        }
    }

    static DiffArray steal(uint32_t index) {
        DRJIT_MARK_USED(index);
        if constexpr (is_jit_v<Type>)
            return Type::steal(index);
        else
            drjit_raise("steal(): not supported in scalar mode!");
    }

    static DiffArray borrow(uint32_t index) {
        DRJIT_MARK_USED(index);
        if constexpr (is_jit_v<Type>)
            return Type::borrow(index);
        else
            drjit_raise("borrow(): not supported in scalar mode!");
    }

    bool grad_enabled_() const {
        if constexpr (IsEnabled) {
            if (!m_index)
                return false;
            else
                return detail::ad_grad_enabled<Type>(m_index);
        } else {
            return false;
        }
    }

    void set_grad_enabled_(bool value) {
        DRJIT_MARK_USED(value);
        if constexpr (IsEnabled) {
            if (value) {
                if (m_index)
                    return;
                m_index = detail::ad_new<Type>(nullptr, width(m_value));
                if constexpr (is_jit_v<Type>) {
                    const char *label = m_value.label_();
                    if (label)
                        detail::ad_set_label<Type>(m_index, label);
                }
            } else {
                if (m_index == 0)
                    return;
                detail::ad_dec_ref<Type>(m_index);
                m_index = 0;
            }
        }
    }

    DiffArray migrate_(AllocType type) const {
        DRJIT_MARK_USED(type);
        if constexpr (is_jit_v<Type_>)
            return m_value.migrate_(type);
        else
            return *this;
    }

    bool schedule_() const {
        if constexpr (is_jit_v<Type_>)
            return m_value.schedule_();
        else
            return false;
    }

    bool eval_() const {
        if constexpr (is_jit_v<Type_>)
            return m_value.eval_();
        else
            return false;
    }

    void enqueue_(ADMode mode) const {
        DRJIT_MARK_USED(mode);
        if constexpr (IsEnabled)
            detail::ad_enqueue<Type>(mode, m_index);
    }

    static void traverse_(ADMode mode, uint32_t flags) {
        DRJIT_MARK_USED(flags);
        if constexpr (IsEnabled)
            detail::ad_traverse<Type>(mode, flags);
    }

    void set_label_(const char *label) {
        set_label(m_value, label);

        if constexpr (IsEnabled) {
            if (m_index)
                detail::ad_set_label<Type>(m_index, label);
        }
    }

    const char *label_() const {
        const char *result = nullptr;
        if constexpr (IsEnabled) {
            if (m_index)
                result = detail::ad_label<Type>(m_index);
        }
        if constexpr (is_jit_v<Type>) {
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
        DRJIT_MARK_USED(fail_if_missing);
        if constexpr (IsEnabled)
            return detail::ad_grad<Type>(m_index, fail_if_missing);
        else
            return zeros<Type>();
    }

    void set_grad_(const Type &value, bool fail_if_missing = false) {
        DRJIT_MARK_USED(value);
        DRJIT_MARK_USED(fail_if_missing);
        if constexpr (IsEnabled)
            detail::ad_set_grad<Type>(m_index, value, fail_if_missing);
    }

    void accum_grad_(const Type &value, bool fail_if_missing = false) {
        DRJIT_MARK_USED(value);
        DRJIT_MARK_USED(fail_if_missing);
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
        DRJIT_MARK_USED(offset);
        if constexpr (std::is_scalar_v<Type>)
            return m_value;
        else
            return m_value.entry(offset);
    }

    void set_entry(size_t offset, Value value) {
        if (m_index)
            drjit_raise("Attempted to overwrite entries of a variable that is "
                        "attached to the AD graph. This is not allowed.");

        if constexpr (is_dynamic_v<Type_>) {
            m_value.set_entry(offset, value);
        } else {
            DRJIT_MARK_USED(offset);
#if !defined(NDEBUG) && !defined(DRJIT_DISABLE_RANGE_CHECK)
            if (offset != 0)
                drjit_raise("Out of range access (tried to access index %u in "
                            "an array of size 1)", offset);
#endif
            m_value = value;
        }
    }

    void resize(size_t size) {
        DRJIT_MARK_USED(size);
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

    static DiffArray create(uint32_t index, Type&& value) {
        DiffArray result;
        result.m_index = index;
        result.m_value = std::move(value);
        return result;
    }

    static DiffArray create_borrow(uint32_t index, const Type &value) {
        DiffArray result;
        result.m_index = index;
        result.m_value = value;
        if constexpr (IsEnabled)
            detail::ad_inc_ref<Type>(index);
        return result;
    }

    void init_(size_t size) {
        DRJIT_MARK_USED(size);
        if constexpr (is_dynamic_v<Type>)
            m_value.init_(size);
    }

    bool is_literal() const {
        if constexpr (is_jit_v<Type>)
            return m_value.is_literal();
        else
            drjit_raise("is_literal(): expected a JIT array type");
    }

    bool is_evaluated() const {
        if constexpr (is_jit_v<Type>)
            return m_value.is_evaluated();
        else
            drjit_raise("is_evaluated(): expected a JIT array type");
    }

    uint32_t index() const {
        if constexpr (is_jit_v<Type>)
            return m_value.index();
        else
            drjit_raise("index(): expected a JIT array type");
    }

    uint32_t* index_ptr() {
        if constexpr (is_jit_v<Type>)
            return m_value.index_ptr();
        else
            drjit_raise("index_ptr(): expected a JIT array type");
    }

    uint32_t index_ad() const { return m_index; }
    uint32_t* index_ad_ptr() { return &m_index; }

    //! @}
    // -----------------------------------------------------------------------

protected:
    Type m_value {};
    uint32_t m_index = 0;
};

#define DRJIT_DECLARE_EXTERN_TEMPLATE(T, Mask, Index)                          \
    DRJIT_AD_EXPORT_TEMPLATE(T)                                                \
    namespace detail {                                                         \
    extern template DRJIT_AD_EXPORT void                                       \
        ad_inc_ref_impl<T>(uint32_t) noexcept (true);                          \
    extern template DRJIT_AD_EXPORT uint32_t                                   \
        ad_inc_ref_cond_impl<T>(uint32_t) noexcept (true);                     \
    extern template DRJIT_AD_EXPORT void                                       \
        ad_dec_ref_impl<T>(uint32_t) noexcept (true);                          \
    extern template DRJIT_AD_EXPORT uint32_t ad_new<T>(const char *, size_t,   \
                                                       uint32_t, uint32_t *,   \
                                                       T *);                   \
    extern template DRJIT_AD_EXPORT T ad_grad<T>(uint32_t, bool);              \
    extern template DRJIT_AD_EXPORT void ad_set_grad<T>(uint32_t, const T &,   \
                                                        bool);                 \
    extern template DRJIT_AD_EXPORT void ad_accum_grad<T>(uint32_t, const T &, \
                                                          bool);               \
    extern template DRJIT_AD_EXPORT void ad_set_label<T>(uint32_t,             \
                                                         const char *);        \
    extern template DRJIT_AD_EXPORT const char *ad_label<T>(uint32_t);         \
    extern template DRJIT_AD_EXPORT const char *ad_graphviz<T>();              \
    extern template DRJIT_AD_EXPORT void ad_enqueue<T>(ADMode, uint32_t);      \
    extern template DRJIT_AD_EXPORT void ad_traverse<T>(ADMode, uint32_t);     \
    extern template DRJIT_AD_EXPORT uint32_t ad_new_select<T, Mask>(           \
        const char *, size_t, const Mask &, uint32_t, uint32_t);               \
    extern template DRJIT_AD_EXPORT uint32_t ad_new_gather<T, Mask, Index>(    \
        const char *, size_t, uint32_t, const Index &, const Mask &, bool);    \
    extern template DRJIT_AD_EXPORT uint32_t ad_new_scatter<T, Mask, Index>(   \
        const char *, size_t, ReduceOp, uint32_t, uint32_t, const Index &,     \
        const Mask &, bool);                                                   \
    extern template DRJIT_AD_EXPORT void ad_add_edge<T>(uint32_t, uint32_t,    \
                                                        DiffCallback *);       \
    extern template DRJIT_AD_EXPORT size_t ad_implicit<T>();                   \
    extern template DRJIT_AD_EXPORT void ad_extract_implicit<T>(size_t,        \
                                                                uint32_t *);   \
    extern template DRJIT_AD_EXPORT void                                       \
    ad_scope_enter<T>(ADScope, size_t, const uint32_t *);                      \
    extern template DRJIT_AD_EXPORT void ad_scope_leave<T>(bool);              \
    extern template DRJIT_AD_EXPORT bool ad_grad_enabled<T>(uint32_t);         \
    extern template DRJIT_AD_EXPORT bool ad_enabled<T>();                      \
    }

DRJIT_DECLARE_EXTERN_TEMPLATE(float,  bool, uint32_t)
DRJIT_DECLARE_EXTERN_TEMPLATE(double, bool, uint32_t)

#if defined(DRJIT_H)
DRJIT_DECLARE_EXTERN_TEMPLATE(CUDAArray<float>,  CUDAArray<bool>, CUDAArray<uint32_t>)
DRJIT_DECLARE_EXTERN_TEMPLATE(CUDAArray<double>, CUDAArray<bool>, CUDAArray<uint32_t>)
DRJIT_DECLARE_EXTERN_TEMPLATE(LLVMArray<float>,  LLVMArray<bool>, LLVMArray<uint32_t>)
DRJIT_DECLARE_EXTERN_TEMPLATE(LLVMArray<double>, LLVMArray<bool>, LLVMArray<uint32_t>)
#endif

extern DRJIT_AD_EXPORT const char *ad_whos();
extern DRJIT_AD_EXPORT void ad_prefix_push(const char *value);
extern DRJIT_AD_EXPORT void ad_prefix_pop();

NAMESPACE_END(drjit)

#if defined(DRJIT_VCALL_H)
#  include <drjit/vcall_autodiff.h>
#endif
