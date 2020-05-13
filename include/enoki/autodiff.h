/*
    enoki/autodiff.h -- Forward/reverse-mode automatic differentiation wrapper

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once
#define ENOKI_AUTODIFF_H

#include <enoki/array.h>
#include <enoki-jit/jit.h>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

// -----------------------------------------------------------------------
//! @{ \name External API compiled as part of libenoki-ad.so
// -----------------------------------------------------------------------
/// Increase the external reference count of a given variable
template <typename Value> void ad_inc_ref(uint32_t index);

/// Decrease the external reference count of a given variable
template <typename Value> void ad_dec_ref(uint32_t index);

/// Create a new variable with the given number of operands and AD weights
template <typename Value> uint32_t ad_new(const char *label, uint32_t size, uint32_t ops,
                                          const uint32_t *indices, Value *weights);

/// Query the gradient associated with a variable
template <typename Value> Value ad_grad(uint32_t index);

/// Overwrite the gradient associated with a variable
template <typename Value> void ad_set_grad(uint32_t index, const Value &v);

/// Schedule a variable and its dependencies
template <typename Value> void ad_schedule(uint32_t index, bool reverse);

/// Perform a forward or reverse mode traversal of scheduled variables
template <typename Value> void ad_traverse(bool reverse, bool retain_graph);

/// Label a variable (useful for debugging via graphviz etc.)
template <typename Value> void ad_set_label(uint32_t index, const char *);

/// Return the label associated with a variable
template <typename Value> const char *ad_label(uint32_t index);

/// Generate a graphviz plot of the subgraph specified via ad_schedule()
template <typename Value> const char *ad_graphviz();

/// Special case of ad_new: create a node for a select() statement.
template <typename Value, typename Mask>
uint32_t ad_new_select(const char *label, uint32_t size, const Mask &m,
                       uint32_t t_index, uint32_t f_index);

/// Special case of ad_new: create a node for a gather() expression
template <typename Value, typename Mask, typename Index>
uint32_t ad_new_gather(const char *label, uint32_t size, uint32_t src_index,
                       const Index &offset, const Mask &mask, bool permute);


/// Special case of ad_new: create a node for a scatter[_add]() statement.
template <typename Value, typename Mask, typename Index>
uint32_t ad_new_scatter(const char *label, uint32_t size, uint32_t src_index,
                        uint32_t dst_index, const Index &offset,
                        const Mask &mask, bool permute, bool scatter_add);

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(detail)

template <typename Type_>
struct DiffArray : ArrayBaseT<value_t<Type_>, is_mask_v<Type_>, DiffArray<Type_>> {
    static_assert(std::is_scalar_v<Type_> || (is_dynamic_array_v<Type_> && array_depth_v<Type_> == 1),
                  "DiffArray template parameter must either be a scalar (e.g. "
                  "float) or a dynamic array of depth 1.");
    using Base = ArrayBaseT<value_t<Type_>, is_mask_v<Type_>, DiffArray<Type_>>;

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

    ENOKI_INLINE ~DiffArray() {
        if constexpr (IsEnabled)
            detail::ad_dec_ref<Type>(m_index);
    }

    ENOKI_INLINE DiffArray(const DiffArray &a) : m_value(a.m_value) {
        if constexpr (IsEnabled) {
            m_index = a.m_index;
            detail::ad_inc_ref<Type>(m_index);
        }
    }

    ENOKI_INLINE DiffArray(DiffArray &&a) noexcept : m_value(std::move(a.m_value)) {
        if constexpr (IsEnabled) {
            m_index = a.m_index;
            a.m_index = 0;
        }
    }

    template <typename T>
    DiffArray(const DiffArray<T> &v) : m_value(v.m_value) { }

    template <typename T>
    DiffArray(const DiffArray<T> &v, detail::reinterpret_flag)
        : m_value(v.m_value) { }

    DiffArray(const Type &value) : m_value(value) { }
    DiffArray(Type &&value) : m_value(std::move(value)) { }

    template <typename T = Value, enable_if_t<!std::is_same_v<T, Type>> = 0>
    DiffArray(const Value &value) : m_value(value) { }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              (!std::is_same_v<Ts, detail::reinterpret_flag> && ...))> = 0>
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
            uint32_t index_new = 0;
            Type result = m_value + a.m_value;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { 1.f, 1.f };
                    index_new = detail::ad_new<Type>(
                        "add", (uint32_t) slices(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sub_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("sub_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = m_value - a.m_value;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { 1.f, -1.f };
                    index_new = detail::ad_new<Type>(
                        "sub", (uint32_t) slices(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray mul_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("mul_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = m_value * a.m_value;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type weights[2] = { a.m_value, m_value };
                    index_new = detail::ad_new<Type>(
                        "mul", (uint32_t) slices(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray div_(const DiffArray &a) const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("div_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = m_value / a.m_value;
            if constexpr (IsEnabled) {
                if (m_index || a.m_index) {
                    uint32_t indices[2] = { m_index, a.m_index };
                    Type rcp_a = rcp(a.m_value);
                    Type weights[2] = { rcp_a, -m_value * sqr(rcp_a) };
                    index_new = detail::ad_new<Type>(
                        "div", (uint32_t) slices(result), 2, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray neg_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("neg_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = -m_value;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -1.f };
                    index_new = detail::ad_new<Type>(
                        "neg", (uint32_t) slices(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fmadd_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = fmadd(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { a.m_value, m_value, 1.f };
                    index_new = detail::ad_new<Type>(
                        "fmadd", (uint32_t) slices(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fmsub_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = fmsub(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { a.m_value, m_value, -1.f };
                    index_new = detail::ad_new<Type>(
                        "fmsub", (uint32_t) slices(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmadd_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fnmadd_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = fnmadd(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { -a.m_value, -m_value, 1.f };
                    index_new = detail::ad_new<Type>(
                        "fnmadd", (uint32_t) slices(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray fnmsub_(const DiffArray &a, const DiffArray &b) const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("fnmsub_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = fnmsub(m_value, a.m_value, b.m_value);
            if constexpr (IsEnabled) {
                if (m_index || a.m_index || b.m_index) {
                    uint32_t indices[3] = { m_index, a.m_index, b.m_index };
                    Type weights[3] = { -a.m_value, -m_value, -1.f };
                    index_new = detail::ad_new<Type>(
                        "fnmsub", (uint32_t) slices(result), 3, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray abs_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("abs_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = enoki::abs(m_value);
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { sign(m_value) };
                    index_new = detail::ad_new<Type>(
                        "abs", (uint32_t) slices(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray sqrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("sqrt_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = enoki::sqrt(m_value);
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { .5f / result };
                    index_new = detail::ad_new<Type>(
                        "sqrt", (uint32_t) slices(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rcp_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("rcp_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = enoki::rcp(m_value);
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -sqr(result) };
                    index_new = detail::ad_new<Type>(
                        "rcp", (uint32_t) slices(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray rsqrt_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("rsqrt_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = enoki::rsqrt(m_value);
            if constexpr (IsEnabled) {
                if (m_index) {
                    Type rsqrt_2 = sqr(result), rsqrt_3 = result * rsqrt_2;
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -.5f * rsqrt_3 };
                    index_new = detail::ad_new<Type>(
                        "rsqrt", (uint32_t) slices(result), 1, indices, weights);
                }
            }
            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray min_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>) {
            enoki_raise("min_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = enoki::min(m_value, a.m_value);

            if constexpr (IsEnabled) {
                if (m_index || a.m_value) {
                    mask_t<Type> m = m_value <= a.m_value;
                    uint32_t indices[2] = { m_index, a.m_value };
                    Type weights[2] = { select(m, Type(1), Type(0)),
                                        select(m, Type(0), Type(1)) };
                    index_new = detail::ad_new<Type>(
                        "min", (uint32_t) slices(result), 2, indices, weights);
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    DiffArray max_(const DiffArray &a) const {
        if constexpr (!std::is_integral_v<Scalar>) {
            enoki_raise("max_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = enoki::max(m_value, a.m_value);
            if constexpr (IsEnabled) {
                if (m_index || a.m_value) {
                    uint32_t indices[2] = { m_index, a.m_value };
                    mask_t<Type> m = m_value > a.m_value;
                    Type weights[2] = { select(m, Type(1), Type(0)),
                                        select(m, Type(0), Type(1)) };
                    index_new = detail::ad_new<Type>(
                        "max", (uint32_t) slices(result), 2, indices, weights);
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
            Type result = enoki::select(m.m_value, t.m_value, f.m_value);
            uint32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (t.m_index || f.m_index) {
                    if (m.m_value.is_literal_one()) {
                        return t;
                    } else if (m.m_value.is_literal_zero()) {
                        return f;
                    } else {
                        index_new = detail::ad_new_select<Type>(
                            "select", (uint32_t) slices(result),
                            m.m_value, t.m_index, f.m_index);
                    }
                }
            }

            return DiffArray::create(index_new, std::move(result));
        }
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray and_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
                return enoki::select(mask, *this, DiffArray(0));
        }
        return DiffArray::create(0, detail::and_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray or_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            const Scalar value = memcpy_cast<Scalar>(int_array_t<Scalar>(-1));
            if (m_index)
                return enoki::select(mask, DiffArray(value), *this);
        }
        return DiffArray::create(0, detail::or_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray xor_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
                enoki_raise("xor_(): operation not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, mask.m_value));
    }

    template <typename T = Type_, enable_if_t<!is_mask_v<T>> = 0>
    DiffArray andnot_(const MaskType &mask) const {
        if constexpr (IsEnabled) {
            if (m_index)
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
            if constexpr (IsEnabled) {
                if (m_index) {
                    auto [s, c] = sincos(m_value);

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { std::move(c) };
                    uint32_t index_new = detail::ad_new<Type>(
                        "sin", (uint32_t) slices(s), 1, indices, weights);

                    return DiffArray::create(index_new, std::move(s));
                }
            }

            return DiffArray::create(0, enoki::sin(m_value));
        }
    }

    DiffArray cos_() const {
        if constexpr (!std::is_floating_point_v<Scalar>) {
            enoki_raise("cos_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index) {
                    auto [s, c] = sincos(m_value);

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { -s };
                    uint32_t index_new = detail::ad_new<Type>(
                        "cos", (uint32_t) slices(c), 1, indices, weights);

                    return DiffArray::create(index_new, std::move(c));
                }
            }

            return DiffArray::create(0, enoki::cos(m_value));
        }
    }

    std::pair<DiffArray, DiffArray> sincos_() const {
        return { sin_(), cos_() };
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
            if (m_index || a.m_index)
                enoki_raise("or_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::or_(m_value, a.m_value));
    }

    DiffArray and_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                enoki_raise("and_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::and_(m_value, a.m_value));
    }

    DiffArray xor_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                enoki_raise("xor_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::xor_(m_value, a.m_value));
    }

    DiffArray andnot_(const DiffArray &a) const {
        if constexpr (is_floating_point_v<Scalar>) {
            if (m_index || a.m_index)
                enoki_raise("andnot_(): bit operations are not permitted for "
                            "floating point arrays attached to the AD graph!");
        }
        return DiffArray::create(0, detail::andnot_(m_value, a.m_value));
    }

    DiffArray floor_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("floor_(): invalid operand type!");
        else
            return DiffArray::create(0, enoki::floor(m_value));
    }

    DiffArray ceil_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("ceil_(): invalid operand type!");
        else
            return DiffArray::create(0, enoki::ceil(m_value));
    }

    DiffArray trunc_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("trunc_(): invalid operand type!");
        else
            return DiffArray::create(0, enoki::trunc(m_value));
    }

    DiffArray round_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("round_(): invalid operand type!");
        else
            return DiffArray::create(0, enoki::round(m_value));
    }

    template <typename T> T ceil2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("ceil2int_(): invalid operand type!");
        else
            return T::create(0, ceil2int<T::Type>(m_value));
    }

    template <typename T> T floor2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("floor2int_(): invalid operand type!");
        else
            return T::create(0, floor2int<T::Type>(m_value));
    }

    template <typename T> T round2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("round2int_(): invalid operand type!");
        else
            return T::create(0, round2int<T::Type>(m_value));
    }

    template <typename T> T trunc2int_() const {
        if constexpr (!std::is_floating_point_v<Scalar>)
            enoki_raise("trunc2int_(): invalid operand type!");
        else
            return T::create(0, trunc2int<T::Type>(m_value));
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
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { 1.f };
                    index_new = detail::ad_new<Type>(
                        "hsum_async", 1, 1, indices, weights);
                }
            }

            return DiffArray::create(index_new, enoki::hsum_async(m_value));
        }
    }

    Value hsum_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hsum_(): invalid operand type!");
        } else {
            if constexpr (IsEnabled) {
                if (m_index)
                    enoki_raise("hsum_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hsum_async() instead, which "
                                "returns a differentiable array.");
            }
            return enoki::hsum(m_value);
        }
    }

    DiffArray hprod_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hprod_async_(): invalid operand type!");
        } else {
            uint32_t index_new = 0;
            Type result = hprod_async(m_value);
            if constexpr (IsEnabled) {
                if (m_index) {
                    uint32_t indices[1] = { m_index };
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
                if (m_index)
                    enoki_raise("hprod_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hprod_async() instead, which "
                                "returns a differentiable array.");
            }
            return enoki::hprod(m_value);
        }
    }

    DiffArray hmin_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hmin_async_(): invalid operand type!");
        } else {
            Type result = enoki::hmin_async(m_value);
            uint32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the minimum , which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */
                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { enoki::select(
                        enoki::eq(m_value, result), Type(1), Type(0)) };
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
                if (m_index)
                    enoki_raise("hmin_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hmin_async() instead, which "
                                "returns a differentiable array.");
            }
            return enoki::hmin(m_value);
        }
    }

    DiffArray hmax_async_() const {
        if constexpr (!std::is_arithmetic_v<Scalar>) {
            enoki_raise("hmax_async_(): invalid operand type!");
        } else {
            Type result = enoki::hmax_async(m_value);
            uint32_t index_new = 0;

            if constexpr (IsEnabled) {
                if (m_index) {
                    /* This gradient has duplicate '1' entries when
                       multiple entries are equal to the maximum, which is
                       strictly speaking not correct (but getting this right
                       would make the operation quite a bit more expensive). */

                    uint32_t indices[1] = { m_index };
                    Type weights[1] = { enoki::select(
                        enoki::eq(m_value, result), Type(1), Type(0)) };
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
                if (m_index)
                    enoki_raise("hmax_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use hmax_async() instead, which "
                                "returns a differentiable array.");
            }
            return enoki::hmax(m_value);
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
                if (m_index)
                    enoki_raise("dot_(): operation returns a detached scalar, "
                                "which is not permitted for arrays attached to "
                                "the AD graph! Use dot_async() instead, which "
                                "returns a differentiable array.");
            }
            return enoki::dot(m_value, a.m_value);
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
            uint32_t index_new = 0;
            if constexpr (IsEnabled) {
                if (src.m_index)
                    index_new = detail::ad_new_gather<Type>(
                        Permute ? "gather[permute]" : "gather",
                        (uint32_t) slices(result), src.m_index, offset.m_value,
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
            enoki::scatter(dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index || dst.m_index) {
                    uint32_t index = detail::ad_new_scatter<Type>(
                        Permute ? "scatter[permute]" : "scatter", slices(dst),
                        m_index, dst.m_index, offset.m_value, mask.m_value,
                        Permute, false);
                    detail::ad_dec_ref<Type>(dst.m_index);
                    dst.m_index = index;
                }
            }
        }
    }

    void scatter_add_(DiffArray &dst, const IndexType &offset,
                  const MaskType &mask = true) const {
        if constexpr (std::is_scalar_v<Type>) {
            enoki_raise("Array scatter_add operation not supported for scalar array type.");
        } else {
            enoki::scatter_add(dst.m_value, m_value, offset.m_value, mask.m_value);
            if constexpr (IsEnabled) {
                if (m_index) { // safe to ignore dst.m_index in the case of scatter_add
                    uint32_t index = detail::ad_new_scatter<Type>(
                        "scatter_add", slices(dst), m_index, dst.m_index,
                        offset.m_value, mask.m_value, false, true);
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
        enoki::scatter(dst, m_value, offset.m_value, mask.m_value);
    }

    void scatter_add_(void *dst, const IndexType &offset,
                      const MaskType &mask = true) const {
        enoki::scatter_add(dst, m_value, offset.m_value, mask.m_value);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Standard initializers
    // -----------------------------------------------------------------------

    static DiffArray empty_(size_t size) {
        return empty<Type>(size);
    }

    static DiffArray zero_(size_t size) {
        return zero<Type>(size);
    }

    static DiffArray full_(Value value, size_t size) {
        return full<Type>(value, size);
    }

    static DiffArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        return arange<Type>(start, stop, step);
    }

    static DiffArray linspace_(Value min, Value max, size_t size) {
        return linspace<Type>(min, max, size);
    }

    // static CUDAArray map_(void *ptr, size_t size, bool free = false) {
    //     return map<Type>(ptr, size, free);
    // }

    static DiffArray load_unaligned_(const void *ptr, size_t size) {
        return load_unaligned<Type>(ptr, size);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    void requires_grad(bool value = true) {
        (void) value;
        if constexpr (IsEnabled) {
            if (m_index) {
                if (!value) {
                    detail::ad_dec_ref<Type>(m_index);
                    m_index = 0;
                }
            } else {
                if (value)
                    m_index = detail::ad_new<Type>(nullptr, (uint32_t) slices(m_value),
                                                   0, nullptr, (Type *) nullptr);
            }
        }
    }

    void set_entry(uint32_t offset, Value value) {
        if constexpr (is_jit_array_v<Type_>) {
            if (m_index)
                enoki_raise(
                    "Attempted to overwrite entries of a variable that is "
                    "attached to the AD graph. This is not allowed.");
            m_value.set_entry(offset, value);
        } else {
            #if !defined(NDEBUG) && !defined(ENOKI_DISABLE_RANGE_CHECK)
            if (offset != 0)
                enoki_raise("Out of range access (tried to access index %u in "
                            "an array of size 1)", offset);
            #endif
            m_value = value;
        }
    }

    void migrate(AllocType type) {
        if constexpr (is_jit_array_v<Type_>)
            m_value.migrate(type);
    }

    void schedule() const {
        if constexpr (is_jit_array_v<Type_>)
            m_value.schedule();
    }

    void ad_schedule(bool reverse) const {
        ENOKI_MARK_USED(reverse);
        if constexpr (IsEnabled)
            enoki::detail::ad_schedule<Type>(m_index, reverse);
    }

    static void traverse(bool reverse, bool retain_graph) {
        ENOKI_MARK_USED(reverse);
        if constexpr (IsEnabled)
            enoki::detail::ad_traverse<Type>(reverse, retain_graph);
    }

    void set_label(const char *label) const {
        if constexpr (IsEnabled)
            enoki::detail::ad_set_label<Type>(m_index, label);
        enoki::set_label(m_value, label);
    }

    const char *label() const {
        if constexpr (IsEnabled) {
            if (m_index)
                enoki::detail::ad_label<Type>(m_index);
        }
        if constexpr (is_jit_array_v<Type>)
            return m_value.label();
        return nullptr;
    }

    static const char *graphviz_() {
        if constexpr (IsEnabled)
            return enoki::detail::ad_graphviz<Type>();
    }

    const Type value() const {
        return m_value;
    }

    const Type grad() const {
        if constexpr (IsEnabled)
            return detail::ad_grad<Type>(m_index);
        else
            return Type();
    }

    void set_grad(const Type &value) {
        if constexpr (IsEnabled)
            detail::ad_set_grad<Type>(m_index, value);
        else
            enoki_raise("set_grad(): gradients not enabled for this type!");
    }

    void set_label(const char *label) {
        enoki::set_label(m_value, label);

        if constexpr (IsEnabled) {
            if (m_index)
                detail::ad_set_label<Type>(m_index, label);
        }
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

    void init_(size_t size) {
        if constexpr (is_dynamic_v<Type>)
            m_value.init_(size);
    }

    uint32_t index() const { return m_index; }

    //! @}
    // -----------------------------------------------------------------------

protected:
    Type m_value {};
    uint32_t m_index = 0;
};

#if defined(ENOKI_BUILD_AUTODIFF)
#  define ENOKI_AUTODIFF_EXPORT ENOKI_EXPORT
#else
#  define ENOKI_AUTODIFF_EXPORT ENOKI_IMPORT
#endif

#define ENOKI_DECLARE_EXTERN_TEMPLATE(T, Mask, Index)                          \
    extern template ENOKI_AUTODIFF_EXPORT void ad_inc_ref<T>(uint32_t);        \
    extern template ENOKI_AUTODIFF_EXPORT void ad_dec_ref<T>(uint32_t);        \
    extern template ENOKI_AUTODIFF_EXPORT uint32_t ad_new<T>(                  \
        const char *, uint32_t, uint32_t,                                      \
        ENOKI_AUTODIFF_EXPORT const uint32_t *, T *);                          \
    extern template ENOKI_AUTODIFF_EXPORT T ad_grad<T>(uint32_t);              \
    extern template ENOKI_AUTODIFF_EXPORT void ad_set_grad<T>(uint32_t,        \
                                                              const T &);      \
    extern template ENOKI_AUTODIFF_EXPORT void ad_set_label<T>(uint32_t,       \
                                                               const char *);  \
    extern template ENOKI_AUTODIFF_EXPORT const char *ad_label<T>(uint32_t);   \
    extern template ENOKI_AUTODIFF_EXPORT const char *ad_graphviz<T>();        \
    extern template ENOKI_AUTODIFF_EXPORT void ad_schedule<T>(uint32_t, bool); \
    extern template ENOKI_AUTODIFF_EXPORT void ad_traverse<T>(bool, bool);     \
    extern template ENOKI_AUTODIFF_EXPORT uint32_t ad_new_select<T, Mask>(     \
        const char *, uint32_t, const Mask &, uint32_t, uint32_t);             \
    extern template ENOKI_AUTODIFF_EXPORT uint32_t                             \
    ad_new_gather<T, Mask, Index>(const char *, uint32_t, uint32_t,            \
                                  const Index &, const Mask &, bool);          \
    extern template ENOKI_AUTODIFF_EXPORT uint32_t                             \
    ad_new_scatter<T, Mask, Index>(const char *, uint32_t, uint32_t, uint32_t, \
                                  const Index &, const Mask &, bool, bool);

NAMESPACE_BEGIN(detail)
ENOKI_DECLARE_EXTERN_TEMPLATE(float, bool, uint32_t)
ENOKI_DECLARE_EXTERN_TEMPLATE(double, bool, uint32_t)
#if defined(ENOKI_CUDA_H)
ENOKI_DECLARE_EXTERN_TEMPLATE(CUDAArray<float>, CUDAArray<bool>, CUDAArray<uint32_t>)
ENOKI_DECLARE_EXTERN_TEMPLATE(CUDAArray<double>, CUDAArray<bool>, CUDAArray<uint32_t>)
#endif
#if defined(ENOKI_LLVM_H)
ENOKI_DECLARE_EXTERN_TEMPLATE(LLVMArray<float>, LLVMArray<bool>, LLVMArray<uint32_t>)
ENOKI_DECLARE_EXTERN_TEMPLATE(LLVMArray<double>, LLVMArray<bool>, LLVMArray<uint32_t>)
#endif
NAMESPACE_END(detail)

extern ENOKI_AUTODIFF_EXPORT const char *ad_whos();
extern ENOKI_AUTODIFF_EXPORT void ad_prefix_push(const char *value);
extern ENOKI_AUTODIFF_EXPORT void ad_prefix_pop();
NAMESPACE_END(enoki)
