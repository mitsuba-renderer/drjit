/*
    enoki/array_static.h -- Base class of all variants of static arrays

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array_base.h>

NAMESPACE_BEGIN(enoki)

namespace detail {
    /// Compute binary OR of 'i' with right-shifted versions
    static constexpr size_t fill(size_t i) {
        return i != 0 ? i | fill(i >> 1) : 0;
    }

    /// Find the largest power of two smaller than 'i'
    static constexpr size_t lpow2(size_t i) {
        return i != 0 ? (fill(i-1) >> 1) + 1 : 0;
    }

    /// Compile-time integer logarithm
    static constexpr size_t clog2i(size_t value) {
        return (value > 1) ? 1 + clog2i(value >> 1) : 0;
    }
}

template <typename Value_, size_t Size_, bool IsMask_, typename Derived_>
struct StaticArrayBase : ArrayBaseT<Value_, Derived_> {
    using Base = ArrayBaseT<Value_, Derived_>;
    using typename Base::Derived;
    using typename Base::Value;
    using typename Base::Scalar;
    using Base::derived;

    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations
    // -----------------------------------------------------------------------

    /// Number of array entries
    static constexpr size_t Size = Size_;

    /// Size of the low array part returned by low()
    static constexpr size_t Size1 = detail::lpow2(Size_);

    /// Size of the high array part returned by high()
    static constexpr size_t Size2 = Size_ - Size1;

    /// Size and ActualSize can be different, e.g. when representing 3D vectors using 4-wide registers
    static constexpr size_t ActualSize = Size;

    /// Is this a mask type?
    static constexpr bool IsMask = Base::IsMask || IsMask_;

    /// Does this array represent a fixed size vector?
    static constexpr bool IsVector = true;

    /// Type of the low array part returned by low()
    using Array1 = std::conditional_t<!IsMask_, Array<Value_, Size1>,
                                                Mask <Value_, Size1>>;

    /// Type of the high array part returned by high()
    using Array2 = std::conditional_t<!IsMask_, Array<Value_, Size2>,
                                                Mask <Value_, Size2>>;

    //! @}
    // -----------------------------------------------------------------------

    ENOKI_INLINE constexpr size_t size() const { return Derived::Size; }

    ENOKI_INLINE void init_(size_t) { }

    static Derived empty_(size_t size) {
        Derived result;

        if constexpr (Derived::IsDynamic) {
            for (size_t i = 0; i < Derived::Size; ++i)
                result.coeff(i) = empty<Value>(size);
        } else {
            ENOKI_MARK_USED(size);
        }

        return result;
    }

    static Derived zero_(size_t size) {
        if constexpr (is_array_v<Value>) {
            Derived result;
            for (size_t i = 0; i < Derived::Size; ++i)
                result.coeff(i) = zero<Value>(size);
            return result;
        } else {
            ENOKI_MARK_USED(size);
            return Derived((Scalar) 0);
        }
    }

    static Derived full_(std::conditional_t<IsMask_, bool, Scalar> value, size_t size) {
        if constexpr (is_array_v<Value>) {
            Derived result;
            for (size_t i = 0; i < Derived::Size; ++i)
                result.coeff(i) = full<Value>(value, size);
            return result;
        } else {
            ENOKI_MARK_USED(size);
            return Derived(value);
        }
    }

private:
    template <size_t Imm, size_t... Is>
    ENOKI_INLINE Derived ror_array_(std::index_sequence<Is...>) const {
        return shuffle<(Is + Derived::Size - Imm) % Derived::Size...>(derived());
    }

    template <size_t Imm, size_t... Is>
    ENOKI_INLINE Derived rol_array_(std::index_sequence<Is...>) const {
        return shuffle<(Is + Imm) % Derived::Size...>(derived());
    }

    template <typename T, size_t Offset, size_t... Is>
    ENOKI_INLINE T sub_array_(std::index_sequence<Is...>) const {
        return T(derived().coeff(Offset + Is)...);
    }

public:
    /// Return the low array part (always a power of two)
    ENOKI_INLINE auto low_() const {
        return sub_array_<typename Derived::Array1, 0>(
            std::make_index_sequence<Derived::Size1>());
    }

    /// Return the high array part
    template <typename T = Derived, enable_if_t<T::Size2 != 0> = 0>
    ENOKI_INLINE auto high_() const {
        return sub_array_<typename Derived::Array2, Derived::Size1>(
            std::make_index_sequence<Derived::Size2>());
    }

    //! @}
    // -----------------------------------------------------------------------

    ENOKI_ARRAY_IMPORT(StaticArrayBase, Base)
};

NAMESPACE_END(enoki)
