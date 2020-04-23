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
struct StaticArrayBase : ArrayBaseT<std::conditional_t<IsMask_, mask_t<Value_>, Value_>, Derived_> {
    using Base = ArrayBaseT<std::conditional_t<IsMask_, mask_t<Value_>, Value_>, Derived_>;
    ENOKI_ARRAY_IMPORT(StaticArrayBase, Base)

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

    static Derived full_(const std::conditional_t<IsMask_, bool, Scalar> &value, size_t size) {
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

    template <typename T = Derived>
    static Derived load_unaligned_(const void *mem, size_t) {
        static_assert(!is_dynamic_v<value_t<T>>,
                      "store_unaligned(): nested dynamic array not "
                      "supported! Did you mean to use enoki::gather?");

        Derived result;

        if constexpr (std::is_scalar_v<Value>) {
            memcpy(derived().data(), mem, sizeof(Value) * Derived::Size);
        } else {
            ENOKI_CHKSCALAR("store_unaligned");
            for (size_t i = 0; i < Derived::Size; ++i)
                derived().coeff(i) =
                    load_unaligned<Value>(static_cast<Value *>(mem) + i);
        }
    }

    template <typename T = Derived>
    void store_unaligned_(void *mem) const {
        static_assert(!is_dynamic_v<value_t<T>>,
                      "store_unaligned(): nested dynamic array not "
                      "supported! Did you mean to use enoki::gather?");

        if constexpr (std::is_scalar_v<Value>) {
            memcpy(mem, derived().data(), sizeof(Value) * Derived::Size);
        } else {
            ENOKI_CHKSCALAR("store_unaligned");
            for (size_t i = 0; i < Derived::Size; ++i)
                store_unaligned(static_cast<Value *>(mem) + i,
                                derived().coeff(i));
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

    template <typename T, size_t... Is>
    static ENOKI_INLINE auto linspace_impl_(std::index_sequence<Is...>, T offset, T step) {
        ENOKI_MARK_USED(step);
        if constexpr (sizeof...(Is) == 0)
            return Derived();
        else if constexpr (sizeof...(Is) == 1)
            return Derived((Scalar) offset);
        else
            return Derived(((Scalar) ((T) Is * step + offset))...);
    }

public:
    /// Construct an evenly spaced integer sequence
    static ENOKI_INLINE Derived arange_(ssize_t start, ssize_t stop, ssize_t step) {
        (void) stop;
        return linspace_impl_(std::make_index_sequence<Derived::Size>(), start,
                              step);
    }

    /// Construct an array that linearly interpolates from min..max
    static ENOKI_INLINE Derived linspace_(Scalar min, Scalar max, size_t) {
        if constexpr (Derived::Size == 1)
            return Derived(min);
        else if constexpr (std::is_floating_point_v<Scalar>)
            return linspace_impl_(
                std::make_index_sequence<Derived::Size>(), min,
                (max - min) / Scalar(Derived::Size - 1));
        else
            return linspace_impl_(
                std::make_index_sequence<Derived::Size>(), (double) min,
                ((double) max - (double) min) / double(Derived::Size - 1));
    }

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
};

NAMESPACE_END(enoki)
