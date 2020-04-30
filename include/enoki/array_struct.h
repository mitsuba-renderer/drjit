/*
    enoki/array.h -- Infrastructure for working with general data structures

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <enoki/array_generic.h>

#pragma once

NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Masked array helper classes
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

template <typename T> struct MaskedValue {
    MaskedValue(T &d, bool m) : d(d), m(m) { }

    template <typename T2> ENOKI_INLINE void operator =(const T2 &value) { if (m) d = value; }
    template <typename T2> ENOKI_INLINE void operator+=(const T2 &value) { if (m) d += value; }
    template <typename T2> ENOKI_INLINE void operator-=(const T2 &value) { if (m) d -= value; }
    template <typename T2> ENOKI_INLINE void operator*=(const T2 &value) { if (m) d *= value; }
    template <typename T2> ENOKI_INLINE void operator/=(const T2 &value) { if (m) d /= value; }
    template <typename T2> ENOKI_INLINE void operator|=(const T2 &value) { if (m) d |= value; }
    template <typename T2> ENOKI_INLINE void operator&=(const T2 &value) { if (m) d &= value; }
    template <typename T2> ENOKI_INLINE void operator^=(const T2 &value) { if (m) d ^= value; }
    template <typename T2> ENOKI_INLINE void operator<<=(const T2 &value) { if (m) d <<= value; }
    template <typename T2> ENOKI_INLINE void operator>>=(const T2 &value) { if (m) d >>= value; }

    T &d;
    bool m;
};

template <typename T> struct MaskedArray : ArrayBaseT<value_t<T>, is_mask_v<T>, MaskedArray<T>> {
    using Mask     = mask_t<T>;
    static constexpr size_t Size = array_size_v<T>;
    static constexpr bool IsMaskedArray = true;

    MaskedArray(T &d, const Mask &m) : d(d), m(m) { }

    template <typename T2> ENOKI_INLINE void operator =(const T2 &value) { d = select(m, value, d); }
    template <typename T2> ENOKI_INLINE void operator+=(const T2 &value) { d = select(m, d + value, d); }
    template <typename T2> ENOKI_INLINE void operator-=(const T2 &value) { d = select(m, d - value, d); }
    template <typename T2> ENOKI_INLINE void operator*=(const T2 &value) { d = select(m, d * value, d); }
    template <typename T2> ENOKI_INLINE void operator/=(const T2 &value) { d = select(m, d / value, d); }
    template <typename T2> ENOKI_INLINE void operator|=(const T2 &value) { d = select(m, d | value, d); }
    template <typename T2> ENOKI_INLINE void operator&=(const T2 &value) { d = select(m, d & value, d); }
    template <typename T2> ENOKI_INLINE void operator^=(const T2 &value) { d = select(m, d ^ value, d); }
    template <typename T2> ENOKI_INLINE void operator<<=(const T2 &value) { d = select(m, d << value, d); }
    template <typename T2> ENOKI_INLINE void operator>>=(const T2 &value) { d = select(m, d >> value, d); }

    /// Type alias for a similar-shaped array over a different type
    template <typename T2> using ReplaceValue = MaskedArray<typename T::template ReplaceValue<T2>>;

    T &d;
    Mask m;
};

NAMESPACE_END(detail)

template <typename Value_, size_t Size_>
struct Array<detail::MaskedArray<Value_>, Size_>
    : detail::MaskedArray<Array<Value_, Size_>> {
    using Base = detail::MaskedArray<Array<Value_, Size_>>;
    using Base::Base;
    using Base::operator=;
    Array(const Base &b) : Base(b) { }
};

template <typename T, typename Mask>
ENOKI_INLINE auto masked(T &value, const Mask &mask) {
    if constexpr (std::is_same_v<Mask, bool>)
        return detail::MaskedValue<T>{ value, mask };
    else if constexpr (is_array_v<T>)
        return detail::MaskedArray<T>{ value, mask };
    else
        return struct_support<T>::masked(value, mask);
}

//! @}
// -----------------------------------------------------------------------

template <typename T, typename> struct struct_support {
};

NAMESPACE_END(enoki)
