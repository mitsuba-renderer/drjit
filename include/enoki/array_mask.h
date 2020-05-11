/*
    enoki/array_mask.h -- Infrastructure for dealing with the special case of
    mask arrays

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array_generic.h>

NAMESPACE_BEGIN(enoki)

template <typename T> struct MaskBit {
public:
    using Value = typename T::Value;

    MaskBit(T &mask, size_t index) : mask(mask), index(index) { }

    operator bool() const {
        return mask.bit_(index);
    }

    MaskBit &operator=(bool b) {
        mask.set_bit_(index, b);
        return *this;
    }

private:
    T &mask;
    size_t index;
};

template <typename Value_, size_t Size_, typename Derived_>
struct MaskBase : StaticArrayImpl<Value_, Size_, true, Derived_> {
    using Base = StaticArrayImpl<Value_, Size_, true, Derived_>;
    using typename Base::Scalar;
    using Base::derived;

    static constexpr bool IsOldStyleMask = Base::IsPacked && !Base::IsKMask;

    MaskBase() = default;
    MaskBase(const MaskBase &) = default;
    MaskBase(MaskBase &&) = default;
    MaskBase &operator=(const MaskBase &) = default;
    MaskBase &operator=(MaskBase &&) = default;

    /// Catch-all move/copy assignment operator
    template <typename T> MaskBase &operator=(T&& value) {
        return operator=(Derived_(std::forward<T>(value)));
    }

    /// Forward to base
    template <typename T> MaskBase(T &&value, detail::reinterpret_flag)
        : Base(std::forward<T>(value), detail::reinterpret_flag()) { }

    /// Forward to base
    template <typename T, typename T2 = MaskBase,
              enable_if_t<!detail::is_bool_v<T>> = 0>
    MaskBase(T &&value)
        : Base(std::forward<T>(value), detail::reinterpret_flag()) { }

    /// Broadcast boolean
    template <typename T, typename T2 = MaskBase,
              enable_if_t<detail::is_bool_v<T> && !T2::IsOldStyleMask> = 0>
    MaskBase(T &&value) : Base(value) { }

    /// Broadcast boolean (SSE/AVX packed format)
    template <typename T, typename T2 = MaskBase,
              enable_if_t<detail::is_bool_v<T> && T2::IsOldStyleMask> = 0>
    MaskBase(T &&value)
        : Base(memcpy_cast<Scalar>(int_array_t<Scalar>(value ? -1 : 0))) { }

    /// Initialize from individual entries (forward)
    template <typename... Ts, typename T2 = MaskBase,
              enable_if_t<sizeof...(Ts) == Size_ && Size_ != 1 && !T2::IsOldStyleMask> = 0>
    MaskBase(Ts&&... ts) : Base(std::forward<Ts>(ts)...) { }

    /// Initialize from boolean values (SSE/AVX packed format)
    template <typename... Ts, typename T2 = MaskBase,
              enable_if_t<sizeof...(Ts) == Size_ && Size_ != 1 && T2::IsOldStyleMask> = 0>
    MaskBase(Ts&&... ts) : Base(memcpy_cast<Scalar>(int_array_t<Scalar>(bool(ts) ? -1 : 0))...) { }

    /// Construct from sub-arrays
    template <typename T1, typename T2, typename T = MaskBase, enable_if_t<
              array_depth_v<T1> == array_depth_v<T> && array_size_v<T1> == Base::Size1 &&
              array_depth_v<T2> == array_depth_v<T> && array_size_v<T2> == Base::Size2 &&
              Base::Size2 != 0> = 0>
    MaskBase(const T1 &a1, const T2 &a2)
        : Base(a1, a2) { }

    using Base::entry;
    ENOKI_INLINE decltype(auto) entry(size_t i) {
        if constexpr (!Base::IsPacked)
            return Base::entry(i);
        else
            return MaskBit<MaskBase>(derived(), i);
    }

    ENOKI_INLINE decltype(auto) entry(size_t i) const {
        if constexpr (!Base::IsPacked)
            return Base::entry(i);
        else
            return MaskBit<const MaskBase>(derived(), i);
    }

    ENOKI_INLINE bool bit_(size_t index) const {
        if constexpr (Base::IsKMask) {
            return Base::bit_(index);
        } else if constexpr (Base::IsPacked) {
            using Int = int_array_t<Value_>;
            return memcpy_cast<Int>(Base::entry(index)) != 0;
        } else {
            return Base::entry(index);
        }
    }

    ENOKI_INLINE void set_bit_(size_t index, bool value) {
        if constexpr (Base::IsKMask) {
            Base::set_bit_(index, value);
        } else if constexpr (Base::IsPacked) {
            using Int = int_array_t<Value_>;
            Base::entry(index) = memcpy_cast<Value_>(Int(value ? -1 : 0));
        } else {
            Base::entry(index) = value;
        }
    }
};

NAMESPACE_END(enoki)
