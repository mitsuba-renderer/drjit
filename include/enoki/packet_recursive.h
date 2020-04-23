/*
    enoki/array_recursive.h -- Template specialization that recursively
    instantiates arrays with smaller sizes when the requested packet size is
    not directly supported by the processor's SIMD instructions

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>

NAMESPACE_BEGIN(enoki)

template <typename Value_, size_t Size_, bool IsMask_, typename Derived_>
struct StaticArrayImpl<Value_, Size_, IsMask_, Derived_,
                       detail::enable_if_recursive<Value_, Size_>>
    : StaticArrayBase<Value_, Size_, IsMask_, Derived_> {

    using Base = StaticArrayBase<Value_, Size_, IsMask_, Derived_>;
    using typename Base::Value;
    using typename Base::Array1;
    using typename Base::Array2;
    using typename Base::Derived;
    using Base::Size1;
    using Base::Size2;
    using Ref = const Derived &;

    static constexpr bool IsRecursive = true;

    ENOKI_ARRAY_IMPORT(StaticArrayImpl, Base)

    // -----------------------------------------------------------------------
    //! @{ \name Constructors
    // -----------------------------------------------------------------------

    /// Construct from a scalar
    ENOKI_INLINE StaticArrayImpl(const Value &v) : a1(v), a2(v) { }

    /// Construct from component values
    template <typename... Ts, enable_if_t<sizeof...(Ts) == Size_ && Size_ != 1 &&
              (!std::is_same_v<Ts, detail::reinterpret_flag> && ...)> = 0>
    ENOKI_INLINE StaticArrayImpl(Ts&&... ts) {
        alignas(alignof(Array1)) Value storage[Size_] = { (Value) ts... };
        a1 = load<Array1>(storage);
        a2 = load<Array2>(storage + Size1);
    }

    /// Construct from two smaller arrays
    template <typename T1, typename T2,
              enable_if_t<array_size_v<T1> == Size1 && array_size_v<T2> == Size2> = 0>
    ENOKI_INLINE StaticArrayImpl(const T1 &a1, const T2 &a2)
        : a1(a1), a2(a2) { }

    /// Copy another array
    template <typename Value2, typename Derived2>
    ENOKI_INLINE StaticArrayImpl(const ArrayBaseT<Value2, Derived2> &a)
        : a1(low(a)), a2(high(a)) { }

    /// Reinterpret another array
    template <typename Value2, typename Derived2>
    ENOKI_INLINE StaticArrayImpl(const ArrayBaseT<Value2, Derived2> &a, detail::reinterpret_flag)
        : a1(low (a), detail::reinterpret_flag()),
          a2(high(a), detail::reinterpret_flag()) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Ref a) const { return Derived(a1 + a.a1, a2 + a.a2); }
    ENOKI_INLINE Derived sub_(Ref a) const { return Derived(a1 - a.a1, a2 - a.a2); }
    ENOKI_INLINE Derived mul_(Ref a) const { return Derived(a1 * a.a1, a2 * a.a2); }
    ENOKI_INLINE Derived div_(Ref a) const { return Derived(a1 / a.a1, a2 / a.a2); }
    ENOKI_INLINE Derived mod_(Ref a) const { return Derived(a1 % a.a1, a2 % a.a2); }

    ENOKI_INLINE Derived mulhi_(Ref a) const {
        return Derived(mulhi(a1, a.a1), mulhi(a2, a.a2));
    }

    ENOKI_INLINE auto lt_ (Ref a) const { return mask_t<Derived>(a1 <  a.a1, a2 <  a.a2); }
    ENOKI_INLINE auto gt_ (Ref a) const { return mask_t<Derived>(a1 >  a.a1, a2 >  a.a2); }
    ENOKI_INLINE auto le_ (Ref a) const { return mask_t<Derived>(a1 <= a.a1, a2 <= a.a2); }
    ENOKI_INLINE auto ge_ (Ref a) const { return mask_t<Derived>(a1 >= a.a1, a2 >= a.a2); }
    ENOKI_INLINE auto eq_ (Ref a) const { return mask_t<Derived>(eq(a1, a.a1), eq(a2, a.a2)); }
    ENOKI_INLINE auto neq_(Ref a) const { return mask_t<Derived>(neq(a1, a.a1), neq(a2, a.a2)); }

    ENOKI_INLINE Derived min_(Ref a) const { return Derived(min(a1, a.a1), min(a2, a.a2)); }
    ENOKI_INLINE Derived max_(Ref a) const { return Derived(max(a1, a.a1), max(a2, a.a2)); }
    ENOKI_INLINE Derived abs_() const { return Derived(abs(a1), abs(a2)); }
    ENOKI_INLINE Derived sqrt_() const { return Derived(sqrt(a1), sqrt(a2)); }
    ENOKI_INLINE Derived ceil_() const { return Derived(ceil(a1), ceil(a2)); }
    ENOKI_INLINE Derived floor_() const { return Derived(floor(a1), floor(a2)); }
    ENOKI_INLINE Derived round_() const { return Derived(round(a1), round(a2)); }
    ENOKI_INLINE Derived trunc_() const { return Derived(trunc(a1), trunc(a2)); }
    ENOKI_INLINE Derived rcp_() const { return Derived(rcp(a1), rcp(a2)); }
    ENOKI_INLINE Derived rsqrt_() const { return Derived(rsqrt(a1), rsqrt(a2)); }
    ENOKI_INLINE Derived not_() const { return Derived(~a1, ~a2); }
    ENOKI_INLINE Derived neg_() const { return Derived(-a1, -a2); }

    ENOKI_INLINE Derived fmadd_(Ref b, Ref c) const {
        return Derived(fmadd(a1, b.a1, c.a1), fmadd(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fnmadd_(Ref b, Ref c) const {
        return Derived(fnmadd(a1, b.a1, c.a1), fnmadd(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fmsub_(Ref b, Ref c) const {
        return Derived(fmsub(a1, b.a1, c.a1), fmsub(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fnmsub_(Ref b, Ref c) const {
        return Derived(fnmsub(a1, b.a1, c.a1), fnmsub(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fmaddsub_(Ref b, Ref c) const {
        return Derived(fmaddsub(a1, b.a1, c.a1), fmaddsub(a2, b.a2, c.a2));
    }

    ENOKI_INLINE Derived fmsubadd_(Ref b, Ref c) const {
        return Derived(fmsubadd(a1, b.a1, c.a1), fmsubadd(a2, b.a2, c.a2));
    }

    template <typename T> ENOKI_INLINE Derived or_(const T &a) const {
        return Derived(a1 | low(a), a2 | high(a));
    }

    template <typename T> ENOKI_INLINE Derived andnot_(const T &a) const {
        return Derived(andnot(a1, low(a)), andnot(a2, high(a)));
    }

    template <typename T> ENOKI_INLINE Derived and_(const T &a) const {
        return Derived(a1 & low(a), a2 & high(a));
    }

    template <typename T> ENOKI_INLINE Derived xor_(const T &a) const {
        return Derived(a1 ^ low(a), a2 ^ high(a));
    }

    template <int Imm> ENOKI_INLINE Derived sl_() const {
        return Derived(sl<Imm>(a1), sl<Imm>(a2));
    }

    ENOKI_INLINE Derived sl_(Ref a) const {
        return Derived(a1 << a.a1, a2 << a.a2);
    }

    template <int Imm> ENOKI_INLINE Derived sr_() const {
        return Derived(sr<Imm>(a1), sr<Imm>(a2));
    }

    ENOKI_INLINE Derived sr_(Ref a) const {
        return Derived(a1 >> a.a1, a2 >> a.a2);
    }

    template <typename Mask>
    static ENOKI_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        return Derived(select(m.a1, t.a1, f.a1),
                       select(m.a2, t.a2, f.a2));
    }

    template <typename T>
    ENOKI_INLINE auto floor2int_() const {
        return T(floor2int<typename T::Array1>(a1),
                 floor2int<typename T::Array2>(a2));
    }

    template <typename T>
    ENOKI_INLINE auto ceil2int_() const {
        return T(ceil2int<typename T::Array1>(a1),
                 ceil2int<typename T::Array2>(a2));
    }

    template <typename T>
    ENOKI_INLINE auto trunc2int_() const {
        return T(trunc2int<typename T::Array1>(a1),
                 trunc2int<typename T::Array2>(a2));
    }

    template <typename T>
    ENOKI_INLINE auto round2int_() const {
        return T(round2int<typename T::Array1>(a1),
                 round2int<typename T::Array2>(a2));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_() const {
        if constexpr (Size1 == Size2)
            return hsum(a1 + a2);
        else
            return hsum(a1) + hsum(a2);
    }

    ENOKI_INLINE Value hprod_() const {
        if constexpr (Size1 == Size2)
            return hprod(a1 * a2);
        else
            return hprod(a1) * hprod(a2);
    }

    ENOKI_INLINE Value hmin_() const {
        if constexpr (Size1 == Size2)
            return hmin(min(a1, a2));
        else
            return min(hmin(a1), hmin(a2));
    }

    ENOKI_INLINE Value hmax_() const {
        if constexpr (Size1 == Size2)
            return hmax(max(a1, a2));
        else
            return max(hmax(a1), hmax(a2));
    }

    ENOKI_INLINE Value dot_(Ref a) const {
        if constexpr (Size1 == Size2)
            return hsum(fmadd(a1, a.a1, a2 * a.a2));
        else
            return dot(a1, a.a1) + dot(a2, a.a2);
    }

    ENOKI_INLINE bool all_() const {
        if constexpr (Size1 == Size2)
            return all(a1 & a2);
        else
            return all(a1) && all(a2);
    }

    ENOKI_INLINE bool any_() const {
        if constexpr (Size1 == Size2)
            return any(a1 | a2);
        else
            return any(a1) || any(a2);
    }

    ENOKI_INLINE size_t count_() const { return count(a1) + count(a2); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *mem, size_t) const {
        store((uint8_t *) mem, a1);
        store((uint8_t *) mem + sizeof(Array1), a2);
    }

    ENOKI_INLINE void store_unaligned_(void *mem, size_t) const {
        store_unaligned((uint8_t *) mem, a1);
        store_unaligned((uint8_t *) mem + sizeof(Array1), a2);
    }

    static ENOKI_INLINE Derived load_(const void *mem, size_t) {
        return Derived(
            load<Array1>((uint8_t *) mem),
            load<Array2>((uint8_t *) mem + sizeof(Array1))
        );
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *a, size_t) {
        return Derived(
            load_unaligned<Array1>((uint8_t *) a),
            load_unaligned<Array2>((uint8_t *) a + sizeof(Array1))
        );
    }

    static ENOKI_INLINE Derived zero_(size_t) {
        return Derived(zero<Array1>(), zero<Array2>());
    }

    static ENOKI_INLINE Derived empty_(size_t) {
        return Derived(empty<Array1>(), empty<Array2>());
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Component access
    // -----------------------------------------------------------------------

    ENOKI_INLINE const Array1& low_()  const { return a1; }
    ENOKI_INLINE const Array2& high_() const { return a2; }

    ENOKI_INLINE decltype(auto) coeff(size_t i) const {
        if constexpr (Size1 == Size2)
            return ((i < Size1) ? a1 : a2).coeff(i % Size1);
        else
            return (i < Size1) ? a1.coeff(i) : a2.coeff(i - Size1);
    }

    ENOKI_INLINE decltype(auto) coeff(size_t i) {
        if constexpr (Size1 == Size2)
            return ((i < Size1) ? a1 : a2).coeff(i % Size1);
        else
            return (i < Size1) ? a1.coeff(i) : a2.coeff(i - Size1);
    }

    //! @}
    // -----------------------------------------------------------------------

    Array1 a1;
    Array2 a2;
};

NAMESPACE_END(enoki)
