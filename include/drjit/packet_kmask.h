/*
    drjit/packet_kmask.h -- Abstraction around AVX512 'k' mask registers

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)

#define DRJIT_REINTERPRET_KMASK(Value, Size)                                   \
    template <typename Value2, typename Derived2,                              \
              enable_if_t<detail::is_same_v<Value2, Value>> = 0>               \
    DRJIT_INLINE KMaskBase(                                                    \
        const StaticArrayBase<Value2, Size, true, Derived2> &a,                \
        detail::reinterpret_flag)

template <typename Value_, size_t Size_> struct KMask;

template <typename Value_, size_t Size_, typename Derived_>
struct KMaskBase : StaticArrayBase<Value_, Size_, true, Derived_> {
    using Register = std::conditional_t<Size_ == 16, __mmask16, __mmask8>;
    using Base = StaticArrayBase<Value_, Size_, true, Derived_>;
    using Base::Size;
    using Base::derived;
    using Derived = Derived_;
    static constexpr bool IsPacked = true;
    static constexpr bool IsKMask = true;
    static constexpr Register BitMask = Register((1 << Size_) - 1);

#if defined(NDEBUG)
    KMaskBase() = default;
#else
    KMaskBase() : k((Register) -1) { }
#endif
    KMaskBase(const KMaskBase &) = default;
    KMaskBase(KMaskBase &&) = default;
    KMaskBase &operator=(const KMaskBase &) = default;
    KMaskBase &operator=(KMaskBase &&) = default;

    /// Initialize from a list of boolean values
    template <typename... Ts, detail::enable_if_components_t<Size_, Ts...> = 0>
    KMaskBase(Ts&&... ts) {
        bool data[] = { (bool) ts... };
        k = load_bool_(data);
    }

    template <typename Array, enable_if_t<std::is_same_v<Register, typename Array::Derived::Register>> = 0>
    DRJIT_INLINE KMaskBase(const Array &other, detail::reinterpret_flag) : k(other.derived().k) { }

    template <typename T, enable_if_t<std::is_same_v<bool, T> || std::is_same_v<int, T>> = 0>
    DRJIT_INLINE KMaskBase(const T &b) : k(bool(b) ? BitMask : Register(0)) { }

    template <typename T>
    KMaskBase(const detail::MaskBit<T> &bit, detail::reinterpret_flag) : KMaskBase((bool) bit) { }

    DRJIT_REINTERPRET_KMASK(bool, Size) : k(load_bool_(a.derived().data())) { }

    DRJIT_REINTERPRET_KMASK(double, 16)   { k = _mm512_kunpackb(high(a).k, low(a).k); }
    DRJIT_REINTERPRET_KMASK(int64_t, 16)  { k = _mm512_kunpackb(high(a).k, low(a).k); }
    DRJIT_REINTERPRET_KMASK(uint64_t, 16) { k = _mm512_kunpackb(high(a).k, low(a).k); }

    static Register load_bool_(const void *data) {
        __m128i value;
        if constexpr (Size == 16)
            value = _mm_loadu_si128((__m128i *) data);
        else if constexpr (Size == 8)
            value = _mm_loadl_epi64((const __m128i *) data);
        else if constexpr (Size == 4 || Size == 3)
            value = _mm_cvtsi32_si128(*((const int *) data));
        else if constexpr (Size == 2)
            value = _mm_cvtsi32_si128((int) *((const short *) data));
        else
            drjit_raise("KMaskBase: unsupported number of elements!");

        return (Register) _mm_test_epi8_mask(value, _mm_set1_epi8((char) 0xFF));
    }

    template <typename T> DRJIT_INLINE static Derived from_k(const T &k) {
        Derived result;
        result.k = (Register) k;
        return result;
    }

    DRJIT_INLINE Derived eq_(const Derived &a) const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kxnor_mask16(k, a.k));
        else
            return Derived::from_k(_kxnor_mask8(k, a.k));
    }

    DRJIT_INLINE Derived neq_(const Derived &a) const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kxor_mask16(k, a.k));
        else
            return Derived::from_k(_kxor_mask8(k, a.k));
    }

    DRJIT_INLINE Derived or_(const Derived &a) const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kor_mask16(k, a.k));
        else
            return Derived::from_k(_kor_mask8(k, a.k));
    }

    DRJIT_INLINE Derived and_(const Derived &a) const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kand_mask16(k, a.k));
        else
            return Derived::from_k(_kand_mask8(k, a.k));
    }

    DRJIT_INLINE Derived andnot_(const Derived &a) const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kandn_mask16(a.k, k));
        else
            return Derived::from_k(_kandn_mask8(a.k, k));
    }

    DRJIT_INLINE Derived xor_(const Derived &a) const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kxor_mask16(k, a.k));
        else
            return Derived::from_k(_kxor_mask8(k, a.k));
    }

    DRJIT_INLINE Derived not_() const {
        if constexpr (Size_ == 16)
            return Derived::from_k(_knot_mask16(k));
        else
            return Derived::from_k(_knot_mask8(k));
    }

    static DRJIT_INLINE Derived select_(const Derived &m, const Derived &t, const Derived &f) {
        if constexpr (Size_ == 16)
            return Derived::from_k(_kor_mask16(_kand_mask16(m.k, t.k),
                                               _kandn_mask16(m.k, f.k)));
        else
            return Derived::from_k(_kor_mask8(_kand_mask8(m.k, t.k),
                                              _kandn_mask8(m.k, f.k)));
    }

    DRJIT_INLINE bool all_() const {
        if constexpr (Size_ == 16)
            return _kortestc_mask16_u8(k, k);
        else if constexpr (Size_ == 8)
            return _kortestc_mask8_u8(k, k);
        else
            return (k & BitMask) == BitMask;
    }

    DRJIT_INLINE bool any_() const {
        if constexpr (Size_ == 16)
            return !_kortestz_mask16_u8(k, k);
        else if constexpr (Size_ == 8)
            return !_kortestz_mask8_u8(k, k);
        else
            return (k & BitMask) != 0;
    }

    DRJIT_INLINE uint32_t bitmask_() const {
        if constexpr (Size_ == 8 || Size_ == 16)
            return (uint32_t) k;
        else
            return (uint32_t) (k & BitMask);
    }

    DRJIT_INLINE size_t count_() const {
        return (size_t) _mm_popcnt_u32(bitmask_());
    }

    DRJIT_INLINE bool bit_(size_t index) const {
        return (k & (1 << index)) != 0;
    }

    DRJIT_INLINE void set_bit_(size_t index, bool value) {
        if (value)
            k |= (Register) (1 << index);
        else
            k &= (Register) ~(1 << index);
    }

    DRJIT_INLINE static Derived zero_(size_t) { return Derived::from_k(0); }
    DRJIT_INLINE static Derived empty_(size_t) { Derived d; return d; }

    DRJIT_INLINE auto low_() const {
        using Return = KMask<Value_, Size_ / 2>;
        if constexpr (Size == 16)
            return Return::from_k(__mmask8(k));
        else
            return Return::from_k(Return::BitMask & k);
    }

    DRJIT_INLINE auto high_()  const {
        return KMask<Value_, Size_ / 2>::from_k(k >> (Size_ / 2));
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return load_(ptr);
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        Derived result;
        memcpy(&result.k, ptr, sizeof(Register));
        return result;
    }

    DRJIT_INLINE void store_aligned_(void *ptr, size_t) const {
        store_(ptr);
    }

    DRJIT_INLINE void store_(void *ptr, size_t) const {
        memcpy(ptr, &k, sizeof(Register));
    }

#if 0 //XXX
    template <size_t Stride, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index_, const Mask &mask) {
        using UInt32 = Array<uint32_t, Size>;

        UInt32 index_32 = UInt32(index_),
               index, offset;

        if constexpr (Size == 2) {
            index  = sr<1>(index_32);
            offset = Index(1) << (index_32 & (uint32_t) 0x1);
        } else if constexpr (Size == 4) {
            index  = sr<2>(index_32);
            offset = Index(1) << (index_32 & (uint32_t) 0x3);
        } else {
            index  = sr<3>(index_32);
            offset = Index(1) << (index_32 & (uint32_t) 0x7);
        }

        return Derived(neq(gather<UInt32, 1>(ptr, index, mask) & offset, (uint32_t) 0));
    }
#endif

#if 0
    template <typename Array, enable_if_t<std::is_same_v<Register, typename Array::Derived::Register>> = 0>
    DRJIT_INLINE Derived& operator=(const Array &other) {
        k = other.derived().k;
        return derived();
    }

    template <typename T, enable_if_t<std::is_same_v<bool, T> || std::is_same_v<int, T>> = 0>
    DRJIT_INLINE Derived& operator=(const T &b) {
        k = bool(b) ? BitMask : Register(0);
        return derived();
    }
#endif

    Register k;
};

template <typename Value_, size_t Size_>
struct KMask : KMaskBase<Value_, Size_, KMask<Value_, Size_>> {
    using Base = KMaskBase<Value_, Size_, KMask<Value_, Size_>>;

    DRJIT_ARRAY_IMPORT(KMask, Base)
};

#define DRJIT_DECLARE_KMASK(Type, Size, Derived, SFINAE)                       \
    struct StaticArrayImpl<Type, Size, true, Derived, SFINAE>                  \
        : KMaskBase<Type, Size, Derived> {                                     \
        using Base = KMaskBase<Type, Size, Derived>;                           \
        DRJIT_ARRAY_IMPORT(StaticArrayImpl, Base)                              \
    };

NAMESPACE_END(drjit)
