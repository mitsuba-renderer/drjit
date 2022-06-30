/*
    drjit/array_avx2.h -- Packed SIMD array (AVX2 specialization)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
DRJIT_PACKET_DECLARE_COND(32, enable_if_t<is_integral_ext_v<Type>>)
DRJIT_PACKET_DECLARE_COND(24, enable_if_int64_t<Type>)

/// Partial overload of StaticArrayImpl using AVX intrinsics (32 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<Value_, 8, IsMask_, Derived_, enable_if_int32_t<Value_>>
  : StaticArrayBase<Value_, 8, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(Value_, 8, __m256i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm256_set1_epi32((int32_t) value)) { }
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3,
                                 Value v4, Value v5, Value v6, Value v7)
        : m(_mm256_setr_epi32((int32_t) v0, (int32_t) v1, (int32_t) v2, (int32_t) v3,
                              (int32_t) v4, (int32_t) v5, (int32_t) v6, (int32_t) v7)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    DRJIT_CONVERT(float) {
        if constexpr (std::is_signed_v<Value>) {
            m = _mm256_cvttps_epi32(a.derived().m);
        } else {
            #if defined(DRJIT_X86_AVX512)
                m = _mm256_cvttps_epu32(a.derived().m);
            #else
                constexpr uint32_t limit = 1u << 31;
                const __m256  limit_f = _mm256_set1_ps((float) limit);
                const __m256i limit_i = _mm256_set1_epi32((int) limit);

                __m256 v = a.derived().m;

                __m256i mask =
                    _mm256_castps_si256(_mm256_cmp_ps(v, limit_f, _CMP_GE_OQ));

                __m256i b2 = _mm256_add_epi32(
                    _mm256_cvttps_epi32(_mm256_sub_ps(v, limit_f)), limit_i);

                __m256i b1 = _mm256_cvttps_epi32(v);

                m = _mm256_blendv_epi8(b1, b2, mask);
            #endif
        }
    }

    DRJIT_CONVERT(int32_t) : m(a.derived().m) { }
    DRJIT_CONVERT(uint32_t) : m(a.derived().m) { }

    DRJIT_CONVERT(double) {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                m = _mm512_cvttpd_epi32(a.derived().m);
            #else
                m = detail::concat(_mm256_cvttpd_epi32(low(a).m),
                                   _mm256_cvttpd_epi32(high(a).m));
            #endif
        } else {
            #if defined(DRJIT_X86_AVX512)
                m = _mm512_cvttpd_epu32(a.derived().m);
            #else
                DRJIT_TRACK_SCALAR("Constructor (converting, double[8] -> [u]int32[8])");
                for (size_t i = 0; i < Size; ++i)
                    entry(i) = Value(a.derived().entry(i));
            #endif
        }
    }

    DRJIT_CONVERT(int64_t) {
        #if defined(DRJIT_X86_AVX512)
            m = _mm512_cvtepi64_epi32(a.derived().m);
        #else
            m = detail::mm512_cvtepi64_epi32(low(a).m, high(a).m);
        #endif
    }

    DRJIT_CONVERT(uint64_t) {
        #if defined(DRJIT_X86_AVX512)
            m = _mm512_cvtepi64_epi32(a.derived().m);
        #else
            m = detail::mm512_cvtepi64_epi32(low(a).m, high(a).m);
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET_MASK(bool) {
        uint64_t ival;
        memcpy(&ival, a.derived().data(), 8);
        __m128i value = _mm_cmpgt_epi8(detail::mm_cvtsi64_si128((long long) ival),
                                       _mm_setzero_si128());
        m = _mm256_cvtepi8_epi32(value);
    }

    DRJIT_REINTERPRET(float) : m(_mm256_castps_si256(a.derived().m)) { }
    DRJIT_REINTERPRET(int32_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(uint32_t) : m(a.derived().m) { }

#if !defined(DRJIT_X86_AVX512)
    DRJIT_REINTERPRET_MASK(double)
        : m(detail::mm512_cvtepi64_epi32(_mm256_castpd_si256(low(a).m),
                                         _mm256_castpd_si256(high(a).m))) { }
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(detail::mm512_cvtepi64_epi32(low(a).m, high(a).m)) { }
    DRJIT_REINTERPRET_MASK(uint64_t)
        : m(detail::mm512_cvtepi64_epi32(low(a).m, high(a).m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    DRJIT_INLINE Array1 low_()  const { return _mm256_castsi256_si128(m); }
    DRJIT_INLINE Array2 high_() const { return _mm256_extractf128_si256(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return _mm256_add_epi32(m, a.m);   }
    DRJIT_INLINE Derived sub_(Ref a) const { return _mm256_sub_epi32(m, a.m);   }
    DRJIT_INLINE Derived mul_(Ref a) const { return _mm256_mullo_epi32(m, a.m); }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            return _mm256_ternarylogic_epi32(m, m, m, 0b01010101);
        #else
            return _mm256_xor_si256(m, _mm256_set1_epi32(-1));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm256_sub_epi32(_mm256_setzero_si256(), m);
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_epi32(m, a.k, _mm256_set1_epi32(-1));
            else
        #endif
        return _mm256_or_si256(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_maskz_mov_epi32(a.k, m);
            else
        #endif
        return _mm256_and_si256(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_xor_epi32(m, a.k, m, _mm256_set1_epi32(-1));
            else
        #endif
        return _mm256_xor_si256(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_epi32(m, a.k, _mm256_setzero_si256());
            else
        #endif
        return _mm256_andnot_si256(a.m, m);
    }

    template <int Imm> DRJIT_INLINE Derived sl_() const {
        return _mm256_slli_epi32(m, Imm);
    }

    template <int Imm> DRJIT_INLINE Derived sr_() const {
        return std::is_signed_v<Value> ? _mm256_srai_epi32(m, Imm)
                                       : _mm256_srli_epi32(m, Imm);
    }

    DRJIT_INLINE Derived sl_(Ref k) const {
        return _mm256_sllv_epi32(m, k.m);
    }

    DRJIT_INLINE Derived sr_(Ref k) const {
        return std::is_signed_v<Value> ? _mm256_srav_epi32(m, k.m)
                                       : _mm256_srlv_epi32(m, k.m);
    }

    DRJIT_INLINE auto eq_(Ref a)  const {
        using Return = mask_t<Derived>;

        #if defined(DRJIT_X86_AVX512)
            return Return::from_k(_mm256_cmpeq_epi32_mask(m, a.m));
        #else
            return Return(_mm256_cmpeq_epi32(m, a.m));
        #endif
    }

    DRJIT_INLINE auto neq_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(_mm256_cmpneq_epi32_mask(m, a.m));
        #else
            return ~eq_(a);
        #endif
    }

    DRJIT_INLINE auto lt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm256_cmpgt_epi32(a.m, m));
            } else {
                const __m256i offset = _mm256_set1_epi32((int32_t) 0x80000000ul);
                return Return(_mm256_cmpgt_epi32(_mm256_sub_epi32(a.m, offset),
                                                 _mm256_sub_epi32(m, offset)));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm256_cmplt_epi32_mask(m, a.m)
                                  : _mm256_cmplt_epu32_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto gt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm256_cmpgt_epi32(m, a.m));
            } else {
                const __m256i offset = _mm256_set1_epi32((int32_t) 0x80000000ul);
                return Return(_mm256_cmpgt_epi32(_mm256_sub_epi32(m, offset),
                                                 _mm256_sub_epi32(a.m, offset)));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm256_cmpgt_epi32_mask(m, a.m)
                                  : _mm256_cmpgt_epu32_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto le_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm256_cmple_epi32_mask(m, a.m)
                                           : _mm256_cmple_epu32_mask(m, a.m));
        #else
            return ~gt_(a);
        #endif
    }

    DRJIT_INLINE auto ge_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm256_cmpge_epi32_mask(m, a.m)
                                           : _mm256_cmpge_epu32_mask(m, a.m));
        #else
            return ~lt_(a);
        #endif
    }

    DRJIT_INLINE Derived minimum_(Ref a) const {
        return std::is_signed_v<Value> ? _mm256_min_epi32(a.m, m)
                                       : _mm256_min_epu32(a.m, m);
    }

    DRJIT_INLINE Derived maximum_(Ref a) const {
        return std::is_signed_v<Value> ? _mm256_max_epi32(a.m, m)
                                       : _mm256_max_epu32(a.m, m);
    }

    DRJIT_INLINE Derived abs_() const {
        return std::is_signed_v<Value> ? _mm256_abs_epi32(m) : m;
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm256_blendv_epi8(f.m, t.m, m.m);
        #else
            return _mm256_mask_blend_epi32(m.k, f.m, t.m);
        #endif
    }

    template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
    DRJIT_INLINE Derived shuffle_() const {
        return _mm256_permutevar8x32_epi32(m,
            _mm256_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7));
    }

    DRJIT_INLINE Derived mulhi_(Ref b) const {
        Derived even, odd;

        if constexpr (std::is_signed_v<Value>) {
            even.m = _mm256_srli_epi64(_mm256_mul_epi32(m, b.m), 32);
            odd.m = _mm256_mul_epi32(_mm256_srli_epi64(m, 32), _mm256_srli_epi64(b.m, 32));
        } else {
            even.m = _mm256_srli_epi64(_mm256_mul_epu32(m, b.m), 32);
            odd.m = _mm256_mul_epu32(_mm256_srli_epi64(m, 32), _mm256_srli_epi64(b.m, 32));
        }

        return select(
            mask_t<Derived>(true, false, true, false, true, false, true, false),
            even, odd);
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived lzcnt_() const { return _mm256_lzcnt_epi32(m); }
    DRJIT_INLINE Derived tzcnt_() const { return Value(32) - lzcnt(~derived() & (derived() - Value(1))); }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Value sum_()  const { return sum(low_() + high_()); }
    DRJIT_INLINE Value prod_() const { return prod(low_() * high_()); }
    DRJIT_INLINE Value min_()  const { return min(minimum(low_(), high_())); }
    DRJIT_INLINE Value max_()  const { return max(maximum(low_(), high_())); }

    DRJIT_INLINE bool all_() const { return _mm256_movemask_ps(_mm256_castsi256_ps(m)) == 0xFF; }
    DRJIT_INLINE bool any_() const { return _mm256_movemask_ps(_mm256_castsi256_ps(m)) != 0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_ps(_mm256_castsi256_ps(m)); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm256_store_si256((__m256i *) DRJIT_ASSUME_ALIGNED(ptr, 32), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm256_storeu_si256((__m256i *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm256_load_si256((const __m256i *) DRJIT_ASSUME_ALIGNED(ptr, 32));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm256_loadu_si256((const __m256i *) ptr);
    }

    static DRJIT_INLINE Derived empty_(size_t) { return _mm256_undefined_si256(); }
    static DRJIT_INLINE Derived zero_(size_t) { return _mm256_setzero_si256(); }

    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mmask_i32gather_epi32(_mm256_setzero_si256(), mask.k, index.m, (const int *) ptr, 4);
            else
                return _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), mask.k, index.m, (const int *) ptr, 4);
        #else
            if constexpr (sizeof(scalar_t<Index>) == 4) {
                return _mm256_mask_i32gather_epi32(
                    _mm256_setzero_si256(), (const int *) ptr, index.m, mask.m, 4);
            } else {
                return Derived(
                    _mm256_mask_i64gather_epi32(_mm_setzero_si128(), (const int *) ptr, low(index).m, low(mask).m, 4),
                    _mm256_mask_i64gather_epi32(_mm_setzero_si128(), (const int *) ptr, high(index).m, high(mask).m, 4)
                );
            }
        #endif
    }

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm256_mask_i32scatter_epi32(ptr, mask.k, index.m, m, 4);
        else
            _mm512_mask_i64scatter_epi32(ptr, mask.k, index.m, m, 4);
    }
#endif

    template <typename Mask>
    DRJIT_INLINE Value extract_(const Mask &mask) const {
        #if defined(DRJIT_X86_AVX512)
            return (Value) _mm_cvtsi128_si32(_mm256_castsi256_si128(
                _mm256_mask_compress_epi32(_mm256_setzero_si256(), mask.k, m)));
        #else
            int k = _mm256_movemask_ps(_mm256_castsi256_ps(mask.m));
            return entry((size_t) (detail::tzcnt(k) & 7));
        #endif
    }


    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl using AVX intrinsics (64 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<Value_, 4, IsMask_, Derived_, enable_if_int64_t<Value_>>
  : StaticArrayBase<Value_, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(Value_, 4, __m256i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value)
        : m(_mm256_set1_epi64x((long long) value)) { }

    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m(_mm256_setr_epi64x((long long) v0, (long long) v1,
                               (long long) v2, (long long) v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(float) {
        m = std::is_signed_v<Value> ? _mm256_cvttps_epi64(a.derived().m)
                                    : _mm256_cvttps_epu64(a.derived().m);
    }

    DRJIT_CONVERT(double) {
        m = std::is_signed_v<Value> ? _mm256_cvttpd_epi64(a.derived().m)
                                    : _mm256_cvttpd_epu64(a.derived().m);
    }
#endif
    DRJIT_CONVERT(int32_t)  : m(_mm256_cvtepi32_epi64(a.derived().m)) { }
    DRJIT_CONVERT(uint32_t) : m(_mm256_cvtepu32_epi64(a.derived().m)) { }

    DRJIT_CONVERT(int64_t) : m(a.derived().m) { }
    DRJIT_CONVERT(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET_MASK(bool) {
        int ival;
        memcpy(&ival, a.derived().data(), 4);
        m = _mm256_cvtepi8_epi64(
            _mm_cmpgt_epi8(_mm_cvtsi32_si128(ival), _mm_setzero_si128()));
    }

#if !defined(DRJIT_X86_AVX512)
    DRJIT_REINTERPRET_MASK(float)
        : m(_mm256_cvtepi32_epi64(_mm_castps_si128(a.derived().m))) { }
    DRJIT_REINTERPRET_MASK(int32_t) : m(_mm256_cvtepi32_epi64(a.derived().m)) { }
    DRJIT_REINTERPRET_MASK(uint32_t) : m(_mm256_cvtepi32_epi64(a.derived().m)) { }
#endif

    DRJIT_REINTERPRET(double) : m(_mm256_castpd_si256(a.derived().m)) { }
    DRJIT_REINTERPRET(int64_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    DRJIT_INLINE Array1 low_()  const { return _mm256_castsi256_si128(m); }
    DRJIT_INLINE Array2 high_() const { return _mm256_extractf128_si256(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return _mm256_add_epi64(m, a.m);   }
    DRJIT_INLINE Derived sub_(Ref a) const { return _mm256_sub_epi64(m, a.m);   }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            return _mm256_ternarylogic_epi64(m, m, m, 0b01010101);
        #else
            return _mm256_xor_si256(m, _mm256_set1_epi64x(-1));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm256_sub_epi64(_mm256_setzero_si256(), m);
    }

    DRJIT_INLINE Derived mul_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return _mm256_mullo_epi64(m, a.m);
        #else
            __m256i h0    = _mm256_srli_epi64(m, 32);
            __m256i h1    = _mm256_srli_epi64(a.m, 32);
            __m256i low   = _mm256_mul_epu32(m, a.m);
            __m256i mix0  = _mm256_mul_epu32(m, h1);
            __m256i mix1  = _mm256_mul_epu32(h0, a.m);
            __m256i mix   = _mm256_add_epi64(mix0, mix1);
            __m256i mix_s = _mm256_slli_epi64(mix, 32);
            return _mm256_add_epi64(mix_s, low);
        #endif
    }

    DRJIT_INLINE Derived mulhi_(Ref b) const {
        if constexpr (std::is_unsigned_v<Value>) {
            const __m256i low_bits = _mm256_set1_epi64x(0xffffffffu);
            __m256i al = m, bl = b.m;

            __m256i ah = _mm256_srli_epi64(al, 32);
            __m256i bh = _mm256_srli_epi64(bl, 32);

            // 4x unsigned 32x32->64 bit multiplication
            __m256i albl = _mm256_mul_epu32(al, bl);
            __m256i albh = _mm256_mul_epu32(al, bh);
            __m256i ahbl = _mm256_mul_epu32(ah, bl);
            __m256i ahbh = _mm256_mul_epu32(ah, bh);

            // Calculate a possible carry from the low bits of the multiplication.
            __m256i carry = _mm256_add_epi64(
                _mm256_srli_epi64(albl, 32),
                _mm256_add_epi64(_mm256_and_si256(albh, low_bits),
                                 _mm256_and_si256(ahbl, low_bits)));

            __m256i s0 = _mm256_add_epi64(ahbh, _mm256_srli_epi64(carry, 32));
            __m256i s1 = _mm256_add_epi64(_mm256_srli_epi64(albh, 32),
                                          _mm256_srli_epi64(ahbl, 32));

            return _mm256_add_epi64(s0, s1);

        } else {
            const Derived mask(0xffffffff);
            const Derived a = derived();
            Derived ah = sr<32>(a), bh = sr<32>(b),
                    al = a & mask, bl = b & mask;

            Derived albl_hi = _mm256_srli_epi64(_mm256_mul_epu32(m, b.m), 32);

            Derived t = ah * bl + albl_hi;
            Derived w1 = al * bh + (t & mask);

            return ah * bh + sr<32>(t) + sr<32>(w1);
        }
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_epi64(m, a.k, _mm256_set1_epi64x(-1));
            else
        #endif
        return _mm256_or_si256(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_maskz_mov_epi64(a.k, m);
            else
        #endif
        return _mm256_and_si256(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_xor_epi64(m, a.k, m, _mm256_set1_epi64x(-1));
            else
        #endif
        return _mm256_xor_si256(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_epi64(m, a.k, _mm256_setzero_si256());
            else
        #endif
        return _mm256_andnot_si256(a.m, m);
    }

    template <int Imm> DRJIT_INLINE Derived sl_() const {
        return _mm256_slli_epi64(m, Imm);
    }

    template <int Imm> DRJIT_INLINE Derived sr_() const {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                return _mm256_srai_epi64(m, Imm);
            #else
                const __m256i offset = _mm256_set1_epi64x((long long) 0x8000000000000000ull);
                __m256i s1 = _mm256_srli_epi64(_mm256_add_epi64(m, offset), Imm);
                __m256i s2 = _mm256_srli_epi64(offset, Imm);
                return _mm256_sub_epi64(s1, s2);
            #endif
        } else {
            return _mm256_srli_epi64(m, Imm);
        }
    }

    DRJIT_INLINE Derived sl_(Ref k) const {
        return _mm256_sllv_epi64(m, k.m);
    }

    DRJIT_INLINE Derived sr_(Ref k) const {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                return _mm256_srav_epi64(m, k.m);
            #else
                const __m256i offset = _mm256_set1_epi64x((long long) 0x8000000000000000ull);
                __m256i s1 = _mm256_srlv_epi64(_mm256_add_epi64(m, offset), k.m);
                __m256i s2 = _mm256_srlv_epi64(offset, k.m);
                return _mm256_sub_epi64(s1, s2);
            #endif
        } else {
            return _mm256_srlv_epi64(m, k.m);
        }
    }

    DRJIT_INLINE auto eq_(Ref a)  const {
        using Return = mask_t<Derived>;

        #if defined(DRJIT_X86_AVX512)
            return Return::from_k(_mm256_cmpeq_epi64_mask(m, a.m));
        #else
            return Return(_mm256_cmpeq_epi64(m, a.m));
        #endif
    }

    DRJIT_INLINE auto neq_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(_mm256_cmpneq_epi64_mask(m, a.m));
        #else
            return ~eq_(a);
        #endif
    }

    DRJIT_INLINE auto lt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm256_cmpgt_epi64(a.m, m));
            } else {
                const __m256i offset =
                    _mm256_set1_epi64x((long long) 0x8000000000000000ull);
                return Return(_mm256_cmpgt_epi64(
                    _mm256_sub_epi64(a.m, offset),
                    _mm256_sub_epi64(m, offset)
                ));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm256_cmplt_epi64_mask(m, a.m)
                                  : _mm256_cmplt_epu64_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto gt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm256_cmpgt_epi64(m, a.m));
            } else {
                const __m256i offset =
                    _mm256_set1_epi64x((long long) 0x8000000000000000ull);
                return Return(_mm256_cmpgt_epi64(
                    _mm256_sub_epi64(m, offset),
                    _mm256_sub_epi64(a.m, offset)
                ));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm256_cmpgt_epi64_mask(m, a.m)
                                  : _mm256_cmpgt_epu64_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto le_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm256_cmple_epi64_mask(m, a.m)
                                           : _mm256_cmple_epu64_mask(m, a.m));
        #else
            return ~gt_(a);
        #endif
    }

    DRJIT_INLINE auto ge_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm256_cmpge_epi64_mask(m, a.m)
                                           : _mm256_cmpge_epu64_mask(m, a.m));
        #else
            return ~lt_(a);
        #endif
    }

    DRJIT_INLINE Derived minimum_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return std::is_signed_v<Value> ? _mm256_min_epi64(a.m, m)
                                           : _mm256_min_epu64(a.m, m);
        #else
            return select(derived() < a, derived(), a);
        #endif
    }

    DRJIT_INLINE Derived maximum_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return std::is_signed_v<Value> ? _mm256_max_epi64(a.m, m)
                                           : _mm256_max_epu64(a.m, m);
        #else
            return select(derived() > a, derived(), a);
        #endif
    }

    DRJIT_INLINE Derived abs_() const {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                return _mm256_abs_epi64(m);
            #else
                return select(derived() < zeros<Derived>(),
                              ~derived() + Derived(Value(1)), derived());
            #endif
        } else {
            return m;
        }
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm256_blendv_epi8(f.m, t.m, m.m);
        #else
            return _mm256_mask_blend_epi64(m.k, f.m, t.m);
        #endif
    }

    template <int I0, int I1, int I2, int I3>
    DRJIT_INLINE Derived shuffle_() const {
        return _mm256_permute4x64_epi64(m, _MM_SHUFFLE(I3, I2, I1, I0));
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived lzcnt_() const { return _mm256_lzcnt_epi64(m); }
    DRJIT_INLINE Derived tzcnt_() const { return Value(64) - lzcnt(~derived() & (derived() - Value(1))); }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------
    //
    DRJIT_INLINE Value sum_()  const { return sum(low_() + high_()); }
    DRJIT_INLINE Value prod_() const { return prod(low_() * high_()); }
    DRJIT_INLINE Value min_()  const { return min(drjit::minimum(low_(), high_())); }
    DRJIT_INLINE Value max_()  const { return max(drjit::maximum(low_(), high_())); }

    DRJIT_INLINE bool all_() const { return _mm256_movemask_pd(_mm256_castsi256_pd(m)) == 0xF; }
    DRJIT_INLINE bool any_() const { return _mm256_movemask_pd(_mm256_castsi256_pd(m)) != 0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_pd(_mm256_castsi256_pd(m)); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm256_store_si256((__m256i *) DRJIT_ASSUME_ALIGNED(ptr, 32), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm256_storeu_si256((__m256i *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr ,size_t) {
        return _mm256_load_si256((const __m256i *) DRJIT_ASSUME_ALIGNED(ptr, 32));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm256_loadu_si256((const __m256i *) ptr);
    }

    static DRJIT_INLINE Derived zero_(size_t) { return _mm256_setzero_si256(); }
    static DRJIT_INLINE Derived empty_(size_t) { return _mm256_undefined_si256(); }

    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mmask_i32gather_epi64(_mm256_setzero_si256(), mask.k, index.m, (const long long *) ptr, 8);
            else
                return _mm256_mmask_i64gather_epi64(_mm256_setzero_si256(), mask.k, index.m, (const long long *) ptr, 8);
        #else
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mask_i32gather_epi64(_mm256_setzero_si256(), (const long long *) ptr, index.m, mask.m, 8);
            else
                return _mm256_mask_i64gather_epi64(_mm256_setzero_si256(), (const long long *) ptr, index.m, mask.m, 8);
        #endif
    }

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm256_mask_i32scatter_epi64(ptr, mask.k, index.m, m, 8);
        else
            _mm256_mask_i64scatter_epi64(ptr, mask.k, index.m, m, 8);
    }
#endif

    template <typename Mask>
    DRJIT_INLINE Value extract_(const Mask &mask) const {
        #if defined(DRJIT_X86_AVX512)
            return (Value) detail::mm_cvtsi128_si64(_mm256_castsi256_si128(
                _mm256_mask_compress_epi64(_mm256_setzero_si256(), mask.k, m)));
        #else
            int k = _mm256_movemask_pd(_mm256_castsi256_pd(mask.m));
            return entry((size_t) (detail::tzcnt(k) & 3));
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl for the n=3 case (64 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<Value_, 3, IsMask_, Derived_, enable_if_int64_t<Value_>>
  : StaticArrayImpl<Value_, 4, IsMask_, Derived_> {
    DRJIT_PACKET_TYPE_3D(Value_)
    using Base::entry;

    template <int I0, int I1, int I2>
    DRJIT_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    DRJIT_INLINE Value sum_() const {
        Value result = entry(0);
        for (size_t i = 1; i < 3; ++i)
            result += entry(i);
        return result;
    }

    DRJIT_INLINE Value prod_() const {
        Value result = entry(0);
        for (size_t i = 1; i < 3; ++i)
            result *= entry(i);
        return result;
    }

    DRJIT_INLINE Value min_() const {
        Value result = entry(0);
        for (size_t i = 1; i < 3; ++i)
            result = drjit::minimum(result, entry(i));
        return result;
    }

    DRJIT_INLINE Value max_() const {
        Value result = entry(0);
        for (size_t i = 1; i < 3; ++i)
            result = drjit::maximum(result, entry(i));
        return result;
    }

    DRJIT_INLINE bool all_()  const { return (_mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7) == 7;}
    DRJIT_INLINE bool any_()  const { return (_mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7) != 0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_pd(_mm256_castsi256_pd(m)) & 7; }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    static DRJIT_INLINE auto mask_() {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k((__mmask8) 7);
        #else
            return mask_t<Derived>(_mm256_setr_epi64x(
                (int64_t) -1, (int64_t) -1, (int64_t) -1, (int64_t) 0));
        #endif
    }

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        memcpy(ptr, &m, sizeof(Value) * 3);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        store_aligned_(ptr);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t size) {
        return Base::load_(ptr, size);
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Value) * 3);
        return result;
    }

    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::template gather_<false>(ptr, index, mask & mask_());
    }

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        Base::template scatter_<false>(ptr, index, mask & mask_());
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

#if defined(DRJIT_X86_AVX512)
template <typename Value_, typename Derived_>
DRJIT_DECLARE_KMASK(Value_, 8, Derived_, enable_if_int32_t<Value_>)
template <typename Value_, typename Derived_>
DRJIT_DECLARE_KMASK(Value_, 4, Derived_, enable_if_int64_t<Value_>)
template <typename Value_, typename Derived_>
DRJIT_DECLARE_KMASK(Value_, 3, Derived_, enable_if_int64_t<Value_>)
#endif

NAMESPACE_END(drjit)
