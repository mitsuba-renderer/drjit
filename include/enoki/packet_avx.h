/*
    enoki/array_avx.h -- Packed SIMD array (AVX specialization)

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(enoki)
ENOKI_PACKET_DECLARE_COND(32, enable_if_t<std::is_floating_point_v<Type>>)
ENOKI_PACKET_DECLARE_COND(24, enable_if_t<(std::is_same_v<Type, double>)>)

/// Partial overload of StaticArrayImpl using AVX intrinsics (single precision)
template <bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<float, 8, IsMask_, Derived_>
  : StaticArrayBase<float, 8, IsMask_, Derived_> {

    ENOKI_PACKET_TYPE(float, 8, __m256)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm256_set1_ps(value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3,
                                 Value v4, Value v5, Value v6, Value v7)
        : m(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

// #if defined(ENOKI_X86_F16C)
//     ENOKI_CONVERT(half)
//         : m(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) a.derived().data()))) { }
// #endif

    ENOKI_CONVERT(float) : m(a.derived().m) { }

#if defined(ENOKI_X86_AVX2)
    ENOKI_CONVERT(int32_t) : m(_mm256_cvtepi32_ps(a.derived().m)) { }
#else
    ENOKI_CONVERT(int32_t)
        : m(detail::concat(_mm_cvtepi32_ps(low(a).m), _mm_cvtepi32_ps(high(a).m))) { }
#endif

    ENOKI_CONVERT(uint32_t) {
        #if defined(ENOKI_X86_AVX512)
            m = _mm256_cvtepu32_ps(a.derived().m);
        #else
            int32_array_t<Derived> ai(a);
            Derived result =
                Derived(ai & 0x7fffffff) +
                (Derived(float(1u << 31)) & mask_t<Derived>(sr<31>(ai)));
            m = result.m;
        #endif
    }

#if defined(ENOKI_X86_AVX512)
    ENOKI_CONVERT(double)
        :m(_mm512_cvtpd_ps(a.derived().m)) { }
#else
    ENOKI_CONVERT(double)
        : m(detail::concat(_mm256_cvtpd_ps(low(a).m),
                           _mm256_cvtpd_ps(high(a).m))) { }
#endif

#if defined(ENOKI_X86_AVX512)
    ENOKI_CONVERT(int64_t) : m(_mm512_cvtepi64_ps(a.derived().m)) { }
    ENOKI_CONVERT(uint64_t) : m(_mm512_cvtepu64_ps(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET_MASK(bool) {
        uint64_t ival;
        memcpy(&ival, a.derived().data(), 8);
        __m128i value = _mm_cmpgt_epi8(
            detail::mm_cvtsi64_si128((long long) ival), _mm_setzero_si128());
        #if defined(ENOKI_X86_AVX2)
            m = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(value));
        #else
            m = _mm256_castsi256_ps(_mm256_insertf128_si256(
                    _mm256_castsi128_si256(_mm_cvtepi8_epi32(value)),
                    _mm_cvtepi8_epi32(_mm_srli_si128(value, 4)), 1));
        #endif
    }

    ENOKI_REINTERPRET(float) : m(a.derived().m) { }

#if defined(ENOKI_X86_AVX2)
    ENOKI_REINTERPRET(int32_t) : m(_mm256_castsi256_ps(a.derived().m)) { }
    ENOKI_REINTERPRET(uint32_t) : m(_mm256_castsi256_ps(a.derived().m)) { }
#else
    ENOKI_REINTERPRET(int32_t)
        : m(detail::concat(_mm_castsi128_ps(low(a).m),
                           _mm_castsi128_ps(high(a).m))) { }

    ENOKI_REINTERPRET(uint32_t)
        : m(detail::concat(_mm_castsi128_ps(low(a).m),
                           _mm_castsi128_ps(high(a).m))) { }
#endif

#if defined(ENOKI_X86_AVX512)
    /// Handled by KMask
#else
    ENOKI_REINTERPRET_MASK(double)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
              _mm256_castpd_si256(low(a).m), _mm256_castpd_si256(high(a).m)))) { }
#  if defined(ENOKI_X86_AVX2)
    ENOKI_REINTERPRET_MASK(int64_t)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(low(a).m, high(a).m))) { }
    ENOKI_REINTERPRET_MASK(uint64_t)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(low(a).m, high(a).m))) { }
#  else
    ENOKI_REINTERPRET_MASK(int64_t)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
             low(low(a)).m, high(low(a)).m,
             low(high(a)).m, high(high(a)).m))) { }
    ENOKI_REINTERPRET_MASK(uint64_t)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
             low(low(a)).m, high(low(a)).m,
             low(high(a)).m, high(high(a)).m))) { }
#  endif
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm256_castps256_ps128(m); }
    ENOKI_INLINE Array2 high_() const { return _mm256_extractf128_ps(m, 1); }


    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Ref a) const    { return _mm256_add_ps(m, a.m); }
    ENOKI_INLINE Derived sub_(Ref a) const    { return _mm256_sub_ps(m, a.m); }
    ENOKI_INLINE Derived mul_(Ref a) const    { return _mm256_mul_ps(m, a.m); }
    ENOKI_INLINE Derived div_(Ref a) const    { return _mm256_div_ps(m, a.m); }

    ENOKI_INLINE Derived not_() const {
        #if defined(ENOKI_X86_AVX512)
            __m256i mi = _mm256_castps_si256(m);
            mi = _mm256_ternarylogic_epi32(mi, mi, mi, 0b01010101);
            return _mm256_castsi256_ps(mi);
        #else
            return _mm256_xor_ps(m, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
        #endif
    }

    ENOKI_INLINE Derived neg_() const {
        return _mm256_sub_ps(_mm256_setzero_ps(), m);
    }

    template <typename T> ENOKI_INLINE Derived or_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_ps(m, a.k, _mm256_set1_ps(memcpy_cast<Value>(int32_t(-1))));
            else
        #endif
        return _mm256_or_ps(m, a.m);
    }

    template <typename T> ENOKI_INLINE Derived and_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_maskz_mov_ps(a.k, m);
            else
        #endif
        return _mm256_and_ps(m, a.m);
    }

    template <typename T> ENOKI_INLINE Derived xor_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_xor_ps(m, a.k, m, _mm256_set1_ps(memcpy_cast<Value>(int32_t(-1))));
            else
        #endif
        return _mm256_xor_ps(m, a.m);
    }

    template <typename T> ENOKI_INLINE Derived andnot_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_ps(m, a.k, _mm256_setzero_ps());
            else
        #endif
        return _mm256_andnot_ps(a.m, m);
    }

    #if defined(ENOKI_X86_AVX512)
        #define ENOKI_COMP(name, NAME) mask_t<Derived>::from_k(_mm256_cmp_ps_mask(m, a.m, _CMP_##NAME))
    #else
        #define ENOKI_COMP(name, NAME) mask_t<Derived>(_mm256_cmp_ps(m, a.m, _CMP_##NAME))
    #endif

    ENOKI_INLINE auto lt_ (Ref a) const { return ENOKI_COMP(lt,  LT_OQ);  }
    ENOKI_INLINE auto gt_ (Ref a) const { return ENOKI_COMP(gt,  GT_OQ);  }
    ENOKI_INLINE auto le_ (Ref a) const { return ENOKI_COMP(le,  LE_OQ);  }
    ENOKI_INLINE auto ge_ (Ref a) const { return ENOKI_COMP(ge,  GE_OQ);  }
    ENOKI_INLINE auto eq_ (Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(eq(Int(derived()), Int(a)));
        else
            return ENOKI_COMP(eq, EQ_OQ);
    }

    ENOKI_INLINE auto neq_(Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(neq(Int(derived()), Int(a)));
        else
            return ENOKI_COMP(neq, NEQ_UQ);
    }

    #undef ENOKI_COMP

    ENOKI_INLINE Derived abs_()      const { return _mm256_andnot_ps(_mm256_set1_ps(-0.f), m); }
    ENOKI_INLINE Derived min_(Ref b) const { return _mm256_min_ps(b.m, m); }
    ENOKI_INLINE Derived max_(Ref b) const { return _mm256_max_ps(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm256_ceil_ps(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm256_floor_ps(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm256_sqrt_ps(m);     }

    ENOKI_INLINE Derived round_() const {
        return _mm256_round_ps(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE Derived trunc_() const {
        return _mm256_round_ps(m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    template <typename Mask>
    static ENOKI_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(ENOKI_X86_AVX512)
            return _mm256_blendv_ps(f.m, t.m, m.m);
        #else
            return _mm256_mask_blend_ps(m.k, f.m, t.m);
        #endif
    }

#if defined(ENOKI_X86_FMA)
    ENOKI_INLINE Derived fmadd_   (Ref b, Ref c) const { return _mm256_fmadd_ps   (m, b.m, c.m); }
    ENOKI_INLINE Derived fmsub_   (Ref b, Ref c) const { return _mm256_fmsub_ps   (m, b.m, c.m); }
    ENOKI_INLINE Derived fnmadd_  (Ref b, Ref c) const { return _mm256_fnmadd_ps  (m, b.m, c.m); }
    ENOKI_INLINE Derived fnmsub_  (Ref b, Ref c) const { return _mm256_fnmsub_ps  (m, b.m, c.m); }
#endif

    template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
    ENOKI_INLINE Derived shuffle_() const {
        #if defined(ENOKI_X86_AVX2)
            return _mm256_permutevar8x32_ps(m,
                _mm256_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7));
        #else
            return Base::template shuffle_<I0, I1, I2, I3, I4, I5, I6, I7>();
        #endif
    }


    template <typename Index>
    ENOKI_INLINE Derived shuffle_(const Index &index_) const {
        #if defined(ENOKI_X86_AVX2)
            return _mm256_permutevar8x32_ps(m, index_.m);
        #else
            __m128i i0 = low(index_).m,
                    i1 = high(index_).m;

            // swap low and high part of table
            __m256 m2 = _mm256_permute2f128_ps(m, m, 1);

            __m256i index = _mm256_insertf128_si256(_mm256_castsi128_si256(i0), i1, 1);

            __m256  r0 = _mm256_permutevar_ps(m,  index),
                    r1 = _mm256_permutevar_ps(m2, index);

            __m128i k0 = _mm_slli_epi32(i0,  29),
                    k1 = _mm_slli_epi32(_mm_xor_si128(i1, _mm_set1_epi32(4)),  29);

            __m256 k = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_castsi128_ps(k0)),
                _mm_castsi128_ps(k1), 1);

            return _mm256_blendv_ps(r0, r1, k);
        #endif
    }

#if defined(ENOKI_X86_AVX512)
    ENOKI_INLINE Derived ldexp_(Ref arg) const { return _mm256_scalef_ps(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm256_getmant_ps(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm256_getexp_ps(m));
    }
#endif

    ENOKI_INLINE Derived rcp_() const {
        __m256 r;
        #if defined(ENOKI_X86_AVX512)
            r = _mm256_rcp14_ps(m); /* rel error < 2^-14 */
        #else
            r = _mm256_rcp_ps(m);   /* rel error < 1.5*2^-12 */
        #endif

        // Refine using one Newton-Raphson iteration
        __m256 t0 = _mm256_add_ps(r, r),
               t1 = _mm256_mul_ps(r, m),
               ro = r;
        (void) ro;

        #if defined(ENOKI_X86_FMA)
            r = _mm256_fnmadd_ps(t1, r, t0);
        #else
            r = _mm256_sub_ps(t0, _mm256_mul_ps(r, t1));
        #endif

        #if defined(ENOKI_X86_AVX512)
            return _mm256_fixupimm_ps(r, m, _mm256_set1_epi32(0x0087A622), 0);
        #else
            return _mm256_blendv_ps(r, ro, t1); /* mask bit is '1' iff t1 == nan */
        #endif
    }

    ENOKI_INLINE Derived rsqrt_() const {
        /* Use best reciprocal square root approximation available
           on the current hardware and refine */
        __m256 r;
        #if defined(ENOKI_X86_AVX512)
            r = _mm256_rsqrt14_ps(m); /* rel error < 2^-14 */
        #else
            r = _mm256_rsqrt_ps(m);   /* rel error < 1.5*2^-12 */
        #endif

        // Refine using one Newton-Raphson iteration
        const __m256 c0 = _mm256_set1_ps(.5f),
                     c1 = _mm256_set1_ps(3.f);

        __m256 t0 = _mm256_mul_ps(r, c0),
               t1 = _mm256_mul_ps(r, m),
               ro = r;
        (void) ro;

        #if defined(ENOKI_X86_FMA)
            r = _mm256_mul_ps(_mm256_fnmadd_ps(t1, r, c1), t0);
        #else
            r = _mm256_mul_ps(_mm256_sub_ps(c1, _mm256_mul_ps(t1, r)), t0);
        #endif

        #if defined(ENOKI_X86_AVX512)
            return _mm256_fixupimm_ps(r, m, _mm256_set1_epi32(0x0383A622), 0);
        #else
            return _mm256_blendv_ps(r, ro, t1); /* mask bit is '1' iff t1 == nan */
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_()  const { return _mm256_movemask_ps(m) == 0xFF;}
    ENOKI_INLINE bool any_()  const { return _mm256_movemask_ps(m) != 0x0; }

    ENOKI_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_ps(m); }
    ENOKI_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    ENOKI_INLINE Value dot_(Ref a) const {
        __m256 dp = _mm256_dp_ps(m, a.m, 0b11110001);
        __m128 m0 = _mm256_castps256_ps128(dp);
        __m128 m1 = _mm256_extractf128_ps(dp, 1);
        __m128 m = _mm_add_ss(m0, m1);
        return _mm_cvtss_f32(m);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const {
        _mm256_store_ps((Value *) ENOKI_ASSUME_ALIGNED(ptr, 32), m);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        _mm256_storeu_ps((Value *) ptr, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr, size_t) {
        return _mm256_load_ps((const Value *) ENOKI_ASSUME_ALIGNED(ptr, 32));
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, size_t) {
        return _mm256_loadu_ps((const Value *) ptr);
    }

    static ENOKI_INLINE Derived empty_(size_t) { return _mm256_undefined_ps(); }
    static ENOKI_INLINE Derived zero_(size_t) { return _mm256_setzero_ps(); }

#if defined(ENOKI_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mmask_i32gather_ps(_mm256_setzero_ps(), mask.k, index.m, (const float *) ptr, 4);
            else
                return _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask.k, index.m, (const float *) ptr, 4);
        #else
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mask_i32gather_ps(_mm256_setzero_ps(), (const float *) ptr, index.m, mask.m, 4);
            else
                return Derived(
                    _mm256_mask_i64gather_ps(_mm_setzero_ps(), (const float *) ptr, low(index).m, low(mask).m, 4),
                    _mm256_mask_i64gather_ps(_mm_setzero_ps(), (const float *) ptr, high(index).m, high(mask).m, 4)
                );
        #endif
    }
#endif

#if defined(ENOKI_X86_AVX512)
    template <bool, typename Index, typename Mask>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm256_mask_i32scatter_ps(ptr, mask.k, index.m, m, 4);
        else
            _mm512_mask_i64scatter_ps(ptr, mask.k, index.m, m, 4);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} ENOKI_MAY_ALIAS;

/// Partial overload of StaticArrayImpl using AVX intrinsics (double precision)
template <bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<double, 4, IsMask_, Derived_>
  : StaticArrayBase<double, 4, IsMask_, Derived_> {

    ENOKI_PACKET_TYPE(double, 4, __m256d)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    ENOKI_INLINE StaticArrayImpl(Value value) : m(_mm256_set1_pd(value)) { }
    ENOKI_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m(_mm256_setr_pd(v0, v1, v2, v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

// #if defined(ENOKI_X86_F16C)
//     ENOKI_CONVERT(half) {
//         m = _mm256_cvtps_pd(
//             _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) a.derived().data())));
//     }
// #endif

    ENOKI_CONVERT(float) : m(_mm256_cvtps_pd(a.derived().m)) { }
    ENOKI_CONVERT(int32_t) : m(_mm256_cvtepi32_pd(a.derived().m)) { }

#if defined(ENOKI_X86_AVX512)
    ENOKI_CONVERT(uint32_t) : m(_mm256_cvtepu32_pd(a.derived().m)) { }
#endif

    ENOKI_CONVERT(double) : m(a.derived().m) { }

#if defined(ENOKI_X86_AVX512)
    ENOKI_CONVERT(int64_t) : m(_mm256_cvtepi64_pd(a.derived().m)) { }
    ENOKI_CONVERT(uint64_t) : m(_mm256_cvtepu64_pd(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    ENOKI_REINTERPRET_MASK(bool) {
        int ival;
        memcpy(&ival, a.derived().data(), 4);
        __m128i value = _mm_cmpgt_epi8(
            _mm_cvtsi32_si128(ival), _mm_setzero_si128());
        #if defined(ENOKI_X86_AVX2)
            m = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(value));
        #else
            m = _mm256_castsi256_pd(_mm256_insertf128_si256(
                    _mm256_castsi128_si256(_mm_cvtepi8_epi64(value)),
                    _mm_cvtepi8_epi64(_mm_srli_si128(value, 2)), 1));
        #endif
    }

    ENOKI_REINTERPRET_MASK(float)
        : m(_mm256_castsi256_pd(
              detail::mm256_cvtepi32_epi64(_mm_castps_si128(a.derived().m)))) { }

    ENOKI_REINTERPRET_MASK(int32_t)
        : m(_mm256_castsi256_pd(detail::mm256_cvtepi32_epi64(a.derived().m))) { }

    ENOKI_REINTERPRET_MASK(uint32_t)
        : m(_mm256_castsi256_pd(detail::mm256_cvtepi32_epi64(a.derived().m))) { }

    ENOKI_REINTERPRET(double) : m(a.derived().m) { }

#if defined(ENOKI_X86_AVX2)
    ENOKI_REINTERPRET(int64_t) : m(_mm256_castsi256_pd(a.derived().m)) { }
    ENOKI_REINTERPRET(uint64_t) : m(_mm256_castsi256_pd(a.derived().m)) { }
#else
    ENOKI_REINTERPRET(int64_t)
        : m(detail::concat(_mm_castsi128_pd(low(a).m),
                           _mm_castsi128_pd(high(a).m))) { }
    ENOKI_REINTERPRET(uint64_t)
        : m(detail::concat(_mm_castsi128_pd(low(a).m),
                           _mm_castsi128_pd(high(a).m))) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(detail::concat(a1.m, a2.m)) { }

    ENOKI_INLINE Array1 low_()  const { return _mm256_castpd256_pd128(m); }
    ENOKI_INLINE Array2 high_() const { return _mm256_extractf128_pd(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Derived add_(Ref a) const { return _mm256_add_pd(m, a.m); }
    ENOKI_INLINE Derived sub_(Ref a) const { return _mm256_sub_pd(m, a.m); }
    ENOKI_INLINE Derived mul_(Ref a) const { return _mm256_mul_pd(m, a.m); }
    ENOKI_INLINE Derived div_(Ref a) const { return _mm256_div_pd(m, a.m); }

    ENOKI_INLINE Derived not_() const {
        #if defined(ENOKI_X86_AVX512)
            __m256i mi = _mm256_castpd_si256(m);
            mi = _mm256_ternarylogic_epi32(mi, mi, mi, 0b01010101);
            return _mm256_castsi256_pd(mi);
        #else
            return _mm256_xor_pd(m, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
        #endif
    }

    ENOKI_INLINE Derived neg_() const {
        return _mm256_sub_pd(_mm256_setzero_pd(), m);
    }

    template <typename T> ENOKI_INLINE Derived or_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_pd(m, a.k, _mm256_set1_pd(memcpy_cast<Value>(int64_t(-1))));
            else
        #endif
        return _mm256_or_pd(m, a.m);
    }

    template <typename T> ENOKI_INLINE Derived and_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_maskz_mov_pd(a.k, m);
            else
        #endif
        return _mm256_and_pd(m, a.m);
    }

    template <typename T> ENOKI_INLINE Derived xor_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_xor_pd(m, a.k, m, _mm256_set1_pd(memcpy_cast<Value>(int64_t(-1))));
            else
        #endif
        return _mm256_xor_pd(m, a.m);
    }

    template <typename T> ENOKI_INLINE Derived andnot_(const T &a) const {
        #if defined(ENOKI_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_pd(m, a.k, _mm256_setzero_pd());
            else
        #endif
        return _mm256_andnot_pd(a.m, m);
    }

    #if defined(ENOKI_X86_AVX512)
        #define ENOKI_COMP(name, NAME) mask_t<Derived>::from_k(_mm256_cmp_pd_mask(m, a.m, _CMP_##NAME))
    #else
        #define ENOKI_COMP(name, NAME) mask_t<Derived>(_mm256_cmp_pd(m, a.m, _CMP_##NAME))
    #endif

    ENOKI_INLINE auto lt_ (Ref a) const { return ENOKI_COMP(lt,  LT_OQ);  }
    ENOKI_INLINE auto gt_ (Ref a) const { return ENOKI_COMP(gt,  GT_OQ);  }
    ENOKI_INLINE auto le_ (Ref a) const { return ENOKI_COMP(le,  LE_OQ);  }
    ENOKI_INLINE auto ge_ (Ref a) const { return ENOKI_COMP(ge,  GE_OQ);  }

    ENOKI_INLINE auto eq_ (Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(eq(Int(derived()), Int(a)));
        else
            return ENOKI_COMP(eq, EQ_OQ);
    }

    ENOKI_INLINE auto neq_(Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(neq(Int(derived()), Int(a)));
        else
            return ENOKI_COMP(neq, NEQ_UQ);
    }

    #undef ENOKI_COMP

    ENOKI_INLINE Derived abs_()      const { return _mm256_andnot_pd(_mm256_set1_pd(-0.), m); }
    ENOKI_INLINE Derived min_(Ref b) const { return _mm256_min_pd(b.m, m); }
    ENOKI_INLINE Derived max_(Ref b) const { return _mm256_max_pd(b.m, m); }
    ENOKI_INLINE Derived ceil_()     const { return _mm256_ceil_pd(m);     }
    ENOKI_INLINE Derived floor_()    const { return _mm256_floor_pd(m);    }
    ENOKI_INLINE Derived sqrt_()     const { return _mm256_sqrt_pd(m);     }

    ENOKI_INLINE Derived round_() const {
        return _mm256_round_pd(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    ENOKI_INLINE Derived trunc_() const {
        return _mm256_round_pd(m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    template <typename Mask>
    static ENOKI_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(ENOKI_X86_AVX512)
            return _mm256_blendv_pd(f.m, t.m, m.m);
        #else
            return _mm256_mask_blend_pd(m.k, f.m, t.m);
        #endif
    }

#if defined(ENOKI_X86_FMA)
    ENOKI_INLINE Derived fmadd_   (Ref b, Ref c) const { return _mm256_fmadd_pd   (m, b.m, c.m); }
    ENOKI_INLINE Derived fmsub_   (Ref b, Ref c) const { return _mm256_fmsub_pd   (m, b.m, c.m); }
    ENOKI_INLINE Derived fnmadd_  (Ref b, Ref c) const { return _mm256_fnmadd_pd  (m, b.m, c.m); }
    ENOKI_INLINE Derived fnmsub_  (Ref b, Ref c) const { return _mm256_fnmsub_pd  (m, b.m, c.m); }
#endif

#if defined(ENOKI_X86_AVX2)
    template <int I0, int I1, int I2, int I3>
    ENOKI_INLINE Derived shuffle_() const {
        return _mm256_permute4x64_pd(m, _MM_SHUFFLE(I3, I2, I1, I0));
    }

    template <typename Index>
    ENOKI_INLINE Derived shuffle_(const Index &index) const {
        return Base::shuffle_(index);
    }
#endif


#if defined(ENOKI_X86_AVX512)
    ENOKI_INLINE Derived ldexp_(Ref arg) const { return _mm256_scalef_pd(m, arg.m); }

    ENOKI_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm256_getmant_pd(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm256_getexp_pd(m));
    }
#endif

#if defined(ENOKI_X86_AVX512)
    ENOKI_INLINE Derived rcp_() const {
        __m256d r = _mm256_rcp14_pd(m); /* rel error < 2^-14 */

        __m256d ro = r, t0, t1;
        (void) ro;

        // Refine using 2 Newton-Raphson iterations
        ENOKI_UNROLL for (int i = 0; i < 2; ++i) {
            t0 = _mm256_add_pd(r, r);
            t1 = _mm256_mul_pd(r, m);
            r = _mm256_fnmadd_pd(t1, r, t0);
        }

        #if defined(ENOKI_X86_AVX512)
            return _mm256_fixupimm_pd(r, m, _mm256_set1_epi32(0x0087A622), 0);
        #else
            return _mm256_blendv_pd(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }

    ENOKI_INLINE Derived rsqrt_() const {
        __m256d r = _mm256_rsqrt14_pd(m); // rel error < 2^-14

        const __m256d c0 = _mm256_set1_pd(0.5),
                      c1 = _mm256_set1_pd(3.0);

        __m256d ro = r, t0, t1;
        (void) ro;

        // Refine using 2 Newton-Raphson iterations
        ENOKI_UNROLL for (int i = 0; i < 2; ++i) {
            t0 = _mm256_mul_pd(r, c0);
            t1 = _mm256_mul_pd(r, m);
            r = _mm256_mul_pd(_mm256_fnmadd_pd(t1, r, c1), t0);
        }

        #if defined(ENOKI_X86_AVX512)
            return _mm256_fixupimm_pd(r, m, _mm256_set1_epi32(0x0383A622), 0);
        #else
            return _mm256_blendv_pd(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    ENOKI_INLINE Value hsum_()  const { return hsum(low_() + high_()); }
    ENOKI_INLINE Value hprod_() const { return hprod(low_() * high_()); }
    ENOKI_INLINE Value hmin_()  const { return hmin(min(low_(), high_())); }
    ENOKI_INLINE Value hmax_()  const { return hmax(max(low_(), high_())); }

    ENOKI_INLINE bool all_()  const { return _mm256_movemask_pd(m) == 0xF;}
    ENOKI_INLINE bool any_()  const { return _mm256_movemask_pd(m) != 0x0; }

    ENOKI_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_pd(m); }
    ENOKI_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    ENOKI_INLINE void store_(void *ptr) const {
        _mm256_store_pd((Value *) ENOKI_ASSUME_ALIGNED(ptr, 32), m);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        _mm256_storeu_pd((Value *) ptr, m);
    }

    static ENOKI_INLINE Derived load_(const void *ptr, size_t) {
        return _mm256_load_pd((const Value *) ENOKI_ASSUME_ALIGNED(ptr, 32));
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, size_t) {
        return _mm256_loadu_pd((const Value *) ptr);
    }

    static ENOKI_INLINE Derived empty_(size_t) { return _mm256_undefined_pd(); }
    static ENOKI_INLINE Derived zero_(size_t) { return _mm256_setzero_pd(); }

#if defined(ENOKI_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if !defined(ENOKI_X86_AVX512)
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mask_i32gather_pd(_mm256_setzero_pd(), (const double *) ptr, index.m, mask.m, 8);
            else
                return _mm256_mask_i64gather_pd(_mm256_setzero_pd(), (const double *) ptr, index.m, mask.m, 8);
        #else
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm256_mmask_i32gather_pd(_mm256_setzero_pd(), mask.k, index.m, (const double *) ptr, 8);
            else
                return _mm256_mmask_i64gather_pd(_mm256_setzero_pd(), mask.k, index.m, (const double *) ptr, 8);
        #endif
    }
#endif

#if defined(ENOKI_X86_AVX512)
    template <bool, typename Index, typename Mask>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm256_mask_i32scatter_pd(ptr, mask.k, index.m, m, 8);
        else
            _mm256_mask_i64scatter_pd(ptr, mask.k, index.m, m, 8);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} ENOKI_MAY_ALIAS;

/// Partial overload of StaticArrayImpl for the n=3 case (double precision)
template <bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<double, 3, IsMask_, Derived_>
  : StaticArrayImpl<double, 4, IsMask_, Derived_> {
    ENOKI_PACKET_TYPE_3D(double)

#if defined(ENOKI_X86_F16C)
    // template <typename Derived2>
    // ENOKI_INLINE StaticArrayImpl(const StaticArrayBase<half, 3, IsMask_, Derived2> &a) {
    //     uint16_t temp[4];
    //     memcpy(temp, a.derived().data(), sizeof(uint16_t) * 3);
    //     temp[3] = 0;
    //     m = _mm256_cvtps_pd(_mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) temp)));
    // }
#endif

    template <int I0, int I1, int I2>
    ENOKI_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    template <typename Index>
    ENOKI_INLINE Derived shuffle_(const Index &index) const {
        return Base::shuffle_(index);
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define ENOKI_HORIZONTAL_OP(name, op)                                    \
        ENOKI_INLINE Value name##_() const {                                 \
            __m128d t1 = _mm256_extractf128_pd(m, 1);                        \
            __m128d t2 = _mm256_castpd256_pd128(m);                          \
            t1 = _mm_##op##_sd(t1, t2);                                      \
            t2 = _mm_permute_pd(t2, 1);                                      \
            t2 = _mm_##op##_sd(t2, t1);                                      \
            return _mm_cvtsd_f64(t2);                                        \
        }

    ENOKI_HORIZONTAL_OP(hsum, add)
    ENOKI_HORIZONTAL_OP(hprod, mul)
    ENOKI_HORIZONTAL_OP(hmin, min)
    ENOKI_HORIZONTAL_OP(hmax, max)

    #undef ENOKI_HORIZONTAL_OP

    ENOKI_INLINE bool all_() const { return (_mm256_movemask_pd(m) & 7) == 7; }
    ENOKI_INLINE bool any_() const { return (_mm256_movemask_pd(m) & 7) != 0; }

    ENOKI_INLINE uint32_t bitmask_() const { return (uint32_t) (_mm256_movemask_pd(m) & 7); }
    ENOKI_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Loading/writing data (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    static ENOKI_INLINE auto mask_() {
        #if defined(ENOKI_X86_AVX512)
            return mask_t<Derived>::from_k((__mmask8) 7);
        #else
            return mask_t<Derived>(
                _mm256_castsi256_pd(_mm256_setr_epi64x(-1, -1, -1, 0)));
        #endif
    }

    ENOKI_INLINE void store_(void *ptr) const {
        memcpy(ptr, &m, sizeof(Value) * 3);
    }

    ENOKI_INLINE void store_unaligned_(void *ptr) const {
        store_(ptr);
    }

    static ENOKI_INLINE Derived load_(const void *ptr, size_t) {
        return Base::load_unaligned_(ptr);
    }

    static ENOKI_INLINE Derived load_unaligned_(const void *ptr, size_t) {
        Derived result;
        memcpy(&result.m, ptr, sizeof(Value) * 3);
        return result;
    }

#if defined(ENOKI_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static ENOKI_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::gather_<false>(ptr, index, mask & mask_());
    }
#endif

#if defined(ENOKI_X86_AVX512)
    template <bool, typename Index, typename Mask>
    ENOKI_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        Base::scatter_<false>(ptr, index, mask & mask_());
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} ENOKI_MAY_ALIAS;

#if defined(ENOKI_X86_AVX512)
template <typename Derived_>
ENOKI_DECLARE_KMASK(float, 8, Derived_, int)
template <typename Derived_>
ENOKI_DECLARE_KMASK(double, 4, Derived_, int)
template <typename Derived_>
ENOKI_DECLARE_KMASK(double, 3, Derived_, int)
#endif

NAMESPACE_END(enoki)
