/*
    drjit/array_avx.h -- Packed SIMD array (AVX specialization)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
DRJIT_PACKET_DECLARE_COND(32, enable_if_t<std::is_floating_point_v<Type>>)
DRJIT_PACKET_DECLARE_COND(24, enable_if_t<(std::is_same_v<Type, double>)>)

/// Partial overload of StaticArrayImpl using AVX intrinsics (single precision)
template <bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<float, 8, IsMask_, Derived_>
  : StaticArrayBase<float, 8, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(float, 8, __m256)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm256_set1_ps((float) value)) {}
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3,
                                 Value v4, Value v5, Value v6, Value v7)
        : m(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

// #if defined(DRJIT_X86_F16C)
//     DRJIT_CONVERT(half)
//         : m(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) a.derived().data()))) { }
// #endif

    DRJIT_CONVERT(float) : m(a.derived().m) { }

#if defined(DRJIT_X86_AVX2)
    DRJIT_CONVERT(int32_t) : m(_mm256_cvtepi32_ps(a.derived().m)) { }
#else
    DRJIT_CONVERT(int32_t)
        : m(detail::concat(_mm_cvtepi32_ps(low(a).m), _mm_cvtepi32_ps(high(a).m))) { }
#endif

    DRJIT_CONVERT(uint32_t) {
        #if defined(DRJIT_X86_AVX512)
            m = _mm256_cvtepu32_ps(a.derived().m);
        #else
            int32_array_t<Derived> ai(a);
            Derived result =
                Derived(ai & 0x7fffffff) +
                (Derived(float(1u << 31)) & mask_t<Derived>(sr<31>(ai)));
            m = result.m;
        #endif
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(double)
        :m(_mm512_cvtpd_ps(a.derived().m)) { }
#else
    DRJIT_CONVERT(double)
        : m(detail::concat(_mm256_cvtpd_ps(low(a).m),
                           _mm256_cvtpd_ps(high(a).m))) { }
#endif

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(int64_t) : m(_mm512_cvtepi64_ps(a.derived().m)) { }
    DRJIT_CONVERT(uint64_t) : m(_mm512_cvtepu64_ps(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET_MASK(bool) {
        uint64_t ival;
        memcpy(&ival, a.derived().data(), 8);
        __m128i value = _mm_cmpgt_epi8(
            detail::mm_cvtsi64_si128((long long) ival), _mm_setzero_si128());
        #if defined(DRJIT_X86_AVX2)
            m = _mm256_castsi256_ps(_mm256_cvtepi8_epi32(value));
        #else
            m = _mm256_castsi256_ps(_mm256_insertf128_si256(
                    _mm256_castsi128_si256(_mm_cvtepi8_epi32(value)),
                    _mm_cvtepi8_epi32(_mm_srli_si128(value, 4)), 1));
        #endif
    }

    DRJIT_REINTERPRET(float) : m(a.derived().m) { }

#if defined(DRJIT_X86_AVX2)
    DRJIT_REINTERPRET(int32_t) : m(_mm256_castsi256_ps(a.derived().m)) { }
    DRJIT_REINTERPRET(uint32_t) : m(_mm256_castsi256_ps(a.derived().m)) { }
#else
    DRJIT_REINTERPRET(int32_t)
        : m(detail::concat(_mm_castsi128_ps(low(a).m),
                           _mm_castsi128_ps(high(a).m))) { }

    DRJIT_REINTERPRET(uint32_t)
        : m(detail::concat(_mm_castsi128_ps(low(a).m),
                           _mm_castsi128_ps(high(a).m))) { }
#endif

#if defined(DRJIT_X86_AVX512)
    /// Handled by KMask
#else
    DRJIT_REINTERPRET_MASK(double)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
              _mm256_castpd_si256(low(a).m), _mm256_castpd_si256(high(a).m)))) { }
#  if defined(DRJIT_X86_AVX2)
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(low(a).m, high(a).m))) { }
    DRJIT_REINTERPRET_MASK(uint64_t)
        : m(_mm256_castsi256_ps(
              detail::mm512_cvtepi64_epi32(low(a).m, high(a).m))) { }
#  else
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(_mm256_castsi256_ps(detail::mm512_cvtepi64_epi32(
             low(low(a)).m, high(low(a)).m,
             low(high(a)).m, high(high(a)).m))) { }
    DRJIT_REINTERPRET_MASK(uint64_t)
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

    DRJIT_INLINE Array1 low_()  const { return _mm256_castps256_ps128(m); }
    DRJIT_INLINE Array2 high_() const { return _mm256_extractf128_ps(m, 1); }


    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const    { return _mm256_add_ps(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const    { return _mm256_sub_ps(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const    { return _mm256_mul_ps(m, a.m); }
    DRJIT_INLINE Derived div_(Ref a) const    { return _mm256_div_ps(m, a.m); }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            __m256i mi = _mm256_castps_si256(m);
            mi = _mm256_ternarylogic_epi32(mi, mi, mi, 0b01010101);
            return _mm256_castsi256_ps(mi);
        #else
            return _mm256_xor_ps(m, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm256_xor_ps(m, _mm256_set1_ps(-0.f));
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_ps(m, a.k, _mm256_set1_ps(memcpy_cast<Value>(int32_t(-1))));
            else
        #endif
        return _mm256_or_ps(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_maskz_mov_ps(a.k, m);
            else
        #endif
        return _mm256_and_ps(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_xor_ps(m, a.k, m, _mm256_set1_ps(memcpy_cast<Value>(int32_t(-1))));
            else
        #endif
        return _mm256_xor_ps(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_ps(m, a.k, _mm256_setzero_ps());
            else
        #endif
        return _mm256_andnot_ps(a.m, m);
    }

    #if defined(DRJIT_X86_AVX512)
        #define DRJIT_COMP(name, NAME) mask_t<Derived>::from_k(_mm256_cmp_ps_mask(m, a.m, _CMP_##NAME))
    #else
        #define DRJIT_COMP(name, NAME) mask_t<Derived>(_mm256_cmp_ps(m, a.m, _CMP_##NAME))
    #endif

    DRJIT_INLINE auto lt_ (Ref a) const { return DRJIT_COMP(lt,  LT_OQ);  }
    DRJIT_INLINE auto gt_ (Ref a) const { return DRJIT_COMP(gt,  GT_OQ);  }
    DRJIT_INLINE auto le_ (Ref a) const { return DRJIT_COMP(le,  LE_OQ);  }
    DRJIT_INLINE auto ge_ (Ref a) const { return DRJIT_COMP(ge,  GE_OQ);  }
    DRJIT_INLINE auto eq_ (Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(eq(Int(derived()), Int(a)));
        else
            return DRJIT_COMP(eq, EQ_OQ);
    }

    DRJIT_INLINE auto neq_(Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(neq(Int(derived()), Int(a)));
        else
            return DRJIT_COMP(neq, NEQ_UQ);
    }

    #undef DRJIT_COMP

    DRJIT_INLINE Derived abs_()      const { return _mm256_andnot_ps(_mm256_set1_ps(-0.f), m); }
    DRJIT_INLINE Derived minimum_(Ref b) const { return _mm256_min_ps(b.m, m); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return _mm256_max_ps(b.m, m); }
    DRJIT_INLINE Derived ceil_()     const { return _mm256_ceil_ps(m);     }
    DRJIT_INLINE Derived floor_()    const { return _mm256_floor_ps(m);    }
    DRJIT_INLINE Derived sqrt_()     const { return _mm256_sqrt_ps(m);     }

    DRJIT_INLINE Derived round_() const {
        return _mm256_round_ps(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    DRJIT_INLINE Derived trunc_() const {
        return _mm256_round_ps(m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm256_blendv_ps(f.m, t.m, m.m);
        #else
            return _mm256_mask_blend_ps(m.k, f.m, t.m);
        #endif
    }

#if defined(DRJIT_X86_FMA)
    DRJIT_INLINE Derived fmadd_   (Ref b, Ref c) const { return _mm256_fmadd_ps   (m, b.m, c.m); }
    DRJIT_INLINE Derived fmsub_   (Ref b, Ref c) const { return _mm256_fmsub_ps   (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmadd_  (Ref b, Ref c) const { return _mm256_fnmadd_ps  (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmsub_  (Ref b, Ref c) const { return _mm256_fnmsub_ps  (m, b.m, c.m); }
#endif

    template <int I0, int I1, int I2, int I3, int I4, int I5, int I6, int I7>
    DRJIT_INLINE Derived shuffle_() const {
        #if defined(DRJIT_X86_AVX2)
            return _mm256_permutevar8x32_ps(m,
                _mm256_setr_epi32(I0, I1, I2, I3, I4, I5, I6, I7));
        #else
            return Base::template shuffle_<I0, I1, I2, I3, I4, I5, I6, I7>();
        #endif
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived ldexp_(Ref arg) const { return _mm256_scalef_ps(m, arg.m); }

    DRJIT_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm256_getmant_ps(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm256_getexp_ps(m));
    }
#endif

    DRJIT_INLINE Derived rcp_() const {
        __m256 r;
        #if defined(DRJIT_X86_AVX512)
            r = _mm256_rcp14_ps(m); /* rel error < 2^-14 */
        #else
            r = _mm256_rcp_ps(m);   /* rel error < 1.5*2^-12 */
        #endif

        // Refine using one Newton-Raphson iteration
        __m256 t0 = _mm256_add_ps(r, r),
               t1 = _mm256_mul_ps(r, m),
               ro = r;
        (void) ro;

        #if defined(DRJIT_X86_FMA)
            r = _mm256_fnmadd_ps(t1, r, t0);
        #else
            r = _mm256_sub_ps(t0, _mm256_mul_ps(r, t1));
        #endif

        #if defined(DRJIT_X86_AVX512)
            return _mm256_fixupimm_ps(r, m, _mm256_set1_epi32(0x0087A622), 0);
        #else
            return _mm256_blendv_ps(r, ro, t1); /* mask bit is '1' iff t1 == nan */
        #endif
    }

    DRJIT_INLINE Derived rsqrt_() const {
        /* Use best reciprocal square root approximation available
           on the current hardware and refine */
        __m256 r;
        #if defined(DRJIT_X86_AVX512)
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

        #if defined(DRJIT_X86_FMA)
            r = _mm256_mul_ps(_mm256_fnmadd_ps(t1, r, c1), t0);
        #else
            r = _mm256_mul_ps(_mm256_sub_ps(c1, _mm256_mul_ps(t1, r)), t0);
        #endif

        #if defined(DRJIT_X86_AVX512)
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

    DRJIT_INLINE Value sum_()  const { return sum(low_() + high_()); }
    DRJIT_INLINE Value prod_() const { return prod(low_() * high_()); }
    DRJIT_INLINE Value min_()  const { return min(minimum(low_(), high_())); }
    DRJIT_INLINE Value max_()  const { return max(maximum(low_(), high_())); }

    DRJIT_INLINE bool all_()  const { return _mm256_movemask_ps(m) == 0xFF;}
    DRJIT_INLINE bool any_()  const { return _mm256_movemask_ps(m) != 0x0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_ps(m); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    DRJIT_INLINE Value dot_(Ref a) const {
        __m256 dp = _mm256_dp_ps(m, a.m, 0b11110001);
        __m128 m0 = _mm256_castps256_ps128(dp);
        __m128 m1 = _mm256_extractf128_ps(dp, 1);
        __m128 mr = _mm_add_ss(m0, m1);
        return _mm_cvtss_f32(mr);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm256_store_ps((Value *) DRJIT_ASSUME_ALIGNED(ptr, 32), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm256_storeu_ps((Value *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm256_load_ps((const Value *) DRJIT_ASSUME_ALIGNED(ptr, 32));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm256_loadu_ps((const Value *) ptr);
    }

    static DRJIT_INLINE Derived empty_(size_t) { return _mm256_undefined_ps(); }
    static DRJIT_INLINE Derived zero_(size_t) { return _mm256_setzero_ps(); }

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if defined(DRJIT_X86_AVX512)
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

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm256_mask_i32scatter_ps(ptr, mask.k, index.m, m, 4);
        else
            _mm512_mask_i64scatter_ps(ptr, mask.k, index.m, m, 4);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl using AVX intrinsics (double precision)
template <bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<double, 4, IsMask_, Derived_>
  : StaticArrayBase<double, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(double, 4, __m256d)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm256_set1_pd((double) value)) {}
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m(_mm256_setr_pd(v0, v1, v2, v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

// #if defined(DRJIT_X86_F16C)
//     DRJIT_CONVERT(half) {
//         m = _mm256_cvtps_pd(
//             _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) a.derived().data())));
//     }
// #endif

    DRJIT_CONVERT(float) : m(_mm256_cvtps_pd(a.derived().m)) { }
    DRJIT_CONVERT(int32_t) : m(_mm256_cvtepi32_pd(a.derived().m)) { }

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(uint32_t) : m(_mm256_cvtepu32_pd(a.derived().m)) { }
#endif

    DRJIT_CONVERT(double) : m(a.derived().m) { }

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(int64_t) : m(_mm256_cvtepi64_pd(a.derived().m)) { }
    DRJIT_CONVERT(uint64_t) : m(_mm256_cvtepu64_pd(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET_MASK(bool) {
        int ival;
        memcpy(&ival, a.derived().data(), 4);
        __m128i value = _mm_cmpgt_epi8(
            _mm_cvtsi32_si128(ival), _mm_setzero_si128());
        #if defined(DRJIT_X86_AVX2)
            m = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(value));
        #else
            m = _mm256_castsi256_pd(_mm256_insertf128_si256(
                    _mm256_castsi128_si256(_mm_cvtepi8_epi64(value)),
                    _mm_cvtepi8_epi64(_mm_srli_si128(value, 2)), 1));
        #endif
    }

    DRJIT_REINTERPRET_MASK(float)
        : m(_mm256_castsi256_pd(
              detail::mm256_cvtepi32_epi64(_mm_castps_si128(a.derived().m)))) { }

    DRJIT_REINTERPRET_MASK(int32_t)
        : m(_mm256_castsi256_pd(detail::mm256_cvtepi32_epi64(a.derived().m))) { }

    DRJIT_REINTERPRET_MASK(uint32_t)
        : m(_mm256_castsi256_pd(detail::mm256_cvtepi32_epi64(a.derived().m))) { }

    DRJIT_REINTERPRET(double) : m(a.derived().m) { }

#if defined(DRJIT_X86_AVX2)
    DRJIT_REINTERPRET(int64_t) : m(_mm256_castsi256_pd(a.derived().m)) { }
    DRJIT_REINTERPRET(uint64_t) : m(_mm256_castsi256_pd(a.derived().m)) { }
#else
    DRJIT_REINTERPRET(int64_t)
        : m(detail::concat(_mm_castsi128_pd(low(a).m),
                           _mm_castsi128_pd(high(a).m))) { }
    DRJIT_REINTERPRET(uint64_t)
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

    DRJIT_INLINE Array1 low_()  const { return _mm256_castpd256_pd128(m); }
    DRJIT_INLINE Array2 high_() const { return _mm256_extractf128_pd(m, 1); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return _mm256_add_pd(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const { return _mm256_sub_pd(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const { return _mm256_mul_pd(m, a.m); }
    DRJIT_INLINE Derived div_(Ref a) const { return _mm256_div_pd(m, a.m); }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            __m256i mi = _mm256_castpd_si256(m);
            mi = _mm256_ternarylogic_epi32(mi, mi, mi, 0b01010101);
            return _mm256_castsi256_pd(mi);
        #else
            return _mm256_xor_pd(m, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm256_xor_pd(m, _mm256_set1_pd(-0.0));
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_pd(m, a.k, _mm256_set1_pd(memcpy_cast<Value>(int64_t(-1))));
            else
        #endif
        return _mm256_or_pd(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_maskz_mov_pd(a.k, m);
            else
        #endif
        return _mm256_and_pd(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_xor_pd(m, a.k, m, _mm256_set1_pd(memcpy_cast<Value>(int64_t(-1))));
            else
        #endif
        return _mm256_xor_pd(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm256_mask_mov_pd(m, a.k, _mm256_setzero_pd());
            else
        #endif
        return _mm256_andnot_pd(a.m, m);
    }

    #if defined(DRJIT_X86_AVX512)
        #define DRJIT_COMP(name, NAME) mask_t<Derived>::from_k(_mm256_cmp_pd_mask(m, a.m, _CMP_##NAME))
    #else
        #define DRJIT_COMP(name, NAME) mask_t<Derived>(_mm256_cmp_pd(m, a.m, _CMP_##NAME))
    #endif

    DRJIT_INLINE auto lt_ (Ref a) const { return DRJIT_COMP(lt,  LT_OQ);  }
    DRJIT_INLINE auto gt_ (Ref a) const { return DRJIT_COMP(gt,  GT_OQ);  }
    DRJIT_INLINE auto le_ (Ref a) const { return DRJIT_COMP(le,  LE_OQ);  }
    DRJIT_INLINE auto ge_ (Ref a) const { return DRJIT_COMP(ge,  GE_OQ);  }

    DRJIT_INLINE auto eq_ (Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(eq(Int(derived()), Int(a)));
        else
            return DRJIT_COMP(eq, EQ_OQ);
    }

    DRJIT_INLINE auto neq_(Ref a) const {
        using Int = int_array_t<Derived>;
        if constexpr (IsMask_)
            return mask_t<Derived>(neq(Int(derived()), Int(a)));
        else
            return DRJIT_COMP(neq, NEQ_UQ);
    }

    #undef DRJIT_COMP

    DRJIT_INLINE Derived abs_()      const { return _mm256_andnot_pd(_mm256_set1_pd(-0.), m); }
    DRJIT_INLINE Derived minimum_(Ref b) const { return _mm256_min_pd(b.m, m); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return _mm256_max_pd(b.m, m); }
    DRJIT_INLINE Derived ceil_()     const { return _mm256_ceil_pd(m);     }
    DRJIT_INLINE Derived floor_()    const { return _mm256_floor_pd(m);    }
    DRJIT_INLINE Derived sqrt_()     const { return _mm256_sqrt_pd(m);     }

    DRJIT_INLINE Derived round_() const {
        return _mm256_round_pd(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    DRJIT_INLINE Derived trunc_() const {
        return _mm256_round_pd(m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm256_blendv_pd(f.m, t.m, m.m);
        #else
            return _mm256_mask_blend_pd(m.k, f.m, t.m);
        #endif
    }

#if defined(DRJIT_X86_FMA)
    DRJIT_INLINE Derived fmadd_   (Ref b, Ref c) const { return _mm256_fmadd_pd   (m, b.m, c.m); }
    DRJIT_INLINE Derived fmsub_   (Ref b, Ref c) const { return _mm256_fmsub_pd   (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmadd_  (Ref b, Ref c) const { return _mm256_fnmadd_pd  (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmsub_  (Ref b, Ref c) const { return _mm256_fnmsub_pd  (m, b.m, c.m); }
#endif

#if defined(DRJIT_X86_AVX2)
    template <int I0, int I1, int I2, int I3>
    DRJIT_INLINE Derived shuffle_() const {
        return _mm256_permute4x64_pd(m, _MM_SHUFFLE(I3, I2, I1, I0));
    }
#endif


#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived ldexp_(Ref arg) const { return _mm256_scalef_pd(m, arg.m); }

    DRJIT_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm256_getmant_pd(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm256_getexp_pd(m));
    }
#endif

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived rcp_() const {
        __m256d r = _mm256_rcp14_pd(m); /* rel error < 2^-14 */

        __m256d ro = r, t0, t1;
        (void) ro;

        // Refine using 2 Newton-Raphson iterations
        DRJIT_UNROLL for (int i = 0; i < 2; ++i) {
            t0 = _mm256_add_pd(r, r);
            t1 = _mm256_mul_pd(r, m);
            r = _mm256_fnmadd_pd(t1, r, t0);
        }

        #if defined(DRJIT_X86_AVX512)
            return _mm256_fixupimm_pd(r, m, _mm256_set1_epi32(0x0087A622), 0);
        #else
            return _mm256_blendv_pd(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }

    DRJIT_INLINE Derived rsqrt_() const {
        __m256d r = _mm256_rsqrt14_pd(m); // rel error < 2^-14

        const __m256d c0 = _mm256_set1_pd(0.5),
                      c1 = _mm256_set1_pd(3.0);

        __m256d ro = r, t0, t1;
        (void) ro;

        // Refine using 2 Newton-Raphson iterations
        DRJIT_UNROLL for (int i = 0; i < 2; ++i) {
            t0 = _mm256_mul_pd(r, c0);
            t1 = _mm256_mul_pd(r, m);
            r = _mm256_mul_pd(_mm256_fnmadd_pd(t1, r, c1), t0);
        }

        #if defined(DRJIT_X86_AVX512)
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

    DRJIT_INLINE Value sum_()  const { return sum(low_() + high_()); }
    DRJIT_INLINE Value prod_() const { return prod(low_() * high_()); }
    DRJIT_INLINE Value min_()  const { return min(minimum(low_(), high_())); }
    DRJIT_INLINE Value max_()  const { return max(maximum(low_(), high_())); }

    DRJIT_INLINE bool all_()  const { return _mm256_movemask_pd(m) == 0xF;}
    DRJIT_INLINE bool any_()  const { return _mm256_movemask_pd(m) != 0x0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm256_movemask_pd(m); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm256_store_pd((Value *) DRJIT_ASSUME_ALIGNED(ptr, 32), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm256_storeu_pd((Value *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm256_load_pd((const Value *) DRJIT_ASSUME_ALIGNED(ptr, 32));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm256_loadu_pd((const Value *) ptr);
    }

    static DRJIT_INLINE Derived empty_(size_t) { return _mm256_undefined_pd(); }
    static DRJIT_INLINE Derived zero_(size_t) { return _mm256_setzero_pd(); }

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if !defined(DRJIT_X86_AVX512)
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

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm256_mask_i32scatter_pd(ptr, mask.k, index.m, m, 8);
        else
            _mm256_mask_i64scatter_pd(ptr, mask.k, index.m, m, 8);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl for the n=3 case (double precision)
template <bool IsMask_, typename Derived_> struct alignas(32)
    StaticArrayImpl<double, 3, IsMask_, Derived_>
  : StaticArrayImpl<double, 4, IsMask_, Derived_> {
    DRJIT_PACKET_TYPE_3D(double)

#if defined(DRJIT_X86_F16C)
    // template <typename Derived2>
    // DRJIT_INLINE StaticArrayImpl(const StaticArrayBase<half, 3, IsMask_, Derived2> &a) {
    //     uint16_t temp[4];
    //     memcpy(temp, a.derived().data(), sizeof(uint16_t) * 3);
    //     temp[3] = 0;
    //     m = _mm256_cvtps_pd(_mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) temp)));
    // }
#endif

    template <int I0, int I1, int I2>
    DRJIT_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op)                                    \
        DRJIT_INLINE Value name##_() const {                                 \
            __m128d t1 = _mm256_extractf128_pd(m, 1);                        \
            __m128d t2 = _mm256_castpd256_pd128(m);                          \
            t1 = _mm_##op##_sd(t1, t2);                                      \
            t2 = _mm_permute_pd(t2, 1);                                      \
            t2 = _mm_##op##_sd(t2, t1);                                      \
            return _mm_cvtsd_f64(t2);                                        \
        }

    DRJIT_HORIZONTAL_OP(sum, add)
    DRJIT_HORIZONTAL_OP(prod, mul)
    DRJIT_HORIZONTAL_OP(min, min)
    DRJIT_HORIZONTAL_OP(max, max)

    #undef DRJIT_HORIZONTAL_OP

    DRJIT_INLINE bool all_() const { return (_mm256_movemask_pd(m) & 7) == 7; }
    DRJIT_INLINE bool any_() const { return (_mm256_movemask_pd(m) & 7) != 0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) (_mm256_movemask_pd(m) & 7); }
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
            return mask_t<Derived>(
                _mm256_castsi256_pd(_mm256_setr_epi64x(-1, -1, -1, 0)));
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

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        return Base::template gather_<false>(ptr, index, mask & mask_());
    }
#endif

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
template <typename Derived_>
DRJIT_DECLARE_KMASK(float, 8, Derived_, int)
template <typename Derived_>
DRJIT_DECLARE_KMASK(double, 4, Derived_, int)
template <typename Derived_>
DRJIT_DECLARE_KMASK(double, 3, Derived_, int)
#endif

NAMESPACE_END(drjit)
