/*
    drjit/packet_sse42.h -- Packet arrays, SSE4.2 specialization

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
DRJIT_PACKET_DECLARE(16)
DRJIT_PACKET_DECLARE(12)

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (single precision)
template <bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<float, 4, IsMask_, Derived_>
  : StaticArrayBase<float, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(float, 4, __m128)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm_set1_ps((float) value)) {}

    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m(_mm_setr_ps(v0, v1, v2, v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

// #if defined(DRJIT_X86_F16C)
    // DRJIT_CONVERT(half) {
    //     m = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) a.derived().data()));
    // }
// #endif

    DRJIT_CONVERT(float) : m(a.derived().m) { }
    DRJIT_CONVERT(int32_t) : m(_mm_cvtepi32_ps(a.derived().m)) { }

    DRJIT_CONVERT(uint32_t) {
        #if defined(DRJIT_X86_AVX512)
            m = _mm_cvtepu32_ps(a.derived().m);
        #else
            int32_array_t<Derived> ai(a);
            Derived result =
                Derived(ai & 0x7fffffff) +
                (Derived(float(1u << 31)) & mask_t<Derived>(sr<31>(ai)));
            m = result.m;
        #endif
    }

#if defined(DRJIT_X86_AVX)
    DRJIT_CONVERT(double) : m(_mm256_cvtpd_ps(a.derived().m)) { }
#else
    DRJIT_CONVERT(double)
        : m(_mm_shuffle_ps(_mm_cvtpd_ps(low(a).m), _mm_cvtpd_ps(high(a).m),
                           _MM_SHUFFLE(1, 0, 1, 0))) { }
#endif

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(int64_t) : m(_mm256_cvtepi64_ps(a.derived().m)) { }
    DRJIT_CONVERT(uint64_t) : m(_mm256_cvtepu64_ps(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET(float) : m(a.derived().m) { }
    DRJIT_REINTERPRET(int32_t) : m(_mm_castsi128_ps(a.derived().m)) { }
    DRJIT_REINTERPRET(uint32_t) : m(_mm_castsi128_ps(a.derived().m)) { }

    DRJIT_REINTERPRET_MASK(bool) {
        int ival;
        memcpy(&ival, a.derived().data(), 4);
        m = _mm_castsi128_ps(_mm_cvtepi8_epi32(
            _mm_cmpgt_epi8(_mm_cvtsi32_si128(ival), _mm_setzero_si128())));
    }

#if !defined(DRJIT_X86_AVX512)
#  if defined(DRJIT_X86_AVX)
    DRJIT_REINTERPRET_MASK(double)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(_mm256_castpd_si256(a.derived().m)))) { }
#  else
    DRJIT_REINTERPRET_MASK(double)
        : m(_mm_castsi128_ps(detail::mm256_cvtepi64_epi32(
              _mm_castpd_si128(low(a).m), _mm_castpd_si128(high(a).m)))) { }
#  endif

#  if defined(DRJIT_X86_AVX2)
    DRJIT_REINTERPRET_MASK(uint64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(a.derived().m))) { }
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(a.derived().m))) { }
#  else
    DRJIT_REINTERPRET_MASK(uint64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(low(a).m, high(a).m))) { }
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(_mm_castsi128_ps(
              detail::mm256_cvtepi64_epi32(low(a).m, high(a).m))) { }
#  endif
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_ps(a1.entry(0), a1.entry(1), a2.entry(0), a2.entry(1))) { }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0), entry(1)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(2), entry(3)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const    { return _mm_add_ps(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const    { return _mm_sub_ps(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const    { return _mm_mul_ps(m, a.m); }
    DRJIT_INLINE Derived div_(Ref a) const    { return _mm_div_ps(m, a.m); }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            __m128i mi = _mm_castps_si128(m);
            mi = _mm_ternarylogic_epi32(mi, mi, mi, 0b01010101);
            return _mm_castsi128_ps(mi);
        #else
            return _mm_xor_ps(m, _mm_castsi128_ps(_mm_set1_epi32(-1)));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm_xor_ps(m, _mm_set1_ps(-0.f));
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_ps(m, a.k, _mm_set1_ps(memcpy_cast<Value>(int32_t(-1))));
            else
        #endif
        return _mm_or_ps(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_maskz_mov_ps(a.k, m);
            else
        #endif
        return _mm_and_ps(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_ps(m, a.k, _mm_setzero_ps());
            else
        #endif
        return _mm_andnot_ps(a.m, m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_xor_ps(m, a.k, m, _mm_set1_ps(memcpy_cast<Value>(int32_t(-1))));
            else
        #endif
        return _mm_xor_ps(m, a.m);
    }

    #if defined(DRJIT_X86_AVX512)
        #define DRJIT_COMP(name, NAME) mask_t<Derived>::from_k(_mm_cmp_ps_mask(m, a.m, _CMP_##NAME))
    #elif defined(DRJIT_X86_AVX)
        #define DRJIT_COMP(name, NAME) mask_t<Derived>(_mm_cmp_ps(m, a.m, _CMP_##NAME))
    #else
        #define DRJIT_COMP(name, NAME) mask_t<Derived>(_mm_cmp##name##_ps(m, a.m))
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

    DRJIT_INLINE Derived abs_()      const { return _mm_andnot_ps(_mm_set1_ps(-0.f), m); }
    DRJIT_INLINE Derived minimum_(Ref b) const { return _mm_min_ps(b.m, m); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return _mm_max_ps(b.m, m); }
    DRJIT_INLINE Derived sqrt_()     const { return _mm_sqrt_ps(m);     }

    DRJIT_INLINE Derived floor_() const {
        return _mm_round_ps(m, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    }

    DRJIT_INLINE Derived ceil_() const {
        return _mm_round_ps(m, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    }

    DRJIT_INLINE Derived round_() const {
        return _mm_round_ps(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    DRJIT_INLINE Derived trunc_() const {
        return _mm_round_ps(m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm_blendv_ps(f.m, t.m, m.m);
        #else
            return _mm_mask_blend_ps(m.k, f.m, t.m);
        #endif
    }

#if defined(DRJIT_X86_FMA)
    DRJIT_INLINE Derived fmadd_   (Ref b, Ref c) const { return _mm_fmadd_ps   (m, b.m, c.m); }
    DRJIT_INLINE Derived fmsub_   (Ref b, Ref c) const { return _mm_fmsub_ps   (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmadd_  (Ref b, Ref c) const { return _mm_fnmadd_ps  (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmsub_  (Ref b, Ref c) const { return _mm_fnmsub_ps  (m, b.m, c.m); }
#endif

    template <int I0, int I1, int I2, int I3>
    DRJIT_INLINE Derived shuffle_() const {
        #if defined(DRJIT_X86_AVX)
            return _mm_permute_ps(m, _MM_SHUFFLE(I3, I2, I1, I0));
        #else
            return _mm_shuffle_ps(m, m, _MM_SHUFFLE(I3, I2, I1, I0));
        #endif
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived ldexp_(Ref arg) const { return _mm_scalef_ps(m, arg.m); }

    DRJIT_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm_getmant_ps(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm_getexp_ps(m));
    }
#endif

    DRJIT_INLINE Derived rcp_() const {
        __m128 r;
        #if defined(DRJIT_X86_AVX512)
            r = _mm_rcp14_ps(m); // rel error < 2^-14
        #else
            r = _mm_rcp_ps(m);   // rel error < 1.5*2^-12
        #endif

        // Refine using one Newton-Raphson iteration
        __m128 t0 = _mm_add_ps(r, r),
               t1 = _mm_mul_ps(r, m),
               ro = r;
        (void) ro;

        #if defined(DRJIT_X86_FMA)
            r = _mm_fnmadd_ps(t1, r, t0);
        #else
            r = _mm_sub_ps(t0, _mm_mul_ps(r, t1));
        #endif

        #if defined(DRJIT_X86_AVX512)
            return _mm_fixupimm_ps(r, m, _mm_set1_epi32(0x0087A622), 0);
        #else
            return _mm_blendv_ps(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }

    DRJIT_INLINE Derived rsqrt_() const {
        __m128 r;
        #if defined(DRJIT_X86_AVX512)
            r = _mm_rsqrt14_ps(m); // rel error < 2^-14
        #else
            r = _mm_rsqrt_ps(m);   // rel error < 1.5*2^-12
        #endif

        // Refine using one Newton-Raphson iteration
        const __m128 c0 = _mm_set1_ps(.5f),
                     c1 = _mm_set1_ps(3.f);

        __m128 t0 = _mm_mul_ps(r, c0),
               t1 = _mm_mul_ps(r, m),
               ro = r;
        (void) ro;

        #if defined(DRJIT_X86_FMA)
            r = _mm_mul_ps(_mm_fnmadd_ps(t1, r, c1), t0);
        #else
            r = _mm_mul_ps(_mm_sub_ps(c1, _mm_mul_ps(t1, r)), t0);
        #endif

        #if defined(DRJIT_X86_AVX512)
            return _mm_fixupimm_ps(r, m, _mm_set1_epi32(0x0383A622), 0);
        #else
            return _mm_blendv_ps(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op)                                    \
        DRJIT_INLINE Value name##_() const {                                 \
            __m128 t1 = _mm_movehdup_ps(m);                                  \
            __m128 t2 = _mm_##op##_ps(m, t1);                                \
            t1 = _mm_movehl_ps(t1, t2);                                      \
            t2 = _mm_##op##_ss(t2, t1);                                      \
            return _mm_cvtss_f32(t2);                                        \
        }

    DRJIT_HORIZONTAL_OP(sum, add)
    DRJIT_HORIZONTAL_OP(prod, mul)
    DRJIT_HORIZONTAL_OP(min, min)
    DRJIT_HORIZONTAL_OP(max, max)

    #undef DRJIT_HORIZONTAL_OP

    DRJIT_INLINE bool all_()  const { return _mm_movemask_ps(m) == 0xF;}
    DRJIT_INLINE bool any_()  const { return _mm_movemask_ps(m) != 0x0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm_movemask_ps(m); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    DRJIT_INLINE Value dot_(Ref a) const {
        return _mm_cvtss_f32(_mm_dp_ps(m, a.m, 0b11110001));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm_store_ps((Value *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm_storeu_ps((Value *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm_load_ps((const Value *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm_loadu_ps((const Value *) ptr);
    }

    static DRJIT_INLINE Derived empty_(size_t) { return _mm_undefined_ps(); }
    static DRJIT_INLINE Derived zero_(size_t) { return _mm_setzero_ps(); }

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm_mmask_i32gather_ps(_mm_setzero_ps(), mask.k, index.m, (const float *) ptr, 4);
            else
                return _mm256_mmask_i64gather_ps(_mm_setzero_ps(), mask.k, index.m, (const float *) ptr, 4);
        #else
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm_mask_i32gather_ps(_mm_setzero_ps(), (const float *) ptr, index.m, mask.m, 4);
            else
                return _mm256_mask_i64gather_ps(_mm_setzero_ps(), (const float *) ptr, index.m, mask.m, 4);
        #endif
    }
#endif

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm_mask_i32scatter_ps(ptr, mask.k, index.m, m, 4);
        else
            _mm256_mask_i64scatter_ps(ptr, mask.k, index.m, m, 4);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (32 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<Value_, 4, IsMask_, Derived_, enable_if_int32_t<Value_>>
  : StaticArrayBase<Value_, 4, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(Value_, 4, __m128i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm_set1_epi32((int32_t) value)) { }
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1, Value v2, Value v3)
        : m(_mm_setr_epi32((int32_t) v0, (int32_t) v1, (int32_t) v2, (int32_t) v3)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    DRJIT_CONVERT(float) {
        if constexpr (std::is_signed_v<Value>) {
            m = _mm_cvttps_epi32(a.derived().m);
        } else {
#if defined(DRJIT_X86_AVX512)
            m = _mm_cvttps_epu32(a.derived().m);
#else
            constexpr uint32_t limit = 1u << 31;
            const __m128  limit_f = _mm_set1_ps((float) limit);
            const __m128i limit_i = _mm_set1_epi32((int) limit);

            __m128 v = a.derived().m;

            __m128i mask =
                _mm_castps_si128(_mm_cmpge_ps(v, limit_f));

            __m128i b2 = _mm_add_epi32(
                _mm_cvttps_epi32(_mm_sub_ps(v, limit_f)), limit_i);

            __m128i b1 = _mm_cvttps_epi32(v);

            m = _mm_blendv_epi8(b1, b2, mask);
#endif
        }
    }

    DRJIT_CONVERT(int32_t) : m(a.derived().m) { }
    DRJIT_CONVERT(uint32_t) : m(a.derived().m) { }

#if defined(DRJIT_X86_AVX)
    DRJIT_CONVERT(double) {
        if constexpr (std::is_signed_v<Value>) {
            m = _mm256_cvttpd_epi32(a.derived().m);
        } else {
#if defined(DRJIT_X86_AVX512)
            m = _mm256_cvttpd_epu32(a.derived().m);
#else
            DRJIT_TRACK_SCALAR("Constructor (converting, double[4] -> uint32[4])");
            for (size_t i = 0; i < Size; ++i)
                entry(i) = Value(a.derived().entry(i));
#endif
        }
    }
#endif

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(int64_t) { m = _mm256_cvtepi64_epi32(a.derived().m); }
    DRJIT_CONVERT(uint64_t) { m = _mm256_cvtepi64_epi32(a.derived().m); }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET(float) : m(_mm_castps_si128(a.derived().m)) { }
    DRJIT_REINTERPRET(int32_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(uint32_t) : m(a.derived().m) { }

    DRJIT_REINTERPRET_MASK(bool) {
        int ival;
        memcpy(&ival, a.derived().data(), 4);
        m = _mm_cvtepi8_epi32(
            _mm_cmpgt_epi8(_mm_cvtsi32_si128(ival), _mm_setzero_si128()));
    }

#if !defined(DRJIT_X86_AVX512)
#  if defined(DRJIT_X86_AVX)
    DRJIT_REINTERPRET_MASK(double)
        : m(detail::mm256_cvtepi64_epi32(_mm256_castpd_si256(a.derived().m))) { }
#  else
    DRJIT_REINTERPRET_MASK(double)
        : m(detail::mm256_cvtepi64_epi32(_mm_castpd_si128(low(a).m),
                                         _mm_castpd_si128(high(a).m))) { }
#  endif

#  if defined(DRJIT_X86_AVX2)
    DRJIT_REINTERPRET_MASK(uint64_t)
        : m(detail::mm256_cvtepi64_epi32(a.derived().m)) { }
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(detail::mm256_cvtepi64_epi32(a.derived().m)) {}
#  else
    DRJIT_REINTERPRET_MASK(uint64_t)
        : m(detail::mm256_cvtepi64_epi32(low(a).m, high(a).m)) { }
    DRJIT_REINTERPRET_MASK(int64_t)
        : m(detail::mm256_cvtepi64_epi32(low(a).m, high(a).m)) { }
#  endif
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_epi32((int32_t) a1.entry(0), (int32_t) a1.entry(1),
                           (int32_t) a2.entry(0), (int32_t) a2.entry(1))) { }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0), entry(1)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(2), entry(3)); }

    //! @}
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return _mm_add_epi32(m, a.m);   }
    DRJIT_INLINE Derived sub_(Ref a) const { return _mm_sub_epi32(m, a.m);   }
    DRJIT_INLINE Derived mul_(Ref a) const { return _mm_mullo_epi32(m, a.m); }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            return _mm_ternarylogic_epi32(m, m, m, 0b01010101);
        #else
            return _mm_xor_si128(m, _mm_set1_epi32(-1));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm_sub_epi32(_mm_setzero_si128(), m);
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_epi32(m, a.k, _mm_set1_epi32(-1));
            else
        #endif
        return _mm_or_si128(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_maskz_mov_epi32(a.k, m);
            else
        #endif
        return _mm_and_si128(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_xor_epi32(m, a.k, m, _mm_set1_epi32(-1));
            else
        #endif
        return _mm_xor_si128(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_epi32(m, a.k, _mm_setzero_si128());
            else
        #endif
        return _mm_andnot_si128(a.m, m);
    }

    template <int Imm> DRJIT_INLINE Derived sl_() const {
        return _mm_slli_epi32(m, Imm);
    }

    template <int Imm> DRJIT_INLINE Derived sr_() const {
        return std::is_signed_v<Value> ? _mm_srai_epi32(m, Imm)
                                       : _mm_srli_epi32(m, Imm);
    }

#if defined(DRJIT_X86_AVX2)
    DRJIT_INLINE Derived sl_(Ref k) const {
         return _mm_sllv_epi32(m, k.m);
    }

    DRJIT_INLINE Derived sr_(Ref k) const {
        return std::is_signed_v<Value> ? _mm_srav_epi32(m, k.m)
                                       : _mm_srlv_epi32(m, k.m);
    }
#else
    using Base::sl_;
    using Base::sr_;
#endif

    DRJIT_INLINE auto eq_(Ref a)  const {
        using Return = mask_t<Derived>;

        #if defined(DRJIT_X86_AVX512)
            return Return::from_k(_mm_cmpeq_epi32_mask(m, a.m));
        #else
            return Return(_mm_cmpeq_epi32(m, a.m));
        #endif
    }

    DRJIT_INLINE auto neq_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(_mm_cmpneq_epi32_mask(m, a.m));
        #else
            return ~eq_(a);
        #endif
    }

    DRJIT_INLINE auto lt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm_cmpgt_epi32(a.m, m));
            } else {
                const __m128i offset = _mm_set1_epi32((int32_t) 0x80000000ul);
                return Return(_mm_cmpgt_epi32(_mm_sub_epi32(a.m, offset),
                                              _mm_sub_epi32(m, offset)));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                      ? _mm_cmplt_epi32_mask(m, a.m)
                                      : _mm_cmplt_epu32_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto gt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm_cmpgt_epi32(m, a.m));
            } else {
                const __m128i offset = _mm_set1_epi32((int32_t) 0x80000000ul);
                return Return(_mm_cmpgt_epi32(_mm_sub_epi32(m, offset),
                                              _mm_sub_epi32(a.m, offset)));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm_cmpgt_epi32_mask(m, a.m)
                                  : _mm_cmpgt_epu32_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto le_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm_cmple_epi32_mask(m, a.m)
                                           : _mm_cmple_epu32_mask(m, a.m));
        #else
            return ~gt_(a);
        #endif
    }

    DRJIT_INLINE auto ge_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm_cmpge_epi32_mask(m, a.m)
                                           : _mm_cmpge_epu32_mask(m, a.m));
        #else
            return ~lt_(a);
        #endif
    }

    DRJIT_INLINE Derived minimum_(Ref a) const {
        return std::is_signed_v<Value> ? _mm_min_epi32(a.m, m)
                                       : _mm_min_epu32(a.m, m);
    }

    DRJIT_INLINE Derived maximum_(Ref a) const {
        return std::is_signed_v<Value> ? _mm_max_epi32(a.m, m)
                                       : _mm_max_epu32(a.m, m);
    }

    DRJIT_INLINE Derived abs_() const {
        return std::is_signed_v<Value> ? _mm_abs_epi32(m) : m;
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm_blendv_epi8(f.m, t.m, m.m);
        #else
            return _mm_mask_blend_epi32(m.k, f.m, t.m);
        #endif
    }

    template <int I0, int I1, int I2, int I3>
    DRJIT_INLINE Derived shuffle_() const {
        return _mm_shuffle_epi32(m, _MM_SHUFFLE(I3, I2, I1, I0));
    }

    DRJIT_INLINE Derived mulhi_(Ref a) const {
        Derived even, odd;

        if constexpr (std::is_signed_v<Value>) {
            even.m = _mm_srli_epi64(_mm_mul_epi32(m, a.m), 32);
            odd.m = _mm_mul_epi32(_mm_srli_epi64(m, 32), _mm_srli_epi64(a.m, 32));
        } else {
            even.m = _mm_srli_epi64(_mm_mul_epu32(m, a.m), 32);
            odd.m = _mm_mul_epu32(_mm_srli_epi64(m, 32), _mm_srli_epi64(a.m, 32));
        }

        return select(mask_t<Derived>(true, false, true, false), even, odd);
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived lzcnt_() const { return _mm_lzcnt_epi32(m); }
    DRJIT_INLINE Derived tzcnt_() const { return Value(32) - lzcnt(~derived() & (derived() - Value(1))); }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op)                                     \
        DRJIT_INLINE Value name##_() const {                                  \
            __m128i t1 = _mm_shuffle_epi32(m, 0x4e);                          \
            __m128i t2 = _mm_##op##_epi32(m, t1);                             \
            t1 = _mm_shufflelo_epi16(t2, 0x4e);                               \
            t2 = _mm_##op##_epi32(t2, t1);                                    \
            return (Value) _mm_cvtsi128_si32(t2);                             \
        }

    #define DRJIT_HORIZONTAL_OP_SIGNED(name, op)                              \
        DRJIT_INLINE Value name##_() const {                                  \
            __m128i t1 = _mm_shuffle_epi32(m, 0x4e);                          \
            __m128i t2 = std::is_signed_v<Value> ? _mm_##op##_epi32(m, t1) :  \
                                                   _mm_##op##_epu32(m, t1);   \
            t1 = _mm_shufflelo_epi16(t2, 0x4e);                               \
            t2 = std::is_signed_v<Value> ? _mm_##op##_epi32(t2, t1) :         \
                                           _mm_##op##_epu32(t2, t1);          \
            return (Value) _mm_cvtsi128_si32(t2);                             \
        }

    DRJIT_HORIZONTAL_OP(sum, add)
    DRJIT_HORIZONTAL_OP(prod, mullo)
    DRJIT_HORIZONTAL_OP_SIGNED(min, min)
    DRJIT_HORIZONTAL_OP_SIGNED(max, max)

    #undef DRJIT_HORIZONTAL_OP
    #undef DRJIT_HORIZONTAL_OP_SIGNED

    DRJIT_INLINE bool all_()  const { return _mm_movemask_ps(_mm_castsi128_ps(m)) == 0xF;}
    DRJIT_INLINE bool any_()  const { return _mm_movemask_ps(_mm_castsi128_ps(m)) != 0x0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm_movemask_ps(_mm_castsi128_ps(m)); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm_store_si128((__m128i *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm_storeu_si128((__m128i *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm_load_si128((const __m128i *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm_loadu_si128((const __m128i *) ptr);
    }

    static DRJIT_INLINE Derived empty_(size_t) { return _mm_undefined_si128(); }
    static DRJIT_INLINE Derived zero_(size_t) { return _mm_setzero_si128(); }

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm_mmask_i32gather_epi32(_mm_setzero_si128(), mask.k, index.m, (const int *) ptr, 4);
            else
                return _mm256_mmask_i64gather_epi32(_mm_setzero_si128(), mask.k, index.m, (const int *) ptr, 4);
        #else
            if constexpr (sizeof(scalar_t<Index>) == 4)
                return _mm_mask_i32gather_epi32(_mm_setzero_si128(), (const int *) ptr, index.m, mask.m, 4);
            else
                return _mm256_mask_i64gather_epi32(_mm_setzero_si128(), (const int *) ptr, index.m, mask.m, 4);
        #endif
    }
#endif

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            _mm_mask_i32scatter_epi32(ptr, mask.k, index.m, m, 4);
        else
            _mm256_mask_i64scatter_epi32(ptr, mask.k, index.m, m, 4);
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (double precision)
template <bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<double, 2, IsMask_, Derived_>
  : StaticArrayBase<double, 2, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(double, 2, __m128d)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm_set1_pd((double) value)) {}
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1)
        : m(_mm_setr_pd(v0, v1)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

    /* No vectorized conversions from float/[u]int32_t (too small) */

    DRJIT_CONVERT(double) : m(a.derived().m) { }

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(int64_t) : m(_mm_cvtepi64_pd(a.derived().m)) { }
    DRJIT_CONVERT(uint64_t) : m(_mm_cvtepu64_pd(a.derived().m)) { }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET_MASK(bool) {
        int16_t ival;
        memcpy(&ival, a.derived().data(), 2);
        m = _mm_castsi128_pd(_mm_cvtepi8_epi64(_mm_cmpgt_epi8(
            _mm_cvtsi32_si128((int) ival), _mm_setzero_si128())));
    }

    DRJIT_REINTERPRET_MASK(float) {
        DRJIT_TRACK_SCALAR("Constructor (reinterpreting, float32[2] -> double[2])");
        auto v0 = a.derived().entry(0), v1 = a.derived().entry(1);
        m = _mm_castps_pd(_mm_setr_ps(v0, v0, v1, v1));
    }

    DRJIT_REINTERPRET_MASK(int32_t) {
        DRJIT_TRACK_SCALAR("Constructor (reinterpreting, int32[2] -> double[2])");
        auto v0 = a.derived().entry(0), v1 = a.derived().entry(1);
        m = _mm_castsi128_pd(_mm_setr_epi32(v0, v0, v1, v1));
    }

    DRJIT_REINTERPRET_MASK(uint32_t) {
        DRJIT_TRACK_SCALAR("Constructor (reinterpreting, uint32[2] -> double[2])");
        auto v0 = a.derived().entry(0), v1 = a.derived().entry(1);
        m = _mm_castsi128_pd(_mm_setr_epi32((int32_t) v0, (int32_t) v0,
                                            (int32_t) v1, (int32_t) v1));
    }

    DRJIT_REINTERPRET(double) : m(a.derived().m) { }
    DRJIT_REINTERPRET(int64_t) : m(_mm_castsi128_pd(a.derived().m)) { }
    DRJIT_REINTERPRET(uint64_t) : m(_mm_castsi128_pd(a.derived().m)) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2)
        : m(_mm_setr_pd(a1.entry(0), a2.entry(0))) { }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return _mm_add_pd(m, a.m); }
    DRJIT_INLINE Derived sub_(Ref a) const { return _mm_sub_pd(m, a.m); }
    DRJIT_INLINE Derived mul_(Ref a) const { return _mm_mul_pd(m, a.m); }
    DRJIT_INLINE Derived div_(Ref a) const { return _mm_div_pd(m, a.m); }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            __m128i mi = _mm_castpd_si128(m);
            mi = _mm_ternarylogic_epi32(mi, mi, mi, 0b01010101);
            return _mm_castsi128_pd(mi);
        #else
            return _mm_xor_pd(m, _mm_castsi128_pd(_mm_set1_epi32(-1)));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm_xor_pd(m, _mm_set1_pd(-0.0));
    }

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_pd(m, a.k, _mm_set1_pd(memcpy_cast<Value>(int64_t(-1))));
            else
        #endif
        return _mm_or_pd(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_maskz_mov_pd(a.k, m);
            else
        #endif
        return _mm_and_pd(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_xor_pd(m, a.k, m, _mm_set1_pd(memcpy_cast<Value>(int64_t(-1))));
            else
        #endif
        return _mm_xor_pd(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_pd(m, a.k, _mm_setzero_pd());
            else
        #endif
        return _mm_andnot_pd(a.m, m);
    }

    #if defined(DRJIT_X86_AVX512)
        #define DRJIT_COMP(name, NAME) mask_t<Derived>::from_k(_mm_cmp_pd_mask(m, a.m, _CMP_##NAME))
    #elif defined(DRJIT_X86_AVX)
        #define DRJIT_COMP(name, NAME) mask_t<Derived>(_mm_cmp_pd(m, a.m, _CMP_##NAME))
    #else
        #define DRJIT_COMP(name, NAME) mask_t<Derived>(_mm_cmp##name##_pd(m, a.m))
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

    DRJIT_INLINE Derived abs_()      const { return _mm_andnot_pd(_mm_set1_pd(-0.), m); }
    DRJIT_INLINE Derived minimum_(Ref b) const { return _mm_min_pd(b.m, m); }
    DRJIT_INLINE Derived maximum_(Ref b) const { return _mm_max_pd(b.m, m); }
    DRJIT_INLINE Derived ceil_()     const { return _mm_ceil_pd(m);     }
    DRJIT_INLINE Derived floor_()    const { return _mm_floor_pd(m);    }
    DRJIT_INLINE Derived sqrt_()     const { return _mm_sqrt_pd(m);     }

    DRJIT_INLINE Derived round_() const {
        return _mm_round_pd(m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    DRJIT_INLINE Derived trunc_() const {
        return _mm_round_pd(m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    template <typename Mask>
    static DRJIT_INLINE Derived select_(const Mask &m, Ref t, Ref f) {
        #if !defined(DRJIT_X86_AVX512)
            return _mm_blendv_pd(f.m, t.m, m.m);
        #else
            return _mm_mask_blend_pd(m.k, f.m, t.m);
        #endif
    }

#if defined(DRJIT_X86_FMA)
    DRJIT_INLINE Derived fmadd_   (Ref b, Ref c) const { return _mm_fmadd_pd   (m, b.m, c.m); }
    DRJIT_INLINE Derived fmsub_   (Ref b, Ref c) const { return _mm_fmsub_pd   (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmadd_  (Ref b, Ref c) const { return _mm_fnmadd_pd  (m, b.m, c.m); }
    DRJIT_INLINE Derived fnmsub_  (Ref b, Ref c) const { return _mm_fnmsub_pd  (m, b.m, c.m); }
#endif

    #if defined(DRJIT_X86_AVX)
        #define DRJIT_SHUFFLE_PD(m, flags) _mm_permute_pd(m, flags)
    #else
        #define DRJIT_SHUFFLE_PD(m, flags) _mm_shuffle_pd(m, m, flags)
    #endif

    template <int I0, int I1>
    DRJIT_INLINE Derived shuffle_() const {
        return DRJIT_SHUFFLE_PD(m, (I1 << 1) | I0);
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived ldexp_(Ref arg) const { return _mm_scalef_pd(m, arg.m); }

    DRJIT_INLINE std::pair<Derived, Derived> frexp_() const {
        return std::make_pair<Derived, Derived>(
            _mm_getmant_pd(m, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src),
            _mm_getexp_pd(m));
    }
#endif

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived rcp_() const {
        /* Use best reciprocal approximation available on the current
           hardware and refine */
        __m128d r = _mm_rcp14_pd(m); /* rel error < 2^-14 */

        __m128d ro = r, t0, t1;
        (void) ro;

        // Refine using 2 Newton-Raphson iterations
        DRJIT_UNROLL for (int i = 0; i < 2; ++i) {
            t0 = _mm_add_pd(r, r);
            t1 = _mm_mul_pd(r, m);
            r = _mm_fnmadd_pd(t1, r, t0);
        }

        #if defined(DRJIT_X86_AVX512)
            return _mm_fixupimm_pd(r, m, _mm_set1_epi32(0x0087A622), 0);
        #else
            return _mm_blendv_pd(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }

    DRJIT_INLINE Derived rsqrt_() const {
        /* Use best reciprocal square root approximation available
           on the current hardware and refine */
        __m128d r = _mm_rsqrt14_pd(m); /* rel error < 2^-14 */

        const __m128d c0 = _mm_set1_pd(0.5),
                      c1 = _mm_set1_pd(3.0);

        __m128d ro = r, t0, t1;
        (void) ro;

        // Refine using 2 Newton-Raphson iterations
        DRJIT_UNROLL for (int i = 0; i < 2; ++i) {
            t0 = _mm_mul_pd(r, c0);
            t1 = _mm_mul_pd(r, m);
            r = _mm_mul_pd(_mm_fnmadd_pd(t1, r, c1), t0);
        }

        #if defined(DRJIT_X86_AVX512)
            return _mm_fixupimm_pd(r, m, _mm_set1_epi32(0x0383A622), 0);
        #else
            return _mm_blendv_pd(r, ro, t1); // mask bit is '1' iff t1 == nan
        #endif
    }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op) \
        DRJIT_INLINE Value name##_() const { \
            __m128d t0 = DRJIT_SHUFFLE_PD(m, 1); \
            __m128d t1 = _mm_##op##_sd(t0, m); \
            return  _mm_cvtsd_f64(t1); \
        }

    DRJIT_HORIZONTAL_OP(sum, add)
    DRJIT_HORIZONTAL_OP(prod, mul)
    DRJIT_HORIZONTAL_OP(min, min)
    DRJIT_HORIZONTAL_OP(max, max)

    #undef DRJIT_HORIZONTAL_OP
    #undef DRJIT_SHUFFLE_PD

    DRJIT_INLINE bool all_()  const { return _mm_movemask_pd(m) == 0x3;}
    DRJIT_INLINE bool any_()  const { return _mm_movemask_pd(m) != 0x0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm_movemask_pd(m); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    DRJIT_INLINE Value dot_(Ref a) const {
        return _mm_cvtsd_f64(_mm_dp_pd(m, a.m, 0b00110001));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm_store_pd((Value *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm_storeu_pd((Value *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm_load_pd((const Value *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm_loadu_pd((const Value *) ptr);
    }

    static DRJIT_INLINE Derived zero_(size_t) { return _mm_setzero_pd(); }
    static DRJIT_INLINE Derived empty_(size_t) { return _mm_undefined_pd(); }

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        if constexpr (sizeof(scalar_t<Index>) == 4) {
            return Base::template gather_<false>(ptr, index, mask);
        } else {
            #if defined(DRJIT_X86_AVX512)
                return _mm_mmask_i64gather_pd(_mm_setzero_pd(), mask.k, index.m, (const double *) ptr, 8);
            #else
                return _mm_mask_i64gather_pd(_mm_setzero_pd(), (const double *) ptr, index.m, mask.m, 8);
            #endif
        }
    }
#endif

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            Base::template scatter_<false>(ptr, index, mask);
        else
            _mm_mask_i64scatter_pd(ptr, mask.k, index.m, m, 8);
    }

    template <typename Mask>
    DRJIT_INLINE Value extract_(const Mask &mask) const {
        return _mm_cvtsd_f64(_mm_mask_compress_pd(_mm_setzero_pd(), mask.k, m));
    }
#endif

    //! @}
    // -----------------------------------------------------------------------
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl using SSE4.2 intrinsics (64 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<Value_, 2, IsMask_, Derived_, enable_if_int64_t<Value_>>
  : StaticArrayBase<Value_, 2, IsMask_, Derived_> {

    DRJIT_PACKET_TYPE(Value_, 2, __m128i)

    // -----------------------------------------------------------------------
    //! @{ \name Value constructors
    // -----------------------------------------------------------------------

    template <typename T, enable_if_scalar_t<T> = 0>
    DRJIT_INLINE StaticArrayImpl(T value) : m(_mm_set1_epi64x((int64_t) value)) { }
    DRJIT_INLINE StaticArrayImpl(Value v0, Value v1) {
        alignas(16) Value data[2];
        data[0] = (Value) v0;
        data[1] = (Value) v1;
        m = _mm_load_si128((__m128i *) data);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Type converting constructors
    // -----------------------------------------------------------------------

#if defined(DRJIT_X86_AVX512)
    DRJIT_CONVERT(double) {
        if constexpr (std::is_signed_v<Value>)
            m = _mm_cvttpd_epi64(a.derived().m);
        else
            m = _mm_cvttpd_epu64(a.derived().m);
    }
#endif

    DRJIT_CONVERT(int64_t) : m(a.derived().m) { }
    DRJIT_CONVERT(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Reinterpreting constructors, mask converters
    // -----------------------------------------------------------------------

    DRJIT_REINTERPRET_MASK(bool) {
        int16_t ival;
        memcpy(&ival, a.derived().data(), 2);
        m = _mm_cvtepi8_epi64(
            _mm_cmpgt_epi8(_mm_cvtsi32_si128((int) ival), _mm_setzero_si128()));
    }

    DRJIT_REINTERPRET_MASK(float) {
        DRJIT_TRACK_SCALAR("Constructor (reinterpreting, float32[2] -> int64[2])");
        auto v0 = a.derived().entry(0), v1 = a.derived().entry(1);
        m = _mm_castps_si128(_mm_setr_ps(v0, v0, v1, v1));
    }

    DRJIT_REINTERPRET_MASK(int32_t) {
        DRJIT_TRACK_SCALAR("Constructor (reinterpreting, int32[2] -> int64[2])");
        auto v0 = a.derived().entry(0), v1 = a.derived().entry(1);
        m = _mm_setr_epi32(v0, v0, v1, v1);
    }

    DRJIT_REINTERPRET_MASK(uint32_t) {
        DRJIT_TRACK_SCALAR("Constructor (reinterpreting, uint32[2] -> int64[2])");
        auto v0 = a.derived().entry(0), v1 = a.derived().entry(1);
        m = _mm_setr_epi32((int32_t) v0, (int32_t) v0, (int32_t) v1,
                           (int32_t) v1);
    }

    DRJIT_REINTERPRET(double) : m(_mm_castpd_si128(a.derived().m)) { }
    DRJIT_REINTERPRET(int64_t) : m(a.derived().m) { }
    DRJIT_REINTERPRET(uint64_t) : m(a.derived().m) { }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Converting from/to half size vectors
    // -----------------------------------------------------------------------

    StaticArrayImpl(const Array1 &a1, const Array2 &a2) {
        alignas(16) Value data[2];
        data[0] = (Value) a1.entry(0);
        data[1] = (Value) a2.entry(0);
        m = _mm_load_si128((__m128i *) data);
    }

    DRJIT_INLINE Array1 low_()  const { return Array1(entry(0)); }
    DRJIT_INLINE Array2 high_() const { return Array2(entry(1)); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    DRJIT_INLINE Derived add_(Ref a) const { return _mm_add_epi64(m, a.m);   }
    DRJIT_INLINE Derived sub_(Ref a) const { return _mm_sub_epi64(m, a.m);   }

    DRJIT_INLINE Derived not_() const {
        #if defined(DRJIT_X86_AVX512)
            return _mm_ternarylogic_epi64(m, m, m, 0b01010101);
        #else
            return _mm_xor_si128(m, _mm_set1_epi64x(-1));
        #endif
    }

    DRJIT_INLINE Derived neg_() const {
        return _mm_sub_epi64(_mm_setzero_si128(), m);
    }

#if defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived mul_(Ref a) const {
        return _mm_mullo_epi64(m, a.m);
    }
#endif

    template <typename T> DRJIT_INLINE Derived or_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_epi64(m, a.k, _mm_set1_epi64x(-1));
            else
        #endif
        return _mm_or_si128(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived and_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_maskz_mov_epi64(a.k, m);
            else
        #endif
        return _mm_and_si128(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived xor_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_xor_epi64(m, a.k, m, _mm_set1_epi64x(-1));
            else
        #endif
        return _mm_xor_si128(m, a.m);
    }

    template <typename T> DRJIT_INLINE Derived andnot_(const T &a) const {
        #if defined(DRJIT_X86_AVX512)
            if constexpr (is_mask_v<T>)
                return _mm_mask_mov_epi64(m, a.k, _mm_setzero_si128());
            else
        #endif
        return _mm_andnot_si128(a.m, m);
    }

    template <int Imm> DRJIT_INLINE Derived sl_() const {
        return _mm_slli_epi64(m, Imm);
    }

    template <int Imm> DRJIT_INLINE Derived sr_() const {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                return _mm_srai_epi64(m, Imm);
            #else
                return Base::template sr_<Imm>();
            #endif
        } else {
            return _mm_srli_epi64(m, Imm);
        }
    }

#if defined(DRJIT_X86_AVX2)
    DRJIT_INLINE Derived sl_(Ref k) const {
        return _mm_sllv_epi64(m, k.m);
    }

    DRJIT_INLINE Derived sr_(Ref a) const {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                return _mm_srav_epi64(m, a.m);
            #else
                return Base::sr_(a);
            #endif
        } else {
            return _mm_srlv_epi64(m, a.m);
        }
    }
#else
    using Base::sl_;
    using Base::sr_;
#endif

    DRJIT_INLINE auto eq_(Ref a)  const {
        using Return = mask_t<Derived>;

        #if defined(DRJIT_X86_AVX512)
            return Return::from_k(_mm_cmpeq_epi64_mask(m, a.m));
        #else
            return Return(_mm_cmpeq_epi64(m, a.m));
        #endif
    }

    DRJIT_INLINE auto neq_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(_mm_cmpneq_epi64_mask(m, a.m));
        #else
            return ~eq_(a);
        #endif
    }

    DRJIT_INLINE auto lt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm_cmpgt_epi64(a.m, m));
            } else {
                const __m128i offset =
                    _mm_set1_epi64x((long long) 0x8000000000000000ull);
                return Return(_mm_cmpgt_epi64(
                    _mm_sub_epi64(a.m, offset),
                    _mm_sub_epi64(m, offset)
                ));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm_cmplt_epi64_mask(m, a.m)
                                  : _mm_cmplt_epu64_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto gt_(Ref a) const {
        using Return = mask_t<Derived>;

        #if !defined(DRJIT_X86_AVX512)
            if constexpr (std::is_signed_v<Value>) {
                return Return(_mm_cmpgt_epi64(m, a.m));
            } else {
                const __m128i offset =
                    _mm_set1_epi64x((long long) 0x8000000000000000ull);
                return Return(_mm_cmpgt_epi64(
                    _mm_sub_epi64(m, offset),
                    _mm_sub_epi64(a.m, offset)
                ));
            }
        #else
            return Return::from_k(std::is_signed_v<Value>
                                  ? _mm_cmpgt_epi64_mask(m, a.m)
                                  : _mm_cmpgt_epu64_mask(m, a.m));
        #endif
    }

    DRJIT_INLINE auto le_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm_cmple_epi64_mask(m, a.m)
                                           : _mm_cmple_epu64_mask(m, a.m));
        #else
            return ~gt_(a);
        #endif
    }

    DRJIT_INLINE auto ge_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return mask_t<Derived>::from_k(std::is_signed_v<Value>
                                           ? _mm_cmpge_epi64_mask(m, a.m)
                                           : _mm_cmpge_epu64_mask(m, a.m));
        #else
            return ~lt_(a);
        #endif
    }

    DRJIT_INLINE Derived minimum_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return std::is_signed_v<Value> ? _mm_min_epi64(a.m, m)
                                           : _mm_min_epu64(a.m, m);
        #else
            return select(derived() < a, derived(), a);
        #endif
    }

    DRJIT_INLINE Derived maximum_(Ref a) const {
        #if defined(DRJIT_X86_AVX512)
            return std::is_signed_v<Value> ? _mm_max_epi64(a.m, m)
                                           : _mm_max_epu64(a.m, m);
        #else
            return select(derived() > a, derived(), a);
        #endif
    }

    DRJIT_INLINE Derived abs_() const {
        if constexpr (std::is_signed_v<Value>) {
            #if defined(DRJIT_X86_AVX512)
                return _mm_abs_epi64(m);
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
            return _mm_blendv_epi8(f.m, t.m, m.m);
        #else
            return _mm_mask_blend_epi64(m.k, f.m, t.m);
        #endif
    }

    template <int I0, int I1>
    DRJIT_INLINE Derived shuffle_() const {
        return _mm_shuffle_epi32(
            m, _MM_SHUFFLE(I1 * 2 + 1, I1 * 2, I0 * 2 + 1, I0 * 2));
    }

#if defined(DRJIT_X86_AVX512) && defined(DRJIT_X86_AVX512)
    DRJIT_INLINE Derived lzcnt_() const { return _mm_lzcnt_epi64(m); }
    DRJIT_INLINE Derived tzcnt_() const { return Value(64) - lzcnt(~derived() & (derived() - Value(1))); }
#endif

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op)                                     \
        DRJIT_INLINE Value name##_() const {                                  \
            Value t1 = Value(detail::mm_extract_epi64<1>(m));                 \
            Value t2 = Value(detail::mm_cvtsi128_si64(m));                    \
            return op;                                                        \
        }

    DRJIT_HORIZONTAL_OP(sum,  t1 + t2)
    DRJIT_HORIZONTAL_OP(prod, t1 * t2)
    DRJIT_HORIZONTAL_OP(min,  minimum(t1, t2))
    DRJIT_HORIZONTAL_OP(max,  maximum(t1, t2))

    #undef DRJIT_HORIZONTAL_OP

    DRJIT_INLINE bool all_()  const { return _mm_movemask_pd(_mm_castsi128_pd(m)) == 0x3;}
    DRJIT_INLINE bool any_()  const { return _mm_movemask_pd(_mm_castsi128_pd(m)) != 0x0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm_movemask_pd(_mm_castsi128_pd(m)); }
    DRJIT_INLINE size_t count_() const { return (size_t) _mm_popcnt_u32(bitmask_()); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Initialization, loading/writing data
    // -----------------------------------------------------------------------

    DRJIT_INLINE void store_aligned_(void *ptr) const {
        _mm_store_si128((__m128i *) DRJIT_ASSUME_ALIGNED(ptr, 16), m);
    }

    DRJIT_INLINE void store_(void *ptr) const {
        _mm_storeu_si128((__m128i *) ptr, m);
    }

    static DRJIT_INLINE Derived load_aligned_(const void *ptr, size_t) {
        return _mm_load_si128((const __m128i *) DRJIT_ASSUME_ALIGNED(ptr, 16));
    }

    static DRJIT_INLINE Derived load_(const void *ptr, size_t) {
        return _mm_loadu_si128((const __m128i *) ptr);
    }

    static DRJIT_INLINE Derived empty_(size_t) { return _mm_undefined_si128(); }
    static DRJIT_INLINE Derived zero_(size_t) { return _mm_setzero_si128(); }

#if defined(DRJIT_X86_AVX2)
    template <bool, typename Index, typename Mask>
    static DRJIT_INLINE Derived gather_(const void *ptr, const Index &index, const Mask &mask) {
        if constexpr (sizeof(scalar_t<Index>) == 4) {
            return Base::template gather_<false>(ptr, index, mask);
        } else {
            #if defined(DRJIT_X86_AVX512)
                return _mm_mmask_i64gather_epi64(_mm_setzero_si128(), mask.k, index.m, (const long long *) ptr, 8);
            #else
                return _mm_mask_i64gather_epi64(_mm_setzero_si128(), (const long long *) ptr, index.m, mask.m, 8);
            #endif
        }
    }
#endif

#if defined(DRJIT_X86_AVX512)
    template <bool, typename Index, typename Mask>
    DRJIT_INLINE void scatter_(void *ptr, const Index &index, const Mask &mask) const {
        if constexpr (sizeof(scalar_t<Index>) == 4)
            Base::template scatter_<false>(ptr, index, mask);
        else
            _mm_mask_i64scatter_epi64(ptr, mask.k, index.m, m, 8);
    }

    template <typename Mask>
    DRJIT_INLINE Value extract_(const Mask &mask) const {
        return (Value) detail::mm_cvtsi128_si64(_mm_mask_compress_epi64(_mm_setzero_si128(), mask.k, m));
    }
#endif
} DRJIT_MAY_ALIAS;

/// Partial overload of StaticArrayImpl for the n=3 case (single precision)
template <bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<float, 3, IsMask_, Derived_>
  : StaticArrayImpl<float, 4, IsMask_, Derived_> {
    DRJIT_PACKET_TYPE_3D(float)

// #if defined(DRJIT_X86_F16C)
    // template <typename Derived2>
    // DRJIT_INLINE StaticArrayImpl(
    //     const StaticArrayBase<half, 3, IsMask_, Derived2> &a) {
    //     uint16_t temp[4];
    //     memcpy(temp, a.derived().data(), sizeof(uint16_t) * 3);
    //     temp[3] = 0;
    //     m = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *) temp));
    // }
// #endif

    template <int I0, int I1, int I2>
    DRJIT_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op)                                     \
        DRJIT_INLINE Value name##_() const {                                  \
            __m128 t1 = _mm_movehl_ps(m, m);                                  \
            __m128 t2 = _mm_##op##_ss(m, t1);                                 \
            t1 = _mm_movehdup_ps(m);                                          \
            t1 = _mm_##op##_ss(t1, t2);                                       \
            return _mm_cvtss_f32(t1);                                         \
        }

    DRJIT_HORIZONTAL_OP(sum, add)
    DRJIT_HORIZONTAL_OP(prod, mul)
    DRJIT_HORIZONTAL_OP(min, min)
    DRJIT_HORIZONTAL_OP(max, max)

    #undef DRJIT_HORIZONTAL_OP

    DRJIT_INLINE Value dot_(Ref a) const {
        return _mm_cvtss_f32(_mm_dp_ps(m, a.m, 0b01110001));
    }

    DRJIT_INLINE bool all_()  const { return (_mm_movemask_ps(m) & 7) == 7; }
    DRJIT_INLINE bool any_()  const { return (_mm_movemask_ps(m) & 7) != 0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm_movemask_ps(m) & 7; }
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
            return mask_t<Derived>(_mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0)));
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

/// Partial overload of StaticArrayImpl for the n=3 case (32 bit integers)
template <typename Value_, bool IsMask_, typename Derived_> struct alignas(16)
    StaticArrayImpl<Value_, 3, IsMask_, Derived_, enable_if_int32_t<Value_>>
  : StaticArrayImpl<Value_, 4, IsMask_, Derived_> {
    DRJIT_PACKET_TYPE_3D(Value_)

    template <int I0, int I1, int I2>
    DRJIT_INLINE Derived shuffle_() const {
        return Base::template shuffle_<I0, I1, I2, 3>();
    }

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations (adapted for the n=3 case)
    // -----------------------------------------------------------------------

    #define DRJIT_HORIZONTAL_OP(name, op)                                     \
        DRJIT_INLINE Value name##_() const {                                  \
            __m128i t1 = _mm_unpackhi_epi32(m, m);                            \
            __m128i t2 = _mm_##op##_epi32(m, t1);                             \
            t1 = _mm_shuffle_epi32(m, 1);                                     \
            t1 = _mm_##op##_epi32(t1, t2);                                    \
            return (Value) _mm_cvtsi128_si32(t1);                             \
        }

    #define DRJIT_HORIZONTAL_OP_SIGNED(name, op)                              \
        DRJIT_INLINE Value name##_() const {                                  \
            __m128i t2, t1 = _mm_unpackhi_epi32(m, m);                        \
            if constexpr (std::is_signed<Value>::value)                       \
                t2 = _mm_##op##_epi32(m, t1);                                 \
            else                                                              \
                t2 = _mm_##op##_epu32(m, t1);                                 \
            t1 = _mm_shuffle_epi32(m, 1);                                     \
            if constexpr (std::is_signed<Value>::value)                       \
                t1 = _mm_##op##_epi32(t1, t2);                                \
            else                                                              \
                t1 = _mm_##op##_epu32(t1, t2);                                \
            return (Value) _mm_cvtsi128_si32(t1);                             \
        }

    DRJIT_HORIZONTAL_OP(sum, add)
    DRJIT_HORIZONTAL_OP(prod, mullo)
    DRJIT_HORIZONTAL_OP_SIGNED(min, min)
    DRJIT_HORIZONTAL_OP_SIGNED(max, max)

    #undef DRJIT_HORIZONTAL_OP
    #undef DRJIT_HORIZONTAL_OP_SIGNED

    DRJIT_INLINE bool all_()  const { return (_mm_movemask_ps(_mm_castsi128_ps(m)) & 7) == 7;}
    DRJIT_INLINE bool any_()  const { return (_mm_movemask_ps(_mm_castsi128_ps(m)) & 7) != 0; }

    DRJIT_INLINE uint32_t bitmask_() const { return (uint32_t) _mm_movemask_ps(_mm_castsi128_ps(m)) & 7; }
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
            return mask_t<Derived>(_mm_setr_epi32(-1, -1, -1, 0));
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
template <typename Derived_> DRJIT_DECLARE_KMASK(float,  4, Derived_, int)
template <typename Derived_> DRJIT_DECLARE_KMASK(float,  3, Derived_, int)
template <typename Derived_> DRJIT_DECLARE_KMASK(double, 2, Derived_, int)
template <typename Value_, typename Derived_>
DRJIT_DECLARE_KMASK(Value_, 4, Derived_, enable_if_int32_t<Value_>)
template <typename Value_, typename Derived_>
DRJIT_DECLARE_KMASK(Value_, 3, Derived_, enable_if_int32_t<Value_>)
template <typename Value_, typename Derived_>
DRJIT_DECLARE_KMASK(Value_, 2, Derived_, enable_if_int64_t<Value_>)
#endif

NAMESPACE_END(drjit)
