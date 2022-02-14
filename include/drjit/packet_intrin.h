/*
    drjit/packet_intrin.h -- Import processor intrinsics and declares utility
    functions built from them

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if !defined(_MSC_VER)
#  if !defined(__IMMINTRIN_H) && defined(__clang__)
/* We want to be able to selectively include intrinsics. For instance, it's
   often not desirable to pull in 1 MB (!) of AVX512 header code unless the
   application is really using those intrinsics. Unfortunately, immintrin.h
   tries to prevent this kind of selectiveness, which we simply circumvent with
   the following define.. */
#    define __IMMINTRIN_H
#  endif

#  if !defined(_IMMINTRIN_H_INCLUDED) && !defined(__clang__) && defined(__GNUC__)
     // And the same once more, for GCC..
#    define _IMMINTRIN_H_INCLUDED
#  endif

#  if defined(DRJIT_X86_SSE42)
#    include <nmmintrin.h>
#  endif

#  if defined(DRJIT_X86_AVX)
#    include <avxintrin.h>
#  endif

#  if defined(DRJIT_X86_AVX2)
#    include <avx2intrin.h>
#  endif

#  if defined(DRJIT_X86_BMI)
#    include <bmiintrin.h>
#  endif

#  if defined(DRJIT_X86_BMI2)
#    include <bmi2intrin.h>
#endif

#  if defined(DRJIT_X86_FMA)
#    include <fmaintrin.h>
#  endif

#  if defined(DRJIT_X86_F16C)
#    include <f16cintrin.h>
#  endif

#  if defined(DRJIT_X86_AVX512)
#    include <avx512fintrin.h>
#    include <avx512vlintrin.h>
#    include <avx512bwintrin.h>
#    include <avx512cdintrin.h>
#    include <avx512dqintrin.h>
#    include <avx512vldqintrin.h>
#    include <avx512vlbwintrin.h>
#    if defined(DRJIT_X86_AVX512VPOPCNTDQ)
#        include <avx512vpopcntdqintrin.h>
#        include <avx512vpopcntdqvlintrin.h>
#    endif
#    if defined(DRJIT_X86_AVX512VBMI)
#        include <avx512vbmiintrin.h>
#        include <avx512vbmivlintrin.h>
#    endif
#    if defined(__clang__)
#      include <avx512vlcdintrin.h>
#    endif
#  endif

#  if defined(DRJIT_ARM_NEON)
#    include <arm_neon.h>
#  endif
#else
#  include <intrin.h>
#endif

// -----------------------------------------------------------------------
//! @{ \name Available instruction sets
// -----------------------------------------------------------------------

#if defined(DRJIT_X86_32)
    static constexpr bool has_x86_32 = true;
#else
    static constexpr bool has_x86_32 = false;
#endif

#if defined(DRJIT_X86_64)
    static constexpr bool has_x86_64 = true;
#else
    static constexpr bool has_x86_64 = false;
#endif

#if defined(DRJIT_ARM_32)
    static constexpr bool has_arm_32 = true;
#else
    static constexpr bool has_arm_32 = false;
#endif

#if defined(DRJIT_ARM_64)
    static constexpr bool has_arm_64 = true;
#else
    static constexpr bool has_arm_64 = false;
#endif

#if defined(DRJIT_X86_SSE42)
    static constexpr bool has_sse42 = true;
#else
    static constexpr bool has_sse42 = false;
#endif

#if defined(DRJIT_X86_FMA) || defined(DRJIT_ARM_FMA)
    static constexpr bool has_fma = true;
#else
    static constexpr bool has_fma = false;
#endif

#if defined(DRJIT_X86_F16C)
    static constexpr bool has_f16c = true;
#else
    static constexpr bool has_f16c = false;
#endif

#if defined(DRJIT_X86_AVX)
    static constexpr bool has_avx = true;
#else
    static constexpr bool has_avx = false;
#endif

#if defined(DRJIT_X86_BMI)
    static constexpr bool has_bmi = true;
#else
    static constexpr bool has_bmi = false;
#endif

#if defined(DRJIT_X86_BMI2)
    static constexpr bool has_bmi2 = true;
#else
    static constexpr bool has_bmi2 = false;
#endif

#if defined(DRJIT_X86_AVX2)
    static constexpr bool has_avx2 = true;
#else
    static constexpr bool has_avx2 = false;
#endif

#if defined(DRJIT_X86_AVX512)
    static constexpr bool has_avx512 = true;
#else
    static constexpr bool has_avx512 = false;
#endif

#if defined(DRJIT_ARM_NEON)
    static constexpr bool has_neon = true;
#else
    static constexpr bool has_neon = false;
#endif

static constexpr bool has_x86 = has_x86_32 || has_x86_64;
static constexpr bool has_arm = has_arm_32 || has_arm_64;
static constexpr bool has_vectorization = has_sse42 || has_neon;

//! @}
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

// -----------------------------------------------------------------------
//! @{ \name Helper routines to merge smaller arrays into larger ones
// -----------------------------------------------------------------------

#if defined(DRJIT_X86_AVX)
DRJIT_INLINE __m256 concat(__m128 l, __m128 h) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(l), h, 1);
}

DRJIT_INLINE __m256d concat(__m128d l, __m128d h) {
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(l), h, 1);
}

DRJIT_INLINE __m256i concat(__m128i l, __m128i h) {
    return _mm256_insertf128_si256(_mm256_castsi128_si256(l), h, 1);
}
#endif

#if defined(DRJIT_X86_AVX512)
DRJIT_INLINE __m512 concat(__m256 l, __m256 h) {
    return _mm512_insertf32x8(_mm512_castps256_ps512(l), h, 1);
}

DRJIT_INLINE __m512d concat(__m256d l, __m256d h) {
    return _mm512_insertf64x4(_mm512_castpd256_pd512(l), h, 1);
}

DRJIT_INLINE __m512i concat(__m256i l, __m256i h) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(l), h, 1);
}
#endif

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Mask conversion routines for various platforms
// -----------------------------------------------------------------------

#if defined(DRJIT_X86_AVX)
DRJIT_INLINE __m256i mm256_cvtepi32_epi64(__m128i x) {
#if defined(DRJIT_X86_AVX2)
    return _mm256_cvtepi32_epi64(x);
#else
    /* This version is only suitable for mask conversions */
    __m128i xl = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 1, 0, 0));
    __m128i xh = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 2, 2));
    return detail::concat(xl, xh);
#endif
}

DRJIT_INLINE __m128i mm256_cvtepi64_epi32(__m256i x) {
#if defined(DRJIT_X86_AVX512)
    return _mm256_cvtepi64_epi32(x);
#else
    __m128i x0 = _mm256_castsi256_si128(x);
    __m128i x1 = _mm256_extractf128_si256(x, 1);
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
#endif
}

DRJIT_INLINE __m256i mm512_cvtepi64_epi32(__m128i x0, __m128i x1, __m128i x2, __m128i x3) {
    __m128i y0 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
    __m128i y1 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x2), _mm_castsi128_ps(x3), _MM_SHUFFLE(2, 0, 2, 0)));
    return detail::concat(y0, y1);
}

DRJIT_INLINE __m256i mm512_cvtepi64_epi32(__m256i x0, __m256i x1) {
    __m128i y0 = _mm256_castsi256_si128(x0);
    __m128i y1 = _mm256_extractf128_si256(x0, 1);
    __m128i y2 = _mm256_castsi256_si128(x1);
    __m128i y3 = _mm256_extractf128_si256(x1, 1);
    return mm512_cvtepi64_epi32(y0, y1, y2, y3);
}
#endif

#if defined(DRJIT_X86_SSE42)
DRJIT_INLINE __m128i mm256_cvtepi64_epi32(__m128i x0, __m128i x1) {
    return _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(x0), _mm_castsi128_ps(x1), _MM_SHUFFLE(2, 0, 2, 0)));
}

DRJIT_INLINE __m128i mm_cvtsi64_si128(long long a)  {
    #if defined(DRJIT_X86_64)
        return _mm_cvtsi64_si128(a);
    #else
        alignas(16) long long x[2] = { a, 0ll };
        return _mm_load_si128((__m128i *) x);
    #endif
}

DRJIT_INLINE long long mm_cvtsi128_si64(__m128i m)  {
    #if defined(DRJIT_X86_64)
        return _mm_cvtsi128_si64(m);
    #else
        alignas(16) long long x[2];
        _mm_store_si128((__m128i *) x, m);
        return x[0];
    #endif
}

template <int Imm8>
DRJIT_INLINE long long mm_extract_epi64(__m128i m)  {
    #if defined(DRJIT_X86_64)
        return _mm_extract_epi64(m, Imm8);
    #else
        alignas(16) long long x[2];
        _mm_store_si128((__m128i *) x, m);
        return x[Imm8];
    #endif
}

#endif

#if defined(DRJIT_X86_AVX2)
template <typename T> DRJIT_INLINE T tzcnt(T v) {
    static_assert(std::is_integral_v<T>, "tzcnt(): requires an integer argument!");
    if (sizeof(T) <= 4) {
        return (T) _tzcnt_u32((unsigned int) v);
    } else {
#if defined(DRJIT_X86_64)
        return (T) _tzcnt_u64((unsigned long long) v);
#else
        unsigned long long v_ = (unsigned long long) v;
        unsigned int lo = (unsigned int) v_;
        unsigned int hi = (unsigned int) (v_ >> 32);
        return (T) (lo != 0 ? _tzcnt_u32(lo) : (_tzcnt_u32(hi) + 32));
#endif
    }
}
#endif

//! @}
// -----------------------------------------------------------------------

#define DRJIT_PACKET_DECLARE(Size)                                             \
    namespace detail {                                                         \
        template <typename Type> struct vectorize<Type, Size> {                \
            static constexpr bool recurse = false;                             \
            static constexpr bool self = true;                                 \
        };                                                                     \
    }

#define DRJIT_PACKET_DECLARE_COND(Size, Cond)                                  \
    namespace detail {                                                         \
        template <typename Type> struct vectorize<Type, Size, Cond> {          \
            static constexpr bool recurse = false;                             \
            static constexpr bool self = true;                                 \
        };                                                                     \
    }

#if defined(NDEBUG)
#  define DRJIT_PACKET_INIT(Name) Name() = default;
#else
#  define DRJIT_PACKET_INIT(Name) Name() : Name(DebugInitialization<Value>) { }
#endif

#define DRJIT_PACKET_TYPE(Type, Size_, Register)                               \
    using Base = StaticArrayBase<Type, Size_, IsMask_, Derived_>;              \
    using typename Base::Derived;                                              \
    using typename Base::Value;                                                \
    using typename Base::Array1;                                               \
    using typename Base::Array2;                                               \
    using Base::derived;                                                       \
    using Base::Size;                                                          \
    using Ref = const Derived &;                                               \
    static constexpr bool IsPacked = true;                                     \
    Register m;                                                                \
    DRJIT_PACKET_INIT(StaticArrayImpl)                                         \
    DRJIT_ARRAY_DEFAULTS(StaticArrayImpl)                                      \
    DRJIT_ARRAY_FALLBACK_CONSTRUCTORS(StaticArrayImpl)                         \
    DRJIT_INLINE StaticArrayImpl(Register m) : m(m) { }                        \
    DRJIT_INLINE StaticArrayImpl(Register m, detail::reinterpret_flag):m(m){}  \
    DRJIT_INLINE Value *data() { return (Value *) &m; }                        \
    DRJIT_INLINE const Value *data() const { return (const Value *) &m; }      \
    DRJIT_INLINE Value &entry(size_t i) { return ((Value *) &m)[i]; }          \
    DRJIT_INLINE const Value &entry(size_t i) const {                          \
        return ((const Value *) &m)[i];                                        \
    }


#define DRJIT_PACKET_TYPE_3D(Type)                                             \
    using Base = StaticArrayImpl<Type, 4, IsMask_, Derived_>;                  \
    DRJIT_ARRAY_IMPORT(StaticArrayImpl, Base)                                  \
    DRJIT_INLINE StaticArrayImpl(Type f1, Type f2, Type f3)                    \
        : Base(f1, f2, f3, (Type) 0) { }                                       \
    using typename Base::Derived;                                              \
    using typename Base::Value;                                                \
    using typename Base::Ref;                                                  \
    using Base::derived;                                                       \
    using Base::m;                                                             \
    static constexpr size_t ActualSize = 4;                                    \
    static constexpr size_t Size = 3;

#define DRJIT_CONVERT(Value)                                                   \
    template <typename Value2, typename Derived2,                              \
              enable_if_t<detail::is_same_v<Value2, Value>> = 0>               \
    DRJIT_INLINE StaticArrayImpl(                                              \
        const StaticArrayBase<Value2, Size, IsMask_, Derived2> &a)

#define DRJIT_REINTERPRET(Value)                                               \
    template <typename Value2, typename Derived2, bool IsMask2,                \
              enable_if_t<detail::is_same_v<Value2, Value>> = 0>               \
    DRJIT_INLINE StaticArrayImpl(                                              \
        const StaticArrayBase<Value2, Size, IsMask2, Derived2> &a,             \
        detail::reinterpret_flag)

#define DRJIT_REINTERPRET_MASK(Value)                                          \
    template <typename Value2, typename Derived2, typename T = Derived,        \
              enable_if_t<T::IsMask && detail::is_same_v<Value2, Value>> = 0>  \
    DRJIT_INLINE StaticArrayImpl(                                              \
        const StaticArrayBase<Value2, Size, true, Derived2> &a,                \
        detail::reinterpret_flag)

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
