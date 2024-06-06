/*
    drjit/array_fallbacks.h -- Scalar utility functions used by Dr.Jit's
    array classes

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array_traits.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include <exception>

NAMESPACE_BEGIN(drjit)

/// Reinterpret the binary represesentation of a data type
template <typename T, typename U> DRJIT_INLINE T memcpy_cast(const U &val) {
    static_assert(sizeof(T) == sizeof(U), "memcpy_cast: sizes did not match!");
    T result;
    memcpy(&result, &val, sizeof(T));
    return result;
}

NAMESPACE_BEGIN(detail)

template <typename T> auto not_(const T &a) {
    using UInt = uint_array_t<T>;

    if constexpr (is_same_v<T, bool>)
        return !a;
    else if constexpr (is_array_v<T> || std::is_integral_v<T>)
        return ~a;
    else
        return memcpy_cast<T>(~memcpy_cast<UInt>(a));
}

template <typename T> auto or_(const T &a1, const T &a2) {
    using UInt = uint_array_t<T>;

    if constexpr (is_array_v<T> || std::is_integral_v<T>)
        return a1 | a2;
    else
        return memcpy_cast<T>(memcpy_cast<UInt>(a1) | memcpy_cast<UInt>(a2));
}

template <typename T> auto and_(const T &a1, const T &a2) {
    using UInt = uint_array_t<T>;

    if constexpr (is_array_v<T> || std::is_integral_v<T>)
        return a1 & a2;
    else
        return memcpy_cast<T>(memcpy_cast<UInt>(a1) & memcpy_cast<UInt>(a2));
}

template <typename T> auto andnot_(const T &a1, const T &a2) {
    using UInt = uint_array_t<T>;

    if constexpr (is_array_v<T>)
        return a1.andnot_(a2);
    else if constexpr (std::is_same_v<T, bool>)
        return a1 && !a2;
    else if constexpr (std::is_integral_v<T>)
        return a1 & ~a2;
    else
        return memcpy_cast<T>(memcpy_cast<UInt>(a1) & ~memcpy_cast<UInt>(a2));
}

template <typename T> auto xor_(const T &a1, const T &a2) {
    using UInt = uint_array_t<T>;

    if constexpr (is_array_v<T> || std::is_integral_v<T>)
        return a1 ^ a2;
    else
        return memcpy_cast<T>(memcpy_cast<UInt>(a1) ^ memcpy_cast<UInt>(a2));
}

template <typename T, enable_if_t<!std::is_same_v<T, bool>> = 0> auto or_(const T &a, const bool &b) {
    using Scalar = scalar_t<T>;
    using UInt   = uint_array_t<Scalar>;
    return or_(a, b ? memcpy_cast<Scalar>(UInt(-1)) : memcpy_cast<Scalar>(UInt(0)));
}

template <typename T, enable_if_t<!std::is_same_v<T, bool>> = 0> auto and_(const T &a, const bool &b) {
    using Scalar = scalar_t<T>;
    using UInt   = uint_array_t<Scalar>;
    return and_(a, b ? memcpy_cast<Scalar>(UInt(-1)) : memcpy_cast<Scalar>(UInt(0)));
}

template <typename T, enable_if_t<!std::is_same_v<T, bool>> = 0> auto andnot_(const T &a, const bool &b) {
    using Scalar = scalar_t<T>;
    using UInt   = uint_array_t<Scalar>;
    return and_(a, b ? memcpy_cast<Scalar>(UInt(0)) : memcpy_cast<Scalar>(UInt(-1)));
}

template <typename T, enable_if_t<!std::is_same_v<T, bool>> = 0> auto xor_(const T &a, const bool &b) {
    using Scalar = scalar_t<T>;
    using UInt   = uint_array_t<Scalar>;
    return xor_(a, b ? memcpy_cast<Scalar>(UInt(-1)) : memcpy_cast<Scalar>(UInt(0)));
}

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto or_(const T1 &a1, const T2 &a2) { return a1 | a2; }

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto and_(const T1 &a1, const T2 &a2) { return a1 & a2; }

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto andnot_(const T1 &a1, const T2 &a2) { return andnot(a1, a2); }

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto xor_(const T1 &a1, const T2 &a2) { return a1 ^ a2; }

#if defined(__GNUC__)
#  define DRJIT_BUILTIN(name) ::__builtin_##name
#else
#  define DRJIT_BUILTIN(name) ::name
#endif

template <typename T> T neg_(const T &a) {
    if constexpr (std::is_unsigned_v<T>)
        return ~a + T(1);
    else
        return -a;
}

template <typename T> T abs_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(fabsf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(fabs)(a);
    else if constexpr (std::is_signed_v<T>)
        return a < 0 ? -a : a;
    else
        return a;
}

template <typename T> T sqrt_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(sqrtf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(sqrt)(a);
    else
        return (T) drjit::detail::sqrt_((float) a);
}

template <typename T> T floor_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(floorf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(floor)(a);
    else
        return (T) drjit::detail::floor_((float) a);
}

template <typename T> T ceil_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(ceilf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(ceil)(a);
    else
        return (T) drjit::detail::ceil_((float) a);
}

template <typename T> T trunc_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(truncf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(trunc)(a);
    else
        return (T) drjit::detail::trunc_((float) a);
}

template <typename T> T round_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(rintf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(rint)(a);
    else
        return (T) drjit::detail::round_((float) a);
}

template <typename T> T maximum_(const T &a, const T &b) {
    return a < b ? b : a;
}

template <typename T> T minimum_(const T &a, const T &b) {
    return b < a ? b : a;
}

template <typename T> T fmadd_(const T &a, const T &b, const T &c) {
#if defined(DRJIT_X86_FMA) || defined(DRJIT_ARM_FMA)
    if constexpr (std::is_same_v<T, float>)
        return DRJIT_BUILTIN(fmaf)(a, b, c);
    else if constexpr (std::is_same_v<T, double>)
        return DRJIT_BUILTIN(fma)(a, b, c);
    else
#endif
    return a * b + c;
}

template <typename T> T rcp_(const T &a) {
    return (T) 1 / a;
}

template <typename T> T rsqrt_(const T &a) {
    return drjit::detail::rcp_(drjit::detail::sqrt_(a));
}

template <typename T> DRJIT_INLINE T popcnt_(T v) {
#if defined(_MSC_VER)
    if constexpr (sizeof(T) <= 4)
        return (T) __popcnt((unsigned int) v);
    else
        return (T) __popcnt64((unsigned long long) v);
#else
    if constexpr (sizeof(T) <= 4)
        return (T) __builtin_popcount((unsigned int) v);
    else
        return (T) __builtin_popcountll((unsigned long long) v);
#endif
}

template <typename T> DRJIT_INLINE T lzcnt_(T v) {
#if defined(_MSC_VER)
    unsigned long result;
    if constexpr (sizeof(T) <= 4) {
        _BitScanReverse(&result, (unsigned long) v);
        return (v != 0) ? (31 - result) : 32;
    } else {
        _BitScanReverse64(&result, (unsigned long long) v);
        return (v != 0) ? (63 - result) : 64;
    }
#else
    if constexpr (sizeof(T) <= 4)
        return (T) (v != 0 ? __builtin_clz((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __builtin_clzll((unsigned long long) v) : 64);
#endif
}

template <typename T> DRJIT_INLINE T tzcnt_(T v) {
#if defined(_MSC_VER)
    unsigned long result;
    if constexpr (sizeof(T) <= 4) {
        _BitScanForward(&result, (unsigned long) v);
        return (v != 0) ? result : 32;
    } else {
        _BitScanForward64(&result, (unsigned long long) v);
        return (v != 0) ? result: 64;
    }
#else
    if constexpr (sizeof(T) <= 4)
        return (T) (v != 0 ? __builtin_ctz((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __builtin_ctzll((unsigned long long) v) : 64);
#endif
}

template <typename T>
DRJIT_INLINE T mulhi_(T x, T y) {
    if constexpr (sizeof(T) == 4) {
        using Wide = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
        return T(((Wide) x * (Wide) y) >> 32);
    } else {
#if defined(_MSC_VER) && defined(DRJIT_X86_64)
        if constexpr (std::is_signed_v<T>)
            return (T) __mulh(x, y);
        else
            return (T) __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
        using Wide = std::conditional_t<std::is_signed_v<T>, __int128_t, __uint128_t>;
        return T(((Wide) x * (Wide) y) >> 64);
#else
        // full 128 bits are x0 * y0 + (x0 * y1 << 32) + (x1 * y0 << 32) + (x1 * y1 << 64)
        uint32_t mask = 0xFFFFFFFF;
        uint32_t x0 = (uint32_t) (x & mask),
                 y0 = (uint32_t) (y & mask);

        if constexpr (std::is_signed_v<T>) {
            int32_t x1 = (int32_t) (x >> 32);
            int32_t y1 = (int32_t) (y >> 32);
            uint32_t x0y0_hi = mulhi_scalar(x0, y0);
            int64_t t = x1 * (int64_t) y0 + x0y0_hi;
            int64_t w1 = x0 * (int64_t) y1 + (t & mask);

            return x1 * (int64_t) y1 + (t >> 32) + (w1 >> 32);
        } else {
            uint32_t x1 = (uint32_t) (x >> 32);
            uint32_t y1 = (uint32_t) (y >> 32);
            uint32_t x0y0_hi = mulhi_(x0, y0);

            uint64_t x0y1 = x0 * (uint64_t) y1;
            uint64_t x1y0 = x1 * (uint64_t) y0;
            uint64_t x1y1 = x1 * (uint64_t) y1;
            uint64_t temp = x1y0 + x0y0_hi;
            uint64_t temp_lo = temp & mask,
                     temp_hi = temp >> 32;

            return x1y1 + temp_hi + ((temp_lo + x0y1) >> 32);
        }
#endif
    }
}

NAMESPACE_END(detail)

#if defined(__cpp_exceptions)
#  if defined(_MSC_VER)
#    define DRJIT_STRDUP _strdup
#  else
#    define DRJIT_STRDUP strdup
#  endif

class Exception : public std::exception {
public:
    Exception(const char* msg) : m_msg(DRJIT_STRDUP(msg)) { }
    Exception(const Exception& e) : m_msg(DRJIT_STRDUP(e.m_msg)) { }
    Exception(Exception&& e) : m_msg(e.m_msg) { e.m_msg = nullptr; }
    Exception& operator=(const Exception&) = delete;
    Exception& operator=(Exception&&) = delete;
    virtual const char* what() const noexcept { return m_msg; }
    virtual ~Exception() { free(m_msg); }
private:
    char *m_msg;
};

#  undef DRJIT_STRDUP
#endif

#if !defined(_MSC_VER)
__attribute__((noreturn,noinline))
#else
__declspec(noreturn,noinline)
#endif
inline void drjit_raise(const char *fmt, ...) {
    char msg[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
#if defined(__cpp_exceptions)
    throw drjit::Exception(msg);
#else
    fprintf(stderr, "%s\n", msg);
    abort();
#endif
}

NAMESPACE_END(drjit)
