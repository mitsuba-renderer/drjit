/*
    enoki/array_fallbacks.h -- Scalar utility functions used by Enoki's
    array classes

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array_traits.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include <exception>

NAMESPACE_BEGIN(enoki)

/// Reinterpret the binary represesentation of a data type
template<typename T, typename U> ENOKI_INLINE T memcpy_cast(const U &val) {
    static_assert(sizeof(T) == sizeof(U), "memcpy_cast: sizes did not match!");
    T result;
    memcpy(&result, &val, sizeof(T));
    return result;
}

NAMESPACE_BEGIN(detail)

template <typename T> auto not_(const T &a) {
    using UInt = uint_array_t<T>;

    if constexpr (is_array_v<T> || std::is_integral_v<T>)
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
auto andnot_(const T1 &a1, const T2 &a2) {
    using E = expr_t<T1, T2>;
    return andnot_((const E &) a1, (const E &) a2);
}

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto xor_(const T1 &a1, const T2 &a2) { return a1 ^ a2; }

template <typename T> T abs_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_fabsf(a);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_fabs(a);
    else if constexpr (std::is_signed_v<T>)
        return a < 0 ? -a : a;
    else
        return a;
}

template <typename T> T sqrt_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_sqrtf(a);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_sqrt(a);
    else
        return (T) enoki::detail::sqrt_((float) a);
}

template <typename T> T floor_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_floorf(a);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_floor(a);
    else
        return (T) enoki::detail::floor_((float) a);
}

template <typename T> T ceil_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_ceilf(a);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_ceil(a);
    else
        return (T) enoki::detail::ceil_((float) a);
}

template <typename T> T trunc_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_truncf(a);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_trunc(a);
    else
        return (T) enoki::detail::trunc_((float) a);
}

template <typename T> T round_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_rintf(a);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_rint(a);
    else
        return (T) enoki::detail::round_((float) a);
}

template <typename T> T max_(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_fmaxf(a, b);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_fmax(a, b);
    else
        return a > b ? a : b;
}

template <typename T> T min_(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, float>)
        return __builtin_fminf(a, b);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_fmin(a, b);
    else
        return a < b ? a : b;
}

template <typename T> T fmadd_(const T &a, const T &b, const T &c) {
#if defined(ENOKI_X86_FMA) || defined(ENOKI_ARM_FMA)
    if constexpr (std::is_same_v<T, float>)
        return __builtin_fmaf(a, b, c);
    else if constexpr (std::is_same_v<T, double>)
        return __builtin_fma(a, b, c);
#endif
    return a * b + c;
}

template <typename T> T rcp_(const T &a) {
    return (T) 1 / a;
}

template <typename T> T rsqrt_(const T &a) {
    return enoki::detail::rcp_(enoki::detail::sqrt_(a));
}

template <typename T> ENOKI_INLINE T popcnt_(T v) {
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

template <typename T> ENOKI_INLINE T lzcnt_(T v) {
#if defined(_MSC_VER)
    if constexpr (sizeof(T) <= 4)
        return (T) (v != 0 ? __lzcnt((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __lzcnt64((unsigned long long) v) : 64);
#else
    if constexpr (sizeof(T) <= 4)
        return (T) (v != 0 ? __builtin_clz((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __builtin_clzll((unsigned long long) v) : 64);
#endif
}

template <typename T> ENOKI_INLINE T tzcnt_(T v) {
#if defined(_MSC_VER)
    if constexpr (sizeof(T) <= 4)
        return (T) (v != 0 ? __tzcnt((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __tzcnt64((unsigned long long) v) : 64);
#else
    if constexpr (sizeof(T) <= 4)
        return (T) (v != 0 ? __builtin_clz((unsigned int) v) : 32);
    else
        return (T) (v != 0 ? __builtin_clzll((unsigned long long) v) : 64);
#endif
}

template <typename T>
ENOKI_INLINE T mulhi_(T x, T y) {
    if (sizeof(T) == 4) {
        using Wide = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
        return T(((Wide) x * (Wide) y) >> 32);
    } else {
#if defined(_MSC_VER) && defined(ENOKI_X86_64)
        if constexpr (std::is_signed_v<T>)
            return __mulh(x, y);
        else
            return __umulh(x, y);
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
class Exception : public std::exception {
public:
    Exception(const char *msg) { m_msg = __builtin_strdup(msg); }
    virtual const char *what() const noexcept { return m_msg; }
    virtual ~Exception() { free(m_msg); }
private:
    char *m_msg;
};
#endif

__attribute__((noreturn,noinline))
inline void enoki_raise(const char *fmt, ...) {
    char msg[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
#if defined(__cpp_exceptions)
    throw enoki::Exception(msg);
#else
    fprintf(stderr, "%s\n", msg);
    abort(EXIT_FAILURE);
#endif
}

NAMESPACE_END(enoki)
