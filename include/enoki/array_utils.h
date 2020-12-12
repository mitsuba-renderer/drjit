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
template <typename T, typename U> ENOKI_INLINE T memcpy_cast(const U &val) {
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
#  define ENOKI_BUILTIN(name) ::__builtin_##name
#else
#  define ENOKI_BUILTIN(name) ::name
#endif

template <typename T> T neg_(const T &a) {
    if constexpr (std::is_unsigned_v<T>)
        return ~a + T(1);
    else
        return -a;
}

template <typename T> T abs_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(fabsf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(fabs)(a);
    else if constexpr (std::is_signed_v<T>)
        return a < 0 ? -a : a;
    else
        return a;
}

template <typename T> T sqrt_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(sqrtf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(sqrt)(a);
    else
        return (T) enoki::detail::sqrt_((float) a);
}

template <typename T> T floor_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(floorf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(floor)(a);
    else
        return (T) enoki::detail::floor_((float) a);
}

template <typename T> T ceil_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(ceilf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(ceil)(a);
    else
        return (T) enoki::detail::ceil_((float) a);
}

template <typename T> T trunc_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(truncf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(trunc)(a);
    else
        return (T) enoki::detail::trunc_((float) a);
}

template <typename T> T round_(const T &a) {
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(rintf)(a);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(rint)(a);
    else
        return (T) enoki::detail::round_((float) a);
}

template <typename T> T max_(const T &a, const T &b) {
    return a < b ? b : a;
}

template <typename T> T min_(const T &a, const T &b) {
    return b < a ? b : a;
}

template <typename T> T fmadd_(const T &a, const T &b, const T &c) {
#if defined(ENOKI_X86_FMA) || defined(ENOKI_ARM_FMA)
    if constexpr (std::is_same_v<T, float>)
        return ENOKI_BUILTIN(fmaf)(a, b, c);
    else if constexpr (std::is_same_v<T, double>)
        return ENOKI_BUILTIN(fma)(a, b, c);
    else
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

template <typename T> ENOKI_INLINE T tzcnt_(T v) {
#if defined(_MSC_VER)
    unsigned long result;
    if (sizeof(T) <= 4) {
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
ENOKI_INLINE T mulhi_(T x, T y) {
    if (sizeof(T) == 4) {
        using Wide = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
        return T(((Wide) x * (Wide) y) >> 32);
    } else {
#if defined(_MSC_VER) && defined(ENOKI_X86_64)
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

// Tiny self-contained unique_ptr to avoid having to import thousands of LOC from <memory>
template <typename T> struct tiny_unique_ptr {
    using Type = std::remove_extent_t<T>;

    tiny_unique_ptr() = default;
    tiny_unique_ptr(Type *data) : data(data) { }
    tiny_unique_ptr(tiny_unique_ptr &&other) : data(other.data) {
        other.data = nullptr;
    }

    tiny_unique_ptr &operator=(tiny_unique_ptr &&other) {
        if constexpr (is_array_v<T>)
            delete[] data;
        else
            delete data;
        data = other.data;
        other.data = nullptr;
        return *this;
    }

    ~tiny_unique_ptr() {
        if constexpr (is_array_v<T>)
            delete[] data;
        else
            delete data;
    }

    Type& operator[](size_t index) { return data[index]; }
    Type* get() { return data; }
    Type* operator->() { return data; }
    const Type* operator->() const { return data; }
    Type* release () {
        Type *tmp = data;
        data = nullptr;
        return tmp;
    }
    Type *data = nullptr;
};

// Tiny self-contained tuple to avoid having to import thousands of LOC from <tuple>
template <typename... Ts> struct tuple;
template <> struct tuple<> {
    template <size_t> using type = void;
};

template <typename T, typename... Ts> struct tuple<T, Ts...> : tuple<Ts...> {
    using Base = tuple<Ts...>;

    tuple() = default;
    tuple(const tuple &) = default;
    tuple(tuple &&) = default;
    tuple& operator=(tuple &&) = default;
    tuple& operator=(const tuple &) = default;

    tuple(const T& value, const Ts&... ts)
        : Base(ts...), value(value) { }

    tuple(T&& value, Ts&&... ts)
        : Base(std::move(ts)...), value(std::move(value)) { }

    template <size_t I> auto& get() {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I> const auto& get() const {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I>
    using type =
        std::conditional_t<I == 0, T, typename Base::template type<I - 1>>;

private:
    T value;
};

template <typename... Ts>
tuple(Ts &&...) -> tuple<std::decay_t<Ts>...>;

NAMESPACE_END(detail)


#if defined(__cpp_exceptions)
class Exception : public std::exception {
public:
    Exception(const char *msg) {
#if defined(_MSC_VER)
        m_msg = _strdup(msg);
#else
        m_msg = strdup(msg);
#endif
    }
    virtual const char *what() const noexcept { return m_msg; }
    virtual ~Exception() { free(m_msg); }
private:
    char *m_msg;
};
#endif

#if !defined(_MSC_VER)
__attribute__((noreturn,noinline))
#else
__declspec(noreturn,noinline)
#endif
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
