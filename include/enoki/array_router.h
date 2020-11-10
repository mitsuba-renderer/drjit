/*
    enoki/array_router.h -- Helper functions which route function calls
    in the enoki namespace to the intended recipients

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array_traits.h>
#include <enoki/array_utils.h>
#include <enoki/array_constants.h>

#if defined(min) || defined(max)
#  error min/max are defined as preprocessor symbols! Define NOMINMAX on MSVC.
#endif

extern "C" {
    /// Evaluate all computation that is scheduled on the current stream
    extern ENOKI_IMPORT void jitc_eval();
    /// Set the currently active device and stream
    extern ENOKI_IMPORT void jitc_set_device(int32_t device, uint32_t stream);
    /// Wait for all computation on the current stream to finish
    extern ENOKI_IMPORT void jitc_sync_stream();
    /// Wait for all computation on the current device to finish
    extern ENOKI_IMPORT void jitc_sync_device();
    /// Wait for all computation on the *all devices* to finish
    extern ENOKI_IMPORT void jitc_sync_all_devices();
};

NAMESPACE_BEGIN(enoki)

/// Define an unary operation
#define ENOKI_ROUTE_UNARY(name, func)                                          \
    template <typename T, enable_if_array_t<T> = 0>                            \
    ENOKI_INLINE auto name(const T &a) {                                       \
        return a.func##_();                                                    \
    }

/// Define an unary operation with a fallback expression for scalar arguments
#define ENOKI_ROUTE_UNARY_FALLBACK(name, func, expr)                           \
    template <typename T> ENOKI_INLINE auto name(const T &a) {                 \
        if constexpr (!is_array_v<T>)                                          \
            return expr; /* Scalar fallback implementation */                  \
        else                                                                   \
            return a.func##_(); /* Forward to array */                         \
    }


/// Define an unary operation with an immediate argument (e.g. sr<5>(...))
#define ENOKI_ROUTE_UNARY_IMM_FALLBACK(name, func, expr)                       \
    template <int Imm, typename T> ENOKI_INLINE auto name(const T &a) {        \
        if constexpr (!is_array_v<T>)                                          \
            return expr; /* Scalar fallback implementation */                  \
        else                                                                   \
            return a.template func##_<Imm>(); /* Forward to array */           \
    }


/// Define an unary operation with a fallback expression for scalar arguments
#define ENOKI_ROUTE_UNARY_TYPE_FALLBACK(name, func, expr)                      \
    template <typename T2, typename T> ENOKI_INLINE auto name(const T &a) {    \
        if constexpr (!is_array_v<T>)                                          \
            return expr; /* Scalar fallback implementation */                  \
        else                                                                   \
            return a.template func##_<T2>(); /* Forward to array */            \
    }

/// Define a binary operation
#define ENOKI_ROUTE_BINARY(name, func)                                         \
    template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>     \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using E = expr_t<T1, T2>;                                              \
        if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)          \
            return a1.derived().func##_(a2.derived());                         \
        else                                                                   \
            return name(static_cast<ref_cast_t<T1, E>>(a1),                    \
                        static_cast<ref_cast_t<T2, E>>(a2));                   \
    }

/// Define a binary operation for bit operations
#define ENOKI_ROUTE_BINARY_BITOP(name, func)                                   \
    template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>     \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using E = expr_t<T1, T2>;                                              \
        using Deepest = detail::deepest_t<T1, T2>;                             \
        using E1 = array_t<replace_scalar_t<Deepest, scalar_t<T1>>>;           \
        using E2 = mask_t<E1>;                                                 \
        if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)          \
            return a1.derived().func##_(a2.derived());                         \
        else if constexpr (std::is_same_v<T1, E1> && std::is_same_v<T2, E2>)   \
            return a1.derived().func##_(a2.derived());                         \
        else if constexpr (!is_mask_v<T1> && is_mask_v<T2>)                    \
            return name(static_cast<ref_cast_t<T1, E1>>(a1),                   \
                        static_cast<ref_cast_t<T2, E2>>(a2));                  \
        else                                                                   \
            return name(static_cast<ref_cast_t<T1, E>>(a1),                    \
                        static_cast<ref_cast_t<T2, E>>(a2));                   \
    }

#define ENOKI_ROUTE_BINARY_SHIFT(name, func)                                   \
    template <typename T1, typename T2,                                        \
              enable_if_t<std::is_arithmetic_v<scalar_t<T1>>> = 0,             \
              enable_if_array_any_t<T1, T2> = 0>                               \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using E = expr_t<T1, T2>;                                              \
        if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)          \
            return a1.derived().func##_(a2.derived());                         \
        else                                                                   \
            return name(static_cast<ref_cast_t<T1, E>>(a1),                    \
                        static_cast<ref_cast_t<T2, E>>(a2));                   \
    }

/// Define a binary operation with a fallback expression for scalar arguments
#define ENOKI_ROUTE_BINARY_FALLBACK(name, func, expr)                          \
    template <typename T1, typename T2>                                        \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using E = expr_t<T1, T2>;                                              \
        if constexpr (is_array_any_v<T1, T2>) {                                \
            if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)      \
                return a1.derived().func##_(a2.derived());                     \
            else                                                               \
                return name(static_cast<ref_cast_t<T1, E>>(a1),                \
                            static_cast<ref_cast_t<T2, E>>(a2));               \
        } else {                                                               \
            return expr;                                                       \
        }                                                                      \
    }

/// Define a ternary operation
#define ENOKI_ROUTE_TERNARY_FALLBACK(name, func, expr)                         \
    template <typename T1, typename T2, typename T3>                           \
    ENOKI_INLINE auto name(const T1 &a1, const T2 &a2, const T3 &a3) {         \
        using E = expr_t<T1, T2, T3>;                                          \
        if constexpr (is_array_any_v<T1, T2, T3>) {                            \
            if constexpr (std::is_same_v<T1, E> &&                             \
                          std::is_same_v<T2, E> &&                             \
                          std::is_same_v<T3, E>)                               \
                return a1.derived().func##_(a2.derived(), a3.derived());       \
            else                                                               \
                return name(static_cast<ref_cast_t<T1, E>>(a1),                \
                            static_cast<ref_cast_t<T2, E>>(a2),                \
                            static_cast<ref_cast_t<T3, E>>(a3));               \
        } else {                                                               \
            return expr;                                                       \
        }                                                                      \
    }

/// Macro for compound assignment operators (operator+=, etc.)
#define ENOKI_ROUTE_COMPOUND_OPERATOR(op)                                      \
    template <typename T1, enable_if_t<is_array_v<T1> &&                       \
                                      !std::is_const_v<T1>> = 0, typename T2>  \
    ENOKI_INLINE T1 &operator op##=(T1 &a1, const T2 &a2) {                    \
        a1 = a1 op a2;                                                         \
        return a1;                                                             \
    }

ENOKI_ROUTE_BINARY(operator+, add)
ENOKI_ROUTE_BINARY(operator-, sub)
ENOKI_ROUTE_BINARY(operator*, mul)
ENOKI_ROUTE_BINARY(operator%, mod)
ENOKI_ROUTE_UNARY(operator-, neg)

ENOKI_ROUTE_BINARY(operator<,  lt)
ENOKI_ROUTE_BINARY(operator<=, le)
ENOKI_ROUTE_BINARY(operator>,  gt)
ENOKI_ROUTE_BINARY(operator>=, ge)

ENOKI_ROUTE_BINARY_SHIFT(operator<<, sl)
ENOKI_ROUTE_BINARY_SHIFT(operator>>, sr)

ENOKI_ROUTE_UNARY_IMM_FALLBACK(sl, sl, a << Imm)
ENOKI_ROUTE_UNARY_IMM_FALLBACK(sr, sr, a >> Imm)

ENOKI_ROUTE_BINARY_BITOP(operator&,  and)
ENOKI_ROUTE_BINARY_BITOP(operator&&, and)
ENOKI_ROUTE_BINARY_BITOP(operator|,  or)
ENOKI_ROUTE_BINARY_BITOP(operator||, or)
ENOKI_ROUTE_BINARY_BITOP(operator^,  xor)
ENOKI_ROUTE_UNARY(operator~, not)
ENOKI_ROUTE_UNARY(operator!, not)

ENOKI_ROUTE_BINARY_BITOP(andnot, andnot)
ENOKI_ROUTE_BINARY_FALLBACK(eq,  eq,  a1 == a2)
ENOKI_ROUTE_BINARY_FALLBACK(neq, neq, a1 != a2)

ENOKI_ROUTE_UNARY_FALLBACK(sqrt,  sqrt,  detail::sqrt_(a))
ENOKI_ROUTE_UNARY_FALLBACK(abs,   abs,   detail::abs_(a))
ENOKI_ROUTE_UNARY_FALLBACK(floor, floor, detail::floor_(a))
ENOKI_ROUTE_UNARY_FALLBACK(ceil,  ceil,  detail::ceil_(a))
ENOKI_ROUTE_UNARY_FALLBACK(round, round, detail::round_(a))
ENOKI_ROUTE_UNARY_FALLBACK(trunc, trunc, detail::trunc_(a))

ENOKI_ROUTE_UNARY_TYPE_FALLBACK(floor2int, floor2int, (T2) detail::floor_(a))
ENOKI_ROUTE_UNARY_TYPE_FALLBACK(ceil2int,  ceil2int,  (T2) detail::ceil_(a))
ENOKI_ROUTE_UNARY_TYPE_FALLBACK(round2int, round2int, (T2) detail::round_(a))
ENOKI_ROUTE_UNARY_TYPE_FALLBACK(trunc2int, trunc2int, (T2) detail::trunc_(a))

ENOKI_ROUTE_UNARY_FALLBACK(rcp, rcp, detail::rcp_(a))
ENOKI_ROUTE_UNARY_FALLBACK(rsqrt, rsqrt, detail::rsqrt_(a))

ENOKI_ROUTE_BINARY_FALLBACK(max, max, detail::max_((E) a1, (E) a2))
ENOKI_ROUTE_BINARY_FALLBACK(min, min, detail::min_((E) a1, (E) a2))

ENOKI_ROUTE_BINARY_FALLBACK(mulhi, mulhi, detail::mulhi_((E) a1, (E) a2))
ENOKI_ROUTE_UNARY_FALLBACK(lzcnt, lzcnt, detail::lzcnt_(a))
ENOKI_ROUTE_UNARY_FALLBACK(tzcnt, tzcnt, detail::tzcnt_(a))
ENOKI_ROUTE_UNARY_FALLBACK(popcnt, popcnt, detail::popcnt_(a))

ENOKI_ROUTE_TERNARY_FALLBACK(fmadd, fmadd,   detail::fmadd_((E) a1, (E) a2, (E) a3))
ENOKI_ROUTE_TERNARY_FALLBACK(fmsub, fmsub,   detail::fmadd_((E) a1, (E) a2, -(E) a3))
ENOKI_ROUTE_TERNARY_FALLBACK(fnmadd, fnmadd, detail::fmadd_(-(E) a1, (E) a2, (E) a3))
ENOKI_ROUTE_TERNARY_FALLBACK(fnmsub, fnmsub, detail::fmadd_(-(E) a1, (E) a2, -(E) a3))

ENOKI_ROUTE_COMPOUND_OPERATOR(+)
ENOKI_ROUTE_COMPOUND_OPERATOR(-)
ENOKI_ROUTE_COMPOUND_OPERATOR(*)
ENOKI_ROUTE_COMPOUND_OPERATOR(/)
ENOKI_ROUTE_COMPOUND_OPERATOR(^)
ENOKI_ROUTE_COMPOUND_OPERATOR(|)
ENOKI_ROUTE_COMPOUND_OPERATOR(&)
ENOKI_ROUTE_COMPOUND_OPERATOR(<<)
ENOKI_ROUTE_COMPOUND_OPERATOR(>>)

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
ENOKI_INLINE auto operator/(const T1 &a1, const T2 &a2) {
    using E  = expr_t<T1, T2>;
    using E2 = expr_t<scalar_t<T1>, T2>;

    if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)
        return a1.derived().div_(a2.derived());
    else if constexpr (std::is_floating_point_v<scalar_t<E>> &&
                       array_depth_v<T1> > array_depth_v<T2>) // reciprocal approximation
        return static_cast<ref_cast_t<T1, E>>(a1) *
               rcp(static_cast<ref_cast_t<T1, E2>>(a2));
    else
        return operator/(static_cast<ref_cast_t<T1, E>>(a1),
                         static_cast<ref_cast_t<T2, E>>(a2));
}

template <typename T, enable_if_not_array_t<T> = 0> T andnot(const T &a1, const T &a2) {
    return detail::andnot_(a1, a2);
}

template <typename M, typename T, typename F>
ENOKI_INLINE auto select(const M &m, const T &t, const F &f) {
    using E = replace_scalar_t<array_t<typename detail::deepest<T, F, M>::type>,
                               typename detail::expr<scalar_t<T>, scalar_t<F>>::type>;
    using EM = mask_t<E>;
    if constexpr (!is_array_v<E>) {
        return (bool) m ? (E) t : (E) f;
    } else if constexpr (std::is_same_v<M, EM> &&
                         std::is_same_v<T, E> &&
                         std::is_same_v<F, E>) {
        return E::select_(m.derived(), t.derived(), f.derived());
    } else if constexpr (is_enoki_struct_v<T> && std::is_same_v<T, F>) {
        T result;
        struct_support_t<T>::apply_3(
            t, f, result,
            [&m](auto const &x1, auto const &x2, auto &x3) ENOKI_INLINE_LAMBDA {
                x3 = select(m, x1, x2);
            });
        return result;
    } else {
        return select(
            static_cast<ref_cast_t<M, EM>>(m),
            static_cast<ref_cast_t<T, E>>(t),
            static_cast<ref_cast_t<F, E>>(f));
    }
}

/// Shuffle the entries of an array
template <size_t... Is, typename T>
ENOKI_INLINE auto shuffle(const T &a) {
    if constexpr (is_array_v<T>) {
        return a.template shuffle_<Is...>();
    } else {
        static_assert(sizeof...(Is) == 1 && (... && (Is == 0)), "Shuffle argument out of bounds!");
        return a;
    }
}

template <typename Target, typename Source>
ENOKI_INLINE decltype(auto) reinterpret_array(const Source &src) {
    if constexpr (std::is_same_v<Source, Target>) {
        return src;
    } else if constexpr (is_array_v<Target>) {
        return Target(src, detail::reinterpret_flag());
    } else if constexpr (std::is_scalar_v<Source> && std::is_scalar_v<Target>) {
        if constexpr (sizeof(Source) == sizeof(Target)) {
            return memcpy_cast<Target>(src);
        } else {
            using SrcInt = int_array_t<Source>;
            using TrgInt = int_array_t<Target>;
            if constexpr (std::is_same_v<Target, bool>)
                return memcpy_cast<SrcInt>(src) != 0 ? true : false;
            else
                return memcpy_cast<Target>(memcpy_cast<SrcInt>(src) != 0 ? TrgInt(-1) : TrgInt(0));
        }
    } else {
        static_assert(detail::false_v<Source, Target>, "reinterpret_array(): don't know what to do!");
    }
}

template <typename Target, typename Source>
ENOKI_INLINE bool reinterpret_array(const detail::MaskBit<Source> &src) {
    static_assert(std::is_same_v<Target, bool>);
    return (bool) src;
}

template <typename T> ENOKI_INLINE auto sqr(const T &value) {
    return value * value;
}

template <typename T> ENOKI_INLINE auto isnan(const T &a) {
    return !eq(a, a);
}

template <typename T> ENOKI_INLINE auto isinf(const T &a) {
    return eq(abs(a), Infinity<scalar_t<T>>);
}

template <typename T> ENOKI_INLINE auto isfinite(const T &a) {
    return abs(a) < Infinity<scalar_t<T>>;
}

/// Linearly interpolate between 'a' and 'b', using 't'
template <typename Value1, typename Value2, typename Value3>
auto lerp(const Value1 &a, const Value2 &b, const Value3 &t) {
    return fmadd(b, t, fnmadd(a, t, a));
}

/// Clamp the value 'value' to the range [min, max]
template <typename Value1, typename Value2, typename Value3>
auto clamp(const Value1 &value, const Value2 &min, const Value3 &max) {
    return enoki::max(enoki::min(value, max), min);
}

namespace detail {
    template <typename T> using has_zero = decltype(T().zero_(0));

    template <typename Array> ENOKI_INLINE Array sign_mask() {
        using Scalar = scalar_t<Array>;
        using UInt = uint_array_t<Scalar>;
        return Array(memcpy_cast<Scalar>(UInt(1) << (sizeof(UInt) * 8 - 1)));
    }
}

template <typename Array> ENOKI_INLINE Array sign(const Array &v) {
    if constexpr (std::is_floating_point_v<scalar_t<Array>> && !is_diff_array_v<Array>)
        return detail::or_(Array(1), detail::and_(detail::sign_mask<Array>(), v));
    else
        return select(v >= 0, Array(1), Array(-1));
}

template <typename Array1, typename Array2>
ENOKI_INLINE Array1 copysign(const Array1 &v1, const Array2 &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array2>> && !is_diff_array_v<Array2>) {
        return detail::or_(abs(v1), detail::and_(detail::sign_mask<Array2>(), v2));
    } else {
        Array1 v1_a = abs(v1);
        return select(v2 >= 0, v1_a, -v1_a);
    }
}

template <typename Array1, typename Array2>
ENOKI_INLINE Array1 copysign_neg(const Array1 &v1, const Array2 &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array2>> && !is_diff_array_v<Array2>) {
        return detail::or_(abs(v1), detail::andnot_(detail::sign_mask<Array2>(), v2));
    } else {
        Array1 v1_a = abs(v1);
        return select(v2 >= 0, -v1_a, v1_a);
    }
}

template <typename Array1, typename Array2>
ENOKI_INLINE Array1 mulsign(const Array1 &v1, const Array2 &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array2>> && !is_diff_array_v<Array2>) {
        return detail::xor_(v1, detail::and_(detail::sign_mask<Array2>(), v2));
    } else {
        return select(v2 >= 0, v1, -v1);
    }
}

template <typename Array1, typename Array2>
ENOKI_INLINE Array1 mulsign_neg(const Array1 &v1, const Array2 &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array2>> && !is_diff_array_v<Array2>) {
        return detail::xor_(v1, detail::andnot_(detail::sign_mask<Array2>(), v2));
    } else {
        return select(v2 >= 0, -v1, v1);
    }
}

/// Fast implementation to compute ``floor(log2(value))`` for integer ``value``
template <typename T> ENOKI_INLINE T log2i(T value) {
    return scalar_t<T>(sizeof(scalar_t<T>) * 8 - 1) - lzcnt(value);
}

template <typename A, typename B> expr_t<A, B> hypot(const A &a, const B &b) {
    if constexpr (!std::is_same_v<A, B>) {
        using E = expr_t<A, B>;
        return hypot(static_cast<ref_cast_t<A, E>>(a),
                     static_cast<ref_cast_t<B, E>>(b));
    } else {
        using Value = A;

        Value abs_a  = abs(a),
              abs_b  = abs(b),
              maxval = max(abs_a, abs_b),
              minval = min(abs_a, abs_b),
              ratio  = minval / maxval;

        scalar_t<Value> inf = Infinity<Value>;

        return select(
            (abs_a < inf) && (abs_b < inf) && (ratio < inf),
            maxval * sqrt(fmadd(ratio, ratio, 1)),
            abs_a + abs_b
        );
    }
}

template <typename Value>
ENOKI_INLINE Value prev_float(const Value &value) {
    using Int = int_array_t<Value>;
    using IntMask = mask_t<Int>;
    using IntScalar = scalar_t<Int>;

    Int exponent_mask, pos_denorm;
    if constexpr (sizeof(IntScalar) == 4) {
        exponent_mask = IntScalar(0x7f800000);
        pos_denorm    = IntScalar(0x80000001);
    } else {
        exponent_mask = IntScalar(0x7ff0000000000000ll);
        pos_denorm    = IntScalar(0x8000000000000001ll);
    }

    Int i = reinterpret_array<Int>(value);

    IntMask is_nan_inf = eq(i & exponent_mask, exponent_mask),
            is_pos_0   = eq(i, 0),
            is_gt_0    = i >= 0,
            is_special = is_nan_inf | is_pos_0;

    Int j1 = i + select(is_gt_0, Int(-1), Int(1)),
        j2 = select(is_pos_0, pos_denorm, i);

    return reinterpret_array<Value>(select(is_special, j2, j1));
}

template <typename Value>
ENOKI_INLINE Value next_float(const Value &value) {
    using Int = int_array_t<Value>;
    using IntMask = mask_t<Int>;
    using IntScalar = scalar_t<Int>;

    Int exponent_mask, sign_mask;
    if constexpr (sizeof(IntScalar) == 4) {
        exponent_mask = IntScalar(0x7f800000);
        sign_mask     = IntScalar(0x80000000);
    } else {
        exponent_mask = IntScalar(0x7ff0000000000000ll);
        sign_mask     = IntScalar(0x8000000000000000ll);
    }

    Int i = reinterpret_array<Int>(value);

    IntMask is_nan_inf = eq(i & exponent_mask, exponent_mask),
            is_neg_0   = eq(i, sign_mask),
            is_gt_0    = i >= 0,
            is_special = is_nan_inf | is_neg_0;

    Int j1 = i + select(is_gt_0, Int(1), Int(-1)),
        j2 = select(is_neg_0, Int(1), i);

    return reinterpret_array<Value>(select(is_special, j2, j1));
}

template <typename X, typename Y> expr_t<X, Y> fmod(const X &x, const Y &y) {
    return fnmadd(trunc(x / y), y, x);
}

// -----------------------------------------------------------------------
//! @{ \name Horizontal operations: shuffle/gather/scatter/reductions..
// -----------------------------------------------------------------------

ENOKI_ROUTE_UNARY_FALLBACK(all,   all,   (bool) a)
ENOKI_ROUTE_UNARY_FALLBACK(any,   any,   (bool) a)
ENOKI_ROUTE_UNARY_FALLBACK(count, count, (uint32_t) ((bool) a ? 1 : 0))
ENOKI_ROUTE_UNARY_FALLBACK(hsum,  hsum,  a)
ENOKI_ROUTE_UNARY_FALLBACK(hprod, hprod, a)
ENOKI_ROUTE_UNARY_FALLBACK(hmin,  hmin,  a)
ENOKI_ROUTE_UNARY_FALLBACK(hmax,  hmax,  a)
ENOKI_ROUTE_BINARY_FALLBACK(dot, dot, (E) a1 * (E) a2)

ENOKI_ROUTE_UNARY_FALLBACK(hsum_async,  hsum_async,  a)
ENOKI_ROUTE_UNARY_FALLBACK(hprod_async, hprod_async, a)
ENOKI_ROUTE_UNARY_FALLBACK(hmin_async,  hmin_async,  a)
ENOKI_ROUTE_UNARY_FALLBACK(hmax_async,  hmax_async,  a)
ENOKI_ROUTE_BINARY_FALLBACK(dot_async, dot_async, (E) a1 * (E) a2)

template <typename Array>
ENOKI_INLINE auto hmean(const Array &a) {
    if constexpr (is_array_v<Array>)
        return hsum(a) * (1.f / a.derived().size());
    else
        return a;
}

template <typename Array>
ENOKI_INLINE auto hmean_async(const Array &a) {
    if constexpr (is_array_v<Array>)
        return hsum_async(a) * (1.f / a.derived().size());
    else
        return a;
}

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
ENOKI_INLINE bool operator==(const T1 &a1, const T2 &a2) {
    return all_nested(eq(a1, a2));
}

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
ENOKI_INLINE bool operator!=(const T1 &a1, const T2 &a2) {
    return any_nested(neq(a1, a2));
}

template <typename T> auto hsum_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else
        return hsum_nested(hsum(a));
}

template <typename T> auto hprod_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else
        return hprod_nested(hprod(a));
}

template <typename T> auto hmin_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else
        return hmin_nested(hmin(a));
}

template <typename T> auto hmax_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else
        return hmax_nested(hmax(a));
}

template <typename T> auto hmean_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else
        return hmean_nested(hmean(a));
}

template <typename T> auto count_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return count(a);
    else
        return hsum_nested(count(a));
}

template <typename T> auto any_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return any(a);
    else
        return any_nested(any(a));
}

template <typename T> auto all_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return all(a);
    else
        return all_nested(all(a));
}

template <typename T> auto none(const T &value) {
    return !any(value);
}

template <typename T> auto none_nested(const T &a) {
    return !any_nested(a);
}

template <typename Array, typename Mask>
value_t<Array> extract(const Array &array, const Mask &mask) {
    return array.extract_(mask);
}

template <typename Mask>
uint32_array_t<array_t<Mask>> compress(const Mask &mask) {
    static_assert(is_dynamic_array_v<Mask>);
    return mask.compress_();
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Miscellaneous routines for vector spaces
// -----------------------------------------------------------------------

template <typename T1, typename T2>
ENOKI_INLINE auto abs_dot(const T1 &a1, const T2 &a2) {
    return abs(dot(a1, a2));
}

template <typename T1, typename T2>
ENOKI_INLINE auto abs_dot_async(const T1 &a1, const T2 &a2) {
    return abs(dot_async(a1, a2));
}

template <typename T> ENOKI_INLINE auto squared_norm(const T &v) {
    if constexpr (array_depth_v<T> == 1 || array_size_v<T> == 0) {
        return hsum(v * v);
    } else {
        value_t<T> result = sqr(v.x());
        for (size_t i = 1; i < v.size(); ++i)
            result = fmadd(v.entry(i), v.entry(i), result);
        return result;
    }
}

template <typename T> ENOKI_INLINE auto norm(const T &v) {
    return sqrt(squared_norm(v));
}

template <typename T> ENOKI_INLINE auto normalize(const T &v) {
    return v * rsqrt(squared_norm(v));
}

template <typename T1, typename T2>
ENOKI_INLINE auto cross(const T1 &v1, const T2 &v2) {
    static_assert(array_size_v<T1> == 3 && array_size_v<T2> == 3,
            "cross(): requires 3D input arrays!");

#if defined(ENOKI_ARM_32) || defined(ENOKI_ARM_64)
    return fnmadd(shuffle<2, 0, 1>(v1), shuffle<1, 2, 0>(v2),
                  shuffle<1, 2, 0>(v1) * shuffle<2, 0, 1>(v2));
#else
    return fmsub(shuffle<1, 2, 0>(v1),  shuffle<2, 0, 1>(v2),
                 shuffle<2, 0, 1>(v1) * shuffle<1, 2, 0>(v2));
#endif
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Reduction operators that return a default argument when
//           invoked using JIT-compiled dynamic arrays
// -----------------------------------------------------------------------

template <bool Default, typename T> auto all_or(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        ENOKI_MARK_USED(value);
        return Default;
    } else {
        return all(value);
    }
}

template <bool Default, typename T> auto any_or(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        ENOKI_MARK_USED(value);
        return Default;
    } else {
        return any(value);
    }
}

template <bool Default, typename T> auto none_or(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        ENOKI_MARK_USED(value);
        return Default;
    } else {
        return none(value);
    }
}

template <bool Default, typename T> auto all_nested_or(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        ENOKI_MARK_USED(value);
        return Default;
    } else {
        return all_nested(value);
    }
}

template <bool Default, typename T> auto any_nested_or(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        ENOKI_MARK_USED(value);
        return Default;
    } else {
        return any_nested(value);
    }
}

template <bool Default, typename T> auto none_nested_or(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        ENOKI_MARK_USED(value);
        return Default;
    } else {
        return none_nested(value);
    }
}

template <typename T1, typename T2>
bool allclose(const T1 &a, const T2 &b, float rtol = 1e-5f, float atol = 1e-8f,
              bool equal_nan = false) {
    auto cond = abs(a - b) <= abs(b) * rtol + atol;

    if constexpr (std::is_floating_point_v<scalar_t<T1>> &&
                  std::is_floating_point_v<scalar_t<T2>>) {
        if (equal_nan)
            cond |= isnan(a) & isnan(b);
    }

    return all_nested(cond);
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Initialization, loading/writing data
// -----------------------------------------------------------------------

template <typename T> ENOKI_INLINE T zero(size_t size = 1) {
    if constexpr (is_array_v<T>) {
        return T::Derived::zero_(size);
    } else if constexpr (is_enoki_struct_v<T>) {
        T result;
        struct_support_t<T>::apply_1(
            result,
            [size](auto &x) ENOKI_INLINE_LAMBDA {
                using X = std::decay_t<decltype(x)>;
                x = zero<X>(size);
            });
        if constexpr (is_detected_v<detail::has_zero, T>)
            result.zero_(size);
        return result;
    } else {
        static_assert(std::is_scalar_v<T>,
                      "Unsupported data structure -- did you forget to include "
                      "'enoki/struct.h' or provide a suitable ENOKI_STRUCT() "
                      "declaration?");
        return T(0);
    }
}

#if defined(_MSC_VER) // don't warn about 'undef' below
#  pragma warning(push)
#  pragma warning(disable: 4700)
#endif

template <typename T> ENOKI_INLINE T empty(size_t size = 1) {
    if constexpr (is_array_v<T>) {
        return T::Derived::empty_(size);
    } else if constexpr (is_enoki_struct_v<T>) {
        T result;
        struct_support_t<T>::apply_1(
            result,
            [size](auto &x) ENOKI_INLINE_LAMBDA {
                using X = std::decay_t<decltype(x)>;
                x = empty<X>(size);
            });
        return result;
    } else {
        static_assert(std::is_scalar_v<T>,
                      "Unsupported data structure -- did you forget to include "
                      "'enoki/struct.h' or provide a suitable ENOKI_STRUCT() "
                      "declaration?");
        T undef;
        return undef;
    }
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

template <typename T, typename T2>
ENOKI_INLINE T full(const T2 &value, size_t size = 1, bool eval = false) {
    if constexpr (is_array_v<T>)
        return T::Derived::full_(value, size, eval);
    else
        return value;
}

template <typename T, enable_if_t<!is_special_v<T>> = 0>
ENOKI_INLINE T identity(size_t size = 1) {
    return full<T>(scalar_t<T>(1), size);
}

template <typename Array>
ENOKI_INLINE Array linspace(scalar_t<Array> min, scalar_t<Array> max, size_t size = 1) {
    if constexpr (is_array_v<Array>)
        return Array::linspace_(min, max, size);
    else
        return min;
}

template <typename Array>
ENOKI_INLINE Array arange(size_t size = 1) {
    if constexpr (is_array_v<Array>)
        return Array::arange_(0, (ssize_t) size, 1);
    else
        return Array(0);
}

template <typename Array>
ENOKI_INLINE Array arange(ssize_t start, ssize_t end, ssize_t step = 1) {
    if constexpr (is_array_v<Array>)
        return Array::arange_(start, end, step);
    else
        return Array(start);
}

/// Load an array from aligned memory
template <typename T> ENOKI_INLINE T load(const void *ptr, size_t size = 1) {
#if !defined(NDEBUG)
    if (ENOKI_UNLIKELY((uintptr_t) ptr % alignof(T) != 0))
        enoki_raise("load(): pointer %p is misaligned (alignment = %zu)!", ptr, alignof(T));
#endif
    if constexpr (is_array_v<T>)
        return T::load_(ptr, size);
    else
        return *static_cast<const T *>(ptr);
}

/// Map an array
template <typename T> ENOKI_INLINE T map(void *ptr, size_t size = 1, bool free = false) {
    static_assert(is_jit_array_v<T> && array_depth_v<T> == 1,
                  "enoki::map(): only flat JIT arrays supported!");
    return T::map_(ptr, size, free);
}

/// Load an array from unaligned memory
template <typename T> ENOKI_INLINE T load_unaligned(const void *ptr, size_t size = 1) {
    if constexpr (is_array_v<T>)
        return T::load_unaligned_(ptr, size);
    else
        return *static_cast<const T *>(ptr);
}

/// Store an array to aligned memory
template <typename T> ENOKI_INLINE void store(void *ptr, const T &value) {
#if !defined(NDEBUG)
    if (ENOKI_UNLIKELY((uintptr_t) ptr % alignof(T) != 0))
        enoki_raise("store(): pointer %p is misaligned (alignment = %zu)!", ptr, alignof(T));
#endif

    if constexpr (is_array_v<T>)
        value.store_(ptr);
    else
        *static_cast<T *>(ptr) = value;
}

/// Store an array to unaligned memory
template <typename T> ENOKI_INLINE void store_unaligned(void *ptr, const T &value) {
    if constexpr (is_array_v<T>)
        value.store_unaligned_(ptr);
    else
        *static_cast<T *>(ptr) = value;
}

namespace detail {
    template <typename Target, typename Index> Target broadcast_index(const Index &index) {
        using Scalar = scalar_t<Index>;
        static_assert(Target::Size != Dynamic);

        Index scaled = index * Scalar(Target::Size);
        Target result;
        for (size_t i = 0; i < Target::Size; ++i) {
            if constexpr (array_depth_v<Target> == array_depth_v<Index> + 1)
                result.entry(i) = scaled + Scalar(i);
            else
                result.entry(i) = broadcast_index<value_t<Target>>(scaled + Scalar(i));
        }
        return result;
    }
}

template <typename Target, bool Permute = false, typename Source, typename Index>
Target gather(Source &&source, const Index &index, const mask_t<Target> &mask = true) {
    if constexpr (array_depth_v<Source> > 1) {
        // Case 1: gather<Vector3fC>(const Vector3fC&, ...)
        static_assert(array_size_v<Source> == array_size_v<Target>,
                      "When gathering from a nested array source, the source "
                      "and target types must be compatible!");
        using Index2 = replace_scalar_t<Target, scalar_t<Index>>;
        Target result;
        if constexpr (Target::Size == Dynamic)
            result = empty<Target>(source.size());
        Index2 index2(index);
        mask_t<Index2> mask2(mask);
        for (size_t i = 0; i < source.size(); ++i)
            result.entry(i) = gather<value_t<Target>, Permute>(
                source.entry(i), index2.entry(i), mask2.entry(i));
        return result;
    } else if constexpr (is_array_v<Target>) {
        static_assert(std::is_pointer_v<std::decay_t<Source>> ||
                          array_depth_v<Source> == 1,
                      "Source argument of gather operation must either be a "
                      "pointer address or a flat array!");
        if constexpr (!is_array_v<Index>) {
            size_t offset = index * sizeof(value_t<Target>) * Target::Size;
            if constexpr (std::is_pointer_v<std::decay_t<Source>>) {
                // Case 2.0.0: gather<Target>(const void *, size_t, ...)
                return load_unaligned<Target>((const uint8_t *)source + offset);
            } else {
                // Case 2.0.1: gather<Target>(const FloatC&, size_t, ...)
                return load_unaligned<Target>((const uint8_t *)source.data() + offset);
            }
        } else if constexpr (array_depth_v<Target> == array_depth_v<Index>) {
            // Case 2.1: gather<FloatC>(const FloatC& / const void *, ...)

            if constexpr (Target::IsPacked && is_array_v<Source>)
                return Target::template gather_<Permute>(source.data(), index, mask);
            else
                return Target::template gather_<Permute>(source, index, mask);
        } else {
            // Case 2.2: gather<Vector3fC>(const FloatC & / const void *, ...)
            using TargetIndex = replace_scalar_t<Target, scalar_t<Index>>;

            return gather<Target, Permute>(
                source, detail::broadcast_index<TargetIndex>(index), mask);
        }
    } else if constexpr (is_enoki_struct_v<Target>) {
        /// Case 3: gather<MyStruct>(const MyStruct &, ...)
        static_assert(is_enoki_struct_v<Source>, "Source must also be a custom data structure!");

        Target result;
        struct_support_t<Target>::apply_2(
            source, result,
            [&index, &mask](auto const &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                using X2 = std::decay_t<decltype(x2)>;
                x2 = gather<X2, Permute>(x1, index, mask);
            });

        return result;
    } else {
        /// Case 4: gather<float>(const float *, ...)
        static_assert(std::is_integral_v<Index> && std::is_scalar_v<Target>,
                      "gather(): unsupported inputs -- did you forget to "
                      "include 'enoki/struct.h' or provide a suitable "
                      "ENOKI_STRUCT() declaration?");

        if constexpr (is_array_v<Source>)
            return mask ? source[index] : Target(0);
        else
            return mask ? ((Target *) source)[index] : Target(0);
    }
}

template <bool Permute = false, typename Target, typename Value, typename Index>
void scatter(Target &&target, const Value &value, const Index &index, const mask_t<Value> &mask = true) {
    if constexpr (array_depth_v<Target> > 1) {
        // Case 1: scatter(Vector3fC&, const Vector3fC &...)
        static_assert(array_size_v<Value> == array_size_v<Target>,
                      "When scattering a nested array value, the source and "
                      "target types must be compatible!");
        using Index2 = replace_scalar_t<Value, scalar_t<Index>>;
        Index2 index2(index);
        mask_t<Index2> mask2(mask);
        for (size_t i = 0; i < value.size(); ++i)
            scatter<Permute>(target.entry(i), value.entry(i),
                             index2.entry(i), mask2.entry(i));
    } else if constexpr (is_array_v<Value>) {
        static_assert(std::is_pointer_v<std::decay_t<Target>> || array_depth_v<Target> == 1,
                      "Target argument of scatter operation must either be a "
                      "pointer address or a flat array!");
        static_assert(is_array_v<Index> && is_integral_v<Index>,
                      "Second argument of gather operation must be an index array!");

        if constexpr (array_depth_v<Value> == array_depth_v<Index>) {
            value.template scatter_<Permute>(target, index, mask);
        } else {
            using TargetIndex = replace_scalar_t<Value, scalar_t<Index>>;
            scatter<Permute>(target, value,
                                    detail::broadcast_index<TargetIndex>(index), mask);
        }
    } else if constexpr (is_enoki_struct_v<Value>) {
        static_assert(is_enoki_struct_v<Target>, "Target must also be a custom data structure!");
        struct_support_t<Value>::apply_2(
            target, value,
            [&index, &mask](auto &x1, const auto &x2) ENOKI_INLINE_LAMBDA {
                scatter<Permute>(x1, x2, index, mask);
            });
    } else {
        static_assert(std::is_integral_v<Index> && std::is_scalar_v<Value>,
                      "scatter(): unsupported inputs -- did you forget to "
                      "include 'enoki/struct.h' or provide a suitable "
                      "ENOKI_STRUCT() declaration?");

        if (mask) {
            if constexpr (is_array_v<Target>)
                target[index] = value;
            else
                ((Value *) target)[index] = value;
        }
    }
}

template <typename Target, typename Value, typename Index>
void scatter_add(Target &&target, const Value &value, const Index &index, const mask_t<Value> &mask = true) {
    if constexpr (is_array_v<Value>) {
        static_assert(std::is_pointer_v<std::decay_t<Target>> || array_depth_v<Target> == 1,
                      "Target argument of scatter_add operation must either be a "
                      "pointer address or a flat array!");
        static_assert(is_array_v<Index> && is_integral_v<Index>,
                      "Second argument of gather operation must be an index array!");

        if constexpr (array_depth_v<Value> == array_depth_v<Index>) {
            value.scatter_add_(target, index, mask);
        } else {
            using TargetIndex = replace_scalar_t<Value, scalar_t<Index>>;
            scatter_add(target, value, detail::broadcast_index<TargetIndex>(index), mask);
        }
    } else if constexpr (is_enoki_struct_v<Value>) {
        struct_support<Value>::scatter_add(target, value, index, mask);
    } else if constexpr (std::is_integral_v<Index> && std::is_arithmetic_v<Value> && !std::is_same_v<Value, bool>) {
        if (mask) {
            if constexpr (is_array_v<Target>)
                target[index] += value;
            else
                ((Value *) target)[index] += value;
        }
    } else {
        static_assert(
            detail::false_v<Index, Value>,
            "scatter_add(): don't know what to do with these inputs. Did you forget "
            "an ENOKI_STRUCT() declaration for the type to be scattered?");
    }
}

template <typename T, typename Type> void migrate(T &value, Type type) {
    static_assert(std::is_enum_v<Type>);
    if constexpr (is_jit_array_v<T>) {
        if constexpr (is_jit_array_v<value_t<T>>) {
            for (size_t i = 0; i < value.derived().size(); ++i)
                migrate(value.derived().entry(i), type);
        } else {
            value.derived().migrate_(type);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [type](auto &x) {
                 migrate(x, type);
            });
    } else {
        ; // do nothing
    }
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Forward declarations of math functions
// -----------------------------------------------------------------------

template <typename T> T sin(const T &a);
template <typename T> T cos(const T &a);
template <typename T> std::pair<T, T> sincos(const T &a);
template <typename T> T csc(const T &a);
template <typename T> T sec(const T &a);
template <typename T> T tan(const T &a);
template <typename T> T cot(const T &a);
template <typename T> T asin(const T &a);
template <typename T> T acos(const T &a);
template <typename T> T atan(const T &a);
template <typename T1, typename T2> expr_t<T1, T2> atan2(const T1 &a, const T2 &b);

template <typename T> std::pair<T, T> frexp(const T &a);
template <typename T1, typename T2> expr_t<T1, T2> ldexp(const T1 &a, const T2 &b);
template <typename T> T exp(const T &a);
template <typename T> T exp2(const T &a);
template <typename T> T log(const T &a);
template <typename T> T log2(const T &a);
template <typename T1, typename T2> expr_t<T1, T2> pow(const T1 &a, const T2 &b);

template <typename T> T sinh(const T &a);
template <typename T> T cosh(const T &a);
template <typename T> std::pair<T, T> sincosh(const T &a);

template <typename T> T tanh(const T &a);
template <typename T> T asinh(const T &a);
template <typename T> T acosh(const T &a);
template <typename T> T atanh(const T &a);

template <typename T> T cbrt(const T &a);
template <typename T> T erf(const T &a);
template <typename T> T erfinv(const T &a);

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Reductions that operate on the inner dimension of an array
// -----------------------------------------------------------------------

#define ENOKI_INNER_REDUCTION(red)                                             \
    template <bool Reduce = false, typename Array>                             \
    ENOKI_INLINE auto red##_inner(const Array &a) {                            \
        if constexpr (array_depth_v<Array> <= 1) {                             \
            if constexpr (Reduce)                                              \
                return red(a);                                                 \
            else                                                               \
                return a;                                                      \
        } else {                                                               \
            using Value = decltype(red##_inner<true>(a.entry(0)));             \
            using Result = typename Array::template ReplaceValue<Value>;       \
            Result result;                                                     \
            if (Result::Size == Dynamic)                                       \
                result = enoki::empty<Result>(a.size());                       \
            for (size_t i = 0; i < a.size(); ++i)                              \
                result.set_entry(i, red##_inner<true>(a.entry(i)));            \
            return result;                                                     \
        }                                                                      \
    }

ENOKI_INNER_REDUCTION(hsum)
ENOKI_INNER_REDUCTION(hprod)
ENOKI_INNER_REDUCTION(hmin)
ENOKI_INNER_REDUCTION(hmax)
ENOKI_INNER_REDUCTION(hmean)

#undef ENOKI_INNER_REDUCTION

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name JIT compilation and autodiff-related
// -----------------------------------------------------------------------

template <typename T> ENOKI_INLINE bool schedule(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        if constexpr (is_jit_array_v<value_t<T>>) {
            bool result = false;
            for (size_t i = 0; i < value.derived().size(); ++i)
                result |= schedule(value.derived().entry(i));
            return result;
        } else {
            return value.derived().schedule_();
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        bool result = false;
        struct_support_t<T>::apply_1(
            value,
            [&](auto const &x) ENOKI_INLINE_LAMBDA {
                result |= schedule(x);
            });
        return result;
    } else {
        ; // do nothing
    }
    return false;
}

ENOKI_INLINE bool schedule() { return false; }

template <typename T1, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
ENOKI_INLINE bool schedule(const T1 &value, const Ts&... values) {
    return schedule(value) | schedule(values...);
}

ENOKI_INLINE void eval() {
    jitc_eval();
}

template <typename... Ts>
ENOKI_INLINE void eval(const Ts&... values) {
    if (schedule(values...))
        eval();
}

ENOKI_INLINE void set_device(int32_t device, uint32_t stream = 0) {
    jitc_set_device(device, stream);
}

ENOKI_INLINE void sync_stream() {
    jitc_sync_stream();
}

ENOKI_INLINE void sync_device() {
    jitc_sync_device();
}

ENOKI_INLINE void sync_all_devices() {
    jitc_sync_all_devices();
}

template <typename T> ENOKI_INLINE size_t width(const T &value) {
    if constexpr (array_size_v<T> == 0) {
        return 0;
    } if constexpr (array_depth_v<T> > 1) {
        size_t result = 0;
        for (size_t i = 0; i < value.derived().size(); ++i) {
            size_t w = width(value.derived().entry(i));
            if (w > result)
                result = w;
        }
        return result;
    } else if constexpr (is_array_v<T>) {
        return value.derived().size();
    } else if constexpr (is_enoki_struct_v<T>) {
        size_t result = 0;

        struct_support_t<T>::apply_1(
            value,
            [&](auto const &x) ENOKI_INLINE_LAMBDA {
                size_t w = width(x);
                if (w > result)
                    result = w;
            });

        return result;
    } else {
        return 1;
    }
}

template <typename T> ENOKI_INLINE void resize(T &value, size_t size) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.size(); ++i)
            resize(value.entry(i), size);
    } else if constexpr (is_jit_array_v<T>) {
        value.resize(size);
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value,
            [size](auto &x) ENOKI_INLINE_LAMBDA {
                resize(x, size);
            });
    }
}

template <typename T> void set_label(T &value, const char *label) {
    if constexpr (is_diff_array_v<T> || is_jit_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            size_t bufsize = strlen(label) + 11;
            char *buf = (char *) alloca(bufsize);
            for (size_t i = 0; i < value.size(); ++i) {
                snprintf(buf, bufsize, "%s_%zu", label, i);
                set_label(value.entry(i), buf);
            }
        } else {
            value.derived().set_label_(label);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        size_t bufsize = strlen(label) + 128;
        char *buf = (char *) alloca(bufsize);

        struct_support_t<T>::apply_label(
            value,
            [buf, bufsize, label](const char *label_2, auto &x) ENOKI_INLINE_LAMBDA {
                snprintf(buf, bufsize, "%s_%s", label, label_2);
                set_label(x, buf);
            });
    }
}

template <typename T> bool grad_enabled(const T &value) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            bool result = false;
            for (size_t i = 0; i < value.size(); ++i)
                result |= grad_enabled(value.entry(i));
            return result;
        } else {
            return value.derived().index() > 0;
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        bool result = false;

        struct_support_t<T>::apply_1(
            value,
            [&](auto const &x) ENOKI_INLINE_LAMBDA {
                result |= grad_enabled(x);
            });

        return result;
    } else {
        return false;
    }
}

template <typename T> bool grad_suspended(const T &value) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            bool result = false;
            for (size_t i = 0; i < value.size(); ++i)
                result |= grad_suspended(value.entry(i));
            return result;
        } else {
            return value.derived().index() < 0;
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        bool result = false;

        struct_support_t<T>::apply_1(
            value,
            [&](auto const &x) ENOKI_INLINE_LAMBDA {
                result |= grad_suspended(x);
            });

        return result;
    } else {
        return false;
    }
}

template <typename T> T replace_grad(const T &a, const T &b) {
    static_assert(is_diff_array_v<T>, "Type does not support gradients!");

    if constexpr (array_depth_v<T> > 1) {
        size_t sa = a.size(), sb = b.size(), sr = sa > sb ? sa : sb;
        T result;
        if constexpr (T::Size == Dynamic) {
            result = enoki::empty<T>(sr);
            if ((sa != sr && sa != 1) || (sb != sr && sb != 1))
                enoki_raise("replace_grad() : mismatched input sizes "
                            "(%zu and %zu)", sa, sb);
        }

        for (size_t i = 0; i < sr; ++i)
            result.entry(i) = replace_grad(a.entry(i), b.entry(i));

        return result;
    } else {
        return T::create_borrow(b.index(), a.detach_());
    }
}

template <typename T> void set_grad_enabled(T &value, bool state) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                set_grad_enabled(value.entry(i), state);
        } else {
            value.derived().set_grad_enabled_(state);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value,
            [state](auto &x) ENOKI_INLINE_LAMBDA {
                set_grad_enabled(x, state);
            });
    }
}

template <typename T> void set_grad_suspended(T &value, bool state) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                set_grad_suspended(value.entry(i), state);
        } else {
            value.derived().set_grad_suspended_(state);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value,
            [state](auto &x) ENOKI_INLINE_LAMBDA {
                set_grad_suspended(x, state);
            });
    }
}

template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1)> = 0>
bool grad_enabled(const Ts& ... ts) {
    return (grad_enabled(ts) || ...);
}

template <typename... Ts> void enable_grad(Ts&... ts) {
    (set_grad_enabled(ts, true), ...);
}

template <typename... Ts> void disable_grad(Ts&... ts) {
    (set_grad_enabled(ts, false), ...);
}

template <typename... Ts> void suspend_grad(Ts&... ts) {
    (set_grad_suspended(ts, true), ...);
}

template <typename... Ts> void resume_grad(Ts&... ts) {
    (set_grad_suspended(ts, false), ...);
}

template <typename T> detached_t<T> detach(const T &value) {
    using Result = detached_t<T>;

    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            Result result;
            if constexpr (Result::Size == Dynamic)
                result = empty<Result>(value.size());

            for (size_t i = 0; i < value.size(); ++i)
                result.entry(i) = detach(value.entry(i));

            return result;
        } else {
            return value.derived().detach_();
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        Result result;

        struct_support_t<T>::apply_2(
            value, result,
            [](auto const &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                x2 = detach(x1);
            });

        return result;
    } else {
        return value;
    }
}

template <typename T> detached_t<T> grad(const T &value) {
    using Result = detached_t<T>;

    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            Result result;
            if constexpr (Result::Size == Dynamic)
                result = empty<Result>(value.size());

            for (size_t i = 0; i < value.size(); ++i)
                result.entry(i) = grad(value.entry(i));

            return result;
        } else {
            return value.derived().grad_();
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        Result result;

        struct_support_t<T>::apply_2(
            value, result,
            [](auto const &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                x2 = grad(x1);
            });

        return result;
    } else {
        return zero<Result>();
    }
}

template <typename T>
void set_grad(T &value, const detached_t<T> &grad) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                set_grad(value.entry(i), grad.entry(i));
        } else {
            value.derived().set_grad_(grad);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_2(
            value, grad,
            [&](auto &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                set_grad(x1, x2);
            });
    }
}

template <typename T>
void accum_grad(T &value, const detached_t<T> &grad) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                accum_grad(value.entry(i), grad.entry(i));
        } else {
            value.derived().accum_grad_(grad);
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_2(
            value, grad,
            [&](auto &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                accum_grad(x1, x2);
            });
    }
}

template <typename T> void enqueue(const T &value) {
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                enqueue(value.entry(i));
        } else {
            value.derived().enqueue_();
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value,
            [](auto const &x) ENOKI_INLINE_LAMBDA {
                enqueue(x);
            });
    }
}

template <typename T1, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
void enqueue(const T1 &value, const Ts&... values) {
    enqueue(value);
    enqueue(values...);
}

ENOKI_INLINE void enqueue() { }

template <typename T> const char *graphviz(const T& value, bool reverse = true) {
    enqueue(value);
    return leaf_array_t<T>::graphviz_(reverse);
}

template <typename T> void traverse(bool reverse = true, bool retain_graph = false) {
    leaf_array_t<T>::traverse_(reverse, retain_graph);
}

template <typename T> void backward(T& value, bool retain_graph = false) {
    if (!grad_enabled(value))
        enoki_raise("backward(): attempted to propagate derivatives through a "
                    "variable that is not registered with the AD backend. Did "
                    "you forget to call enable_grad()?");
    set_grad(value, 1.f);
    enqueue(value);
    traverse<T>(true, retain_graph);
}

template <typename T> void forward(T& value, bool retain_graph = false) {
    if (!grad_enabled(value))
        enoki_raise("forward(): attempted to propagate derivatives through a "
                    "variable that is not registered with the AD backend. Did "
                    "you forget to call enable_grad()?");
    set_grad(value, 1.f);
    enqueue(value);
    traverse<T>(false, retain_graph);
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name "Safe" functions that avoid domain errors due to rounding
// -----------------------------------------------------------------------

template <typename Value> ENOKI_INLINE Value safe_sqrt(const Value &a) {
    Value result = sqrt(max(a, 0));

    if constexpr (is_diff_array_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, sqrt(max(a, Epsilon<Value>)));
    }

    return result;
}

template <typename Value> ENOKI_INLINE Value safe_rsqrt(const Value &a) {
    Value result = rsqrt(max(a, 0));

    if constexpr (is_diff_array_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, rsqrt(max(a, Epsilon<Value>)));
    }

    return result;
}

template <typename Value> ENOKI_INLINE Value safe_cbrt(const Value &a) {
    Value result = cbrt(max(a, 0));

    if constexpr (is_diff_array_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, cbrt(max(a, Epsilon<Value>)));
    }

    return result;
}

template <typename Value> ENOKI_INLINE Value safe_asin(const Value &a) {
    Value result = asin(clamp(a, -1, 1));

    if constexpr (is_diff_array_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, asin(clamp(a, -OneMinusEpsilon<Value>,
                                                     OneMinusEpsilon<Value>)));
    }

    return result;
}

template <typename Value> ENOKI_INLINE Value safe_acos(const Value &a) {
    Value result = acos(clamp(a, -1, 1));

    if constexpr (is_diff_array_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, acos(clamp(a, -OneMinusEpsilon<Value>,
                                                     OneMinusEpsilon<Value>)));
    }

    return result;
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Masked array helper classes
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

template <typename T>
struct MaskedArray : ArrayBase<value_t<T>, is_mask_v<T>, MaskedArray<T>> {
    using Mask     = mask_t<T>;
    static constexpr size_t Size = array_size_v<T>;
    static constexpr bool IsMaskedArray = true;

    MaskedArray() = default;
    MaskedArray(const MaskedArray<T> &value) : d(value.d), m(value.m) { }
    template <typename T2> MaskedArray(T2) { }
    MaskedArray(T &d, const Mask &m) : d(&d), m(m) { }

    ENOKI_INLINE void operator=(const MaskedArray<T> &value) { d = value.d; m = value.m; }

    #define ENOKI_MASKED_OPERATOR(name, expr)                                 \
        template <typename T2> ENOKI_INLINE void name(const T2 &value) {      \
            if constexpr (is_same_v<Mask, bool>) {                            \
                if (m)                                                        \
                    *d = expr;                                                \
            } else {                                                          \
                *d = select(m, expr, *d);                                     \
            }                                                                 \
        }

    ENOKI_MASKED_OPERATOR(operator=,         value)
    ENOKI_MASKED_OPERATOR(operator+=,  *d +  value)
    ENOKI_MASKED_OPERATOR(operator-=,  *d -  value)
    ENOKI_MASKED_OPERATOR(operator*=,  *d *  value)
    ENOKI_MASKED_OPERATOR(operator/=,  *d /  value)
    ENOKI_MASKED_OPERATOR(operator|=,  *d |  value)
    ENOKI_MASKED_OPERATOR(operator&=,  *d &  value)
    ENOKI_MASKED_OPERATOR(operator^=,  *d ^  value)
    ENOKI_MASKED_OPERATOR(operator<<=, *d << value)
    ENOKI_MASKED_OPERATOR(operator>>=, *d >> value)

    #undef ENOKI_MASKED_OPERATOR

    /// Type alias for a similar-shaped array over a different type
    template <typename T2> using ReplaceValue = MaskedArray<replace_value_t<T, T2>>;

    T *d = nullptr;
    Mask m = false;
};

NAMESPACE_END(detail)

template <typename T, typename Mask>
ENOKI_INLINE auto masked(T &value, const Mask &mask) {
    if constexpr (is_array_v<T> || std::is_scalar_v<Mask>) {
        return detail::MaskedArray<T>{ value, mask };
    } else if constexpr (is_enoki_struct_v<T>) {
        masked_t<T> result;

        struct_support_t<T>::apply_2(
            value, result,
            [&mask](auto &x1, auto &x2) ENOKI_INLINE_LAMBDA {
                x2 = masked(x1, mask);
            });

        return result;
    } else {
        static_assert(
            detail::false_v<T, Mask>,
            "masked(): don't know what to do with these inputs. Did you forget "
            "an ENOKI_STRUCT() declaration for type to be masked?");
    }
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Functions for accessing sub-regions of an array
// -----------------------------------------------------------------------

/// Extract the low elements from an array of even size
template <typename Array, enable_if_t<(array_size_v<Array>> 1 &&
                                       array_size_v<Array> != Dynamic)> = 0>
ENOKI_INLINE auto low(const Array &a) {
    return a.derived().low_();
}

/// Extract the high elements from an array of even size
template <typename Array, enable_if_t<(array_size_v<Array>> 1 &&
                                       array_size_v<Array> != Dynamic)> = 0>
ENOKI_INLINE auto high(const Array &a) {
    return a.derived().high_();
}

namespace detail {
    template <typename Return, size_t Offset, typename T, size_t... Index>
    static ENOKI_INLINE Return extract(const T &a, std::index_sequence<Index...>) {
        return Return(a.entry(Index + Offset)...);
    }
}

template <size_t Size, typename T> ENOKI_INLINE Array<value_t<T>, Size> head(const T &a) {
    static_assert(is_static_array_v<T>, "head(): input must be a static Enoki array!");

    if constexpr (T::Size == Size ||
                  Array<value_t<T>, Size>::ActualSize == T::ActualSize) {
        return a;
    } else if constexpr (T::Size1 == Size) {
        return low(a);
    } else {
        static_assert(Size <= array_size_v<T>, "Array size mismatch");
        return detail::extract<Array<value_t<T>, Size>, 0>(
            a, std::make_index_sequence<Size>());
    }
}

template <size_t Size, typename T> ENOKI_INLINE Array<value_t<T>, Size> tail(const T &a) {
    static_assert(is_static_array_v<T>, "tail(): input must be a static Enoki array!");

    if constexpr (T::Size == Size) {
        return a;
    } else if constexpr (T::Size2 == Size) {
        return high(a);
    } else {
        static_assert(Size <= array_size_v<T>, "Array size mismatch");
        return detail::extract<Array<value_t<T>, Size>, T::Size - Size>(
            a, std::make_index_sequence<Size>());
    }
}

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto concat(const T1 &a1, const T2 &a2) {
    static_assert(is_array_any_v<T1, T2>,
                  "concat(): at least one of the inputs must be an array!");
    static_assert(std::is_same_v<scalar_t<T1>, scalar_t<T2>>,
                  "concat(): Scalar types must be identical");

    constexpr size_t Size1 = array_size_v<T1>,
                     Size2 = array_size_v<T2>;

    using Result = Array<value_t<expr_t<T1, T2>>, Size1 + Size2>;

    if constexpr (Result::Size1 == T1::Size && Result::Size2 == T2::Size2) {
        return Result(a1, a2);
    } else {
        Result result;

        if constexpr (is_array_v<T1>) {
            if constexpr (Result::Size == T1::ActualSize) {
                result = a1.derived();
            } else {
                for (size_t i = 0; i < Size1; ++i)
                    result.entry(i) = a1.derived().entry(i);
            }
        } else {
            result.entry(0) = a1;
        }

        if constexpr (is_array_v<T2>) {
            for (size_t i = 0; i < Size2; ++i)
                result.entry(i + Size1) = a2.derived().entry(i);
        } else {
            result.entry(Size1) = a2;
        }

        return result;
    }
}

template <int Imm, typename T, enable_if_static_array_t<T> = 0>
ENOKI_INLINE T rotate_left(const T &a) {
    return a.template rotate_left_<Imm>();
}

template <int Imm, typename T, enable_if_static_array_t<T> = 0>
ENOKI_INLINE T rotate_right(const T &a) {
    return a.template rotate_right_<Imm>();
}

//! @}
// -----------------------------------------------------------------------

#undef ENOKI_ROUTE_UNARY
#undef ENOKI_ROUTE_UNARY_FALLBACK
#undef ENOKI_ROUTE_UNARY_IMM_FALLBACK
#undef ENOKI_ROUTE_UNARY_TYPE_FALLBACK
#undef ENOKI_ROUTE_BINARY
#undef ENOKI_ROUTE_BINARY_BITOP
#undef ENOKI_ROUTE_BINARY_SHIFT
#undef ENOKI_ROUTE_BINARY_FALLBACK
#undef ENOKI_ROUTE_TERNARY_FALLBACK
#undef ENOKI_ROUTE_COMPOUND_OPERATOR

NAMESPACE_END(enoki)
