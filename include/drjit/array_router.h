/*
    drjit/array_router.h -- Helper functions which route function calls
    in the drjit namespace to the intended recipients

    (This file isn't meant to be included as-is. Please use 'drjit/array.h',
     which bundles all the 'array_*' headers in the right order.)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(min) || defined(max)
#  error min/max are defined as preprocessor symbols! Define NOMINMAX on MSVC.
#endif

NAMESPACE_BEGIN(drjit)

/// Define an unary operation
#define DRJIT_ROUTE_UNARY(name, func)                                          \
    template <typename T, enable_if_array_t<T> = 0>                            \
    DRJIT_INLINE auto name(const T &a) {                                       \
        return a.func##_();                                                    \
    }

/// Define an unary operation with a fallback expression for scalar arguments
#define DRJIT_ROUTE_UNARY_FALLBACK(name, func, expr)                           \
    template <typename T> DRJIT_INLINE auto name(const T &a) {                 \
        if constexpr (!is_array_v<T>)                                          \
            return expr; /* Scalar fallback implementation */                  \
        else                                                                   \
            return a.func##_(); /* Forward to array */                         \
    }


/// Define an unary operation with an immediate argument (e.g. sr<5>(...))
#define DRJIT_ROUTE_UNARY_IMM_FALLBACK(name, func, expr)                       \
    template <int Imm, typename T> DRJIT_INLINE auto name(const T &a) {        \
        if constexpr (!is_array_v<T>)                                          \
            return expr; /* Scalar fallback implementation */                  \
        else                                                                   \
            return a.template func##_<Imm>(); /* Forward to array */           \
    }


/// Define an unary operation with a fallback expression for scalar arguments
#define DRJIT_ROUTE_UNARY_TYPE_FALLBACK(name, func, expr)                      \
    template <typename T2, typename T> DRJIT_INLINE auto name(const T &a) {    \
        if constexpr (!is_array_v<T>)                                          \
            return expr; /* Scalar fallback implementation */                  \
        else                                                                   \
            return a.template func##_<T2>(); /* Forward to array */            \
    }

/// Define a binary operation
#define DRJIT_ROUTE_BINARY(name, func)                                         \
    template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>     \
    DRJIT_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using E = expr_t<T1, T2>;                                              \
        if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)          \
            return a1.derived().func##_(a2.derived());                         \
        else                                                                   \
            return name(static_cast<ref_cast_t<T1, E>>(a1),                    \
                        static_cast<ref_cast_t<T2, E>>(a2));                   \
    }

/// Define a binary operation for bit operations
#define DRJIT_ROUTE_BINARY_BITOP(name, func)                                   \
    template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>     \
    DRJIT_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
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

#define DRJIT_ROUTE_BINARY_SHIFT(name, func)                                   \
    template <typename T1, typename T2,                                        \
              enable_if_t<is_arithmetic_v<scalar_t<T1>>> = 0,                  \
              enable_if_array_any_t<T1, T2> = 0>                               \
    DRJIT_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
        using E = expr_t<T1, T2>;                                              \
        if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)          \
            return a1.derived().func##_(a2.derived());                         \
        else                                                                   \
            return name(static_cast<ref_cast_t<T1, E>>(a1),                    \
                        static_cast<ref_cast_t<T2, E>>(a2));                   \
    }

/// Define a binary operation with a fallback expression for scalar arguments
#define DRJIT_ROUTE_BINARY_FALLBACK(name, func, expr)                          \
    template <typename T1, typename T2>                                        \
    DRJIT_INLINE auto name(const T1 &a1, const T2 &a2) {                       \
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
#define DRJIT_ROUTE_TERNARY_FALLBACK(name, func, expr)                         \
    template <typename T1, typename T2, typename T3>                           \
    DRJIT_INLINE auto name(const T1 &a1, const T2 &a2, const T3 &a3) {         \
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
#define DRJIT_ROUTE_COMPOUND_OPERATOR(op)                                      \
    template <typename T1, enable_if_t<is_array_v<T1> &&                       \
                                      !std::is_const_v<T1>> = 0, typename T2>  \
    DRJIT_INLINE T1 &operator op##=(T1 &a1, const T2 &a2) {                    \
        a1 = a1 op a2;                                                         \
        return a1;                                                             \
    }

DRJIT_ROUTE_BINARY(operator+, add)
DRJIT_ROUTE_BINARY(operator-, sub)
DRJIT_ROUTE_BINARY(operator*, mul)
DRJIT_ROUTE_BINARY(operator%, mod)
DRJIT_ROUTE_UNARY(operator-, neg)

DRJIT_ROUTE_BINARY(operator<,  lt)
DRJIT_ROUTE_BINARY(operator<=, le)
DRJIT_ROUTE_BINARY(operator>,  gt)
DRJIT_ROUTE_BINARY(operator>=, ge)
DRJIT_ROUTE_BINARY(operator==, eq)
DRJIT_ROUTE_BINARY(operator!=, neq)

DRJIT_ROUTE_BINARY_SHIFT(operator<<, sl)
DRJIT_ROUTE_BINARY_SHIFT(operator>>, sr)

DRJIT_ROUTE_UNARY_IMM_FALLBACK(sl, sl, (T) (std::make_unsigned_t<T>(a) << Imm))
DRJIT_ROUTE_UNARY_IMM_FALLBACK(sr, sr, a >> Imm)

DRJIT_ROUTE_BINARY_BITOP(operator&,  and)
DRJIT_ROUTE_BINARY_BITOP(operator&&, and)
DRJIT_ROUTE_BINARY_BITOP(operator|,  or)
DRJIT_ROUTE_BINARY_BITOP(operator||, or)
DRJIT_ROUTE_BINARY_BITOP(operator^,  xor)
DRJIT_ROUTE_UNARY(operator~, not)
DRJIT_ROUTE_UNARY(operator!, not)

DRJIT_ROUTE_BINARY_BITOP(andnot, andnot)

DRJIT_ROUTE_UNARY_FALLBACK(sqrt,  sqrt,  detail::sqrt_(a))
DRJIT_ROUTE_UNARY_FALLBACK(abs,   abs,   detail::abs_(a))
DRJIT_ROUTE_UNARY_FALLBACK(floor, floor, detail::floor_(a))
DRJIT_ROUTE_UNARY_FALLBACK(ceil,  ceil,  detail::ceil_(a))
DRJIT_ROUTE_UNARY_FALLBACK(round, round, detail::round_(a))
DRJIT_ROUTE_UNARY_FALLBACK(trunc, trunc, detail::trunc_(a))

DRJIT_ROUTE_UNARY_TYPE_FALLBACK(floor2int, floor2int, (T2) detail::floor_(a))
DRJIT_ROUTE_UNARY_TYPE_FALLBACK(ceil2int,  ceil2int,  (T2) detail::ceil_(a))
DRJIT_ROUTE_UNARY_TYPE_FALLBACK(round2int, round2int, (T2) detail::round_(a))
DRJIT_ROUTE_UNARY_TYPE_FALLBACK(trunc2int, trunc2int, (T2) detail::trunc_(a))

DRJIT_ROUTE_UNARY_FALLBACK(rcp, rcp, detail::rcp_(a))
DRJIT_ROUTE_UNARY_FALLBACK(rsqrt, rsqrt, detail::rsqrt_(a))

DRJIT_ROUTE_BINARY_FALLBACK(maximum, maximum, detail::maximum_((E) a1, (E) a2))
DRJIT_ROUTE_BINARY_FALLBACK(minimum, minimum, detail::minimum_((E) a1, (E) a2))

DRJIT_ROUTE_BINARY_FALLBACK(mulhi, mulhi, detail::mulhi_((E) a1, (E) a2))
DRJIT_ROUTE_UNARY_FALLBACK(lzcnt, lzcnt, detail::lzcnt_(a))
DRJIT_ROUTE_UNARY_FALLBACK(tzcnt, tzcnt, detail::tzcnt_(a))
DRJIT_ROUTE_UNARY_FALLBACK(popcnt, popcnt, detail::popcnt_(a))
DRJIT_ROUTE_UNARY_FALLBACK(brev, brev, detail::brev_(a))

DRJIT_ROUTE_TERNARY_FALLBACK(fmadd, fmadd,   detail::fmadd_((E) a1, (E) a2, (E) a3))
DRJIT_ROUTE_TERNARY_FALLBACK(fmsub, fmsub,   detail::fmadd_((E) a1, (E) a2, -(E) a3))
DRJIT_ROUTE_TERNARY_FALLBACK(fnmadd, fnmadd, detail::fmadd_(-(E) a1, (E) a2, (E) a3))
DRJIT_ROUTE_TERNARY_FALLBACK(fnmsub, fnmsub, detail::fmadd_(-(E) a1, (E) a2, -(E) a3))

DRJIT_ROUTE_COMPOUND_OPERATOR(+)
DRJIT_ROUTE_COMPOUND_OPERATOR(-)
DRJIT_ROUTE_COMPOUND_OPERATOR(*)
DRJIT_ROUTE_COMPOUND_OPERATOR(/)
DRJIT_ROUTE_COMPOUND_OPERATOR(^)
DRJIT_ROUTE_COMPOUND_OPERATOR(|)
DRJIT_ROUTE_COMPOUND_OPERATOR(&)
DRJIT_ROUTE_COMPOUND_OPERATOR(<<)
DRJIT_ROUTE_COMPOUND_OPERATOR(>>)

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
DRJIT_INLINE auto operator/(const T1 &a1, const T2 &a2) {
    using E  = expr_t<T1, T2>;
    using E2 = expr_t<scalar_t<T1>, T2>;

    if constexpr (std::is_same_v<T1, E> && std::is_same_v<T2, E>)
        return a1.derived().div_(a2.derived());
    else if constexpr (is_floating_point_v<scalar_t<E>> &&
                       depth_v<T1> > depth_v<T2>) // reciprocal approximation
        return static_cast<ref_cast_t<T1, E>>(a1) *
               rcp(static_cast<ref_cast_t<T1, E2>>(a2));
    else
        return operator/(static_cast<ref_cast_t<T1, E>>(a1),
                         static_cast<ref_cast_t<T2, E>>(a2));
}

template <typename T1, typename T2>
[[deprecated("drjit::eq is deprecated, please use the normal '==' operator.")]]
auto eq(const T1 &a, const T2 &b) {
    return a == b;
}

template <typename T1, typename T2>
[[deprecated("drjit::neq is deprecated, please use the normal '!=' operator.")]]
auto neq(const T1 &a, const T2 &b) {
    return a != b;
}

template <typename T, enable_if_not_array_t<T> = 0> T andnot(const T &a1, const T &a2) {
    return detail::andnot_(a1, a2);
}

template <typename T> DRJIT_INLINE T zeros(size_t size = 1);

template <typename M, typename T, typename F>
DRJIT_INLINE auto select(const M &m, const T &t, const F &f) {
    if constexpr (!is_array_v<F> && is_traversable_v<F> && std::is_same_v<T, F>) {
        T result;
        traverse_3(
            fields(t), fields(f), fields(result),
            [&m](auto &vt, auto &vf, auto &r) {
                using R = std::decay_t<decltype(r)>;
                if constexpr (is_array_v<M> && !(is_array_v<R> || is_drjit_struct_v<R>))
                    r = zeros<R>();
                else
                    r = select(m, vt, vf);
            });
        return result;
    } else {
        using E_ = replace_scalar_t<array_t<typename detail::deepest<T, F, M>::type>,
                                    typename detail::expr<scalar_t<T>, scalar_t<F>>::type>;
        using E = std::conditional_t<is_mask_v<T>, mask_t<E_>, E_>;
        using EM = mask_t<E>;

        if constexpr (!is_array_v<E>) {
            return (bool) m ? (E) t : (E) f;
        } else if constexpr (std::is_same_v<M, EM> &&
                             std::is_same_v<T, E> &&
                             std::is_same_v<F, E>) {
            return E::select_(m.derived(), t.derived(), f.derived());
        } else {
            return select(
                static_cast<ref_cast_t<M, EM>>(m),
                static_cast<ref_cast_t<T, E>>(t),
                static_cast<ref_cast_t<F, E>>(f));
        }
    }
}

/// Shuffle the entries of an array
template <size_t... Is, typename T>
DRJIT_INLINE auto shuffle(const T &a) {
    if constexpr (is_array_v<T>) {
        return a.template shuffle_<Is...>();
    } else {
        static_assert(sizeof...(Is) == 1 && (... && (Is == 0)), "Shuffle argument out of bounds!");
        return a;
    }
}

template <typename Target, typename Source>
DRJIT_INLINE decltype(auto) reinterpret_array(const Source &src) {
    if constexpr (std::is_same_v<Source, Target>) {
        return src;
    } else if constexpr (is_array_v<Target>) {
        return Target(src, detail::reinterpret_flag());
    } else if constexpr (detail::is_scalar_v<Source> && detail::is_scalar_v<Target>) {
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
DRJIT_INLINE bool reinterpret_array(const detail::MaskBit<Source> &src) {
    static_assert(std::is_same_v<Target, bool>);
    return (bool) src;
}

template <typename T>
[[deprecated("drjit::sqr was replaced by drjit::square (to be consistent with NumPy and to avoid typos/mixups with drjit::sqrt).")]]
DRJIT_INLINE auto sqr(const T &value) {
    return value * value;
}

template <typename T> DRJIT_INLINE auto square(const T &value) {
    return value * value;
}

template <typename T> DRJIT_INLINE auto isnan(const T &a) {
    return !(a == a);
}

template <typename T> DRJIT_INLINE auto isinf(const T &a) {
    return abs(a) == Infinity<scalar_t<T>>;
}

template <typename T> DRJIT_INLINE auto isfinite(const T &a) {
    return abs(a) < Infinity<scalar_t<T>>;
}

/// Linearly interpolate between 'a' and 'b', using 't'
template <typename T1, typename T2, typename T3>
auto lerp(const T1 &a, const T2 &b, const T3 &t) {
    return fmadd(b, t, fnmadd(a, t, a));
}

/// Clip the value 'value' to the range [min, max]
template <typename T1, typename T2, typename T3>
auto clip(const T1 &value, const T2 &min, const T3 &max) {
    return maximum(minimum(value, max), min);
}

template <typename T1, typename T2, typename T3>
[[deprecated("drjit::clamp was replaced by drjit::clip (to be consistent with NumPy).")]]
auto clamp(const T1 &value, const T2 &min, const T3 &max) {
    return clip(value, min, max);
}

namespace detail {
    template <typename T> using det_zero   = decltype(T().zero_(0));
    template <typename T> using det_schedule_force = decltype(T().schedule_force_());

    template <typename T> DRJIT_INLINE T sign_mask() {
        using Scalar = scalar_t<T>;
        using UInt = uint_array_t<Scalar>;
        return T(memcpy_cast<Scalar, UInt>(UInt(1) << (sizeof(UInt) * 8 - 1)));
    }
}

template <typename T> DRJIT_INLINE mask_t<T> signbit(const T &v) {
    if constexpr (is_floating_point_v<scalar_t<T>>)
        return reinterpret_array<int_array_t<T>>(v) < 0;
    else
        return v < 0;
}


template <typename T1, typename T2>
DRJIT_INLINE T1 copysign(const T1 &v1, const T2 &v2) {
    T1 v1_a = abs(v1);

    if constexpr (is_floating_point_v<scalar_t<T2>> && !is_diff_v<T2>)
        return detail::or_(v1_a, detail::and_(detail::sign_mask<T2>(), v2));
    else
        return select(signbit(v2), -v1_a, v1_a);
}

template <typename T1, typename T2>
DRJIT_INLINE T1 copysign_neg(const T1 &v1, const T2 &v2) {
    T1 v1_a = abs(v1);

    if constexpr (is_floating_point_v<scalar_t<T2>> && !is_diff_v<T2>)
        return detail::or_(v1_a, detail::andnot_(detail::sign_mask<T2>(), v2));
    else
        return select(signbit(v2), v1_a, -v1_a);
}

template <typename T1, typename T2>
DRJIT_INLINE T1 mulsign(const T1 &v1, const T2 &v2) {
    if constexpr (is_floating_point_v<scalar_t<T2>> && !is_diff_v<T2>)
        return detail::xor_(v1, detail::and_(detail::sign_mask<T2>(), v2));
    else
        return select(signbit(v2), -v1, v1);
}

template <typename T1, typename T2>
DRJIT_INLINE T1 mulsign_neg(const T1 &v1, const T2 &v2) {
    if constexpr (drjit::is_floating_point_v<scalar_t<T2>> && !is_diff_v<T2>)
        return detail::xor_(v1, detail::andnot_(detail::sign_mask<T2>(), v2));
    else
        return select(signbit(v2), v1, -v1);
}

template <typename T> DRJIT_INLINE T sign(const T &v) {
    return copysign(T(1), v);
}

/// Fast implementation to compute ``floor(log2(value))`` for integer ``value``
template <typename T> DRJIT_INLINE T log2i(const T &value) {
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
              maxval = maximum(abs_a, abs_b),
              minval = minimum(abs_a, abs_b),
              ratio  = minval / maxval;

        scalar_t<Value> inf = Infinity<Value>;

        return select(
            (abs_a < inf) && (abs_b < inf) && (ratio < inf),
            maxval * sqrt(fmadd(ratio, ratio, 1)),
            abs_a + abs_b
        );
    }
}

template <typename T>
DRJIT_INLINE T prev_float(const T &value) {
    using Int = int_array_t<T>;
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

    IntMask is_nan_inf = (i & exponent_mask) == exponent_mask,
            is_pos_0   = i == 0,
            is_gt_0    = i >= 0,
            is_special = is_nan_inf | is_pos_0;

    Int j1 = i + select(is_gt_0, Int(-1), Int(1)),
        j2 = select(is_pos_0, pos_denorm, i);

    return reinterpret_array<T>(select(is_special, j2, j1));
}

template <typename T>
DRJIT_INLINE T next_float(const T &value) {
    using Int = int_array_t<T>;
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

    IntMask is_nan_inf = (i & exponent_mask) == exponent_mask,
            is_neg_0   = i == sign_mask,
            is_gt_0    = i >= 0,
            is_special = is_nan_inf | is_neg_0;

    Int j1 = i + select(is_gt_0, Int(1), Int(-1)),
        j2 = select(is_neg_0, Int(1), i);

    return reinterpret_array<T>(select(is_special, j2, j1));
}

template <typename X, typename Y> expr_t<X, Y> fmod(const X &x, const Y &y) {
    return fnmadd(trunc(x / y), y, x);
}

// -----------------------------------------------------------------------
//! @{ \name Horizontal operations: shuffle/gather/scatter/reductions..
// -----------------------------------------------------------------------

DRJIT_ROUTE_UNARY_FALLBACK(all,   all,   (bool) a)
DRJIT_ROUTE_UNARY_FALLBACK(any,   any,   (bool) a)
DRJIT_ROUTE_UNARY_FALLBACK(count, count, (uint32_t) ((bool) a ? 1 : 0))
DRJIT_ROUTE_UNARY_FALLBACK(sum,  sum,  a)
DRJIT_ROUTE_UNARY_FALLBACK(prod, prod, a)
DRJIT_ROUTE_UNARY_FALLBACK(min,  min,  a)
DRJIT_ROUTE_UNARY_FALLBACK(max,  max,  a)
DRJIT_ROUTE_BINARY_FALLBACK(dot, dot, (E) a1 * (E) a2)

template <typename Array>
DRJIT_INLINE auto mean(const Array &a) {
    if constexpr (is_array_v<Array>)
        return sum(a) * (1.f / a.derived().size());
    else
        return a;
}

template <typename T0 = void, typename T = void>
auto sum_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return sum_nested<T>(sum(a));
}

template <typename T0 = void, typename T = void>
auto prod_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return prod_nested<T>(prod(a));
}

template <typename T0 = void, typename T = void>
auto min_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return min_nested<T>(min(a));
}

template <typename T0 = void, typename T = void>
auto max_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return max_nested<T>(max(a));
}

template <typename T0 = void, typename T = void>
auto mean_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return a;
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return mean_nested<T>(mean(a));
}

template <typename T>
auto count_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return count(a);
    else
        return sum_nested(count(a));
}

template <typename T0 = void, typename T = void>
auto any_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return any(a);
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return any_nested<T>(any(a));
}

template <typename T0 = void, typename T = void>
auto all_nested(const T &a) {
    if constexpr (!is_array_v<T>)
        return all(a);
    else if constexpr (std::is_same_v<T, T0>)
        return a.entry(0);
    else
        return all_nested<T>(all(a));
}

template <typename T> auto none(const T &value) {
    return !any(value);
}

template <typename T>
auto none_nested(const T &a) {
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
DRJIT_INLINE auto abs_dot(const T1 &a1, const T2 &a2) {
    return abs(dot(a1, a2));
}

template <typename T> DRJIT_INLINE auto squared_norm(const T &v) {
    if constexpr (depth_v<T> == 1 || size_v<T> == 0) {
        return sum(v * v);
    } else {
        value_t<T> result = square(v.x());
        for (size_t i = 1; i < v.size(); ++i)
            result = fmadd(v.entry(i), v.entry(i), result);
        return result;
    }
}

template <typename T> DRJIT_INLINE auto norm(const T &v) {
    return sqrt(squared_norm(v));
}

template <typename T> DRJIT_INLINE auto normalize(const T &v) {
    return v * rsqrt(squared_norm(v));
}

template <typename T1, typename T2>
DRJIT_INLINE auto cross(const T1 &v1, const T2 &v2) {
    static_assert(size_v<T1> == 3 && size_v<T2> == 3,
            "cross(): requires 3D input arrays!");

#if defined(DRJIT_ARM_32) || defined(DRJIT_ARM_64)
    return fnmadd(shuffle<2, 0, 1>(v1),  shuffle<1, 2, 0>(v2),
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
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        DRJIT_MARK_USED(value);
        return Default;
    } else {
        return all(value);
    }
}

template <bool Default, typename T> auto any_or(const T &value) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        DRJIT_MARK_USED(value);
        return Default;
    } else {
        return any(value);
    }
}

template <bool Default, typename T> auto none_or(const T &value) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        DRJIT_MARK_USED(value);
        return Default;
    } else {
        return none(value);
    }
}

template <bool Default, typename T> auto all_nested_or(const T &value) {
    if constexpr (is_jit_v<T>) {
        DRJIT_MARK_USED(value);
        return Default;
    } else {
        return all_nested(value);
    }
}

template <bool Default, typename T> auto any_nested_or(const T &value) {
    if constexpr (is_jit_v<T>) {
        DRJIT_MARK_USED(value);
        return Default;
    } else {
        return any_nested(value);
    }
}

template <bool Default, typename T> auto none_nested_or(const T &value) {
    if constexpr (is_jit_v<T>) {
        DRJIT_MARK_USED(value);
        return Default;
    } else {
        return none_nested(value);
    }
}

template <typename T1, typename T2>
bool allclose(const T1 &a, const T2 &b, float rtol = 1e-5f, float atol = 1e-8f,
              bool equal_nan = false) {
    auto cond = abs(a - b) <= abs(b) * rtol + atol;

    if constexpr (is_floating_point_v<scalar_t<T1>> &&
                  is_floating_point_v<scalar_t<T2>>) {
        if (equal_nan)
            cond |= isnan(a) & isnan(b);
    }

    return all_nested(cond);
}

template <typename T> T block_reduce(ReduceOp op, const T &value, size_t block_size, int symbolic = -1) {
    if constexpr (is_traversable_v<T>) {
        T result;
        traverse_2(
            fields(result), fields(value),
            [op, block_size, symbolic](auto &x, const auto &y) {
                x = block_reduce(op, y, block_size, symbolic);
            });
        return result;
    } if constexpr (is_dynamic_array_v<T>) {
        return value.block_reduce_(op, block_size, symbolic);
    } else {
        return value;
    }
}

template <typename T> T block_sum(const T &value, size_t block_size, int symbolic = -1) {
    return block_reduce(ReduceOp::Add, value, block_size, symbolic);
}

template <typename T> T block_prefix_reduce(ReduceOp op, const T &value, uint32_t block_size, bool exclusive = true, bool reverse = false) {
    if constexpr (is_traversable_v<T>) {
        T result;
        traverse_2(
            fields(result), fields(value),
            [op, block_size, exclusive, reverse](auto &x, const auto &y) {
                x = block_prefix_reduce(op, y, block_size, exclusive, reverse);
            });
        return result;
    } if constexpr (is_dynamic_array_v<T>) {
        return value.block_prefix_reduce_(op, block_size, exclusive, reverse);
    } else {
        return value;
    }
}

template <typename T> T block_prefix_sum(const T &value, uint32_t block_size, bool exclusive = true, bool reverse = false) {
    return block_prefix_reduce(ReduceOp::Add, value, block_size, exclusive, reverse);
}

template <typename T> T prefix_sum(const T &value, bool exclusive = true, bool reverse = false) {
    return block_prefix_reduce(ReduceOp::Add, value, value.size(), exclusive, reverse);
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Initialization, loading/writing data
// -----------------------------------------------------------------------

/// Forward declarations
template <typename T, typename... Ts> size_t width(const T &, const Ts& ...);
template <typename T> bool schedule(const T &value);

template <typename T> DRJIT_INLINE T zeros(size_t size) {
    DRJIT_MARK_USED(size);
    if constexpr (std::is_same_v<T, std::nullptr_t>) {
        return nullptr;
    } else if constexpr (is_array_v<T>) {
        return T::Derived::zero_(size);
    } else if constexpr (is_traversable_v<T>) {
        T result;
        traverse_1(
            fields(result),
            [size](auto &x) {
                x = zeros<std::decay_t<decltype(x)>>(size);
            });
        if constexpr (is_detected_v<detail::det_zero, T>)
            result.zero_(size);
        return result;
    } else if constexpr (detail::is_scalar_v<T>) {
        return T(0);
    } else {
        return T();
    }
}

#  if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#  endif

template <typename T> DRJIT_INLINE T empty(size_t size = 1) {
    DRJIT_MARK_USED(size);
    if constexpr (is_array_v<T>) {
        return T::Derived::empty_(size);
    } else if constexpr (is_traversable_v<T>) {
        T result;
        traverse_1(
            fields(result),
            [size](auto &x) {
                x = empty<std::decay_t<decltype(x)>>(size);
            });
        return result;
    } else if constexpr (detail::is_scalar_v<T>) {
        return T(0);
    } else {
        return T();
    }
}

#  if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#  endif

template <typename T, typename T2>
DRJIT_INLINE T full(const T2 &value, size_t size = 1) {
    DRJIT_MARK_USED(size);
    if constexpr (is_array_v<T>)
        return T::Derived::full_(value, size);
    else
        return T(value);
}

template <typename T, typename T2>
DRJIT_INLINE T opaque(const T2 &value, size_t size = 1) {
    DRJIT_MARK_USED(size);
    if constexpr (!is_jit_v<T>) {
        return full<T>(value, size);
    } else if constexpr (is_jit_v<T>) {
        return T::Derived::opaque_(scalar_t<T>(value), size);
    } else if constexpr (is_traversable_v<T>) {
        T result;
        traverse_2(
            fields(result), fields(value),
            [size](auto &x1, auto &x2) {
                x1 = opaque(x2, size);
            });
        return result;
    } else {
        return T(value);
    }
}
template <typename T, enable_if_t<!is_special_v<T>> = 0>
DRJIT_INLINE T identity(size_t size = 1) {
    return full<T>(scalar_t<T>(1), size);
}

template <typename Array>
DRJIT_INLINE Array linspace(scalar_t<Array> min, scalar_t<Array> max,
                            size_t size = 1, bool endpoint = true) {
    DRJIT_MARK_USED(max);
    DRJIT_MARK_USED(size);
    DRJIT_MARK_USED(endpoint);

    if constexpr (is_array_v<Array>)
        return Array::linspace_(min, max, size, endpoint);
    else
        return min;
}

template <typename Array>
DRJIT_INLINE Array arange(size_t size = 1) {
    DRJIT_MARK_USED(size);
    if constexpr (is_array_v<Array>)
        return Array::arange_(0, (ssize_t) size, 1);
    else
        return Array(0);
}

template <typename Array>
DRJIT_INLINE Array arange(ssize_t start, ssize_t end, ssize_t step = 1) {
    DRJIT_MARK_USED(end);
    DRJIT_MARK_USED(step);

    if constexpr (is_array_v<Array>)
        return Array::arange_(start, end, step);
    else
        return Array(start);
}

/// Load an array from aligned memory
template <typename T> DRJIT_INLINE T load_aligned(const void *ptr, size_t size = 1) {
#if !defined(NDEBUG)
    if (DRJIT_UNLIKELY((uintptr_t) ptr % alignof(T) != 0))
        drjit_fail("load_aligned(): pointer %p is misaligned (alignment = %zu)!", ptr, alignof(T));
#endif
    DRJIT_MARK_USED(size);
    if constexpr (is_array_v<T>)
        return T::load_aligned_(ptr, size);
    else
        return *static_cast<const T *>(ptr);
}

/// Map an array
template <typename T> DRJIT_INLINE T map(void *ptr, size_t size = 1, bool free = false) {
    static_assert(is_jit_v<T> && depth_v<T> == 1,
                  "drjit::map(): only flat JIT arrays supported!");
    return T::map_(ptr, size, free);
}

/// Load an array from unaligned memory
template <typename T> DRJIT_INLINE T load(const void *ptr, size_t size = 1) {
    DRJIT_MARK_USED(size);
    if constexpr (is_array_v<T>)
        return T::load_(ptr, size);
    else
        return *static_cast<const T *>(ptr);
}

/// Store an array to aligned memory
template <typename T> DRJIT_INLINE void store_aligned(void *ptr, const T &value) {
#if !defined(NDEBUG)
    if (DRJIT_UNLIKELY((uintptr_t) ptr % alignof(T) != 0))
        drjit_fail("store_aligned(): pointer %p is misaligned (alignment = %zu)!", ptr, alignof(T));
#endif

    if constexpr (is_array_v<T>)
        value.store_aligned_(ptr);
    else
        *static_cast<T *>(ptr) = value;
}

/// Store an array to unaligned memory
template <typename T> DRJIT_INLINE void store(void *ptr, const T &value) {
    if constexpr (is_array_v<T>)
        value.store_(ptr);
    else
        *static_cast<T *>(ptr) = value;
}

template <typename Target, typename Source, typename Index, typename Mask = mask_t<Index>>
Target gather(Source &&source, const Index &index, const Mask &mask_ = true,
              ReduceMode mode = ReduceMode::Auto) {
    // Broadcast mask to match shape of Index
    mask_t<plain_t<Index>> mask = mask_;
    DRJIT_MARK_USED(mask);
    if constexpr (depth_v<Source> > 1) {
        // Case 1: gather<Vector3fC>(const Vector3fC&, ...)
        static_assert(size_v<Source> == size_v<Target>,
                      "When gathering from a nested array source, the source "
                      "and target types must be compatible!");
        using Index2 = plain_t<replace_scalar_t<Target, scalar_t<Index>>>;
        Target result;
        if constexpr (Target::Size == Dynamic)
            result = empty<Target>(source.size());
        Index2 index2(index);
        mask_t<Index2> mask2(mask);
        for (size_t i = 0; i < source.size(); ++i)
            result.entry(i) = gather<value_t<Target>>(
                source.entry(i), index2.entry(i), mask2.entry(i), mode);
        return result;
    } else if constexpr (is_array_v<Target>) {
        static_assert(std::is_pointer_v<std::decay_t<Source>> || depth_v<Source> == 1,
                      "Source argument of gather operation must either be a "
                      "pointer address or a flat array!");
        if constexpr (!is_array_v<Index>) {
            if constexpr (is_jit_v<Target> && is_jit_v<Source>) {
                // Case 2.0.0: gather<FloatC>(const FloatC&, size_t, ...)
                return Target::gather_(
                    source, uint32_array_t<Source>(index), mask, mode);
            } else {
                size_t offset = index * sizeof(value_t<Target>) * Target::Size;
                if constexpr (std::is_pointer_v<std::decay_t<Source>>) {
                    // Case 2.0.1: gather<Target>(const void *, size_t, ...)
                    return select(mask, load<Target>((const uint8_t *)source + offset), 0);
                } else {
                    #if !defined(NDEBUG)
                        if (DRJIT_UNLIKELY((size_t) index >= source.size()))
                            drjit_fail("gather(): out of range access (offset=%zu, size=%zu)!",
                                        (size_t) offset, source.size());
                    #endif
                    // Case 2.0.2: gather<Target>(const FloatP&, size_t, ...)
                    return select(mask, load<Target>((const uint8_t *)source.data() + offset), Target(0));
                }
            }
        } else if constexpr (depth_v<Target> == depth_v<Index>) {
            using Index2 = plain_t<replace_scalar_t<Target, uint32_t>>;
            mask_t<plain_t<replace_scalar_t<Index, scalar_t<Target>>>> mask2 = mask_;
            static_assert(is_array_v<Index> &&
                          is_integral_v<Index> &&
                          sizeof(typename Index::Scalar) <= 4,
                          "Second argument of gather operation must be a 32 bit index array!");
            if constexpr ((Target::IsPacked || Target::IsRecursive) && is_array_v<Source>)
                // Case 2.1.0: gather<FloatC>(const FloatP&, ...)
                return Target::template gather_<>(source.data(), Index2(index), mask2, mode);
            else
                // Case 2.1.1: gather<FloatC>(const FloatC& / const void *, ...)
                return Target::template gather_<>(source, Index2(index), mask2, mode);
        } else {
            // Case 2.2: gather<Vector3fC>(const FloatC & / const void *, ...)
            return value_t<Target>::template gather_packet_<Target::Size>(source, index, mask, mode);
        }
    } else if constexpr (is_drjit_struct_v<Target>) {
        /// Case 3: gather<MyStruct>(const MyStruct &, ...)
        static_assert(is_drjit_struct_v<Source>,
                      "Source must also be a custom data structure!");
        Target result;
        traverse_2(
            fields(source),
            fields(result),
            [&index, &mask, mode](auto &x1, auto &x2) {
                x2 = gather<std::decay_t<decltype(x2)>>(x1, index, mask, mode);
            });
        return result;
    } else {
        // Case 4: gather<float>(const float *, ...)
        static_assert(is_integral_v<Index> && detail::is_scalar_v<Target>,
                      "gather(): unsupported inputs -- did you forget to "
                      "provide a suitable DRJIT_STRUCT() declaration?");

        if constexpr (is_array_v<Source>)
            return mask ? source[index] : Target(0);
        else
            return mask ? ((Target *) source)[index] : Target(0);
    }
    (void) mode;
}

template <typename Target, typename Value, typename Index, typename Mask = mask_t<Index>>
void scatter(Target &target, const Value &value, const Index &index,
             const Mask &mask_ = true, ReduceMode mode = ReduceMode::Auto) {
    DRJIT_MARK_USED(mode);
    // Broadcast mask to match shape of Index
    mask_t<plain_t<Index>> mask = mask_;
    if constexpr (std::is_same_v<std::decay_t<Target>, std::nullptr_t>) {
        return; // Used by virtual function call dispatch when there is no return value
    } else if constexpr (depth_v<Target> > 1) {
        // Case 1: scatter(Vector3fC&, const Vector3fC &...)
        static_assert(size_v<Value> == size_v<Target>,
                      "When scattering a nested array value, the source and "
                      "target types must be compatible!");
        using Index2 = plain_t<replace_scalar_t<Value, scalar_t<Index>>>;
        Index2 index2(index);
        mask_t<Index2> mask2(mask);
        for (size_t i = 0; i < value.size(); ++i)
            scatter(target.entry(i), value.entry(i),
                    index2.entry(i), mask2.entry(i), mode);
    } else if constexpr (is_array_v<Value>) {
        static_assert(std::is_pointer_v<std::decay_t<Target>> ||
                          depth_v<Target> == 1,
                      "Target argument of scatter operation must either be a "
                      "pointer address or a flat array!");
        static_assert(is_array_v<Index> &&
                      is_integral_v<Index> &&
                      sizeof(typename Index::Scalar) <= 4,
                      "Third argument of scatter operation must be a 32 bit index array!");

        if constexpr (depth_v<Value> == depth_v<Index>) {
            value.scatter_(target, uint32_array_t<Value>(index), mask, mode);
        } else {
            Target::template scatter_packet_<Value::Size>(
                target, value, uint32_array_t<Target>(index), mask, mode);
        }
    } else if constexpr (is_drjit_struct_v<Value>) {
        static_assert(is_drjit_struct_v<Target>,
                      "Target must also be a custom data structure!");
        traverse_2(
            fields(target),
            fields(value),
            [&index, &mask, mode](auto &x1, auto &x2) {
                scatter(x1, x2, index, mask, mode);
            });
    } else {
        static_assert(is_integral_v<Index> && detail::is_scalar_v<Value>,
                      "scatter(): unsupported inputs -- did you forget to "
                      "provide a suitable DRJIT_STRUCT() declaration?");

        if (mask) {
            if constexpr (is_array_v<Target>)
                target.set_entry(index, value);
            else
                ((Value *) target)[index] = value;
        }
    }
}

template <typename Index>
Index scatter_inc(Index &target, const Index &index, const mask_t<Index> &value = true) {
    static_assert(is_jit_v<Index> && std::is_same_v<scalar_t<Index>, uint32_t> && depth_v<Index> == 1);
    return target.scatter_inc_(index, value);
}

template <typename Target, typename Value, typename Index, typename Mask = mask_t<Index>>
void scatter_reduce(ReduceOp op, Target &target, const Value &value,
                    const Index &index, const Mask &mask = true,
                    ReduceMode mode = ReduceMode::Auto) {
    if constexpr (is_array_v<Value>) {
        static_assert(std::is_pointer_v<std::decay_t<Target>> || depth_v<Target> == 1,
                      "Target argument of scatter_reduce operation must either be a "
                      "pointer address or a flat array!");
        static_assert(is_array_v<Index> && is_integral_v<Index>,
                      "Second argument of gather operation must be an index array!");

        if constexpr (depth_v<Value> == depth_v<Index>) {
            value.scatter_reduce_(target, index, mask, op, mode);
        } else {
            Target::template scatter_reduce_packet_<Value::Size>(
                target, value, uint32_array_t<Target>(index), mask, op, mode);
        }
    } else if constexpr (is_integral_v<Index> && detail::is_arithmetic_v<scalar_t<Value>>) {
        if (mask) {
            auto func = [op](const Value &a, const Value &b) -> Value {
                switch (op) {
                    case ReduceOp::Identity: return b;
                    case ReduceOp::Add:
                        if constexpr (!std::is_same_v<Value, bool>)
                            return a + b;
                        break;
                    case ReduceOp::Mul:
                        if constexpr (!std::is_same_v<Value, bool>)
                            return a * b;
                        break;
                    case ReduceOp::Min: return minimum(a, b);
                    case ReduceOp::Max: return maximum(a, b);
                    case ReduceOp::And:
                        if constexpr (is_integral_v<Value>)
                            return a & b;
                        break;
                    case ReduceOp::Or:
                        if constexpr (is_integral_v<Value>)
                            return a | b;
                        break;
                    default:
                        break;
                }

                drjit_fail("Reduce operation not supported");
            };

            if constexpr (is_array_v<Target>)
                target[index] = func(target[index], value);
            else
                ((Value *) target)[index] = func(((Value *) target)[index], value);
        }
    } else {
        static_assert(
            detail::false_v<Index, Value>,
            "scatter_reduce(): don't know what to do with these inputs.");
    }
    (void) mode;
}

template <typename Target, typename Value, typename Index,
          typename Mask = mask_t<Index>>
void scatter_add(Target &target, const Value &value, const Index &index,
                 const Mask &mask = true, ReduceMode mode = ReduceMode::Auto) {
    scatter_reduce(ReduceOp::Add, target, value, index, mask, mode);
}

template <typename Target, typename Value, typename Index, typename Mask = mask_t<Index>>
void scatter_add_kahan(Target &&target_1, Target &&target_2,
                       const Value &value, const Index &index,
                       const Mask &mask = true) {
    static_assert(
        is_jit_v<Target> &&
        is_jit_v<Value> &&
        is_jit_v<Index> &&
        is_floating_point_v<Value> &&
        depth_v<Value> == depth_v<Index> &&
        depth_v<Value> == 1,
        "Only flat JIT arrays are supported at the moment");

    value.scatter_add_kahan_(target_1, target_2, index, mask);
}

template <typename T, typename TargetType>
decltype(auto) migrate(const T &value, TargetType target) {
    static_assert(std::is_enum_v<TargetType>);
    DRJIT_MARK_USED(target);

    if constexpr (is_jit_v<T>) {
        if constexpr (depth_v<T> > 1) {
            T result;
            if constexpr (T::Size == Dynamic)
                result = empty<T>(value.size());

            for (size_t i = 0; i < value.size(); ++i)
                result.entry(i) = migrate(value.entry(i), target);

            return result;
        } else {
            return value.derived().migrate_(target);
        }
    } else if constexpr (is_drjit_struct_v<T>) {
        T result;

        traverse_2(
            fields(value),
            fields(result),
            [target](auto &x1, auto &x2) {
                x2 = migrate(x1, target);
            });

        return result;
    } else {
        return (const T &) value;
    }
}

template <typename ResultType = void, typename T>
decltype(auto) slice(const T &value, size_t index = 0) {
    schedule(value);

    if constexpr (depth_v<T> > 1) {
        using Value = std::decay_t<decltype(slice(value.entry(0), index))>;
        using Result = typename T::template ReplaceValue<Value>;
        Result result;
        if constexpr (Result::Size == Dynamic)
            result = empty<Result>(value.size());

        for (size_t i = 0; i < value.size(); ++i)
            result.set_entry(i, slice(value.entry(i), index));

        return result;
    } else if constexpr (is_drjit_struct_v<T>) {
        static_assert(!std::is_same_v<ResultType, void>,
                      "slice(): return type must be specified when used with "
                      "custom data structures!");
        ResultType result;
        traverse_2(
            fields(value), fields(result),
            [index](auto &x1, auto &x2) {
                x2 = slice(x1, index);
            });
        return result;
    } else if constexpr (is_dynamic_array_v<T>) {
        return scalar_t<T>(value[index]);
    } else {
        return value;
    }
}

template <typename Array> Array reverse(const Array &value) {
    uint32_t size = (uint32_t) value.size();

    if constexpr (depth_v<Array> == 1 && is_dynamic_v<Array>) {
        using UInt32 = uint32_array_t<Array>;
        return gather<Array>(value, size - 1 - arange<UInt32>(size));
    } else {
        Array result;
        if constexpr (Array::Size == Dynamic)
            result = empty<Array>(size);

        for (size_t i = 0; i < size; ++i)
            result.entry(i) = value.entry(size - 1 - i);

        return result;
    }
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Forward declarations of math functions
// -----------------------------------------------------------------------

template <typename T, bool = true> T sin(const T &a);
template <typename T, bool = true> T cos(const T &a);
template <typename T, bool = true> std::pair<T, T> sincos(const T &a);
template <typename T, bool = true> T csc(const T &a);
template <typename T, bool = true> T sec(const T &a);
template <typename T, bool = true> T tan(const T &a);
template <typename T, bool = true> T cot(const T &a);
template <typename T, bool = true> T asin(const T &a);
template <typename T, bool = true> T acos(const T &a);
template <typename T, bool = true> T atan(const T &a);
template <typename T1, typename T2, bool = true> expr_t<T1, T2> atan2(const T1 &a, const T2 &b);

template <typename T, bool = true> std::pair<T, T> frexp(const T &a);
template <typename T1, typename T2, bool = true> expr_t<T1, T2> ldexp(const T1 &a, const T2 &b);
template <typename T, bool = true> T exp(const T &a);
template <typename T, bool = true> T exp2(const T &a);
template <typename T, bool = true> T log(const T &a);
template <typename T, bool = true> T log2(const T &a);
template <typename T1, typename T2> expr_t<T1, T2> pow(const T1 &a, const T2 &b);

template <typename T, bool = true> T sinh(const T &a);
template <typename T, bool = true> T cosh(const T &a);
template <typename T, bool = true> std::pair<T, T> sincosh(const T &a);

template <typename T, bool = true> T tanh(const T &a);
template <typename T, bool = true> T asinh(const T &a);
template <typename T, bool = true> T acosh(const T &a);
template <typename T, bool = true> T atanh(const T &a);

template <typename T, bool = true> T cbrt(const T &a);
template <typename T, bool = true> T erf(const T &a);
template <typename T> T erfinv(const T &a);

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Reductions that operate on the inner dimension of an array
// -----------------------------------------------------------------------

#define DRJIT_INNER_REDUCTION(red)                                             \
    template <bool Reduce = false, typename Array>                             \
    DRJIT_INLINE auto red##_inner(const Array &a) {                            \
        if constexpr (depth_v<Array> <= 1) {                                   \
            if constexpr (Reduce)                                              \
                return red(a);                                                 \
            else                                                               \
                return a;                                                      \
        } else {                                                               \
            using Value = decltype(red##_inner<true>(a.entry(0)));             \
            using Result = typename Array::template ReplaceValue<Value>;       \
            Result result;                                                     \
            if constexpr (Result::Size == Dynamic)                             \
                result = empty<Result>(a.size());                              \
            for (size_t i = 0; i < a.size(); ++i)                              \
                result.set_entry(i, red##_inner<true>(a.entry(i)));            \
            return result;                                                     \
        }                                                                      \
    }

DRJIT_INNER_REDUCTION(sum)
DRJIT_INNER_REDUCTION(prod)
DRJIT_INNER_REDUCTION(min)
DRJIT_INNER_REDUCTION(max)
DRJIT_INNER_REDUCTION(mean)

#undef DRJIT_INNER_REDUCTION

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name JIT compilation and autodiff-related
// -----------------------------------------------------------------------

template <typename T> DRJIT_INLINE bool schedule(const T &value) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        return value.derived().schedule_();
    } else if constexpr (is_traversable_v<T>) {
        bool result = false;
        traverse_1(fields(value),
                   [&](auto &x) { result |= schedule(x); });
        return result;
    } else {
        return false;
    }
}

DRJIT_INLINE bool schedule() { return false; }

template <typename T1, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
DRJIT_INLINE bool schedule(const T1 &value, const Ts&... values) {
    // bool<->int conversion to perform logical AND without broadcasting or compiler warnings
    return (bool) ((int) schedule(value) | (int) schedule(values...));
}

DRJIT_INLINE void eval() {
    jit_eval();
}

template <typename... Ts>
DRJIT_INLINE void eval(const Ts&... values) {
    (DRJIT_MARK_USED(values), ...);
    if constexpr (((is_jit_v<Ts> || is_drjit_struct_v<Ts>) || ...)) {
        if (schedule(values...))
            eval();
    }
}

namespace detail {
    template <typename T> DRJIT_INLINE bool schedule_force(T &value) {
        if constexpr (is_jit_v<T> && depth_v<T> == 1) {
            return value.derived().schedule_force_();
        } else if constexpr (is_traversable_v<T>) {
            bool result = false;
            traverse_1(fields(value), [&](auto &x) {
                result |= schedule_force(x);
            });
            return result;
        } else if constexpr (is_detected_v<detail::det_schedule_force, T>) {
            return value.schedule_force_();
        } else {
            return false;
        }
    }

    DRJIT_INLINE bool schedule_force() { return false; }

    template <typename T1, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
    DRJIT_INLINE bool schedule_force(T1 &value, Ts&... values) {
        // bool<->int conversion to perform logical AND without broadcasting or compiler warnings
        return (bool) ((int) schedule_force(value) | (int) schedule_force(values...));
    }
}

template <typename... Ts>
DRJIT_INLINE void make_opaque(Ts&... values) {
    (DRJIT_MARK_USED(values), ...);
    if constexpr (((is_jit_v<Ts> || is_drjit_struct_v<Ts> ||
                    is_detected_v<detail::det_schedule_force, Ts>) || ...)) {
        if (detail::schedule_force(values...))
            eval();
    }
}

/**
 * \brief Helper to modify JIT flags within a given scope
 */
struct scoped_set_flag {
    scoped_set_flag(JitFlag flag, bool value) :
        flag(flag),
        original_value(jit_flag(flag)) {
        jit_set_flag(flag, value);
    }

    ~scoped_set_flag() {
        jit_set_flag(flag, original_value);
    }

    JitFlag flag;
    bool original_value;
};

DRJIT_INLINE void set_device(int32_t device) {
    jit_cuda_set_device(device);
}

DRJIT_INLINE void sync_thread() {
    jit_sync_thread();
}

DRJIT_INLINE void sync_device() {
    jit_sync_device();
}

DRJIT_INLINE void sync_all_devices() {
    jit_sync_all_devices();
}

template <typename T, typename... Ts> size_t width(const T &value, const Ts& ...values) {
    DRJIT_MARK_USED(value);
    size_t result = 0;
    if constexpr (size_v<T> == 0) {
        ;
    } else if constexpr (is_array_v<T> && depth_v<T> == 1) {
        result = value.derived().size();
    } else if constexpr (is_traversable_v<T>) {
        traverse_1(fields(value), [&](auto &x) {
            size_t w = width(x);
            if (w > result)
                result = w;
        });
    } else {
        result = 1;
    }

    if constexpr (sizeof...(Ts) > 0) {
        size_t other = width(values...);
        if (other > result)
            result = other;
    }

    return result;
}

template <typename T> void resize(T &value, size_t size) {
    DRJIT_MARK_USED(value);
    DRJIT_MARK_USED(size);

    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        value.resize(size);
    } else if constexpr (is_traversable_v<T>) {
        traverse_1(fields(value), [size](auto &x) { resize(x, size); });
    }
}

template <typename T, typename... Labels>
void set_label(T &value, Labels... prefix) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        value.derived().set_label_(prefix...);
    } else if constexpr (is_traversable_v<T>) {
        traverse_2(
            fields(value), labels(value),
            [&](auto &x, auto &l) {
            set_label(x, prefix..., l.c_str()); });
    }
}

/**
 * \brief Helper guard to mark scopes that are independent of any
 * ongoing symbolic computation
 *
 * Some scope of code might be traced as part of a symbolic section of code,
 * even though its functionality is independent of the actual symbolic
 * computations it is surrounded by. For example, consider a virtual function
 * call which initializes some data structure the first time it is called. This
 * initilization is completely detached from the symbolic inputs and is guarded
 * only by a scalar runtime check. In such a case, any evaluation which happens
 * in the initilization should be valid. Using this RAII helper will guarantee
 * this behavior.
 */
template <typename T>
struct scoped_symbolic_independence {
    scoped_symbolic_independence() {
        if constexpr(drjit::is_jit_v<T>) {
            uint32_t index = jit_var_mask_default(T::Backend, 1);
            jit_var_mask_push(T::Backend, index);
            jit_var_dec_ref(index);
        }
    }

    ~scoped_symbolic_independence() {
        if constexpr (drjit::is_jit_v<T>)
            jit_var_mask_pop(T::Backend);
    }
};

template <typename T> bool grad_enabled(const T &value) {
    if constexpr (is_diff_v<T> && depth_v<T> == 1) {
        return value.derived().grad_enabled_();
    } else if constexpr (is_traversable_v<T>) {
        bool result = false;
        traverse_1(fields(value), [&](auto &x) { result |= grad_enabled(x); });
        return result;
    } else {
        DRJIT_MARK_USED(value);
        return false;
    }
}

template <typename T> void set_grad_enabled(T &value, bool state) {
    DRJIT_MARK_USED(value);
    DRJIT_MARK_USED(state);

    if constexpr (is_diff_v<T> && depth_v<T> == 1) {
        value.set_grad_enabled_(state);
    } else if constexpr (is_traversable_v<T>) {
        traverse_1(fields(value),
                   [state](auto &x) { set_grad_enabled(x, state); });
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

template <typename T> T replace_grad(const T &a, const T &b) {
    static_assert(is_diff_v<T>, "Type does not support gradients!");
    size_t sa = a.size(), sb = b.size(), sr = sa > sb ? sa : sb;

    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))
        drjit_fail("replace_grad() : incompatible input sizes "
                    "(%zu and %zu)", sa, sb);

    if constexpr (depth_v<T> > 1) {
        T result;
        if constexpr (T::Size == Dynamic)
            result = empty<T>(sr);

        for (size_t i = 0; i < sr; ++i)
            result.entry(i) = replace_grad(a.entry(i), b.entry(i));

        return result;
    } else {
        T va = a, vb = b;
        if (sa != sb) {
            if (sa == 1)
                va += zeros<T>(sb);
            else if (sb == 1)
                vb += zeros<T>(sa);
            else
                drjit_fail("replace_grad(): internal error!");
        }

        return T::borrow(a.index() | (((uint64_t) b.index_ad()) << 32));
    }
}

template <bool PreserveType = true, typename T> decltype(auto) detach(const T &value) {
    using Result = std::decay_t<std::conditional_t<PreserveType, T, detached_t<T>>>;

    if constexpr (is_diff_v<T> && depth_v<T> == 1) {
        return Result::borrow(value.index());
    } else if constexpr (is_diff_v<T> && is_tensor_v<T>) {
        return Result(detach<PreserveType>(value.array()),
                      value.ndim(), value.shape());
    } else if constexpr (is_traversable_v<T>) {
        Result result;

        traverse_2(
            fields(value),
            fields(result),
            [](auto &v, auto &r) { r = detach<PreserveType>(v); }
        );

        return result;
    } else {
        return value;
    }
}

template <bool PreserveType = true, typename T>
auto grad(const T &value) {
    using Result = std::conditional_t<PreserveType, T, detached_t<T>>;

    if constexpr (is_diff_v<T> && depth_v<T> == 1) {
        return Result::steal(value.derived().grad_());
    } else if constexpr (is_diff_v<T> && is_tensor_v<T>) {
        return Result(grad<PreserveType>(value.array()), value.shape());
    } else if constexpr (is_traversable_v<T>) {
        Result result;

        traverse_2(
            fields(value),
            fields(result),
            [](auto &v, auto &r) { r = grad<PreserveType>(v); }
        );

        return result;
    } else {
        return zeros<Result>();
    }
}

template <typename T> void clear_grad(T &value) {
    if constexpr (is_diff_v<T> && depth_v<T> == 1)
        value.clear_grad_();
    else if constexpr (is_traversable_v<T>)
        traverse_1(fields(value), [](auto &v) { clear_grad(v); });
    else
        (void) value;
}

template <typename T, typename T2> void accum_grad(T &value, const T2 &grad) {
    if constexpr (is_diff_v<T> && depth_v<T> == 1) {
        if constexpr (!std::is_same_v<T, T2>)
            value.accum_grad_(T(grad).index());
        else
            value.accum_grad_(grad.index());
    } else if constexpr (is_traversable_v<T>) {
        traverse_2(fields(value), fields(grad),
                   [](auto &v1, auto &v2) { accum_grad(v1, v2); });
    }
}

template <typename T, typename T2>
void set_grad(T &value, const T2 &grad) {
    clear_grad(value);
    accum_grad(value, grad);
}

enum class ADScope { Invalid = 0, Suspend = 1, Resume = 2, Isolate = 3 };

template <typename T> const char *graphviz() {
    using Type = leaf_array_t<T>;

    if constexpr (is_diff_v<Type>)
        return Type::graphviz_();
    else if constexpr (is_jit_v<Type>)
        return jit_var_graphviz();
    else
        return "";
}

template <typename T> const char *graphviz(const T&) {
    return graphviz<T>();
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name "Safe" functions that avoid domain errors due to rounding
// -----------------------------------------------------------------------

template <typename Value> DRJIT_INLINE Value safe_sqrt(const Value &a) {
    Value result = sqrt(maximum(a, 0));

    if constexpr (is_diff_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, sqrt(maximum(a, Epsilon<Value>)));
    }

    return result;
}

template <typename Value> DRJIT_INLINE Value safe_rsqrt(const Value &a) {
    Value result = rsqrt(maximum(a, 0));

    if constexpr (is_diff_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, rsqrt(maximum(a, Epsilon<Value>)));
    }

    return result;
}

template <typename Value> DRJIT_INLINE Value safe_cbrt(const Value &a) {
    Value result = cbrt(maximum(a, 0));

    if constexpr (is_diff_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, cbrt(maximum(a, Epsilon<Value>)));
    }

    return result;
}

template <typename Value> DRJIT_INLINE Value safe_asin(const Value &a) {
    Value result = asin(clip(a, -1, 1));

    if constexpr (is_diff_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, asin(clip(a, -OneMinusEpsilon<Value>,
                                                     OneMinusEpsilon<Value>)));
    }

    return result;
}

template <typename Value> DRJIT_INLINE Value safe_acos(const Value &a) {
    Value result = acos(clip(a, -1, 1));

    if constexpr (is_diff_v<Value>) {
        if (grad_enabled(a))
            result = replace_grad(result, acos(clip(a, -OneMinusEpsilon<Value>,
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
struct MaskedArray : ArrayBaseT<value_t<T>, is_mask_v<T>, MaskedArray<T>> {
    using Mask     = mask_t<T>;
    static constexpr size_t Size = size_v<T>;
    static constexpr bool IsMaskedArray = true;

    MaskedArray() = default;
    MaskedArray(const MaskedArray<T> &value) : d(value.d), m(value.m) { }
    template <typename T2> MaskedArray(T2) { }
    MaskedArray(T &d, const Mask &m) : d(&d), m(m) { }

    DRJIT_INLINE void operator=(const MaskedArray<T> &value) { d = value.d; m = value.m; }

    #define DRJIT_MASKED_OPERATOR(name, expr)                                 \
        template <typename T2> DRJIT_INLINE void name(const T2 &value) {      \
            if constexpr (is_same_v<Mask, bool>) {                            \
                if (m)                                                        \
                    *d = expr;                                                \
            } else {                                                          \
                *d = select(m, expr, *d);                                     \
            }                                                                 \
        }

    DRJIT_MASKED_OPERATOR(operator=,         value)
    DRJIT_MASKED_OPERATOR(operator+=,  *d +  value)
    DRJIT_MASKED_OPERATOR(operator-=,  *d -  value)
    DRJIT_MASKED_OPERATOR(operator*=,  *d *  value)
    DRJIT_MASKED_OPERATOR(operator/=,  *d /  value)
    DRJIT_MASKED_OPERATOR(operator|=,  *d |  value)
    DRJIT_MASKED_OPERATOR(operator&=,  *d &  value)
    DRJIT_MASKED_OPERATOR(operator^=,  *d ^  value)
    DRJIT_MASKED_OPERATOR(operator<<=, *d << value)
    DRJIT_MASKED_OPERATOR(operator>>=, *d >> value)

    #undef DRJIT_MASKED_OPERATOR

    /// Type alias for a similar-shaped array over a different type
    template <typename T2> using ReplaceValue = MaskedArray<replace_value_t<T, T2>>;

    T *d = nullptr;
    Mask m = false;
};

template <typename T, typename Mask> struct MaskedTraversable {
    MaskedTraversable(T &value, Mask mask) : value(value), mask(mask) { }

    void operator=(const T &value_2) {
        value = select(mask, value_2, value);
    }

    T &value;
    Mask mask;
};

NAMESPACE_END(detail)

template <typename T, typename Mask>
DRJIT_INLINE auto masked(T &value, const Mask &mask) {
    if constexpr (is_array_v<T> || detail::is_scalar_v<Mask>) {
        return detail::MaskedArray<T>{ value, mask };
    } else if constexpr (is_traversable_v<T>) {
        return detail::MaskedTraversable<T, Mask> { value, mask };
    } else {
        static_assert(
            detail::false_v<T, Mask>,
            "masked(): don't know what to do with these inputs. Did you forget "
            "an DRJIT_STRUCT() declaration for type to be masked?");
    }
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Functions for accessing sub-regions of an array
// -----------------------------------------------------------------------

/// Extract the low elements from an array of even size
template <typename Array, enable_if_t<(size_v<Array>> 1 &&
                                       size_v<Array> != Dynamic)> = 0>
DRJIT_INLINE auto low(const Array &a) {
    return a.derived().low_();
}

/// Extract the high elements from an array of even size
template <typename Array, enable_if_t<(size_v<Array>> 1 &&
                                       size_v<Array> != Dynamic)> = 0>
DRJIT_INLINE auto high(const Array &a) {
    return a.derived().high_();
}

namespace detail {
    template <typename Return, size_t Offset, typename T, size_t... Index>
    static DRJIT_INLINE Return extract(const T &a, std::index_sequence<Index...>) {
        return Return(a.entry(Index + Offset)...);
    }
}

template <size_t Size, typename T> DRJIT_INLINE Array<value_t<T>, Size> head(const T &a) {
    static_assert(is_static_array_v<T>, "head(): input must be a static Dr.Jit array!");

    if constexpr (T::Size == Size ||
                  Array<value_t<T>, Size>::ActualSize == T::ActualSize) {
        return a;
    } else if constexpr (T::Size1 == Size) {
        return low(a);
    } else {
        static_assert(Size <= size_v<T>, "Array size mismatch");
        return detail::extract<Array<value_t<T>, Size>, 0>(
            a, std::make_index_sequence<Size>());
    }
}

template <size_t Size, typename T> DRJIT_INLINE Array<value_t<T>, Size> tail(const T &a) {
    static_assert(is_static_array_v<T>, "tail(): input must be a static Dr.Jit array!");

    if constexpr (T::Size == Size) {
        return a;
    } else if constexpr (T::Size2 == Size) {
        return high(a);
    } else {
        static_assert(Size <= size_v<T>, "Array size mismatch");
        return detail::extract<Array<value_t<T>, Size>, T::Size - Size>(
            a, std::make_index_sequence<Size>());
    }
}

template <typename T1, typename T2, enable_if_array_any_t<T1, T2> = 0>
auto concat(const T1 &a1, const T2 &a2) {
    constexpr size_t Size1 = size_v<T1>,
                     Size2 = size_v<T2>;

    static_assert(is_array_any_v<T1, T2>,
                  "concat(): at least one of the inputs must be an array!");
    static_assert(std::is_same_v<scalar_t<T1>, scalar_t<T2>>,
                  "concat(): Scalar types must be identical");
    static_assert((Size1 == Dynamic) == (Size2 == Dynamic),
                  "concat(): cannot mix dynamic and static arrays");

    if constexpr (Size1 != Dynamic) {
        using Result = Array<value_t<expr_t<T1, T2>>, Size1 + Size2>;

        if constexpr (Result::Size1 == Size1 && Result::Size2 == Size2) {
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
    } else {
        static_assert(std::is_same_v<T1, T2> && depth_v<T1> == 1);
        using Result = T1;
        using UInt32 = uint32_array_t<T1>;

        size_t s1 = a1.size(), s2 = a2.size();
        Result result = empty<Result>(s1 + s2);

        if (!grad_enabled(a1, a2)) {
            constexpr size_t ScalarSize = sizeof(scalar_t<Result>);
            uint8_t *ptr = (uint8_t *) result.data();

            if constexpr (is_jit_v<Result>) {
                constexpr JitBackend backend = detached_t<Result>::Backend;
                jit_memcpy_async(backend, ptr, a1.data(), s1 * ScalarSize);
                jit_memcpy_async(backend, ptr + s1 * ScalarSize, a2.data(), s2 * ScalarSize);
            } else {
                memcpy(ptr, a1.data(), s1 * ScalarSize);
                memcpy(ptr + s1 * ScalarSize, a2.data(), s2 * ScalarSize);
            }

            return result;
        }

        UInt32 offset = opaque<UInt32>((uint32_t) s1),
               index1 = arange<UInt32>(s1),
               index2 = arange<UInt32>(s2) + offset;

        scatter(result, a1, index1);
        scatter(result, a2, index2);

        return result;
    }
}

template <int Imm, typename T, enable_if_static_array_t<T> = 0>
DRJIT_INLINE T rotate_left(const T &a) {
    return a.template rotate_left_<Imm>();
}

template <int Imm, typename T, enable_if_static_array_t<T> = 0>
DRJIT_INLINE T rotate_right(const T &a) {
    return a.template rotate_right_<Imm>();
}

//! @}
// -----------------------------------------------------------------------

inline uint32_t flags() { return jit_flags(); }

#undef DRJIT_ROUTE_UNARY
#undef DRJIT_ROUTE_UNARY_FALLBACK
#undef DRJIT_ROUTE_UNARY_IMM_FALLBACK
#undef DRJIT_ROUTE_UNARY_TYPE_FALLBACK
#undef DRJIT_ROUTE_BINARY
#undef DRJIT_ROUTE_BINARY_BITOP
#undef DRJIT_ROUTE_BINARY_SHIFT
#undef DRJIT_ROUTE_BINARY_FALLBACK
#undef DRJIT_ROUTE_TERNARY_FALLBACK
#undef DRJIT_ROUTE_COMPOUND_OPERATOR

NAMESPACE_END(drjit)
