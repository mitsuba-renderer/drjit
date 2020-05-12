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
    extern void jitc_eval();
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
        else if constexpr (is_mask_v<T2>)                                      \
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
ENOKI_ROUTE_BINARY(operator/, div)
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

template <typename T, enable_if_not_array_t<T> = 0> T andnot(const T &a1, const T &a2) {
    return detail::andnot_(a1, a2);
}

template <typename M, typename T, typename F>
ENOKI_INLINE auto select(const M &m, const T &t, const F &f) {
    using E = replace_scalar_t<array_t<typename detail::deepest<M, T, F>::type>,
                               typename detail::expr<scalar_t<T>, scalar_t<F>>::type>;
    using EM = mask_t<E>;
    if constexpr (!is_array_v<E>)
        return (bool) m ? (E) t : (E) f;
    else if constexpr (std::is_same_v<M, EM> &&
                       std::is_same_v<T, E> &&
                       std::is_same_v<F, E>)
        return E::select_(m.derived(), t.derived(), f.derived());
    else
        return select(
            static_cast<ref_cast_t<M, EM>>(m),
            static_cast<ref_cast_t<T, E>>(t),
            static_cast<ref_cast_t<F, E>>(f));
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

template <typename T> ENOKI_INLINE auto sqr(const T &value) {
    return value * value;
}

template <typename T> ENOKI_INLINE auto isnan(const T &a) {
    return !eq(a, a);
}

template <typename T> ENOKI_INLINE auto isinf(const T &a) {
    return enoki::eq(enoki::abs(a), Infinity<scalar_t<T>>);
}

template <typename T> ENOKI_INLINE auto isfinite(const T &a) {
    return enoki::abs(a) < Infinity<scalar_t<T>>;
}

namespace detail {
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

template <typename Array> ENOKI_INLINE Array copysign(const Array &v1, const Array &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array>> && !is_diff_array_v<Array>) {
        return detail::or_(enoki::abs(v1), detail::and_(detail::sign_mask<Array>(), v2));
    } else {
        Array v1_a = abs(v1);
        return select(v2 >= 0, v1_a, -v1_a);
    }
}

template <typename Array> ENOKI_INLINE Array copysign_neg(const Array &v1, const Array &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array>> && !is_diff_array_v<Array>) {
        return detail::or_(enoki::abs(v1), detail::andnot_(detail::sign_mask<Array>(), v2));
    } else {
        Array v1_a = abs(v1);
        return select(v2 >= 0, -v1_a, v1_a);
    }
}

template <typename Array> ENOKI_INLINE Array mulsign(const Array &v1, const Array &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array>> && !is_diff_array_v<Array>) {
        return detail::xor_(v1, detail::and_(detail::sign_mask<Array>(), v2));
    } else {
        return select(v2 >= 0, v1, -v1);
    }
}

template <typename Array> ENOKI_INLINE Array mulsign_neg(const Array &v1, const Array &v2) {
    if constexpr (std::is_floating_point_v<scalar_t<Array>> && !is_diff_array_v<Array>) {
        return detail::xor_(v1, detail::andnot_(detail::sign_mask<Array>(), v2));
    } else {
        return select(v2 >= 0, -v1, v1);
    }
}

/// Fast implementation for computing the base 2 log of an integer.
template <typename T> ENOKI_INLINE T log2i(T value) {
    return scalar_t<T>(sizeof(scalar_t<T>) * 8 - 1) - lzcnt(value);
}

// -----------------------------------------------------------------------
//! @{ \name Horizontal operations: shuffle/gather/scatter/reductions..
// -----------------------------------------------------------------------

ENOKI_ROUTE_UNARY_FALLBACK(all,   all,   (bool) a)
ENOKI_ROUTE_UNARY_FALLBACK(any,   any,   (bool) a)
ENOKI_ROUTE_UNARY_FALLBACK(count, count, (size_t) ((bool) a ? 1 : 0))
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

/// Extract the low elements from an array of even size
template <typename Array, enable_if_t<(array_size_v<Array> > 1 && array_size_v<Array> != Dynamic)> = 0>
ENOKI_INLINE auto low(const Array &a) { return a.derived().low_(); }

/// Extract the high elements from an array of even size
template <typename Array, enable_if_t<(array_size_v<Array> > 1 && array_size_v<Array> != Dynamic)> = 0>
ENOKI_INLINE auto high(const Array &a) { return a.derived().high_(); }


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

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Initialization, loading/writing data
// -----------------------------------------------------------------------

template <typename T> ENOKI_INLINE T zero(size_t size = 1) {
    if constexpr (is_array_v<T>) {
        return T::Derived::zero_(size);
    } else if constexpr (has_struct_support_v<T>) {
        return struct_support<T>::zero(size);
    } else {
        return T(0);
    }
}

template <typename T> ENOKI_INLINE T empty(size_t size = 1) {
    if constexpr (is_array_v<T>) {
        return T::Derived::empty_(size);
    } else if constexpr (has_struct_support_v<T>) {
        return struct_support<T>::empty(size);
    } else {
        T undef;
        return undef;
    }
}

template <typename T> ENOKI_INLINE T full(scalar_t<T> value, size_t size = 1) {
    if constexpr (is_array_v<T>)
        return T::Derived::full_(value, size);
    else
        return value;
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

/// Load an array from unaligned memory
template <typename T> ENOKI_INLINE T load_unaligned(const void *ptr, size_t size = 1) {
    if constexpr (is_array_v<T>)
        return T::load_unaligned_(ptr, size);
    else
        return *static_cast<const T *>(ptr);
}

/// Store an array to aligned memory
template <typename T> ENOKI_INLINE void store(void *ptr, const T &value, size_t size = 1) {
#if !defined(NDEBUG)
    if (ENOKI_UNLIKELY((uintptr_t) ptr % alignof(T) != 0))
        enoki_raise("store(): pointer %p is misaligned (alignment = %zu)!", ptr, alignof(T));
#endif

    if constexpr (is_array_v<T>)
        value.store_(ptr, size);
    else
        *static_cast<T *>(ptr) = value;
}

/// Store an array to unaligned memory
template <typename T> ENOKI_INLINE void store_unaligned(void *ptr, const T &value, size_t size = 1) {
    if constexpr (is_array_v<T>)
        value.store_unaligned_(ptr, size);
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
    if constexpr (is_array_v<Target>) {
        static_assert(std::is_pointer_v<std::decay_t<Source>> || array_depth_v<Source> == 1,
                      "Source argument of gather operation must either be a "
                      "pointer address or a flat array!");
        static_assert(is_array_v<Index> && is_integral_v<Index>,
                      "Second argument of gather operation must be an index array!");

        if constexpr (array_depth_v<Target> == array_depth_v<Index>) {
            return Target::template gather_<Permute>(source, index, mask);
        } else {
            using TargetIndex = replace_scalar_t<Target, scalar_t<Index>>;

            return enoki::gather<Target, Permute>(
                source, detail::broadcast_index<TargetIndex>(index), mask);
        }
    } else if constexpr (has_struct_support_v<Target>) {
        return struct_support<Target>::template gather<Permute>(source, index, mask);
    } else {
        static_assert(
            std::is_integral_v<Index>,
            "gather(): don't know what to do with these inputs. Did you forget "
            "an ENOKI_STRUCT() declaration for type to be gathered?");

        if constexpr (is_array_v<Source>)
            return mask ? source[index] : Target(0);
        else
            return mask ? ((Target *) source)[index] : Target(0);
    }
}

template <bool Permute = false, typename Target, typename Value, typename Index>
void scatter(Target &&target, const Value &value, const Index &index, const mask_t<Value> &mask = true) {
    if constexpr (is_array_v<Value>) {
        static_assert(std::is_pointer_v<std::decay_t<Target>> || array_depth_v<Target> == 1,
                      "Target argument of scatter operation must either be a "
                      "pointer address or a flat array!");
        static_assert(is_array_v<Index> && is_integral_v<Index>,
                      "Second argument of gather operation must be an index array!");

        if constexpr (array_depth_v<Value> == array_depth_v<Index>) {
            value.template scatter_<Permute>(target, index, mask);
        } else {
            using TargetIndex = replace_scalar_t<Value, scalar_t<Index>>;
            enoki::scatter<Permute>(target, value,
                                    detail::broadcast_index<TargetIndex>(index), mask);
        }
    } else if constexpr (has_struct_support_v<Value>) {
        struct_support<Value>::template scatter<Permute>(target, value, index, mask);
    } else {
        static_assert(
            std::is_integral_v<Index>,
            "scatter(): don't know what to do with these inputs. Did you forget "
            "an ENOKI_STRUCT() declaration for type to be scattered?");

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
            enoki::scatter_add(target, value, detail::broadcast_index<TargetIndex>(index), mask);
        }
    } else if constexpr (has_struct_support_v<Value>) {
        struct_support<Value>::scatter_add(target, value, index, mask);
    } else {
        static_assert(
            std::is_integral_v<Index>,
            "scatter_add(): don't know what to do with these inputs. Did you forget "
            "an ENOKI_STRUCT() declaration for type to be scattered?");

        if (mask) {
            if constexpr (is_array_v<Target>)
                target[index] += value;
            else
                ((Value *) target)[index] += value;
        }
    }
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Forward declarations of math functions
// -----------------------------------------------------------------------

template <typename Value> Value exp(const Value &value);
template <typename Value> Value log(const Value &value);
template <typename Value> Value sin(const Value &value);
template <typename Value> Value cos(const Value &value);
template <typename Value> std::pair<Value, Value> sincos(const Value &value);

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name JIT compilation and autodiff-related
// -----------------------------------------------------------------------


template <typename T> ENOKI_INLINE void schedule(const T &value) {
    if constexpr (is_jit_array_v<T>) {
        if constexpr (is_jit_array_v<value_t<T>>) {
            for (size_t i = 0; i < value.derived().size(); ++i)
                schedule(value.derived().entry(i));
        } else {
            value.derived().schedule();
        }
    } else if constexpr (has_struct_support_v<T>) {
        struct_support<T>::schedule(value);
    } else {
        ; // do nothing
    }
}

template <typename T1, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
ENOKI_INLINE void schedule(const T1 &value, const Ts&... values) {
    schedule(value);
    schedule(values...);
}

ENOKI_INLINE void schedule() { }

template <typename... Ts>
ENOKI_INLINE void eval(const Ts&... values) {
    schedule(values...);
    jitc_eval();
}

template <typename T> ENOKI_INLINE size_t slices(const T &value) {
    if constexpr (array_depth_v<T> > 1)
        return enoki::slices(value.entry(0));
    else if constexpr (is_array_v<T>)
        return value.size();
    else if constexpr (has_struct_support_v<T>)
        return struct_support<T>::slices(value);
    else
        return 1;
}

template <typename T> ENOKI_INLINE auto detach(const T &value) {
    if constexpr (is_diff_array_v<T> && array_depth_v<T> > 1) {
        using ValueGrad = decltype(detach(std::declval<value_t<T>>()));
        using Result = typename T::template ReplaceType<ValueGrad>;
        Result result;
        result.init_(value.size());
        for (size_t i = 0; i < value.size(); ++i)
            result.entry(i) = detach(value.entry(i));
        return result;
    } else if constexpr (is_diff_array_v<T>) {
        return value.value();
    } else if constexpr (has_struct_support_v<T>) {
        return struct_support<T>::detach(value);
    } else {
        return value;
    }
}

template <typename T> ENOKI_INLINE auto grad(const T &value) {
    if constexpr (is_diff_array_v<T> && array_depth_v<T> > 1) {
        using ValueGrad = decltype(grad(std::declval<value_t<T>>()));
        using Result = typename T::template ReplaceType<ValueGrad>;
        Result result;
        result.init_(value.size());
        for (size_t i = 0; i < value.size(); ++i)
            result.entry(i) = grad(value.entry(i));
        return result;
    } else if constexpr (is_diff_array_v<T>) {
        return value.grad();
    } else if constexpr (has_struct_support_v<T>) {
        return struct_support<T>::grad(value);
    } else {
        return; // return void
    }
}

template <typename T1, typename T2> ENOKI_INLINE void set_grad(T1 &value, const T2 &grad) {
    if constexpr (array_depth_v<T1> > 1) {
        for (size_t i = 0; i < value.size(); ++i) {
            if constexpr (array_size_v<T1> == array_size_v<T2>)
                set_grad(value.entry(i), grad.entry(i));
            else
                set_grad(value.entry(i), grad);
        }
    } else if constexpr (is_diff_array_v<T1>) {
        value.set_grad(grad);
    } else {
        static_assert(detail::false_v<T1, T2>, "Unsupported inputs!");
    }
}

template <typename T1> ENOKI_INLINE void set_label(T1 &value, const char *label) {
    if constexpr (array_depth_v<T1> > 1) {
        size_t label_size = strlen(label) + 11;
        char *buf = (char *) alloca(label_size);
        for (size_t i = 0; i < value.size(); ++i) {
            snprintf(buf, label_size, "%s_%zu", label, i);
            set_grad(value.entry(i), buf);
        }
    } else if constexpr (is_diff_array_v<T1> || is_jit_array_v<T1>) {
        value.set_label(label);
    } else if constexpr (has_struct_support_v<T1>) {
        struct_support<T1>::set_label(value);
    }
}

template <typename T> ENOKI_INLINE void requires_grad(T &value, bool state = true) {
    if constexpr (is_diff_array_v<T>) {
        value.requires_grad(state);
    } else if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.size(); ++i)
            requires_grad(value.entry(i), state);
    } else if constexpr (has_struct_support_v<T>) {
        struct_support<T>::requires_grad(value, state);
    } else {
        ; // do nothing
    }
}

template <typename... Ts>
ENOKI_INLINE void requires_grad(const Ts&... values) {
    bool unused[] = { (requires_grad(values), false)... };
    (void) unused;
}

template <typename T> ENOKI_INLINE void ad_schedule(T &value, bool reverse = true) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.size(); ++i)
            ad_schedule(value.entry(i), reverse);
    } else if constexpr (is_diff_array_v<T>) {
        value.ad_schedule(reverse);
    } else if constexpr (has_struct_support_v<T>) {
        struct_support<T>::ad_schedule(value, reverse);
    } else {
        ; // do nothing
    }
}

template <typename T>
ENOKI_INLINE const char *graphviz(const T& value) {
    using DiffArray = diff_array_t<T>;
    ad_schedule(value);
    return DiffArray::graphviz();
}

template <typename T> ENOKI_INLINE void backward(T& value, bool retain_graph = false) {
    using DiffArray = diff_array_t<T>;
    set_grad(value, 1.f);
    ad_schedule(value, true);
    DiffArray::traverse(true, retain_graph);
}

template <typename T> ENOKI_INLINE void forward(T& value, bool retain_graph = false) {
    using DiffArray = diff_array_t<T>;
    set_grad(value, 1.f);
    ad_schedule(value, false);
    DiffArray::traverse(false, retain_graph);
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Masked array helper classes
// -----------------------------------------------------------------------

NAMESPACE_BEGIN(detail)

template <typename T> struct MaskedValue {
    MaskedValue() = default;
    MaskedValue(T &d, bool m) : d(&d), m(m) { }

    template <typename T2> ENOKI_INLINE void operator =(const T2 &value) { if (m) *d = value; }
    template <typename T2> ENOKI_INLINE void operator+=(const T2 &value) { if (m) *d += value; }
    template <typename T2> ENOKI_INLINE void operator-=(const T2 &value) { if (m) *d -= value; }
    template <typename T2> ENOKI_INLINE void operator*=(const T2 &value) { if (m) *d *= value; }
    template <typename T2> ENOKI_INLINE void operator/=(const T2 &value) { if (m) *d /= value; }
    template <typename T2> ENOKI_INLINE void operator|=(const T2 &value) { if (m) *d |= value; }
    template <typename T2> ENOKI_INLINE void operator&=(const T2 &value) { if (m) *d &= value; }
    template <typename T2> ENOKI_INLINE void operator^=(const T2 &value) { if (m) *d ^= value; }
    template <typename T2> ENOKI_INLINE void operator<<=(const T2 &value) { if (m) *d <<= value; }
    template <typename T2> ENOKI_INLINE void operator>>=(const T2 &value) { if (m) *d >>= value; }

    T *d = nullptr;
    bool m = false;
};

template <typename T> struct MaskedArray : ArrayBaseT<value_t<T>, is_mask_v<T>, MaskedArray<T>> {
    using Mask     = mask_t<T>;
    static constexpr size_t Size = array_size_v<T>;
    static constexpr bool IsMaskedArray = true;

    MaskedArray() = default;
    MaskedArray(T &d, const Mask &m) : d(&d), m(m) { }

    template <typename T2> ENOKI_INLINE void operator =(const T2 &value) { *d = select(m, value, *d); }
    template <typename T2> ENOKI_INLINE void operator+=(const T2 &value) { *d = select(m, *d + value, *d); }
    template <typename T2> ENOKI_INLINE void operator-=(const T2 &value) { *d = select(m, *d - value, *d); }
    template <typename T2> ENOKI_INLINE void operator*=(const T2 &value) { *d = select(m, *d * value, *d); }
    template <typename T2> ENOKI_INLINE void operator/=(const T2 &value) { *d = select(m, *d / value, *d); }
    template <typename T2> ENOKI_INLINE void operator|=(const T2 &value) { *d = select(m, *d | value, *d); }
    template <typename T2> ENOKI_INLINE void operator&=(const T2 &value) { *d = select(m, *d & value, *d); }
    template <typename T2> ENOKI_INLINE void operator^=(const T2 &value) { *d = select(m, *d ^ value, *d); }
    template <typename T2> ENOKI_INLINE void operator<<=(const T2 &value) { *d = select(m, *d << value, *d); }
    template <typename T2> ENOKI_INLINE void operator>>=(const T2 &value) { *d = select(m, *d >> value, *d); }

    /// Type alias for a similar-shaped array over a different type
    template <typename T2> using ReplaceValue = MaskedArray<typename T::template ReplaceValue<T2>>;

    T *d = nullptr;
    Mask m = false;
};

NAMESPACE_END(detail)

template <typename T> struct struct_support {
    static constexpr bool Defined = false;
};

template <typename T, typename Mask>
ENOKI_INLINE auto masked(T &value, const Mask &mask) {
    if constexpr (is_array_v<T>) {
        return detail::MaskedArray<T>{ value, mask };
    } else if constexpr (has_struct_support_v<T>) {
        return struct_support<T>::masked(value, mask);
    } else {
        static_assert(
            std::is_same_v<Mask, bool>,
            "masked(): don't know what to do with these inputs. Did you forget "
            "an ENOKI_STRUCT() declaration for type to be masked?");
        return detail::MaskedValue<T>{ value, mask };
    }
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
