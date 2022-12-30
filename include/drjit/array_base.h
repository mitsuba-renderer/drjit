/*
    drjit/array_base.h -- Base class of all Dr.Jit arrays

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array_router.h>
#include <drjit/array_constants.h>

NAMESPACE_BEGIN(drjit)

#define DRJIT_ARRAY_DEFAULTS(Name)                                             \
    Name(const Name &) = default;                                              \
    Name(Name &&) = default;                                                   \
    Name &operator=(const Name &) = default;                                   \
    Name &operator=(Name &&) = default;

#define DRJIT_ARRAY_IMPORT(Name, Base)                                         \
    Name() = default;                                                          \
    DRJIT_ARRAY_DEFAULTS(Name)                                                 \
    using Base::Base;

#define DRJIT_ARRAY_FALLBACK_CONSTRUCTORS(Name)                                \
    template <typename Value2, typename D2, typename D = Derived_,             \
              enable_if_t<D::Size == D2::Size && D::Depth == D2::Depth> = 0>   \
    Name(const ArrayBase<Value2, false, D2> &v) {                              \
        DRJIT_CHKSCALAR("Copy constructor (conversion)");                      \
        for (size_t i = 0; i < derived().size(); ++i)                          \
            derived().entry(i) = (Value) v.derived().entry(i);                 \
    }                                                                          \
    template <typename Value2, typename D2, typename D = Derived_,             \
              enable_if_t<D::Size == D2::Size && D::Depth == D2::Depth> = 0>   \
    Name(const ArrayBase<Value2, IsMask_, D2> &v, detail::reinterpret_flag) {  \
        DRJIT_CHKSCALAR("Copy constructor (reinterpret_cast)");                \
        for (size_t i = 0; i < derived().size(); ++i)                          \
            derived().entry(i) = reinterpret_array<Value>(v[i]);               \
    }


/// Array base class templated via the curiously recurring template pattern
template <typename Value_, bool IsMask_, typename Derived_> struct ArrayBase {
    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations (may be overridden in subclasses)
    // -----------------------------------------------------------------------

    /// Type underlying the array
    using Value = Value_;

    /// Scalar data type all the way at the lowest level
    using Scalar = scalar_t<Value_>;

    /// Helper structure for dispatching vectorized method calls
    using CallSupport =
        call_support<std::decay_t<std::remove_pointer_t<Value_>>, Derived_>;

    /// Is this an Dr.Jit array?
    static constexpr bool IsDrJit = true;

    /// Specifies how deeply nested this array is
    static constexpr size_t Depth = 1 + array_depth_v<Value>;

    /// Is this a mask array?
    static constexpr bool IsMask = IsMask_;

    /// Is this an array of values that can be added, multiplied, etc.?
    static constexpr bool IsArithmetic = std::is_arithmetic_v<Scalar> && !IsMask;

    /// Is this an array of signed or unsigned integer values?
    static constexpr bool IsIntegral = std::is_integral_v<Scalar> && !IsMask;

    /// Is this an array of floating point values?
    static constexpr bool IsFloat = std::is_floating_point_v<Scalar> && !IsMask;

    /// Does this array map operations onto packed vector instructions?
    static constexpr bool IsPacked = false;

    /// Is this an AVX512-style 'k' mask register?
    static constexpr bool IsKMask = false;

    /// Is the storage representation of this array implemented recursively?
    static constexpr bool IsRecursive = false;

    /// Always prefer broadcasting to the outer dimensions of a N-D array
    static constexpr bool BroadcastOuter = true;

    /// Does this array represent a fixed size vector?
    static constexpr bool IsVector = false;

    /// Does this array represent a complex number?
    static constexpr bool IsComplex = false;

    /// Does this array represent a quaternion?
    static constexpr bool IsQuaternion = false;

    /// Does this array represent a matrix?
    static constexpr bool IsMatrix = false;

    /// Does this array represent a tensorial wrapper?
    static constexpr bool IsTensor = false;

    /// Does this array represent the result of a 'masked(...)' expression?
    static constexpr bool IsMaskedArray = false;

    /// Does this array compute derivatives using automatic differentiation?
    static constexpr bool IsDiff = is_diff_v<Value_>;

    /// Are elements of this array implemented using the LLVM backend?
    static constexpr bool IsLLVM = is_llvm_v<Value_>;

    /// Are elements of this array implemented using the CUDA backend?
    static constexpr bool IsCUDA = is_cuda_v<Value_>;

    /// Are elements of this array implemented using a JIT-compiled backend?
    static constexpr bool IsJIT = IsLLVM || IsCUDA;

    /// Are elements of this array dynamic?
    static constexpr bool IsDynamic = is_dynamic_v<Value_>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Curiously Recurring Template design pattern
    // -----------------------------------------------------------------------

    /// Alias to the derived type
    using Derived = Derived_;

    /// Cast to derived type
    DRJIT_INLINE Derived &derived()             { return (Derived &) *this; }

    /// Cast to derived type (const version)
    DRJIT_INLINE const Derived &derived() const { return (Derived &) *this; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Iterators
    // -----------------------------------------------------------------------

    DRJIT_INLINE auto begin() const { return derived().data(); }
    DRJIT_INLINE auto begin()       { return derived().data(); }
    DRJIT_INLINE auto end()   const { return derived().data() + derived().size(); }
    DRJIT_INLINE auto end()         { return derived().data() + derived().size(); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Element access
    // -----------------------------------------------------------------------

    /// Recursive array indexing operator
    template <typename... Indices, enable_if_t<(sizeof...(Indices) >= 1)> = 0>
    DRJIT_INLINE decltype(auto) entry(size_t i0, Indices... indices) {
        return derived().entry(i0).entry(indices...);
    }

    /// Recursive array indexing operator (const)
    template <typename... Indices, enable_if_t<(sizeof...(Indices) >= 1)> = 0>
    DRJIT_INLINE decltype(auto) entry(size_t i0, Indices... indices) const {
        return derived().entry(i0).entry(indices...);
    }

    /// Array indexing operator with bounds checks in debug mode
    DRJIT_INLINE decltype(auto) operator[](size_t i) {
        #if !defined(NDEBUG) && !defined(DRJIT_DISABLE_RANGE_CHECK)
        if (i >= derived().size())
            drjit_raise("ArrayBase: out of range access (tried to "
                        "access index %zu in an array of size %zu)",
                        i, derived().size());
        #endif
        return derived().entry(i);
    }

    /// Array indexing operator with bounds checks in debug mode, const version
    DRJIT_INLINE decltype(auto) operator[](size_t i) const {
        #if !defined(NDEBUG) && !defined(DRJIT_DISABLE_RANGE_CHECK)
        if (i >= derived().size())
            drjit_raise("ArrayBase: out of range access (tried to "
                        "access index %zu in an array of size %zu)",
                        i, derived().size());
        #endif
        return derived().entry(i);
    }

    template <typename T>
    DRJIT_INLINE void set_entry(size_t i, T &&value) {
        derived().entry(i) = std::forward<T>(value);
    }

    template <typename Mask, enable_if_mask_t<Mask> = 0>
    DRJIT_INLINE auto operator[](const Mask &m) {
        return detail::MaskedArray<Derived>{ derived(),
                                             (const mask_t<Derived> &) m };
    }

    DRJIT_INLINE bool empty() const { return derived().size() == 0; }

    const CallSupport operator->() const {
        return CallSupport(derived());
    }

    Derived& operator++() {
        derived() += 1;
        return derived();
    }

    Derived& operator--() {
        derived() -= 1;
        return derived();
    }

    Derived operator++(int) {
        Derived value = derived();
        derived() += 1;
        return value;
    }

    Derived operator--(int) {
        Derived value = derived();
        derived() -= 1;
        return value;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fallback implementations of vertical operations
    // -----------------------------------------------------------------------


    #define DRJIT_IMPLEMENT_UNARY(name, op, cond)                            \
        Derived name##_() const {                                            \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size();                                \
                                                                             \
               Derived result;                                               \
                if constexpr (Derived::Size == Dynamic)                      \
                    result = drjit::empty<Derived>(sa);                      \
                                                                             \
                for (size_t i = 0; i < sa; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    result.set_entry(i, op);                                 \
                }                                                            \
                                                                             \
                return result;                                               \
            } else {                                                         \
                drjit_raise(#name "_(): invalid operand type!");             \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_UNARY_REC(name, op, cond)                        \
        template <typename T = Value, enable_if_array_t<T> = 0>              \
        DRJIT_IMPLEMENT_UNARY(name, op, cond)

    #define DRJIT_IMPLEMENT_UNARY_TEMPLATE(name, arg, op, cond)              \
        template <arg> Derived name##_() const {                             \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size();                                \
                                                                             \
               Derived result;                                               \
                if constexpr (Derived::Size == Dynamic)                      \
                    result = drjit::empty<Derived>(sa);                      \
                                                                             \
                for (size_t i = 0; i < sa; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    result.set_entry(i, op);                                 \
                }                                                            \
                                                                             \
                return result;                                               \
            } else {                                                         \
                drjit_raise(#name "_(): invalid operand type!");             \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_UNARY_PAIR_REC(name, op, cond)                   \
        template <typename T = Value, enable_if_array_t<T> = 0>              \
        std::pair<Derived, Derived> name##_() const {                        \
            if constexpr (cond) {                                            \
                size_t sa = derived().size();                                \
                                                                             \
                Derived result_1, result_2;                                  \
                if constexpr (Derived::Size == Dynamic) {                    \
                    result_1 = drjit::empty<Derived>(sa);                    \
                    result_2 = drjit::empty<Derived>(sa);                    \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sa; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    auto result = op;                                        \
                    result_1.set_entry(i, std::move(result.first));          \
                    result_2.set_entry(i, std::move(result.second));         \
                }                                                            \
                                                                             \
                return std::pair<Derived, Derived>(std::move(result_1),      \
                                                   std::move(result_2));     \
            } else {                                                         \
                drjit_raise(#name "_(): invalid operand type!");             \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_ROUND2INT(name)                                  \
        template <typename T> T name##2int_() const {                        \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (!IsFloat) {                                        \
                drjit_raise(#name "_(): invalid operand type!");             \
            } else if constexpr (!std::is_scalar_v<Value>) {                 \
                size_t sa = derived().size();                                \
                                                                             \
                T result;                                                    \
                if constexpr (T::Size == Dynamic)                            \
                    result = drjit::empty<T>(sa);                            \
                                                                             \
                for (size_t i = 0; i < sa; ++i)                              \
                    result.set_entry(i,                                      \
                        name##2int<value_t<T>>(derived().entry(i)));         \
                                                                             \
                return result;                                               \
            } else {                                                         \
                return T(name(derived()));                                   \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_BINARY(name, op, cond)                           \
        Derived name##_(const Derived &v) const {                            \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v.size(),                 \
                       sr = sa > sb ? sa : sb;                               \
                                                                             \
                Derived result;                                              \
                if constexpr (Derived::Size == Dynamic) {                    \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))      \
                        drjit_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu and %zu)", sa, sb);                 \
                    result = drjit::empty<Derived>(sr);                      \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    const Value &b = v.entry(i);                             \
                    result.set_entry(i, op);                                 \
                }                                                            \
                                                                             \
                return result;                                               \
            } else {                                                         \
                drjit_raise(#name "_(): invalid operand type!");             \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_BINARY_REC(name, op, cond)                       \
        template <typename T = Value, enable_if_array_t<T> = 0>              \
        DRJIT_IMPLEMENT_BINARY(name, op, cond)


    #define DRJIT_IMPLEMENT_BINARY_BITOP(name, op, cond)                     \
        template <typename Mask> Derived name##_(const Mask &v) const {      \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v.size(),                 \
                       sr = sa > sb ? sa : sb;                               \
                                                                             \
                Derived result;                                              \
                if constexpr (Derived::Size == Dynamic) {                    \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))      \
                        drjit_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu and %zu)", sa, sb);                 \
                    result = drjit::empty<Derived>(sr);                      \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    const auto &b = v.entry(i);                              \
                    result.set_entry(i, op);                                 \
                }                                                            \
                                                                             \
                return result;                                               \
            } else {                                                         \
                drjit_raise(#name "_(): invalid operand type!");             \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_BINARY_MASK(name, op, cond)                      \
        DRJIT_INLINE auto name##_(const Derived &v) const {                  \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v.size(),                 \
                       sr = sa > sb ? sa : sb;                               \
                                                                             \
                mask_t<Derived> result;                                      \
                if constexpr (Derived::Size == Dynamic) {                    \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))      \
                        drjit_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu and %zu)", sa, sb);                 \
                    result = drjit::empty<mask_t<Derived>>(sr);              \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    const Value &b = v.entry(i);                             \
                    result.set_entry(i, op);                                 \
                }                                                            \
                                                                             \
                return result;                                               \
            } else {                                                         \
                drjit_raise(#name "_(): invalid operand type!");             \
            }                                                                \
        }

    #define DRJIT_IMPLEMENT_TERNARY_ALT(name, op, alt, cond)                 \
        Derived name##_(const Derived &v1, const Derived &v2) const {        \
            DRJIT_CHKSCALAR(#name "_");                                      \
                                                                             \
            if constexpr (!cond) {                                           \
                drjit_raise(#name "_(): invalid operand type!");             \
            } else if constexpr (!std::is_scalar_v<Value> &&                 \
                                 !is_special_v<Derived>) {                   \
                size_t sa = derived().size(), sb = v1.size(), sc = v2.size(),\
                       sd = sa > sb ? sa : sb, sr = sc > sd ? sc : sd;       \
                                                                             \
                Derived result;                                              \
                if constexpr (Derived::Size == Dynamic) {                    \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1) ||    \
                        (sc != sr && sc != 1))                               \
                        drjit_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu, %zu, and %zu)", sa, sb, sc);       \
                    result = drjit::empty<Derived>(sr);                      \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().entry(i);                     \
                    const Value &b = v1.entry(i);                            \
                    const Value &c = v2.entry(i);                            \
                    result.set_entry(i, op);                                 \
                }                                                            \
                                                                             \
                return result;                                               \
            } else {                                                         \
                return alt;                                                  \
            }                                                                \
        }

    DRJIT_IMPLEMENT_BINARY(add,   a + b,       IsArithmetic)
    DRJIT_IMPLEMENT_BINARY(sub,   a - b,       IsArithmetic)
    DRJIT_IMPLEMENT_BINARY(mul,   a * b,       IsArithmetic)
    DRJIT_IMPLEMENT_BINARY(mulhi, mulhi(a, b), IsIntegral)
    DRJIT_IMPLEMENT_BINARY(div,   a / b,       IsArithmetic)
    DRJIT_IMPLEMENT_BINARY(mod,   a % b,       IsIntegral)

    DRJIT_IMPLEMENT_BINARY_BITOP(or,     detail::or_(a, b),     true)
    DRJIT_IMPLEMENT_BINARY_BITOP(and,    detail::and_(a, b),    true)
    DRJIT_IMPLEMENT_BINARY_BITOP(andnot, detail::andnot_(a, b), true)
    DRJIT_IMPLEMENT_BINARY_BITOP(xor,    detail::xor_(a, b),    true)

    DRJIT_IMPLEMENT_BINARY(sl, a << b, IsIntegral)
    DRJIT_IMPLEMENT_BINARY(sr, a >> b, IsIntegral)

    DRJIT_IMPLEMENT_UNARY_TEMPLATE(sl, int Imm, a << Imm, IsIntegral)
    DRJIT_IMPLEMENT_UNARY_TEMPLATE(sr, int Imm, a >> Imm, IsIntegral)

    DRJIT_IMPLEMENT_BINARY_MASK(eq,  eq(a, b), true)
    DRJIT_IMPLEMENT_BINARY_MASK(neq, neq(a, b), true)
    DRJIT_IMPLEMENT_BINARY_MASK(lt, a < b,  IsArithmetic)
    DRJIT_IMPLEMENT_BINARY_MASK(le, a <= b, IsArithmetic)
    DRJIT_IMPLEMENT_BINARY_MASK(gt, a > b,  IsArithmetic)
    DRJIT_IMPLEMENT_BINARY_MASK(ge, a >= b, IsArithmetic)

    DRJIT_IMPLEMENT_UNARY(neg, detail::neg_(a), IsArithmetic)
    DRJIT_IMPLEMENT_UNARY(not, detail::not_(a), !IsFloat)

    DRJIT_IMPLEMENT_UNARY(sqrt,  sqrt(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY(abs,   abs(a), IsArithmetic)

    DRJIT_IMPLEMENT_UNARY(floor, floor(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY(ceil,  ceil(a),  IsFloat)
    DRJIT_IMPLEMENT_UNARY(trunc, trunc(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY(round, round(a), IsFloat)

    DRJIT_IMPLEMENT_ROUND2INT(trunc)
    DRJIT_IMPLEMENT_ROUND2INT(ceil)
    DRJIT_IMPLEMENT_ROUND2INT(floor)
    DRJIT_IMPLEMENT_ROUND2INT(round)

    DRJIT_IMPLEMENT_BINARY(minimum, minimum(a, b), IsArithmetic)
    DRJIT_IMPLEMENT_BINARY(maximum, maximum(a, b), IsArithmetic)

    DRJIT_IMPLEMENT_UNARY(rcp, rcp(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY(rsqrt, rsqrt(a), IsFloat)

    DRJIT_IMPLEMENT_TERNARY_ALT(fmadd,  fmadd(a, b, c),   derived()*v1+v2, IsArithmetic)
    DRJIT_IMPLEMENT_TERNARY_ALT(fmsub,  fmsub(a, b, c),   derived()*v1-v2, IsArithmetic)
    DRJIT_IMPLEMENT_TERNARY_ALT(fnmadd, fnmadd(a, b, c), -derived()*v1+v2, IsArithmetic)
    DRJIT_IMPLEMENT_TERNARY_ALT(fnmsub, fnmsub(a, b, c), -derived()*v1-v2, IsArithmetic)

    DRJIT_IMPLEMENT_UNARY_REC(cbrt, cbrt(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(erf, erf(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(sin, sin(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(cos, cos(a), IsFloat)

    DRJIT_IMPLEMENT_UNARY_REC(csc, csc(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(sec, sec(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(tan, tan(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(cot, cot(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(asin, asin(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(acos, acos(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(atan, atan(a), IsFloat)
    DRJIT_IMPLEMENT_BINARY_REC(atan2, atan2(a, b), IsFloat)
    DRJIT_IMPLEMENT_BINARY_REC(ldexp, ldexp(a, b), IsFloat)

    DRJIT_IMPLEMENT_UNARY_REC(exp2, exp2(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(exp, exp(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(log2, log2(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(log, log(a), IsFloat)

    DRJIT_IMPLEMENT_UNARY_REC(sinh, sinh(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(cosh, cosh(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(tanh, tanh(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(asinh, asinh(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(acosh, acosh(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_REC(atanh, atanh(a), IsFloat)

    DRJIT_IMPLEMENT_UNARY(tzcnt, tzcnt(a), IsIntegral)
    DRJIT_IMPLEMENT_UNARY(lzcnt, lzcnt(a), IsIntegral)
    DRJIT_IMPLEMENT_UNARY(popcnt, popcnt(a), IsIntegral)

    DRJIT_IMPLEMENT_UNARY_PAIR_REC(sincos, sincos(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_PAIR_REC(sincosh, sincosh(a), IsFloat)
    DRJIT_IMPLEMENT_UNARY_PAIR_REC(frexp, frexp(a), IsFloat)

    #undef DRJIT_IMPLEMENT_UNARY
    #undef DRJIT_IMPLEMENT_UNARY_REC
    #undef DRJIT_IMPLEMENT_UNARY_TEMPLATE
    #undef DRJIT_IMPLEMENT_UNARY_PAIR_REC
    #undef DRJIT_IMPLEMENT_ROUND2INT
    #undef DRJIT_IMPLEMENT_BINARY
    #undef DRJIT_IMPLEMENT_BINARY_REC
    #undef DRJIT_IMPLEMENT_BINARY_BITOP
    #undef DRJIT_IMPLEMENT_BINARY_MASK
    #undef DRJIT_IMPLEMENT_TERNARY_ALT

    template <typename Mask>
    static DRJIT_INLINE
        auto select_(const Mask &m, const Derived &t, const Derived &f) {
        DRJIT_CHKSCALAR("select_");
        size_t sm = m.size(), st = t.size(), sf = f.size(),
               sd = sm > st ? sm : st, sr = sf > sd ? sf : sd;
        Derived result;

        if constexpr (Derived::Size == Dynamic) {
            if ((sm != sr && sm != 1) || (st != sr && st != 1) ||
                (sf != sr && sf != 1))
                drjit_raise("select_() : mismatched input sizes "
                           "(%zu, %zu, and %zu)", sm, st, sf);
            result = drjit::empty<Derived>(sr);
        }

        for (size_t i = 0; i < sr; ++i) {
            const auto &v_m = m.entry(sm > 1 ? i : 0);
            const Value &v_t = t.entry(st > 1 ? i : 0);
            const Value &v_f = f.entry(sf > 1 ? i : 0);
            result.entry(i) = select(v_m, v_t, v_f);
        }

        return result;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fallback implementations of horizontal operations
    // -----------------------------------------------------------------------

    template <size_t... Indices> DRJIT_INLINE Derived shuffle_() const {
        static_assert(sizeof...(Indices) == Derived::Size ||
                      sizeof...(Indices) == Derived::ActualSize, "shuffle(): Invalid size!");
        DRJIT_CHKSCALAR("shuffle_");
        size_t idx = 0; (void) idx;
        Derived out;
        ((out.entry(idx++) = derived().entry(Indices % Derived::Size)), ...);
        return out;
    }

    Value dot_(const Derived &a) const {
        Value result;
        if constexpr (IsArithmetic) {
			if constexpr (is_array_v<Value>) {
                size_t sa = derived().size(), sb = a.size(),
                       sr = sa > sb ? sa : sb;

                if constexpr (Derived::Size == Dynamic) {
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))
                        drjit_raise("dot_() : mismatched input sizes "
                                    "(%zu and %zu)", sa, sb);
                    else if (sr == 0)
                        return Value(0);
                }

				result = derived().entry(0) * a.entry(0);
				if constexpr (std::is_floating_point_v<Scalar>) {
                    for (size_t i = 1; i < sr; ++i)
                        result = fmadd(derived().entry(i), a.entry(i), result);
                } else {
                    for (size_t i = 1; i < sr; ++i)
                        result += derived().entry(i) * a.entry(i);
                }
            } else {
				result = slice<Value>(sum(derived() * a));
			}
		}
        return result;
    }

    Value sum_() const {
        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    return Value(0);
            }

            Value value = derived().entry(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value += derived().entry(i);
            return value;
        } else {
            drjit_raise("sum_(): invalid operand type!");
        }
    }

    Value prod_() const {
        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    return Value(1);
            }

            Value value = derived().entry(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value *= derived().entry(i);
            return value;
        } else {
            drjit_raise("prod_(): invalid operand type!");
        }
    }

    Value min_() const {
        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    drjit_raise("min_(): zero-sized array!");
            }

            Value value = derived().entry(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = minimum(value, derived().entry(i));
            return value;
        } else {
            drjit_raise("min_(): invalid operand type!");
        }
    }

    Value max_() const {
        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    drjit_raise("max_(): zero-sized array!");
            }

            Value value = derived().entry(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = maximum(value, derived().entry(i));
            return value;
        } else {
            drjit_raise("max_(): invalid operand type!");
        }
    }

    mask_t<Value> all_() const {
        if constexpr (IsMask) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    return true;
            }

            mask_t<Value> value = derived().entry(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = value && derived().entry(i);
            return value;
        } else {
            drjit_raise("all_(): invalid operand type!");
        }
    }

    mask_t<Value> any_() const {
        if constexpr (IsMask) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    return false;
            }

            mask_t<Value> value = derived().entry(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = value || derived().entry(i);
            return value;
        } else {
            drjit_raise("any_(): invalid operand type!");
        }
    }

    uint32_array_t<array_t<Value>> count_() const {
        if constexpr (IsMask) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    return 0;
            }
            uint32_array_t<array_t<Value>> value =
                select(derived().entry(0), 1, 0);
            for (size_t i = 1; i < derived().size(); ++i)
                value += select(derived().entry(i), 1, 0);
            return value;
        } else {
            drjit_raise("count_(): invalid operand type!");
        }
    }

    template <typename Mask, enable_if_t<Mask::Depth == 1> = 0>
    DRJIT_INLINE Value extract_(const Mask &mask) const {
        size_t sa = derived().size(), sb = mask.size(),
               sr = sa > sb ? sa : sb;

        for (size_t i = 0; i < sr; ++i) {
            bool m = mask.entry(i);
            if (m)
                return derived().entry(i);
        }

        return zeros<Value>();
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fallback implementations of scatter/gather/load/store ops
    // -----------------------------------------------------------------------

    template <bool Permute, typename Source, typename Index, typename Mask>
    static Derived gather_(Source &&source, const Index &index, const Mask &mask) {
        DRJIT_CHKSCALAR("gather_");
        Derived result;

        size_t sa = index.size(), sb = mask.size(),
               sr = sa > sb ? sa : sb;

        if constexpr (Derived::Size == Dynamic) {
            if ((sa != sr && sa != 1) || (sb != sr && sb != 1))
                drjit_raise("gather_() : mismatched input sizes "
                            "(%zu and %zu)", sa, sb);
            result = drjit::empty<Derived>(sr);
        }

        for (size_t i = 0; i < sr; ++i)
            result.entry(i) = gather<Value, Permute>(
                source, index.entry(i),
                mask.entry(i));

        return result;
    }

    template <bool Permute, typename Target, typename Index, typename Mask>
    void scatter_(Target &&target, const Index &index, const Mask &mask) const {
        DRJIT_CHKSCALAR("scatter_");

        size_t sa = derived().size(), sb = index.size(), sc = mask.size(),
               sd = sa > sb ? sa : sb, sr = sc > sd ? sc : sd;

        for (size_t i = 0; i < sr; ++i)
            scatter<Permute>(target, derived().entry(i),
                             index.entry(i),
                             mask.entry(i));
    }

    template <typename Target, typename Index, typename Mask>
    void scatter_reduce_(ReduceOp op,
                         Target &&target,
                         const Index &index,
                         const Mask &mask) const {
        DRJIT_CHKSCALAR("scatter_reduce_");

        size_t sa = derived().size(), sb = index.size(), sc = mask.size(),
               sd = sa > sb ? sa : sb, sr = sc > sd ? sc : sd;

        for (size_t i = 0; i < sr; ++i)
            scatter_reduce(op, target, derived().entry(i), index.entry(i),
                           mask.entry(i));
    }

    static Derived load_aligned_(const void *mem, size_t size) {
        return Derived::load_(mem, size);
    }

    void store_aligned_(void *mem) const {
        return derived().store_(mem);
    }

    template <typename T> void migrate_(T type) {
        if constexpr (is_jit_v<Value_>) {
            for (size_t i = 0; i < derived().size(); ++i)
                derived().entry(i).migrate_(type);
        }
    }

    //! @}
    // -----------------------------------------------------------------------
};

NAMESPACE_END(drjit)
