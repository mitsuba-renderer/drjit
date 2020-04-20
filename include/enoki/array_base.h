/*
    enoki/array_base.h -- Base class of all Enoki arrays

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array_router.h>
#include <enoki/array_constants.h>

NAMESPACE_BEGIN(enoki)

#define ENOKI_ARRAY_DEFAULTS(Name)                                             \
    Name() = default;                                                          \
    Name(const Name &) = default;                                              \
    Name(Name &&) = default;                                                   \
    Name &operator=(const Name &) = default;                                   \
    Name &operator=(Name &&) = default;

#define ENOKI_ARRAY_IMPORT(Name, Base)                                         \
    ENOKI_ARRAY_DEFAULTS(Name)                                                 \
    using Base::Base;                                                          \
    using Base::operator=;

/// Generic array base class
struct ArrayBase {
    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations (may be overridden in subclasses)
    // -----------------------------------------------------------------------

    /// Is this an Enoki array?
    static constexpr bool IsEnoki = true;

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

    /// Does this array represent the result of a 'masked(...)' epxpression?
    static constexpr bool IsMaskedArray = false;

    //! @}
    // -----------------------------------------------------------------------
};

/// Array base class templated via the curiously recurring template pattern
template <typename Value_, typename Derived_> struct ArrayBaseT : ArrayBase {
    using Base = ArrayBase;

    // -----------------------------------------------------------------------
    //! @{ \name Basic declarations (may be overridden in subclasses)
    // -----------------------------------------------------------------------

    /// Type underlying the array
    using Value = Value_;

    /// Scalar data type all the way at the lowest level
    using Scalar = scalar_t<Value_>;

    /// Specifies how deeply nested this array is
    static constexpr size_t Depth = 1 + array_depth_v<Value>;

    /// Is this a mask array?
    static constexpr bool IsMask = is_mask_v<Value_>;

    /// Is this an array of values that can be added, multiplied, etc.?
    static constexpr bool IsArithmetic = std::is_arithmetic_v<Scalar>;

    /// Is this an array of signed or unsigned integer values?
    static constexpr bool IsIntegral = std::is_integral_v<Scalar>;

    /// Is this an array of floating point values?
    static constexpr bool IsFloat = std::is_floating_point_v<Scalar>;

    /// Does this array compute derivatives using automatic differentation?
    static constexpr bool IsDiff = is_diff_array_v<Value_>;

    /// Are elements of this array implemented using the LLVM backend?
    static constexpr bool IsLLVM = is_llvm_array_v<Value_>;

    /// Are elements of this array implemented using the CUDA backend?
    static constexpr bool IsCUDA = is_cuda_array_v<Value_>;

    /// Are elements of this array implemented using a JIT-compiled backend?
    static constexpr bool IsJIT = IsLLVM || IsCUDA;

    /// Are elements of this array dynamic?
    static constexpr bool IsDynamic = is_dynamic_array_v<Value_>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Curiously Recurring Template design pattern
    // -----------------------------------------------------------------------

    /// Alias to the derived type
    using Derived = Derived_;

    /// Cast to derived type
    ENOKI_INLINE Derived &derived()             { return (Derived &) *this; }

    /// Cast to derived type (const version)
    ENOKI_INLINE const Derived &derived() const { return (Derived &) *this; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors
    // -----------------------------------------------------------------------

    ENOKI_ARRAY_IMPORT(ArrayBaseT, ArrayBase)

    template <typename Value2, typename Derived2>
    ArrayBaseT(const ArrayBaseT<Value2, Derived2> &v) {
        ENOKI_CHKSCALAR("Copy constructor (conversion)");
        derived().init_(v.derived().size());
        for (size_t i = 0; i < derived().size(); ++i)
            derived().coeff(i) = (Value) v.derived().coeff(i);
    }

    template <typename Value2, typename Derived2>
    ArrayBaseT(const ArrayBaseT<Value2, Derived2> &v, detail::reinterpret_flag) {
        ENOKI_CHKSCALAR("Copy constructor (reinterpret_cast)");
        derived().init_(v.derived().size());
        for (size_t i = 0; i < derived().size(); ++i)
            derived().coeff(i) = reinterpret_array<Value>(v[i]);
    }

    ArrayBaseT(const Value &v) {
        ENOKI_CHKSCALAR("Constructor (scalar broadcast)");
        derived().init_(1);
        for (size_t i = 0; i < derived().size(); ++i)
            derived().coeff(i) = v;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Iterators
    // -----------------------------------------------------------------------

    ENOKI_INLINE auto begin() const { return derived().data(); }
    ENOKI_INLINE auto begin()       { return derived().data(); }
    ENOKI_INLINE auto end()   const { return derived().data() + derived().size(); }
    ENOKI_INLINE auto end()         { return derived().data() + derived().size(); }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Element access
    // -----------------------------------------------------------------------

    /// Recursive array indexing operator
    template <typename... Indices, enable_if_t<(sizeof...(Indices) >= 1)> = 0>
    ENOKI_INLINE decltype(auto) coeff(size_t i0, Indices... indices) {
        return derived().coeff(i0).coeff(indices...);
    }

    /// Recursive array indexing operator (const)
    template <typename... Indices, enable_if_t<(sizeof...(Indices) >= 1)> = 0>
    ENOKI_INLINE decltype(auto) coeff(size_t i0, Indices... indices) const {
        return derived().coeff(i0).coeff(indices...);
    }

    /// Array indexing operator with bounds checks in debug mode
    ENOKI_INLINE decltype(auto) operator[](size_t i) {
        #if !defined(NDEBUG) && !defined(ENOKI_DISABLE_RANGE_CHECK)
        if (i >= derived().size())
            enoki_raise("ArrayBase: out of range access (tried to "
                        "access index %zu in an array of size %zu)",
                        i, derived().size());
#endif
        return derived().coeff(i);
    }

    /// Array indexing operator with bounds checks in debug mode, const version
    ENOKI_INLINE decltype(auto) operator[](size_t i) const {
        #if !defined(NDEBUG) && !defined(ENOKI_DISABLE_RANGE_CHECK)
        if (i >= derived().size())
            enoki_raise("ArrayBase: out of range access (tried to "
                        "access index %zu in an array of size %zu)",
                        i, derived().size());
        #endif
        return derived().coeff(i);
    }

    // template <typename Mask, enable_if_mask_t<Mask> = 0>
    // ENOKI_INLINE auto operator[](const Mask &m) {
    //     return detail::MaskedArray<Derived>{ derived(), (const mask_t<Derived> &) m };
    // }

    ENOKI_INLINE bool empty() const { return derived().size() == 0; }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fallback implementations of vertical operations
    // -----------------------------------------------------------------------

    #define ENOKI_IMPLEMENT_UNARY(name, op, cond)                            \
        Derived name##_() const {                                            \
            ENOKI_CHKSCALAR(#name "_");                                      \
            Derived result;                                                  \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size();                                \
                                                                             \
                if constexpr (Derived::Size == Dynamic)                      \
                    result = enoki::empty<Derived>(sa);                      \
                                                                             \
                for (size_t i = 0; i < sa; ++i) {                            \
                    const Value &a = derived().coeff(i);                     \
                    result.coeff(i) = op;                                    \
                }                                                            \
            } else {                                                         \
                enoki_raise(#name "_(): unsupported operation!");            \
            }                                                                \
                                                                             \
            return result;                                                   \
        }

    #define ENOKI_IMPLEMENT_UNARY_TEMPLATE(name, arg, op, cond)              \
        template <arg> Derived name##_() const {                             \
            ENOKI_CHKSCALAR(#name "_");                                      \
            Derived result;                                                  \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size();                                \
                                                                             \
                if constexpr (Derived::Size == Dynamic)                      \
                    result = enoki::empty<Derived>(sa);                      \
                                                                             \
                for (size_t i = 0; i < sa; ++i) {                            \
                    const Value &a = derived().coeff(i);                     \
                    result.coeff(i) = op;                                    \
                }                                                            \
            } else {                                                         \
                enoki_raise(#name "_(): unsupported operation!");            \
            }                                                                \
                                                                             \
            return result;                                                   \
        }

    #define ENOKI_IMPLEMENT_BINARY(name, op, cond)                           \
        Derived name##_(const Derived &v) const {                            \
            ENOKI_CHKSCALAR(#name "_");                                      \
            Derived result;                                                  \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v.size(),                 \
                       sr = sa > sb ? sa : sb;                               \
                                                                             \
                if constexpr (Derived::Size == Dynamic) {                    \
                    result = enoki::empty<Derived>(sr);                      \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))      \
                        enoki_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu and %zu)", sa, sb);                 \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().coeff(sa > 1 ? i : 0);        \
                    const Value &b = v.coeff(sb > 1 ? i : 0);                \
                    result.coeff(i) = op;                                    \
                }                                                            \
            } else {                                                         \
                enoki_raise(#name "_(): unsupported operation!");            \
            }                                                                \
                                                                             \
            return result;                                                   \
        }

    #define ENOKI_IMPLEMENT_BINARY_BITOP(name, op, cond)                     \
        template <typename Mask> Derived name##_(const Mask &v) const {      \
            ENOKI_CHKSCALAR(#name "_");                                      \
            Derived result;                                                  \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v.size(),                 \
                       sr = sa > sb ? sa : sb;                               \
                                                                             \
                if constexpr (Derived::Size == Dynamic) {                    \
                    result = enoki::empty<Derived>(sr);                      \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))      \
                        enoki_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu and %zu)", sa, sb);                 \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().coeff(sa > 1 ? i : 0);        \
                    const auto &b = v.coeff(sb > 1 ? i : 0);                 \
                    result.coeff(i) = op;                                    \
                }                                                            \
            } else {                                                         \
                enoki_raise(#name "_(): unsupported operation!");            \
            }                                                                \
                                                                             \
            return result;                                                   \
        }

    #define ENOKI_IMPLEMENT_BINARY_MASK(name, op, cond)                      \
        ENOKI_INLINE auto name##_(const Derived &v) const {                  \
            ENOKI_CHKSCALAR(#name "_");                                      \
            mask_t<Derived> result;                                          \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v.size(),                 \
                       sr = sa > sb ? sa : sb;                               \
                                                                             \
                if constexpr (Derived::Size == Dynamic) {                    \
                    result = enoki::empty<mask_t<Derived>>(sr);              \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1))      \
                        enoki_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu and %zu)", sa, sb);                 \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().coeff(sa > 1 ? i : 0);        \
                    const Value &b = v.coeff(sb > 1 ? i : 0);                \
                    result.coeff(i) = op;                                    \
                }                                                            \
            } else {                                                         \
                enoki_raise(#name "_(): unsupported operation!");            \
            }                                                                \
                                                                             \
            return result;                                                   \
        }

    #define ENOKI_IMPLEMENT_TERNARY(name, op, cond)                          \
        Derived name##_(const Derived &v1, const Derived &v2) const {        \
            ENOKI_CHKSCALAR(#name "_");                                      \
            Derived result;                                                  \
                                                                             \
            if constexpr (cond) {                                            \
                size_t sa = derived().size(), sb = v1.size(), sc = v2.size(),\
                       sr = sa > sb ? sa : (sb > sc ? sb : sc);              \
                                                                             \
                if constexpr (Derived::Size == Dynamic) {                    \
                    result = enoki::empty<Derived>(sr);                      \
                    if ((sa != sr && sa != 1) || (sb != sr && sb != 1) ||    \
                        (sc != sr && sc != 1))                               \
                        enoki_raise(#name "_() : mismatched input sizes "    \
                                   "(%zu, %zu, and %zu)", sa, sb, sc);       \
                }                                                            \
                                                                             \
                for (size_t i = 0; i < sr; ++i) {                            \
                    const Value &a = derived().coeff(sa > 1 ? i : 0);        \
                    const Value &b = v1.coeff(sb > 1 ? i : 0);               \
                    const Value &c = v2.coeff(sc > 1 ? i : 0);               \
                    result.coeff(i) = op;                                    \
                }                                                            \
            } else {                                                         \
                enoki_raise(#name "_(): unsupported operation!");            \
            }                                                                \
                                                                             \
            return result;                                                   \
        }


    ENOKI_IMPLEMENT_BINARY(add,   a + b,       IsArithmetic)
    ENOKI_IMPLEMENT_BINARY(sub,   a - b,       IsArithmetic)
    ENOKI_IMPLEMENT_BINARY(mul,   a * b,       IsArithmetic)
    ENOKI_IMPLEMENT_BINARY(mulhi, mulhi(a, b), IsIntegral)
    ENOKI_IMPLEMENT_BINARY(div,   a / b,       IsArithmetic)
    ENOKI_IMPLEMENT_BINARY(mod,   a % b,       IsIntegral)

    ENOKI_IMPLEMENT_BINARY_BITOP(or,     detail::or_(a, b),     true)
    ENOKI_IMPLEMENT_BINARY_BITOP(and,    detail::and_(a, b),    true)
    ENOKI_IMPLEMENT_BINARY_BITOP(andnot, detail::andnot_(a, b), true)
    ENOKI_IMPLEMENT_BINARY_BITOP(xor,    detail::xor_(a, b),    true)

    ENOKI_IMPLEMENT_BINARY(eq,  enoki::eq(a, b), true)
    ENOKI_IMPLEMENT_BINARY(neq, enoki::neq(a, b), true)

    ENOKI_IMPLEMENT_BINARY(sl, a << b, IsIntegral)
    ENOKI_IMPLEMENT_BINARY(sr, a >> b, IsIntegral)

    ENOKI_IMPLEMENT_UNARY_TEMPLATE(sl, int Imm, a << Imm, IsIntegral)
    ENOKI_IMPLEMENT_UNARY_TEMPLATE(sr, int Imm, a >> Imm, IsIntegral)

    ENOKI_IMPLEMENT_BINARY_MASK(lt, a < b,  IsArithmetic)
    ENOKI_IMPLEMENT_BINARY_MASK(le, a <= b, IsArithmetic)
    ENOKI_IMPLEMENT_BINARY_MASK(gt, a > b,  IsArithmetic)
    ENOKI_IMPLEMENT_BINARY_MASK(ge, a >= b, IsArithmetic)

    ENOKI_IMPLEMENT_UNARY(neg, -a, IsArithmetic)
    ENOKI_IMPLEMENT_UNARY(not, detail::not_(a), !IsFloat)

    ENOKI_IMPLEMENT_UNARY(sqrt,  enoki::sqrt(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY(abs,   enoki::abs(a), IsArithmetic)

    ENOKI_IMPLEMENT_UNARY(floor, enoki::floor(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY(ceil,  enoki::ceil(a),  IsFloat)
    ENOKI_IMPLEMENT_UNARY(trunc, enoki::trunc(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY(round, enoki::round(a), IsFloat)

    ENOKI_IMPLEMENT_UNARY_TEMPLATE(floor2int, typename T, enoki::floor2int<value_t<T>>(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY_TEMPLATE(ceil2int,  typename T, enoki::ceil2int<value_t<T>>(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY_TEMPLATE(trunc2int, typename T, enoki::trunc2int<value_t<T>>(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY_TEMPLATE(round2int, typename T, enoki::round2int<value_t<T>>(a), IsFloat)

    ENOKI_IMPLEMENT_BINARY(min, enoki::min(a, b), IsArithmetic)
    ENOKI_IMPLEMENT_BINARY(max, enoki::max(a, b), IsArithmetic)

    ENOKI_IMPLEMENT_UNARY(rcp, enoki::rcp(a), IsFloat)
    ENOKI_IMPLEMENT_UNARY(rsqrt, enoki::rsqrt(a), IsFloat)

    ENOKI_IMPLEMENT_TERNARY(fmadd, enoki::fmadd(a, b, c), IsFloat)
    ENOKI_IMPLEMENT_TERNARY(fmsub, enoki::fmsub(a, b, c), IsFloat)
    ENOKI_IMPLEMENT_TERNARY(fnmadd, enoki::fnmadd(a, b, c), IsFloat)
    ENOKI_IMPLEMENT_TERNARY(fnmsub, enoki::fnmsub(a, b, c), IsFloat)

    #undef ENOKI_IMPLEMENT_UNARY
    #undef ENOKI_IMPLEMENT_UNARY_TEMPLATE
    #undef ENOKI_IMPLEMENT_BINARY
    #undef ENOKI_IMPLEMENT_BINARY_BITOP
    #undef ENOKI_IMPLEMENT_BINARY_MASK
    #undef ENOKI_IMPLEMENT_TERNARY

    template <typename Mask>
    static ENOKI_INLINE auto select_(const Mask &m, const Derived &t, const Derived &f) {
        ENOKI_CHKSCALAR("select_");
        size_t sm = m.size(), st = t.size(), sf = f.size(),
               sr = sm > st ? sm : (st > sf ? st : sf);
        Derived result;

        if constexpr (Derived::Size == Dynamic) {
            result = enoki::empty<Derived>(sr);
            if ((sm != sr && sm != 1) || (st != sr && st != 1) ||
                (sf != sr && sf != 1))
                enoki_raise("select_() : mismatched input sizes "
                           "(%zu, %zu, and %zu)", sm, st, sf);
        }

        for (size_t i = 0; i < sr; ++i) {
            const Value &v_m = m.coeff(sm > 1 ? i : 0);
            const Value &v_t = t.coeff(st > 1 ? i : 0);
            const Value &v_f = f.coeff(sf > 1 ? i : 0);
            result.coeff(i) = enoki::select(v_m, v_t, v_f);
        }

        return result;
    }

    template <size_t... Indices> ENOKI_INLINE Derived shuffle_() const {
        static_assert(sizeof...(Indices) == Derived::Size, "shuffle(): Invalid size!");
        ENOKI_CHKSCALAR("shuffle_");
        Derived out;
        size_t idx = 0;
        bool result[] = { (out.coeff(idx++) = derived().coeff(Indices % Derived::Size), false)... };
        (void) idx; (void) result;
        return out;
    }

    Value hsum_() const {
        Value value;

        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("hsum_(): zero-sized array!");
            }

            value = derived().coeff(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value += derived().coeff(i);
        } else {
            enoki_raise("hsum_(): unsupported operation!");
        }

        return value;
    }

    Value hprod_() const {
        Value value;

        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("hprod_(): zero-sized array!");
            }

            value = derived().coeff(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value *= derived().coeff(i);
        } else {
            enoki_raise("hprod_(): unsupported operation!");
        }

        return value;
    }

    Value hmin_() const {
        Value value;

        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("hmin_(): zero-sized array!");
            }

            value = derived().coeff(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = enoki::min(value, derived().coeff(i));
        } else {
            enoki_raise("hmin_(): unsupported operation!");
        }

        return value;
    }

    Value hmax_() const {
        Value value;

        if constexpr (IsArithmetic) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("hmax_(): zero-sized array!");
            }

            value = derived().coeff(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = enoki::max(value, derived().coeff(i));
        } else {
            enoki_raise("hmax_(): unsupported operation!");
        }

        return value;
    }

    mask_t<Value> all_() const {
        mask_t<Value> value;

        if constexpr (IsMask) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("all_(): zero-sized array!");
            }

            value = derived().coeff(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = value && derived().coeff(i);
        } else {
            enoki_raise("all_(): unsupported operation!");
        }

        return value;
    }

    mask_t<Value> any_() const {
        mask_t<Value> value;

        if constexpr (IsMask) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("any_(): zero-sized array!");
            }

            value = derived().coeff(0);
            for (size_t i = 1; i < derived().size(); ++i)
                value = value || derived().coeff(i);
        } else {
            enoki_raise("any_(): unsupported operation!");
        }

        return value;
    }

    uint32_array_t<Value> count_() const {
        uint32_array_t<Value> value;

        if constexpr (IsMask) {
            if constexpr (Derived::Size == Dynamic) {
                if (empty())
                    enoki_raise("count_(): zero-sized array!");
            }
            value = select(derived().coeff(0), 1, 0);
            for (size_t i = 1; i < derived().size(); ++i)
                value += select(derived().coeff(i), 1, 0);
        } else {
            enoki_raise("count_(): unsupported operation!");
        }

        return value;
    }
};

NAMESPACE_END(enoki)
