/*
    enoki/cuda.h -- CUDA-backed Enoki dynamic array with JIT compilation

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>
#include <enoki-jit/jit.h>
#include <enoki-jit/traits.h>

NAMESPACE_BEGIN(enoki)

template <typename Value_>
struct CUDAArray : ArrayBaseT<Value_, is_mask_v<Value_>, CUDAArray<Value_>> {
    static_assert(std::is_scalar_v<Value_>,
                  "CUDA Arrays can only be created over scalar types!");

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Value = Value_;
    using MaskType = CUDAArray<bool>;
    using ArrayType = CUDAArray;

    static constexpr bool IsCUDA = true;
    static constexpr bool IsJIT = true;
    static constexpr VarType Type = var_type_v<Value>;
    static constexpr size_t Size = Dynamic;

    template <typename T> using ReplaceValue = CUDAArray<T>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    CUDAArray() = default;

    ~CUDAArray() { jitc_var_dec_ref_ext(m_index); }

    CUDAArray(const CUDAArray &a) : m_index(a.m_index) {
        jitc_var_inc_ref_ext(m_index);
    }

    CUDAArray(CUDAArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T> CUDAArray(const CUDAArray<T> &v) {
        static_assert(!std::is_same_v<T, Value>,
                      "Conversion constructor called with arguments that don't "
                      "correspond to a conversion!");
        static_assert(!std::is_same_v<T, bool>, "Conversion from mask not permitted.");
        const char *op;

        if constexpr (std::is_floating_point_v<Value> &&
                      std::is_floating_point_v<T> && sizeof(Value) > sizeof(T))
            op = "cvt.$t0.$t1 $r0, $r1";
        else if constexpr (std::is_floating_point_v<Value>)
            op = "cvt.rn.$t0.$t1 $r0, $r1";
        else if constexpr (std::is_floating_point_v<T> && std::is_integral_v<Value>)
            op = "cvt.rzi.$t0.$t1 $r0, $r1";
        else
            op = "cvt.$t0.$t1 $r0, $r1";

        m_index = jitc_var_new_1(Type, op, 1, 1, v.index());
    }

    template <typename T> CUDAArray(const CUDAArray<T> &v, detail::reinterpret_flag) {
        static_assert(
            sizeof(T) == sizeof(Value),
            "reinterpret_array requires arrays with equal-sized element types!");

        if constexpr (std::is_integral_v<Value> != std::is_integral_v<T>) {
            m_index = jitc_var_new_1(Type, "mov.$b0 $r0, $r1", 1, 1, v.index());
        } else {
            m_index = v.index();
            jitc_var_inc_ref_ext(m_index);
        }
    }

    CUDAArray(Value value) {
        m_index = mkfull_(value, 1);
    }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              (!std::is_same_v<Ts, detail::reinterpret_flag> && ...))> = 0>
    CUDAArray(Ts&&... ts) {
        Value data[] = { (Value) ts... };
        m_index = jitc_var_copy(AllocType::Host, Type, 1, data,
                                (uint32_t) sizeof...(Ts));
    }

    CUDAArray &operator=(const CUDAArray &a) {
        jitc_var_inc_ref_ext(a.m_index);
        jitc_var_dec_ref_ext(m_index);
        m_index = a.m_index;
        return *this;
    }

    CUDAArray &operator=(CUDAArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Vertical operations
    // -----------------------------------------------------------------------

    CUDAArray add_(const CUDAArray &v) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        // Simple constant propagation
        if (is_literal_zero())
            return v;
        else if (v.is_literal_zero())
            return *this;

        const char *op = std::is_same_v<Value, float>
                             ? "add.ftz.$t0 $r0, $r1, $r2"
                             : "add.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray sub_(const CUDAArray &v) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_same_v<Value, float>
                             ? "sub.ftz.$t0 $r0, $r1, $r2"
                             : "sub.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray mul_(const CUDAArray &v) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        // Simple constant propagation
        if (is_literal_one())
            return v;
        else if (v.is_literal_one())
            return *this;

        const char *op;
        if constexpr (std::is_floating_point_v<Value>)
            op = std::is_same_v<Value, float> ? "mul.ftz.$t0 $r0, $r1, $r2"
                                              : "mul.$t0 $r0, $r1, $r2";
        else
            op = "mul.lo.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray div_(const CUDAArray &v) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        // Simple constant propagation
        if (v.is_literal_one())
            return *this;

        const char *op;
        if constexpr (std::is_same_v<Value, float>)
            op = "div.rn.ftz.$t0 $r0, $r1, $r2";
        else if constexpr (std::is_same_v<Value, double>)
            op = "div.rn.$t0 $r0, $r1, $r2";
        else
            op = "div.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.m_index));
    }

    CUDAArray mod_(const CUDAArray &v) const {
        if constexpr (!jitc_is_integral(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_2(Type, "rem.$t0 $r0, $r1, $r2", 1, 1,
                                         m_index, v.m_index));
    }

    CUDAArray<bool> gt_(const CUDAArray &a) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_signed_v<Value>
                             ? "setp.gt.$t1 $r0, $r1, $r2"
                             : "setp.hi.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray<bool> ge_(const CUDAArray &a) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_signed_v<Value>
                             ? "setp.ge.$t1 $r0, $r1, $r2"
                             : "setp.hs.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }


    CUDAArray<bool> lt_(const CUDAArray &a) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_signed_v<Value>
                             ? "setp.lt.$t1 $r0, $r1, $r2"
                             : "setp.lo.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray<bool> le_(const CUDAArray &a) const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_signed_v<Value>
                             ? "setp.le.$t1 $r0, $r1, $r2"
                             : "setp.lo.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray<bool> eq_(const CUDAArray &b) const {
        const char *op = !std::is_same_v<Value, bool>
            ? "setp.eq.$t1 $r0, $r1, $r2" :
              "xor.$t1 $r0, $r1, $r2$n"
              "not.$t1 $r0, $r0";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, b.index()));
    }

    CUDAArray<bool> neq_(const CUDAArray &b) const {
        const char *op = !std::is_same_v<Value, bool>
            ? "setp.ne.$t1 $r0, $r1, $r2" :
              "xor.$t1 $r0, $r1, $r2";

        return CUDAArray<bool>::from_index(jitc_var_new_2(
            CUDAArray<bool>::Type, op, 1, 1, m_index, b.index()));
    }

    CUDAArray neg_() const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_same_v<Value, float>
                             ? "neg.ftz.$t0 $r0, $r1"
                             : "neg.$t0 $r0, $r1";

        return from_index(jitc_var_new_1(Type, op, 1, 1, m_index));
    }

    CUDAArray not_() const {
        return from_index(jitc_var_new_1(Type, "not.$b0 $r0, $r1", 1, 1, m_index));
    }

    template <typename T> CUDAArray or_(const T &a) const {
        if constexpr (std::is_same_v<T, CUDAArray>) {
            // Simple constant propagation
            if constexpr (std::is_same_v<Value, bool>) {
                if (is_literal_one() || a.is_literal_zero())
                    return *this;
                else if (a.is_literal_one() || is_literal_zero())
                    return a;
            }

            return from_index(jitc_var_new_2(Type, "or.$b0 $r0, $r1, $r2", 1, 1,
                                             m_index, a.index()));
        } else {
            // Simple constant propagation
            if (a.is_literal_zero())
                return *this;
            else if (a.is_literal_one())
                return CUDAArray(memcpy_cast<Value>(int_array_t<Value>(-1)));

            return from_index(jitc_var_new_2(Type, "selp.$b0 $r0, -1, $r1, $r2", 1, 1,
                                             m_index, a.index()));
        }
    }

    template <typename T> CUDAArray and_(const T &a) const {
        if constexpr (std::is_same_v<T, CUDAArray>) {
            // Simple constant propagation
            if constexpr (std::is_same_v<Value, bool>) {
                if (is_literal_one() || a.is_literal_zero())
                    return a;
                else if (a.is_literal_one() || is_literal_zero())
                    return *this;
            }

            return from_index(jitc_var_new_2(Type, "and.$b0 $r0, $r1, $r2", 1, 1,
                                             m_index, a.index()));
        } else {
            // Simple constant propagation
            if (a.is_literal_one())
                return *this;
            else if (a.is_literal_zero())
                return CUDAArray(Value(0));

            return from_index(jitc_var_new_2(Type, "selp.$b0 $r0, $r1, 0, $r2", 1, 1,
                                             m_index, a.index()));
        }
    }

    template <typename T> CUDAArray xor_(const T &a) const {
        if constexpr (std::is_same_v<T, CUDAArray>) {
            // Simple constant propagation
            if constexpr (std::is_same_v<Value, bool>) {
                if (is_literal_zero())
                    return a;
                else if (a.is_literal_zero())
                    return *this;
            }

            return from_index(jitc_var_new_2(Type, "xor.$b0 $r0, $r1, $r2", 1, 1,
                                             m_index, a.index()));
        } else {
            // Simple constant propagation
            if (a.is_literal_zero())
                return *this;

            return select(a, ~*this, *this);
        }
    }

    template <typename T> CUDAArray andnot_(const T &a) const {
        return and_(a.not_());
    }

    template <int Imm> CUDAArray sl_() const {
        return sl_((uint32_t) Imm);
    }

    CUDAArray sl_(const CUDAArray<uint32_t> &v) const {
        if constexpr (!jitc_is_integral(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_2(
            Type, "shl.$b0 $r0, $r1, $r2", 1, 1, m_index, v.index()));
    }

    template <int Imm> CUDAArray sr_() const {
        return sr_((uint32_t) Imm);
    }

    CUDAArray sr_(const CUDAArray<uint32_t> &v) const {
        if constexpr (!jitc_is_integral(Type))
            enoki_raise("Unsupported operand type");

        const char *op;
        if constexpr (std::is_signed<Value>::value)
            op = "shr.$t0 $r0, $r1, $r2";
        else
            op = "shr.$b0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, v.index()));
    }

    CUDAArray abs_() const {
        if constexpr (!jitc_is_arithmetic(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_same_v<Value, float>
                             ? "abs.ftz.$t0 $r0, $r1"
                             : "abs.$t0 $r0, $r1";

        return from_index(jitc_var_new_1(Type, op, 1, 1, m_index));
    }

    CUDAArray sqrt_() const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_same_v<Value, float>
                             ? "sqrt.rn.ftz.$t0 $r0, $r1"
                             : "sqrt.rn.$t0 $r0, $r1";

        return from_index(jitc_var_new_1(Type, op, 1, 1, m_index));
    }

    CUDAArray rcp_() const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_same_v<Value, float>
                             ? "rcp.approx.ftz.$t0 $r0, $r1"
                             : "div.rn.$t0 $r0, 1.0, $r1";

        return from_index(jitc_var_new_1(Type, op, 1, 1, m_index));
    }

    CUDAArray rsqrt_() const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        const char *op = std::is_same_v<Value, float>
                             ? "rsqrt.approx.ftz.$t0 $r0, $r1"
                             : "sqrt.rn.$t0 $r0, $r1$n"
                               "div.rn.$t0 $r0, 1.0, $r0";

        return from_index(jitc_var_new_1(Type, op, 1, 1, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float>> = 0>
    CUDAArray exp_() const {
        return from_index(
            jitc_var_new_1(Type,
                           "mul.ftz.$t0 $r0, $r1, 1.4426950408889634074$n"
                           "ex2.approx.ftz.$t0 $r0, $r0",
                           1, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float>> = 0>
    CUDAArray log_() const {
        return from_index(
            jitc_var_new_1(Type,
                           "mul.ftz.$t0 $r0, $r1, 0.69314718055994530942$n"
                           "lg2.approx.ftz.$t1 $r0, $r0",
                           1, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float>> = 0>
    CUDAArray sin_() const {
        return from_index(
            jitc_var_new_1(Type, "sin.approx.ftz.$t1 $r0, $r1", 1, 1, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float>> = 0>
    CUDAArray cos_() const {
        return from_index(
            jitc_var_new_1(Type, "cos.approx.ftz.$t1 $r0, $r1", 1, 1, m_index));
    }

    template <typename T = Value, enable_if_t<std::is_same_v<T, float>> = 0>
    std::pair<CUDAArray, CUDAArray> sincos_() const {
        return { sin_(), cos_() };
    }

    CUDAArray min_(const CUDAArray &a) const {
        const char *op = std::is_same_v<Value, float>
                             ? "min.ftz.$t0 $r0, $r1, $r2"
                             : "min.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray max_(const CUDAArray &a) const {
        const char *op = std::is_same_v<Value, float>
                             ? "max.ftz.$t0 $r0, $r1, $r2"
                             : "max.$t0 $r0, $r1, $r2";

        return from_index(jitc_var_new_2(Type, op, 1, 1, m_index, a.index()));
    }

    CUDAArray round_() const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_1(Type,
            "cvt.rni.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    template <typename T> T round2int_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type) ||
                      !jitc_is_integral(T::Type))
            enoki_raise("Unsupported operand type");

        return T::from_index(jitc_var_new_1(T::Type,
            "cvt.rni.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray floor_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_1(Type,
            "cvt.rmi.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    template <typename T> T floor2int_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type) ||
                      !jitc_is_integral(T::Type))
            enoki_raise("Unsupported operand type");

        return T::from_index(jitc_var_new_1(T::Type,
            "cvt.rmi.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray ceil_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_1(Type,
            "cvt.rpi.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    template <typename T> T ceil2int_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type) ||
                      !jitc_is_integral(T::Type))
            enoki_raise("Unsupported operand type");

        return T::from_index(jitc_var_new_1(T::Type,
            "cvt.rpi.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray trunc_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_1(Type,
            "cvt.rzi.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    template <typename T> T trunc2int_(const CUDAArray &a) const {
        if constexpr (!jitc_is_floating_point(Type) ||
                      !jitc_is_integral(T::Type))
            enoki_raise("Unsupported operand type");

        return T::from_index(jitc_var_new_1(T::Type,
            "cvt.rzi.$t0.$t0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray fmadd_(const CUDAArray &b, const CUDAArray &c) const {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        // Simple constant propagation
        if (is_literal_one()) {
            return b + c;
        } else if (b.is_literal_one()) {
            return *this + c;
        } else if (is_literal_zero() || b.is_literal_zero()) {
            return c;
        } else if (c.is_literal_zero()) {
            return *this * b;
        }

        const char *op = std::is_same_v<Value, float>
                             ? "fma.rn.ftz.$t0 $r0, $r1, $r2, $r3"
                             : "fma.rn.$t0 $r0, $r1, $r2, $r3";

        return from_index(
            jitc_var_new_3(Type, op, 1, 1, m_index, b.index(), c.index()));
    }

    CUDAArray fmsub_(const CUDAArray &b, const CUDAArray &c) const {
        return fmadd_(b, -c);
    }

    CUDAArray fnmadd_(const CUDAArray &b, const CUDAArray &c) const {
        return fmadd_(-b, c);
    }

    CUDAArray fnmsub_(const CUDAArray &b, const CUDAArray &c) const {
        return fmsub_(-b, -c);
    }

    static CUDAArray select_(const CUDAArray<bool> &m, const CUDAArray &t,
                             const CUDAArray &f) {
        // Simple constant propagation
        if (m.is_literal_one())
            return t;
        else if (m.is_literal_zero())
            return f;

        if constexpr (!std::is_same_v<Value, bool>) {
            return from_index(jitc_var_new_3(Type,
                                             "selp.$t0 $r0, $r1, $r2, $r3", 1, 1,
                                             t.index(), f.index(), m.index()));
        } else {
            return (m & t) | (~m & f);
        }
    }

    CUDAArray popcnt_() const {
        if constexpr (!jitc_is_integral(Type))
            enoki_raise("Unsupported operand type");

        return from_index(
            jitc_var_new_1(Type, "popc.$b0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray lzcnt_() const {
        if constexpr (!jitc_is_integral(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_1(Type, "clz.$b0 $r0, $r1", 1, 1, m_index));
    }

    CUDAArray tzcnt_() const {
        if constexpr (!jitc_is_integral(Type))
            enoki_raise("Unsupported operand type");

        return from_index(jitc_var_new_1(
            Type, "brev.$b0 $r0, $r1$nclz.$b0 $r0, $r0", 1, 1, m_index));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Horizontal operations
    // -----------------------------------------------------------------------

    bool all_() const {
        if constexpr (!jitc_is_mask(Type))
            enoki_raise("Unsupported operand type");

        if (size() == 0) {
            enoki_raise("all_(): zero-sized array!");
        } else if (is_literal_one()) {
            return true;
        } else if (is_literal_zero()) {
            return false;
        } else {
            eval();
            return (bool) jitc_all((uint8_t *) data(), (uint32_t) size());
        }
    }

    bool any_() const {
        if constexpr (!jitc_is_mask(Type))
            enoki_raise("Unsupported operand type");

        if (size() == 0) {
            enoki_raise("any_(): zero-sized array!");
        } else if (is_literal_one()) {
            return true;
        } else if (is_literal_zero()) {
            return false;
        } else {
            eval();
            return (bool) jitc_any((uint8_t *) data(), (uint32_t) size());
        }
    }

    #define ENOKI_HORIZONTAL_OP(name, op)                                     \
        CUDAArray name##_async_() const {                                     \
            if constexpr (!jitc_is_arithmetic(Type))                          \
                enoki_raise("Unsupported operand type");                      \
            if (size() == 0)                                                  \
                enoki_raise(#name "_async_(): zero-sized array!");            \
            else if (size() == 1)                                             \
                return *this;                                                 \
                                                                              \
            eval();                                                           \
            CUDAArray result = empty<CUDAArray>(1);                           \
            jitc_reduce(Type, op, data(), (uint32_t) size(), result.data());  \
            return result;                                                    \
        }                                                                     \
        Value name##_() const { return name##_async_().coeff(0); }

    ENOKI_HORIZONTAL_OP(hsum,  ReductionType::Add)
    ENOKI_HORIZONTAL_OP(hprod, ReductionType::Mul)
    ENOKI_HORIZONTAL_OP(hmin,  ReductionType::Min)
    ENOKI_HORIZONTAL_OP(hmax,  ReductionType::Max)

    #undef ENOKI_HORIZONTAL_OP

    Value dot_(const CUDAArray &a) const {
        return enoki::hsum(*this * a);
    }

    CUDAArray dot_async_(const CUDAArray &a) const {
        return enoki::hsum_async(*this * a);
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Fancy array initialization
    // -----------------------------------------------------------------------

    static CUDAArray empty_(size_t size) {
        if (size == 0)
            return CUDAArray();
        size_t byte_size = size * sizeof(Value);
        void *ptr = jitc_malloc(AllocType::Device, byte_size);
        return from_index(jitc_var_map(Type, 1, ptr, (uint32_t) size, 1));
    }

    static CUDAArray zero_(size_t size) {
        if (size == 0)
            return CUDAArray();
        return from_index(mkfull_(Value(0), (uint32_t) size));
    }

    static CUDAArray full_(Value value, size_t size) {
        if (size == 0)
            return CUDAArray();
        return from_index(mkfull_(value, (uint32_t) size));
    }

    static CUDAArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        using UInt32 = CUDAArray<uint32_t>;

        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
        UInt32 index = UInt32::from_index(
            jitc_var_new_0(VarType::UInt32, "mov.u32 $r0, $i", 1, 1, (uint32_t) size));

        if (start == 0 && step == 1) {
            return CUDAArray(index);
        } else {
            if constexpr (std::is_floating_point_v<Value>)
                return fmadd(index, (Value) step, (Value) start);
            else
                return index * (Value) step + (Value) start;
        }
    }

    static CUDAArray linspace_(Value min, Value max, size_t size) {
        if constexpr (!jitc_is_floating_point(Type))
            enoki_raise("Unsupported operand type");

        using UInt32 = CUDAArray<uint32_t>;

        UInt32 index = UInt32::from_index(
            jitc_var_new_0(VarType::UInt32, "mov.u32 $r0, $i", 1, 1, (uint32_t) size));

        Value step = (max - min) / Value(size - 1);
        return fmadd(index, step, min);
    }

    static CUDAArray map_(void *ptr, size_t size, bool free = false) {
        return from_index(
            jitc_var_map(Type, 1, ptr, (uint32_t) size, free ? 1 : 0));
    }

    static CUDAArray load_unaligned_(const void *ptr, size_t size) {
        return from_index(jitc_var_copy(AllocType::Host, Type, 1, ptr, (uint32_t) size));
    }

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Scatter/gather support
    // -----------------------------------------------------------------------
private:
    template <typename Index>
    static CUDAArray gather_impl_(const void *src_ptr, uint32_t src_index,
                                  const CUDAArray<Index> &index,
                                  const CUDAArray<bool> &mask = true) {
        if constexpr (sizeof(Index) != 4) {
            /* Prefer 32 bit index arithmetic, 64 bit multiplies are
               emulated and thus very expensive on NVIDIA GPUs.. */
            using Int32 = int32_array_t<CUDAArray<Index>>;
            return gather_impl_(src_ptr, src_index, Int32(index), mask);
        } else {
            CUDAArray<void *> base = CUDAArray<void *>::from_index(
                jitc_var_copy_ptr(src_ptr, src_index));

            uint32_t var = 0;
            if (mask.is_literal_one()) {
                var = jitc_var_new_2(Type,
                                     !std::is_same_v<Value, bool>
                                         ? "mul.wide.$t2 %rd3, $r2, $s0$n"
                                           "add.$t1 %rd3, %rd3, $r1$n"
                                           "ld.global.nc.$t0 $r0, [%rd3]"
                                         : "mul.wide.$t2 %rd3, $r2, $s0$n"
                                           "add.$t1 %rd3, %rd3, $r1$n"
                                           "ld.global.nc.u8 %w0, [%rd3]$n"
                                           "setp.ne.u16 $r0, %w0, 0",
                                     1, 1, base.index(), index.index());
            } else {
                var = jitc_var_new_3(Type,
                                     !std::is_same_v<Value, bool>
                                         ? "mul.wide.$t2 %rd3, $r2, $s0$n"
                                           "add.$t1 %rd3, %rd3, $r1$n"
                                           "@$r3 ld.global.nc.$t0 $r0, [$r1]$n"
                                           "@!$r3 mov.$b0 $r0, 0"
                                         : "mul.wide.$t2 %rd3, $r2, $s0$n"
                                           "add.$t1 %rd3, %rd3, $r1$n"
                                           "@$r3 ld.global.nc.u8 %w0, [$r1]$n"
                                           "@!$r3 mov.u16 %w0, 0$n"
                                           "setp.ne.u16 $r0, %w0, 0",
                                     1, 1, base.index(), index.index(),
                                     mask.index());
            }

            return from_index(var);
        }
    }

    template <typename Index>
    void scatter_impl_(void *dst, uint32_t dst_index,
                       const CUDAArray<Index> &index,
                       const CUDAArray<bool> &mask = true) {
        if constexpr (sizeof(Index) != 4) {
            /* Prefer 32 bit index arithmetic, 64 bit multiplies are
               emulated and thus very expensive on NVIDIA GPUs.. */
            using Int32 = int32_array_t<CUDAArray<Index>>;
            return scatter_impl_(dst, dst_index, Int32(index), mask);
        } else {
            CUDAArray<void *> base = CUDAArray<void *>::from_index(
                jitc_var_copy_ptr(dst, dst_index));

            uint32_t var;
            if (mask.is_literal_one()) {
                if constexpr (!std::is_same_v<Value, bool>) {
                    var = jitc_var_new_3(VarType::Invalid,
                                         "mul.wide.$t3 %rd3, $r3, $s2$n"
                                         "add.$t1 %rd3, %rd3, $r1$n"
                                         "st.global.$t2 [%rd3], $r2",
                                         1, 1, base.index(), m_index,
                                         index.index());
                } else {
                    var = jitc_var_new_3(VarType::Invalid,
                                         "mul.wide.$t3 %rd3, $r3, $s2$n"
                                         "add.$t1 %rd3, %rd3, $r1$n"
                                         "selp.u16 %w0, 1, 0, $r2$n"
                                         "st.global.u8 [%rd3], %w0",
                                         1, 1, base.index(), m_index,
                                         index.index());
                }
            } else {
                if (!std::is_same_v<Value, bool>) {
                    var = jitc_var_new_4(VarType::Invalid,
                                         "mul.wide.$t3 %rd3, $r3, $s2$n"
                                         "add.$t1 %rd3, %rd3, $r1$n"
                                         "@$r4 st.global.$t2 [%rd3], $r2",
                                         1, 1, base.index(), m_index,
                                         index.index(), mask.index());
                } else {
                    var = jitc_var_new_4(VarType::Invalid,
                                         "mul.wide.$t3 %rd3, $r3, $s2$n"
                                         "add.$t1 %rd3, %rd3, $r1$n"
                                         "selp.u16 %w0, 1, 0, $r2$n"
                                         "@$r4 st.global.u8 [%rd3], %w0",
                                         1, 1, base.index(), m_index,
                                         index.index(), mask.index());
                }
            }
            jitc_var_mark_scatter(var, dst_index);
        }
    }

    template <typename Index>
    void scatter_add_impl_(void *dst, uint32_t dst_index,
                           const CUDAArray<Index> &index,
                           const CUDAArray<bool> &mask = true) {
        if constexpr (sizeof(Index) != 4) {
            /* Prefer 32 bit index arithmetic, 64 bit multiplies are
               emulated and thus very expensive on NVIDIA GPUs.. */
            using Int32 = int32_array_t<CUDAArray<Index>>;
            return scatter_add_impl_(dst, Int32(index), mask);
        } else {
            CUDAArray<void *> base = CUDAArray<void *>::from_index(
                jitc_var_copy_ptr(dst, dst_index));

            uint32_t var;
            if (mask.is_literal_one()) {
                var = jitc_var_new_3(VarType::Invalid,
                                     "mul.wide.$t3 %rd3, $r3, $s2$n"
                                     "add.$t1 %rd3, %rd3, $r1$n"
                                     "red.global.add.$t2 [%rd3], $r2",
                                     1, 1, base.index(), m_index,
                                     index.index());
            } else {
                var = jitc_var_new_4(VarType::Invalid,
                                     "mul.wide.$t3 %rd3, $r3, $s2$n"
                                     "add.$t1 %rd3, %rd3, $r1$n"
                                     "@$r4 red.global.add.$t2 [%rd3], $r2",
                                     1, 1, base.index(), m_index, index.index(),
                                     mask.index());
            }

            jitc_var_mark_scatter(var, dst_index);
        }
    }

public:
    template <typename Index>
    static CUDAArray gather_raw_(const void *src, const CUDAArray<Index> &index,
                                 const CUDAArray<bool> &mask = true) {
        if (mask.is_literal_zero())
            return Value(0);

        return gather_impl_(src, 0, index, mask);
    }

    template <typename Index>
    static CUDAArray gather_(const CUDAArray &src, const CUDAArray<Index> &index,
                             const CUDAArray<bool> &mask = true) {
        if (mask.is_literal_zero())
            return Value(0);

        src.eval();
        return gather_impl_(src.data(), src.index(), index, mask);
    }

    template <typename Index>
    void scatter_raw_(void *dst, const CUDAArray<Index> &index,
                      const CUDAArray<bool> &mask = true) {
        if (mask.is_literal_zero())
            return;

        scatter_impl_(dst, 0, index, mask);
    }

    template <typename Index>
    void scatter_(CUDAArray &dst, const CUDAArray<Index> &index,
                  const CUDAArray<bool> &mask = true) {
        if (mask.is_literal_zero())
            return;

        void *ptr = dst.data();

        if (!ptr) {
            dst.eval();
            ptr = dst.data();
        }

        if (jitc_var_int_ref(dst.index()) > 0) {
            dst = CUDAArray<Value>::from_index(
                jitc_var_copy(AllocType::Device, CUDAArray<Value>::Type,
                              1, ptr, (uint32_t) dst.size()));
            ptr = dst.data();
        }

        scatter_impl_(ptr, dst.index(), index, mask);
    }

    template <typename Index>
    void scatter_add_raw_(void *dst, const CUDAArray<Index> &index,
                          const CUDAArray<bool> &mask = true) {
        if (mask.is_literal_zero())
            return;

        scatter_add_impl_(dst, 0, index, mask);
    }

    template <typename Index>
    void scatter_add_(CUDAArray &dst, const CUDAArray<Index> &index,
                      const CUDAArray<bool> &mask = true) {
        if (mask.is_literal_zero())
            return;

        void *ptr = dst.data();

        if (!ptr) {
            dst.eval();
            ptr = dst.data();
        }

        if (jitc_var_int_ref(dst.index()) > 0) {
            dst = CUDAArray<Value>::from_index(
                jitc_var_copy(AllocType::Device, CUDAArray<Value>::Type,
                              1, ptr, (uint32_t) dst.size()));
            ptr = dst.data();
        }

        scatter_add_impl_(ptr, dst.index(), index, mask);
    }

    // -----------------------------------------------------------------------
    //! @{ \name Miscellaneous
    // -----------------------------------------------------------------------

    CUDAArray& schedule() {
        jitc_var_schedule(m_index);
        return *this;
    }

    const CUDAArray& schedule() const {
        jitc_var_schedule(m_index);
        return *this;
    }

    CUDAArray& eval() {
        jitc_var_eval(m_index);
        return *this;
    }

    const CUDAArray& eval() const {
        jitc_var_eval(m_index);
        return *this;
    }

    bool valid() const { return m_index != 0; }
    size_t size() const { return jitc_var_size(m_index); }
    uint32_t index() const { return m_index; }

    const Value *data() const { return (const Value *) jitc_var_ptr(m_index); }
    Value *data() { return (Value *) jitc_var_ptr(m_index); }

    bool is_literal_one() const { return (bool) jitc_var_is_literal_one(m_index); }
    bool is_literal_zero() const { return (bool) jitc_var_is_literal_zero(m_index); }

    Value coeff(size_t offset) const {
        Value out;
        jitc_var_read(m_index, (uint32_t) offset, &out);
        return out;
    }

    void write(uint32_t offset, Value value) {
        if (jitc_var_int_ref(m_index) > 0) {
            eval();
            *this = CUDAArray::from_index(
                jitc_var_copy(AllocType::Device, CUDAArray<Value>::Type, 1,
                              data(), (uint32_t) size()));
        }

        jitc_var_write(m_index, offset, &value);
    }

    void migrate(AllocType type) {
        jitc_var_migrate(m_index, type);
    }

    void set_label_(const char *label) const {
        jitc_var_set_label(m_index, label);
    }

    const char *label_() const {
        return jitc_var_label(m_index);
    }

    //! @}
    // -----------------------------------------------------------------------

    static CUDAArray from_index(uint32_t index) {
        CUDAArray result;
        result.m_index = index;
        return result;
    }

    static uint32_t mkfull_(Value value, uint32_t size) {
        const char *fmt = nullptr;

        switch (Type) {
            case VarType::Float16:
                fmt = "mov.$b0 $r0, 0x%04x";
                break;

            case VarType::Float32:
                fmt = "mov.$t0 $r0, 0f%08x";
                break;

            case VarType::Float64:
                fmt = "mov.$t0 $r0, 0d%016llx";
                break;

            case VarType::Bool:
                fmt = "mov.$t0 $r0, %i";
                break;

            case VarType::Int8:
            case VarType::UInt8:
                fmt = "mov.b16 %%w1, 0x%02x$ncvt.u8.u16 $r0, %%w1";
                break;

            case VarType::Int16:
            case VarType::UInt16:
                fmt = "mov.$b0 $r0, 0x%04x";
                break;

            case VarType::Int32:
            case VarType::UInt32:
                fmt = "mov.$b0 $r0, 0x%08x";
                break;

            case VarType::Pointer:
            case VarType::Int64:
            case VarType::UInt64:
                fmt = "mov.$b0 $r0, 0x%016llx";
                break;

            default:
                fmt = "<<invalid format during cast>>";
                break;
        }

        uint_array_t<Value> value_uint;
        char value_str[48];
        memcpy(&value_uint, &value, sizeof(Value));
        snprintf(value_str, 48, fmt, value_uint);

        return jitc_var_new_0(Type, value_str, 0, 1, size);
    }

    void init_(size_t size) {
        *this = empty_(size);
    }

protected:
    uint32_t m_index = 0;
};


NAMESPACE_END(enoki)
