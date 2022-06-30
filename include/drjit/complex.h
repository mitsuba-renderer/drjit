/*
    drjit/complex.h -- Complex number data structure

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_>
struct Complex : StaticArrayImpl<Value_, 2, false, Complex<Value_>> {
    using Base = StaticArrayImpl<Value_, 2, false, Complex<Value_>>;
    DRJIT_ARRAY_DEFAULTS(Complex)

    static constexpr bool IsComplex = true;
    static constexpr bool IsSpecial = true;
    static constexpr bool IsVector = false;

    using ArrayType = Complex;
    using PlainArrayType = Array<Value_, 2>;
    using MaskType = Mask<Value_, 2>;
    using typename Base::Scalar;

    template <typename T> using ReplaceValue = Complex<T>;

    Complex() = default;

    template <typename T, enable_if_t<is_complex_v<T> || array_depth_v<T> == Base::Depth> = 0>
    DRJIT_INLINE Complex(T&& z) : Base(std::forward<T>(z)) { }

    template <typename T, enable_if_t<!is_complex_v<T> && array_depth_v<T> != Base::Depth &&
                                       (is_array_v<T> || std::is_scalar_v<std::decay_t<T>>)> = 0>
    DRJIT_INLINE Complex(T&& z) : Base(std::forward<T>(z), zeros<Value_>()) { }

    template <typename T, enable_if_t<!is_array_v<T> && !std::is_scalar_v<std::decay_t<T>>> = 0> // __m128d
    DRJIT_INLINE Complex(T&& z) : Base(z) { }

    DRJIT_INLINE Complex(const Value_ &v1, const Value_ &v2) : Base(v1, v2) { }
    DRJIT_INLINE Complex(Value_ &&v1, Value_ &&v2)
        : Base(std::move(v1), std::move(v2)) { }
};

template <typename T> DRJIT_INLINE T real(const Complex<T> &z) { return z.x(); }
template <typename T> DRJIT_INLINE T imag(const Complex<T> &z) { return z.y(); }

template <typename T, enable_if_complex_t<T> = 0>
T identity(size_t size = 1) {
    using Value = value_t<T>;
    return { identity<Value>(size), zeros<Value>(size) };
}

template <typename T> T squared_norm(const Complex<T> &z) {
    return squared_norm(Array<T, 2>(z));
}

template <typename T> T norm(const Complex<T> &z) {
    return norm(Array<T, 2>(z));
}

template <typename T> Complex<T> normalize(const Complex<T> &z) {
    return normalize(Array<T, 2>(z));
}

template <typename T> Complex<T> conj(const Complex<T> &z) {
    if constexpr (!is_array_v<T>)
        return z ^ Complex<T>(0.f, -0.f);
    else
        return { z.x(), -z.y() };
}

template <typename T0, typename T1>
Complex<expr_t<T0, T1>> operator*(const Complex<T0> &z0,
                                  const Complex<T1> &z1) {
    return {
        fmsub(z0.x(), z1.x(), z0.y()*z1.y()),
        fmadd(z0.x(), z1.y(), z0.y()*z1.x())
    };
}

template <typename T0, typename T1>
Complex<expr_t<T0, T1>> operator*(const Complex<T0> &z0,
                                  const T1 &v1) {
    return Array<T0, 2>(z0) * v1;
}

template <typename T0, typename T1>
Complex<expr_t<T0, T1>> operator*(const T0 &v0,
                                  const Complex<T1> &z1) {
    return v0 * Array<T1, 2>(z1);
}

template <typename T> Complex<T> rcp(const Complex<T> &z) {
    return conj(z) * rcp(squared_norm(z));
}

template <typename T0, typename T1>
Complex<expr_t<T0, T1>> operator/(const Complex<T0> &z0,
                                  const Complex<T1> &z1) {
    return z0 * rcp(z1);
}

template <typename T0, typename T1>
Complex<expr_t<T0, T1>> operator/(const Complex<T0> &z0,
                                  const T1 &v1) {
    return Array<T0, 2>(z0) / v1;
}

template <typename T> T abs(const Complex<T> &z) {
    return norm(z);
}

template <typename T> Complex<T> rsqrt(const Complex<T> &z) {
    return rcp(sqrt(z));
}

template <typename T> Complex<T> exp(const Complex<T> &z) {
    T exp_r = exp(real(z));
    auto [s, c] = sincos(imag(z));
    return { exp_r * c, exp_r * s };
}

template <typename T> Complex<T> exp2(const Complex<T> &z) {
    T exp_r = exp2(real(z));
    auto [s, c] = sincos(imag(z) * LogTwo<T>);
    return { exp_r * c, exp_r * s };
}

template <typename T> Complex<T> log(const Complex<T> &z) {
    return { .5f * log(squared_norm(z)), arg(z) };
}

template <typename T> Complex<T> log2(const Complex<T> &z) {
    return { .5f * log2(squared_norm(z)), arg(z) * InvLogTwo<T> };
}

template <typename T> T arg(const Complex<T> &z) {
    return atan2(imag(z), real(z));
}

template <typename T1, typename T2, typename T = expr_t<T1, T2>>
std::pair<T, T> sincos_arg_diff(const Complex<T1> &z1, const Complex<T2> &z2) {
    T normalization = rsqrt(squared_norm(z1) * squared_norm(z2));
    Complex<T> v = z1 * conj(z2) * normalization;
    return { imag(v), real(v) };
}

template <typename T0, typename T1>
Complex<expr_t<T0, T1>> pow(const Complex<T0> &z0, const Complex<T1> &z1) {
    return exp(log(z0) * z1);
}

template <typename T> Complex<T> sqrt(const Complex<T> &z) {
    T n  = abs(z),
      t1 = sqrt(.5f * (n + abs(real(z)))),
      t2 = .5f * imag(z) / t1;

    mask_t<T> zero = eq(n, 0.f);
    mask_t<T> m = real(z) >= 0.f;

    return {
        select(m, t1, abs(t2)),
        select(zero, 0.f, select(m, t2, copysign(t1, imag(z))))
    };
}

template <typename T> Complex<T> sin(const Complex<T> &z) {
    auto [s, c]   = sincos(real(z));
    auto [sh, ch] = sincosh(imag(z));
    return { s * ch, c * sh };
}

template <typename T> Complex<T> cos(const Complex<T> &z) {
    auto [s, c]   = sincos(real(z));
    auto [sh, ch] = sincosh(imag(z));
    return { c * ch, -s * sh };
}

template <typename T>
std::pair<Complex<T>, Complex<T>> sincos(const Complex<T> &z) {
    auto [s, c]   = sincos(real(z));
    auto [sh, ch] = sincosh(imag(z));
    return {
        Complex<T>(s * ch, c * sh),
        Complex<T>(c * ch, -s * sh)
    };
}

template <typename T> Complex<T> tan(const Complex<T> &z) {
    auto [s, c] = sincos(z);
    return s / c;
}

template <typename T> Complex<T> cot(const Complex<T> &z) {
    auto [s, c] = sincos(z);
    return c / s;
}

template <typename T>
Complex<T> asin(const Complex<T> &z) {
    Complex<T> tmp = log(Complex<T>(-imag(z), real(z)) + sqrt(1.f - sqr(z)));
    return { imag(tmp), -real(tmp) };
}

template <typename T> Complex<T> acos(const Complex<T> &z) {
    Complex<T> tmp = sqrt(1.f - sqr(z));
    tmp = log(z + Complex<T>(-imag(tmp), real(tmp)));
    return Complex<T>{ imag(tmp), -real(tmp) };
}

template <typename T>
Complex<T> atan(const Complex<T> &z) {
    const Complex<T> I(0.f, 1.f);
    Complex<T> tmp = log((I - z) / (I + z));
    return { imag(tmp) * .5f, -real(tmp) * .5f };
}

template <typename T> Complex<T> sinh(const Complex<T> &z) {
    auto [s, c]  = sincos(imag(z));
    auto [sh, ch] = sincosh(real(z));
    return { sh * c, ch * s };
}

template <typename T> Complex<T> cosh(const Complex<T> &z) {
    auto [s, c]   = sincos(imag(z));
    auto [sh, ch] = sincosh(real(z));
    return { ch * c, sh * s };
}

template <typename T>
std::pair<Complex<T>, Complex<T>> sincosh(const Complex<T> &z) {
    auto [s, c] = sincos(imag(z));
    auto [sh, ch] = sincosh(real(z));
    return {
        Complex<T>(sh * c, ch * s),
        Complex<T>(ch * c, sh * s)
    };
}

template <typename T>
Complex<T> tanh(const Complex<T> &z) {
    auto [sh, ch] = sincosh(z);
    return sh / ch;
}

template <typename T>
Complex<T> asinh(const Complex<T> &z) {
    return log(z + sqrt(sqr(z) + 1.f));
}

template <typename T>
Complex<T> acosh(const Complex<T> &z) {
    return 2 * log(sqrt(.5f * (z + 1.f)) + sqrt(.5f * (z - 1.f)));
}

template <typename T>
Complex<T> atanh(const Complex<T> &z) {
    return log((1.f + z) / (1.f - z)) * .5f;
}

template <typename T, typename Stream>
DRJIT_NOINLINE Stream &operator<<(Stream &os, const Complex<T> &z) {
    if constexpr (is_array_v<T>) {
        os << "[";
        size_t size = real(z).size();
        for (size_t i = 0; i < size; ++i) {
            os << Complex<typename T::Value>(real(z).entry(i), imag(z).entry(i));
            if (i + 1 < size)
                os << ",\n ";
        }
        os << "]";
    } else {
        os << real(z);
        os << (imag(z) < 0 ? " - " : " + ") << abs(imag(z)) << "i";
    }
    return os;
}

NAMESPACE_END(drjit)
