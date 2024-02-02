/*
    drjit/array_constants.h -- Common constants and other useful quantities

    (This file isn't meant to be included as-is. Please use 'drjit/array.h',
     which bundles all the 'array_*' headers in the right order.)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

template <typename T> struct constants {
    static constexpr T E               = T(2.71828182845904523536);
    static constexpr T LogTwo          = T(0.69314718055994530942);
    static constexpr T InvLogTwo       = T(1.44269504088896340736);

    static constexpr T Pi              = T(3.14159265358979323846);
    static constexpr T InvPi           = T(0.31830988618379067154);
    static constexpr T SqrtPi          = T(1.77245385090551602793);
    static constexpr T InvSqrtPi       = T(0.56418958354775628695);

    static constexpr T TwoPi           = T(6.28318530717958647692);
    static constexpr T InvTwoPi        = T(0.15915494309189533577);
    static constexpr T SqrtTwoPi       = T(2.50662827463100050242);
    static constexpr T InvSqrtTwoPi    = T(0.39894228040143267794);

    static constexpr T FourPi          = T(12.5663706143591729539);
    static constexpr T InvFourPi       = T(0.07957747154594766788);
    static constexpr T SqrtFourPi      = T(3.54490770181103205460);
    static constexpr T InvSqrtFourPi   = T(0.28209479177387814347);

    static constexpr T SqrtTwo         = T(1.41421356237309504880);
    static constexpr T InvSqrtTwo      = T(0.70710678118654752440);

#if defined(__GNUC__)
    static constexpr T Infinity        = T(__builtin_inf());
#else
    static constexpr T Infinity        = T(__builtin_huge_val());
#endif
    static constexpr T NaN             = T(__builtin_nan(""));

    /// Machine epsilon
    static constexpr T Epsilon         = T(sizeof(T) == 8
                                                ? 0x1p-53
                                                : 0x1p-24);
    /// 1 - Machine epsilon
    static constexpr T OneMinusEpsilon = T(sizeof(T) == 8
                                                ? 0x1.fffffffffffffp-1
                                                : 0x1.fffffep-1);

    /// Any numbers below this threshold will overflow to infinity when a reciprocal is evaluated
    static constexpr T RecipOverflow   = T(sizeof(T) == 8
                                                ? 0x1p-1024 : 0x1p-128);

    /// Smallest normalized floating point value
    static constexpr T Smallest        = T(sizeof(T) == 8
                                                ? 0x1p-1022 : 0x1p-126);

    /// Largest normalized floating point value
    static constexpr T Largest         = T(sizeof(T) == 8
                                                ? 0x1.fffffffffffffp+1023
                                                : 0x1.fffffep+127);
};

NAMESPACE_END(detail)


template <typename T> constexpr auto E               = detail::constants<scalar_t<T>>::E;
template <typename T> constexpr auto LogTwo          = detail::constants<scalar_t<T>>::LogTwo;
template <typename T> constexpr auto InvLogTwo       = detail::constants<scalar_t<T>>::InvLogTwo;

template <typename T> constexpr auto Pi              = detail::constants<scalar_t<T>>::Pi;
template <typename T> constexpr auto InvPi           = detail::constants<scalar_t<T>>::InvPi;
template <typename T> constexpr auto SqrtPi          = detail::constants<scalar_t<T>>::SqrtPi;
template <typename T> constexpr auto InvSqrtPi       = detail::constants<scalar_t<T>>::InvSqrtPi;

template <typename T> constexpr auto TwoPi           = detail::constants<scalar_t<T>>::TwoPi;
template <typename T> constexpr auto InvTwoPi        = detail::constants<scalar_t<T>>::InvTwoPi;
template <typename T> constexpr auto SqrtTwoPi       = detail::constants<scalar_t<T>>::SqrtTwoPi;
template <typename T> constexpr auto InvSqrtTwoPi    = detail::constants<scalar_t<T>>::InvSqrtTwoPi;

template <typename T> constexpr auto FourPi          = detail::constants<scalar_t<T>>::FourPi;
template <typename T> constexpr auto InvFourPi       = detail::constants<scalar_t<T>>::InvFourPi;
template <typename T> constexpr auto SqrtFourPi      = detail::constants<scalar_t<T>>::SqrtFourPi;
template <typename T> constexpr auto InvSqrtFourPi   = detail::constants<scalar_t<T>>::InvSqrtFourPi;

template <typename T> constexpr auto SqrtTwo         = detail::constants<scalar_t<T>>::SqrtTwo;
template <typename T> constexpr auto InvSqrtTwo      = detail::constants<scalar_t<T>>::InvSqrtTwo;

template <typename T> constexpr auto Infinity        = detail::constants<scalar_t<T>>::Infinity;
template <typename T> constexpr auto NaN             = detail::constants<scalar_t<T>>::NaN;

/// Machine epsilon
template <typename T> constexpr auto Epsilon         = detail::constants<scalar_t<T>>::Epsilon;
/// 1 - Machine epsilon
template <typename T> constexpr auto OneMinusEpsilon = detail::constants<scalar_t<T>>::OneMinusEpsilon;

/// Any numbers below this threshold will overflow to infinity when a reciprocal is evaluated
template <typename T> constexpr auto RecipOverflow   = detail::constants<scalar_t<T>>::RecipOverflow;

/// Smallest normalized floating point value
template <typename T> constexpr auto Smallest        = detail::constants<scalar_t<T>>::Smallest;
/// Largest normalized floating point value
template <typename T> constexpr auto Largest         = detail::constants<scalar_t<T>>::Largest;

NAMESPACE_BEGIN(detail)

template <typename T> struct debug_init {
    static constexpr T value = T(int_array_t<scalar_t<T>>(-1));
};
template <> struct debug_init<float> {
    static constexpr float value = NaN<float>;
};
template <> struct debug_init<double> {
    static constexpr double value = NaN<double>;
};
template <typename T> struct debug_init<T*> {
    static constexpr T *value = nullptr;
};
NAMESPACE_END(detail)

template <typename T>
constexpr auto DebugInit = detail::debug_init<scalar_t<T>>::value;

NAMESPACE_END(drjit)
