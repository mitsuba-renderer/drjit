/*
    drjit/array_constants.h -- Common constants and other useful quantities

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/half.h>
#include <drjit/array_traits.h>

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)

template <typename T, typename = void>
struct constants {
    using scalar_t = drjit::scalar_t<T>;

    static constexpr auto E               = scalar_t(2.71828182845904523536);
    static constexpr auto LogTwo          = scalar_t(0.69314718055994530942);
    static constexpr auto InvLogTwo       = scalar_t(1.44269504088896340736);

    static constexpr auto Pi              = scalar_t(3.14159265358979323846);
    static constexpr auto InvPi           = scalar_t(0.31830988618379067154);
    static constexpr auto SqrtPi          = scalar_t(1.77245385090551602793);
    static constexpr auto InvSqrtPi       = scalar_t(0.56418958354775628695);

    static constexpr auto TwoPi           = scalar_t(6.28318530717958647692);
    static constexpr auto InvTwoPi        = scalar_t(0.15915494309189533577);
    static constexpr auto SqrtTwoPi       = scalar_t(2.50662827463100050242);
    static constexpr auto InvSqrtTwoPi    = scalar_t(0.39894228040143267794);

    static constexpr auto FourPi          = scalar_t(12.5663706143591729539);
    static constexpr auto InvFourPi       = scalar_t(0.07957747154594766788);
    static constexpr auto SqrtFourPi      = scalar_t(3.54490770181103205460);
    static constexpr auto InvSqrtFourPi   = scalar_t(0.28209479177387814347);

    static constexpr auto SqrtTwo         = scalar_t(1.41421356237309504880);
    static constexpr auto InvSqrtTwo      = scalar_t(0.70710678118654752440);

#if defined(__GNUC__)
    static constexpr auto Infinity        = scalar_t(__builtin_inf());
#else
    static constexpr auto Infinity        = scalar_t(__builtin_huge_val());
#endif
    static constexpr auto NaN             = scalar_t(__builtin_nan(""));

    /// Machine epsilon
    static constexpr auto Epsilon         = scalar_t(sizeof(scalar_t) == 8
                                                ? 0x1p-53
                                                : 0x1p-24);
    /// 1 - Machine epsilon
    static constexpr auto OneMinusEpsilon = scalar_t(sizeof(scalar_t) == 8
                                                ? 0x1.fffffffffffffp-1
                                                : 0x1.fffffep-1);

    /// Any numbers below this threshold will overflow to infinity when a reciprocal is evaluated
    static constexpr auto RecipOverflow   = scalar_t(sizeof(scalar_t) == 8
                                                ? 0x1p-1024 : 0x1p-128);

    /// Smallest normalized floating point value
    static constexpr auto Smallest        = scalar_t(sizeof(scalar_t) == 8
                                                ? 0x1p-1022 : 0x1p-126);

    /// Largest normalized floating point value
    static constexpr auto Largest         = scalar_t(sizeof(scalar_t) == 8
                                                ? 0x1.fffffffffffffp+1023
                                                : 0x1.fffffep+127);
};

template <typename T>
struct constants<T, typename std::enable_if_t<std::is_same_v<drjit::scalar_t<T>, drjit::half>>> {

    using half = drjit::half;

    static constexpr half E               = half::from_binary(0x4170);
    static constexpr half LogTwo          = half::from_binary(0x398c);
    static constexpr half InvLogTwo       = half::from_binary(0x3dc5);

    static constexpr half Pi              = half::from_binary(0x4248);
    static constexpr half InvPi           = half::from_binary(0x3518);
    static constexpr half SqrtPi          = half::from_binary(0x3f17);
    static constexpr half InvSqrtPi       = half::from_binary(0x3883);

    static constexpr half TwoPi           = half::from_binary(0x4648);
    static constexpr half InvTwoPi        = half::from_binary(0x3118);
    static constexpr half SqrtTwoPi       = half::from_binary(0x4103);
    static constexpr half InvSqrtTwoPi    = half::from_binary(0x3662);

    static constexpr half FourPi          = half::from_binary(0x4a48);
    static constexpr half InvFourPi       = half::from_binary(0x2d18);
    static constexpr half SqrtFourPi      = half::from_binary(0x4317);
    static constexpr half InvSqrtFourPi   = half::from_binary(0x3483);

    static constexpr half SqrtTwo         = half::from_binary(0x3da8);
    static constexpr half InvSqrtTwo      = half::from_binary(0x39a8);

    static constexpr half Infinity        = half::from_binary(0xfc00);
    static constexpr half NaN             = half::from_binary(0xffff);
};

NAMESPACE_END(detail)


template <typename T> constexpr auto E               = detail::constants<T>::E;
template <typename T> constexpr auto LogTwo          = detail::constants<T>::LogTwo;
template <typename T> constexpr auto InvLogTwo       = detail::constants<T>::InvLogTwo;

template <typename T> constexpr auto Pi              = detail::constants<T>::Pi;
template <typename T> constexpr auto InvPi           = detail::constants<T>::InvPi;
template <typename T> constexpr auto SqrtPi          = detail::constants<T>::SqrtPi;
template <typename T> constexpr auto InvSqrtPi       = detail::constants<T>::InvSqrtPi;

template <typename T> constexpr auto TwoPi           = detail::constants<T>::TwoPi;
template <typename T> constexpr auto InvTwoPi        = detail::constants<T>::InvTwoPi;
template <typename T> constexpr auto SqrtTwoPi       = detail::constants<T>::SqrtTwoPi;
template <typename T> constexpr auto InvSqrtTwoPi    = detail::constants<T>::InvSqrtTwoPi;

template <typename T> constexpr auto FourPi          = detail::constants<T>::FourPi;
template <typename T> constexpr auto InvFourPi       = detail::constants<T>::InvFourPi;
template <typename T> constexpr auto SqrtFourPi      = detail::constants<T>::SqrtFourPi;
template <typename T> constexpr auto InvSqrtFourPi   = detail::constants<T>::InvSqrtFourPi;

template <typename T> constexpr auto SqrtTwo         = detail::constants<T>::SqrtTwo;
template <typename T> constexpr auto InvSqrtTwo      = detail::constants<T>::InvSqrtTwo;

template <typename T> constexpr auto Infinity        = detail::constants<T>::Infinity;
template <typename T> constexpr auto NaN             = detail::constants<T>::NaN;

/// Machine epsilon
template <typename T> constexpr auto Epsilon         = detail::constants<T>::Epsilon;
/// 1 - Machine epsilon
template <typename T> constexpr auto OneMinusEpsilon = detail::constants<T>::OneMinusEpsilon;

/// Any numbers below this threshold will overflow to infinity when a reciprocal is evaluated
template <typename T> constexpr auto RecipOverflow   = detail::constants<T>::RecipOverflow;

/// Smallest normalized floating point value
template <typename T> constexpr auto Smallest        = detail::constants<T>::Smallest;
/// Largest normalized floating point value
template <typename T> constexpr auto Largest         = detail::constants<T>::Largest;

NAMESPACE_BEGIN(detail)

template <typename T> struct debug_initialization {
    static constexpr T value = T(int_array_t<T>(-1));
};
template <> struct debug_initialization<half> {
    static constexpr half value = NaN<half>;
};
template <> struct debug_initialization<float> {
    static constexpr float value = NaN<float>;
};
template <> struct debug_initialization<double> {
    static constexpr double value = NaN<double>;
};
template <typename T> struct debug_initialization<T*> {
    static constexpr T *value = nullptr;
};
NAMESPACE_END(detail)

template <typename T>
constexpr auto DebugInitialization = detail::debug_initialization<T>::value;

NAMESPACE_END(drjit)
