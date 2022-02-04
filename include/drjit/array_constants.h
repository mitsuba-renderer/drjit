/*
    drjit/array_constants.h -- Common constants and other useful quantities

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array_traits.h>

NAMESPACE_BEGIN(drjit)

template <typename T> constexpr auto E               = scalar_t<T>(2.71828182845904523536);
template <typename T> constexpr auto LogTwo          = scalar_t<T>(0.69314718055994530942);
template <typename T> constexpr auto InvLogTwo       = scalar_t<T>(1.44269504088896340736);

template <typename T> constexpr auto Pi              = scalar_t<T>(3.14159265358979323846);
template <typename T> constexpr auto InvPi           = scalar_t<T>(0.31830988618379067154);
template <typename T> constexpr auto SqrtPi          = scalar_t<T>(1.77245385090551602793);
template <typename T> constexpr auto InvSqrtPi       = scalar_t<T>(0.56418958354775628695);

template <typename T> constexpr auto TwoPi           = scalar_t<T>(6.28318530717958647692);
template <typename T> constexpr auto InvTwoPi        = scalar_t<T>(0.15915494309189533577);
template <typename T> constexpr auto SqrtTwoPi       = scalar_t<T>(2.50662827463100050242);
template <typename T> constexpr auto InvSqrtTwoPi    = scalar_t<T>(0.39894228040143267794);

template <typename T> constexpr auto FourPi          = scalar_t<T>(12.5663706143591729539);
template <typename T> constexpr auto InvFourPi       = scalar_t<T>(0.07957747154594766788);
template <typename T> constexpr auto SqrtFourPi      = scalar_t<T>(3.54490770181103205460);
template <typename T> constexpr auto InvSqrtFourPi   = scalar_t<T>(0.28209479177387814347);

template <typename T> constexpr auto SqrtTwo         = scalar_t<T>(1.41421356237309504880);
template <typename T> constexpr auto InvSqrtTwo      = scalar_t<T>(0.70710678118654752440);

#if defined(__GNUC__)
template <typename T> constexpr auto Infinity        = scalar_t<T>(__builtin_inf());
#else
template <typename T> constexpr auto Infinity        = scalar_t<T>(__builtin_huge_val());
#endif
template <typename T> constexpr auto NaN             = scalar_t<T>(__builtin_nan(""));

/// Machine epsilon
template <typename T> constexpr auto Epsilon         = scalar_t<T>(sizeof(scalar_t<T>) == 8
                                                                   ? 0x1p-53
                                                                   : 0x1p-24);
/// 1 - Machine epsilon
template <typename T> constexpr auto OneMinusEpsilon = scalar_t<T>(sizeof(scalar_t<T>) == 8
                                                                   ? 0x1.fffffffffffffp-1
                                                                   : 0x1.fffffep-1);

/// Any numbers below this threshold will overflow to infinity when a reciprocal is evaluated
template <typename T> constexpr auto RecipOverflow   = scalar_t<T>(sizeof(scalar_t<T>) == 8
                                                                   ? 0x1p-1024 : 0x1p-128);

/// Smallest normalized floating point value
template <typename T> constexpr auto Smallest        = scalar_t<T>(sizeof(scalar_t<T>) == 8
                                                                   ? 0x1p-1022 : 0x1p-126);

/// Largest normalized floating point value
template <typename T> constexpr auto Largest         = scalar_t<T>(sizeof(scalar_t<T>) == 8
                                                                   ? 0x1.fffffffffffffp+1023
                                                                   : 0x1.fffffep+127);

NAMESPACE_BEGIN(detail)
template <typename T> struct debug_initialization {
    static constexpr T value = T(int_array_t<T>(-1));
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
