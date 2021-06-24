/*
    tests/matrix.cpp -- tests matrix arrays

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <enoki/dynamic.h>
#include <enoki/matrix.h>

namespace ek = enoki;

using Float     = ek::DynamicArray<float>;
using UInt32    = ek::DynamicArray<uint32_t>;
using Mask      = ek::mask_t<Float>;
using Array2f   = ek::Array<Float, 2>;
using Matrix2f  = ek::Matrix<Float, 2>;
using Matrix22f = ek::Matrix<Array2f, 2>;

template <typename Type> void test_scatter() {
    int n = 999;

    auto idx  = ek::arange<UInt32>(n);
    auto mask = idx >= (n / 2);

    auto target = ek::zero<Type>(n);
    auto source = ek::full<Type>(4, n);
    ek::scatter(target, source, idx, mask);

    assert(ek::all_nested(ek::eq(target, ek::select(mask, source, 0))));
}

ENOKI_TEST(test01_scatter) {
    test_scatter<Array2f>();
    test_scatter<Matrix2f>();
    test_scatter<Matrix22f>();
}

template <typename Type> void test_gather() {
    int n = 999;

    auto idx  = ek::arange<UInt32>(n);
    auto mask = idx >= (n / 2);

    auto source = ek::full<Type>(4, n);
    auto res = ek::gather<Type>(source, idx, mask);

    assert(ek::all_nested(ek::eq(res, ek::select(mask, source, 0))));
}

ENOKI_TEST(test02_gather) {
    test_gather<Array2f>();
    test_gather<Matrix2f>();
    test_gather<Matrix22f>();
}
