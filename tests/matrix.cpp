/*
    tests/matrix.cpp -- tests matrix arrays

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <drjit/dynamic.h>
#include <drjit/matrix.h>
#include <drjit/quaternion.h>

namespace dr = drjit;

using Float        = dr::DynamicArray<float>;
using UInt32       = dr::DynamicArray<uint32_t>;
using Mask         = dr::mask_t<Float>;
using Array2f      = dr::Array<Float, 2>;
using Matrix2f     = dr::Matrix<Float, 2>;
using Matrix22f    = dr::Matrix<Array2f, 2>;
using Quaternion4f = dr::Quaternion<Float>;

template <typename Type> void test_scatter() {
    int n = 999;

    auto idx  = dr::arange<UInt32>(n);
    auto mask = idx >= (n / 2);

    auto target = dr::zeros<Type>(n);
    auto source = dr::full<Type>(4, n);
    dr::scatter(target, source, idx, mask);

    assert(dr::all_nested(dr::eq(target, dr::select(mask, source, 0))));
}

DRJIT_TEST(test01_scatter) {
    test_scatter<Array2f>();
    test_scatter<Matrix2f>();
    test_scatter<Matrix22f>();
    test_scatter<Quaternion4f>();
}

template <typename Type> void test_gather() {
    int n = 999;

    auto idx  = dr::arange<UInt32>(n);
    auto mask = idx >= (n / 2);

    auto source = dr::full<Type>(4, n);
    auto res = dr::gather<Type>(source, idx, mask);

    assert(dr::all_nested(dr::eq(res, dr::select(mask, source, 0))));
}

DRJIT_TEST(test02_gather) {
    test_gather<Array2f>();
    test_gather<Matrix2f>();
    test_gather<Matrix22f>();
    test_gather<Quaternion4f>();
}
