/*
    tests/custom.cpp -- tests operations involving custom data structures

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <drjit/dynamic.h>
#include <drjit/struct.h>

template <typename Value_> struct Custom {
    using Value = Value_;
    using FloatVector = Array<Value, 3>;
    using DoubleVector = float64_array_t<FloatVector>;
    using IntVector = int32_array_t<Value>;

    FloatVector o;
    DoubleVector d;
    IntVector i = 0;

    Custom(const FloatVector &o, const DoubleVector &d, const IntVector &i)
        : o(o), d(d), i(i) { }

    bool operator==(const Custom &c) const {
        return o == c.o && d == c.d && i == c.i;
    }

    DRJIT_STRUCT(Custom, o, d, i)
};

using FloatX = DynamicArray<float>;
using Float64X = DynamicArray<double>;
using Int32X = DynamicArray<int32_t>;
using Vector3f = Array<float, 3>;
using Custom3f = Custom<float>;
using Custom3fX = Custom<FloatX>;

DRJIT_TEST(test01_init) {
    auto v = zeros<Custom3fX>();

    auto a = empty<std::pair<Custom3f, Custom3fX>>(100);
    auto b = zeros<std::tuple<Custom3f, Custom3fX, Custom3f>>(100);

    assert(width(a) == 100);
    assert(width(b) == 100);

    set_label(b, "asdf"); // this doesn't do anything, but it should compile
}

DRJIT_TEST(test02_masked_assignment) {
    Custom3fX c = zeros<Custom3fX>(5);
    c.o.x() = linspace<FloatX>(0, 1, 5);
    c.i = arange<Int32X>(5);

    masked(c, c.i < 3) = zeros<Custom3fX>();
    assert(c.o.x() == FloatX(0.f, 0.f, 0.f, 0.75f, 1.f));
}

DRJIT_TEST(test03_scatter_gather) {
    Custom3fX c1 = zeros<Custom3fX>(5);
    Custom3f c2(1, 2, 3);

    scatter(c1, c2, 2);
    assert (gather<Custom3f>(c1, 2) == c2);

    Custom3fX c3 = zeros<Custom3fX>(2);
    c3.d.x() = Float64X(1, 2);
    c3.d.y() = Float64X(3, 4);
    c3.d.z() = Float64X(5, 6);

    scatter(c1, c3, Int32X(0, 1));
    assert(c1.d.y() == Float64X(3, 4, 2, 0, 0));
    assert(c1.o.x() == FloatX(0, 0, 1, 0, 0));
}

DRJIT_TEST(test04_slice) {
    Custom3fX c = zeros<Custom3fX>(5);
    c.o.x() = linspace<FloatX>(0, 1, 5);
    c.i = arange<Int32X>(5);

    Custom3f c3 = slice<Custom3f>(c, 3);

    assert(c3.o.x() == 0.75f);
    assert(c3.i == 3);
}
