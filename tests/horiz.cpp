/*
    tests/horiz.cpp -- tests horizontal operators involving different types

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

DRJIT_TEST_ALL(test01_sum) {
    auto sample = test::sample_values<Value>();

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return sum(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result += a[i];
            return result;
        },
        Value(1e-5f));
}

DRJIT_TEST_ALL(test02_prod) {
    auto sample = test::sample_values<Value>();

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return prod(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result *= a[i];
            return result;
        },
        Value(1e-5f));
}

DRJIT_TEST_ALL(test03_min) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return min(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result = std::min(result, a[i]);
            return result;
        }
    );
}

DRJIT_TEST_ALL(test04_max) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return max(a); },
        [](const T &a) -> Value {
            Value result = a[0];
            for (size_t i = 1; i < Size; ++i)
                result = std::max(result, a[i]);
            return result;
        }
    );
}

DRJIT_TEST_ALL(test05_all) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(all(a >= a[0]) ? 1 : 0); },
        [](const T &a) -> Value {
            bool result = true;
            for (size_t i = 0; i < Size; ++i)
                result &= a[i] >= a[0];
            return Value(result ? 1 : 0);
        }
    );

    assert(all(mask_t<T>(true)));
    assert(!all(mask_t<T>(false)));
}

DRJIT_TEST_ALL(test06_none) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(none(a > a[0]) ? 1 : 0); },
        [](const T &a) -> Value {
            bool result = false;
            for (size_t i = 0; i < Size; ++i)
                result |= a[i] > a[0];
            return Value(result ? 0 : 1);
        }
    );

    assert(!none(mask_t<T>(true)));
    assert(none(mask_t<T>(false)));
}

DRJIT_TEST_ALL(test07_any) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(any(a > a[0]) ? 1 : 0); },
        [](const T &a) -> Value {
            bool result = false;
            for (size_t i = 0; i < Size; ++i)
                result |= a[i] > a[0];
            return Value(result ? 1 : 0);
        }
    );
    assert(any(mask_t<T>(true)));
    assert(!any(mask_t<T>(false)));
}

DRJIT_TEST_ALL(test08_count) {
    auto sample = test::sample_values<Value>(false);

    test::validate_horizontal<T>(sample,
        [](const T &a) -> Value { return Value(count(a > a[0])); },
        [](const T &a) -> Value {
            int result = 0;
            for (size_t i = 0; i < Size; ++i)
                result += (a[i] > a[0]) ? 1 : 0;
            return Value(result);
        }
    );
    assert(Size == count(mask_t<T>(true)));
    assert(0 == count(mask_t<T>(false)));
}

DRJIT_TEST_ALL(test09_dot) {
    auto sample = test::sample_values<Value>();
    T value1, value2;
    Value ref = 0;

    size_t idx = 0;

    for (size_t i = 0; i < sample.size(); ++i) {
        for (size_t j = 0; j < sample.size(); ++j) {
            Value arg_i = sample[i], arg_j = sample[j];
            value1[idx] = arg_i; value2[idx] = arg_j;
            ref += arg_i * arg_j;
            idx++;

            if (idx == value1.size()) {
                Value result = dot(value1, value2);
                test::assert_close(result, ref, Value(1e-6f));
                idx = 0; ref = 0;
            }
        }
    }

    if (idx > 0) {
        while (idx < Size) {
            value1[idx] = 0;
            value2[idx] = 0;
            idx++;
        }
        Value result = dot(value1, value2);
        test::assert_close(result, ref, Value(1e-6f));
    }
}

DRJIT_TEST_ALL(test10_sum_inner_nested) {
    using Array3 = Array<T, 3>;

    Array3 x(
        arange<T>() + scalar_t<T>(1),
        arange<T>() + scalar_t<T>(2),
        arange<T>() + scalar_t<T>(3)
    );

    Array<scalar_t<T>, 3> y(
        sum(x.x()),
        sum(x.y()),
        sum(x.z())
    );

    assert(sum_inner(x) == y);
    assert(sum_nested(x) == sum(y));
}

DRJIT_TEST_ALL(test11_prod_inner_nested) {
    using Array3 = Array<T, 3>;

    Array3 x(
        arange<T>() + scalar_t<T>(1),
        arange<T>() + scalar_t<T>(2),
        arange<T>() + scalar_t<T>(3)
    );

    Array<scalar_t<T>, 3> y(
        prod(x.x()),
        prod(x.y()),
        prod(x.z())
    );

    assert(prod_inner(x) == y);
    assert(prod_nested(x) == prod(y) || T::Size > 4);
}

DRJIT_TEST_ALL(test12_min_inner_nested) {
    using Array3 = Array<T, 3>;

    Array3 x(
        arange<T>() + scalar_t<T>(1),
        arange<T>() + scalar_t<T>(2),
        arange<T>() + scalar_t<T>(3)
    );

    Array<scalar_t<T>, 3> y(
        min(x.x()),
        min(x.y()),
        min(x.z())
    );

    assert(min_inner(x) == y);
    assert(min_nested(x) == min(y));
}

DRJIT_TEST_ALL(test13_max_inner_nested) {
    using Array3 = Array<T, 3>;

    Array3 x(
        arange<T>() + scalar_t<T>(1),
        arange<T>() + scalar_t<T>(2),
        arange<T>() + scalar_t<T>(3)
    );

    Array<scalar_t<T>, 3> y(
        max(x.x()),
        max(x.y()),
        max(x.z())
    );

    assert(max_inner(x) == y);
    assert(max_nested(x) == max(y));
}
