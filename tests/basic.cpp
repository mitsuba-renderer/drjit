/*
    tests/basic.cpp -- tests basic operators involving different types

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"

#if defined(_MSC_VER)
#  pragma warning(disable: 4146) //  warning C4146: unary minus operator applied to unsigned type, result still unsigned
#endif

DRJIT_TEST_ALL(test00_align) {
#if defined(DRJIT_X86_SSE42)
    if (sizeof(Value)*Size == 16) {
        assert(sizeof(T) == 16);
        assert(alignof(T) == 16);
    }
#elif defined(DRJIT_X86_AVX)
    if (sizeof(Value)*Size == 32) {
        assert(sizeof(T) == 32);
        assert(alignof(T) == 32);
    }
#elif defined(DRJIT_X86_AVX512F)
    if (sizeof(Value)*Size == 64) {
        assert(sizeof(T) == 64);
        assert(alignof(T) == 64);
    }
#endif

    using Packet     = T;
    using Vector4x   = Array<Value, 4>;
    using Vector4xP  = Array<Packet, 4>;

    static_assert(std::is_same<value_t<Value>,      Value>::value, "value_t failure");
    static_assert(std::is_same<value_t<Vector4x>,   Value>::value, "value_t failure");
    static_assert(std::is_same<value_t<Vector4xP>,  Packet>::value, "value_t failure");

    using Vector4d   = Array<double, 4>;

    /* Non-array input */
    static_assert(std::is_same<expr_t<Value>,               Value>::value, "expr_t failure");
    static_assert(std::is_same<expr_t<Value&>,              Value>::value, "expr_t failure");
    static_assert(std::is_same<expr_t<Value, double>,       double>::value, "expr_t failure");
    static_assert(std::is_same<expr_t<Value&, double&>,     double>::value, "expr_t failure");

    /* Array input */
    static_assert(std::is_same<expr_t<Vector4x>,            Vector4x>::value, "expr_t failure");
    static_assert(std::is_same<expr_t<Vector4xP>,           Vector4xP>::value, "expr_t failure");

    static_assert(std::is_same<expr_t<Vector4x, double>,    Vector4d>::value, "expr_t failure");

    /* Non-array input */
    static_assert(std::is_same<scalar_t<Value>,             Value>::value, "scalar_t failure");
    static_assert(std::is_same<scalar_t<Value&>,            Value>::value, "scalar_t failure");

    /* Array input */
    static_assert(std::is_same<scalar_t<Vector4x>,          Value>::value, "scalar_t failure");
    static_assert(std::is_same<scalar_t<Vector4xP>,         Value>::value, "scalar_t failure");

    /* Pointers */
    struct Test;
    static_assert(std::is_same<Test*, expr_t<Test*, Test*>>::value, "expr_t failure");
    static_assert(std::is_same<const Test*, expr_t<const Test*, const Test*>>::value, "expr_t failure");
    static_assert(std::is_same<const Test*, expr_t<Test*, const Test*>>::value, "expr_t failure");
    static_assert(std::is_same<const Test*, expr_t<const Test*, Test*>>::value, "expr_t failure");
    static_assert(std::is_same<const Test*, expr_t<const Test*, std::nullptr_t>>::value, "expr_t failure");
    static_assert(std::is_same<Test*, expr_t<std::nullptr_t, Test*>>::value, "expr_t failure");
}

DRJIT_TEST_ALL(test01_add) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a + b; },
        [](Value a, Value b) -> Value { return a + b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x += b; return x; },
        [](Value a, Value b) -> Value { return a + b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a + Value(3); },
        [](Value a) -> Value { return a + Value(3); }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Value(3) + a; },
        [](Value a) -> Value { return Value(3) + a; }
    );
}

DRJIT_TEST_ALL(test02_sub) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a - b; },
        [](Value a, Value b) -> Value { return a - b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x -= b; return x; },
        [](Value a, Value b) -> Value { return a - b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a - Value(3); },
        [](Value a) -> Value { return a - Value(3); }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Value(3) - a; },
        [](Value a) -> Value { return Value(3) - a; }
    );
}

DRJIT_TEST_ALL(test03_mul) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return a * b; },
        [](Value a, Value b) -> Value { return a * b; }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T x(a); x *= b; return x; },
        [](Value a, Value b) -> Value { return a * b; }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return a * Value(3); },
        [](Value a) -> Value { return a * Value(3); }
    );

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return Value(3) * a; },
        [](Value a) -> Value { return Value(3) * a; }
    );
}

DRJIT_TEST_ALL(test05_neg) {
    auto sample = test::sample_values<Value>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return -a; },
        [](Value a) -> Value { return -a; }
    );
}

DRJIT_TEST_ALL(test06_lt) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a < b, T(0), T(1)); },
        [](Value a, Value b) -> Value { return Value(a < b ? 0 : 1); }
    );

    assert(select(T(1) < T(2), T(1), T(2)) == T(1));
    assert(select(T(1) > T(2), T(1), T(2)) == T(2));
}

DRJIT_TEST_ALL(test07_le) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a <= b, T(0), T(1)); },
        [](Value a, Value b) -> Value { return Value(a <= b ? 0 : 1); }
    );
}

DRJIT_TEST_ALL(test08_gt) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a > b, T(0), T(1)); },
        [](Value a, Value b) -> Value { return Value(a > b ? 0 : 1); }
    );
}

DRJIT_TEST_ALL(test09_ge) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(a >= b, T(0), T(1)); },
        [](Value a, Value b) -> Value { return Value(a >= b ? 0 : 1); }
    );
}

DRJIT_TEST_ALL(test10_eq) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(eq(a, b), T(0), T(1)); },
        [](Value a, Value b) -> Value { return Value(a == b ? 0 : 1); }
    );

    // Equality of mask values.
    mask_t<Value> m1(true), m2(true | true), m3(false);
    assert(all_nested(m1 & m2) && all_nested(eq(m1, m2)) && all_nested(eq(m2, m2)));
    assert(all_nested(m1 | m2) && none_nested(m3 & m2)
           && none_nested(eq(m3, m2)) && all_nested(eq(m3, m3)));
}

DRJIT_TEST_ALL(test11_neq) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return select(neq(a, b), T(0), T(1)); },
        [](Value a, Value b) -> Value { return Value(a != b ? 0 : 1); }
    );

    // Equality of mask values.
    mask_t<Value> m1(true), m2(true | true), m3(false);
    assert(none_nested(neq(m1, m2)) && none_nested(neq(m2, m2))
           && none_nested(m3 & m2)  && none_nested(eq(m3, m2)));
}

DRJIT_TEST_ALL(test12_min) {
#if !defined(DRJIT_ARM_NEON)
    auto sample = test::sample_values<Value>();
#else
    // ARM NEON propagates NaN more aggressively than x86..
    auto sample = test::sample_values<Value>(false);
#endif

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return minimum(a, b); },
        [](Value a, Value b) -> Value { return std::min(a, b); }
    );
}

DRJIT_TEST_ALL(test13_max) {
#if !defined(DRJIT_ARM_NEON)
    auto sample = test::sample_values<Value>();
#else
    // ARM NEON propagates NaN more aggressively than x86..
    auto sample = test::sample_values<Value>(false);
#endif

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return maximum(a, b); },
        [](Value a, Value b) -> Value { return std::max(a, b); }
    );
}

DRJIT_TEST_ALL(test14_abs) {
    auto sample = test::sample_values<Value>();

    test::validate_unary<T>(sample,
        [](const T &a) -> T { return abs(a); },
        [](Value a) -> Value { return drjit::abs(a); }
    );
}

DRJIT_TEST_ALL(test15_fmadd) {
    if (!T::IsFloat)
        return;
    auto sample = test::sample_values<Value>();

    test::validate_ternary<T>(sample,
        [](const T &a, const T &b, const T& c) -> T { return fmadd(a, b, c); },
        [](Value a, Value b, Value c) -> Value {
            return a*b + c;
        },
        Value(1e-6)
    );
}

DRJIT_TEST_ALL(test16_fmsub) {
    if (!T::IsFloat)
        return;
    auto sample = test::sample_values<Value>();

    test::validate_ternary<T>(sample,
        [](const T &a, const T &b, const T& c) -> T { return fmsub(a, b, c); },
        [](Value a, Value b, Value c) -> Value {
            return a*b - c;
        },
        Value(1e-6)
    );
}

DRJIT_TEST_ALL(test17_select) {
    auto sample = test::sample_values<Value>();

    test::validate_ternary<T>(sample,
        [](const T &a, const T &b, const T& c) -> T { return select(a > Value(5), b, c); },
        [](Value a, Value b, Value c) -> Value {
            return a > 5 ? b : c;
        }
    );
}

template <typename T, std::enable_if_t<T::Size == 1, int> = 0> void test18_shuffle_impl() {
    assert((drjit::shuffle<0>(T(1)) == T(1)));
}

template <typename T, std::enable_if_t<T::Size == 2, int> = 0> void test18_shuffle_impl() {
    assert((drjit::shuffle<0, 1>(T(1, 2)) == T(1, 2)));
    assert((drjit::shuffle<1, 0>(T(1, 2)) == T(2, 1)));
}

template <typename T, std::enable_if_t<T::Size == 3, int> = 0> void test18_shuffle_impl() {
    assert((drjit::shuffle<0, 1, 2>(T(1, 2, 3)) == T(1, 2, 3)));
    assert((drjit::shuffle<2, 1, 0>(T(1, 2, 3)) == T(3, 2, 1)));
}

template <typename T, std::enable_if_t<T::Size == 4, int> = 0> void test18_shuffle_impl() {
    assert((drjit::shuffle<0, 1, 2, 3>(T(1, 2, 3, 4)) == T(1, 2, 3, 4)));
    assert((drjit::shuffle<3, 2, 1, 0>(T(1, 2, 3, 4)) == T(4, 3, 2, 1)));
}

template <typename T, std::enable_if_t<T::Size == 8, int> = 0> void test18_shuffle_impl() {
    auto shuf1 = shuffle<0, 1, 2, 3, 4, 5, 6, 7>(T(1, 2, 3, 4, 5, 6, 7, 8));
    auto shuf2 = shuffle<7, 6, 5, 4, 3, 2, 1, 0>(T(1, 2, 3, 4, 5, 6, 7, 8));

    assert(shuf1 == T(1, 2, 3, 4, 5, 6, 7, 8));
    assert(shuf2 == T(8, 7, 6, 5, 4, 3, 2, 1));
}

template <typename T, std::enable_if_t<T::Size == 16, int> = 0> void test18_shuffle_impl() {
    auto shuf1 = shuffle<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>(
        T(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    auto shuf2 = shuffle<15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0>(
        T(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));

    assert(shuf1 == T(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    assert(shuf2 == T(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
}

template <typename T, std::enable_if_t<(T::Size > 16), int> = 0> void test18_shuffle_impl() {
    std::cout << "[skipped] ";
}

DRJIT_TEST_ALL(test18_shuffle) {
    test18_shuffle_impl<T>();
}

template <typename T,
          std::enable_if_t<T::Size != (T::Size / 2) * 2, int> = 0>
void test19_lowhi_impl() {
    std::cout << "[skipped] ";
}
template <typename T,
          std::enable_if_t<T::Size == (T::Size / 2) * 2, int> = 0>
void test19_lowhi_impl() {
    auto value = arange<T>();
    assert(T(low(value), high(value)) == value);
    assert(concat(low(value), high(value)) == value);
}
DRJIT_TEST_ALL(test19_lowhi) { test19_lowhi_impl<T>(); }

DRJIT_TEST_ALL(test20_iterator) {
    Value j(0);
    for (Value i : arange<T>()) {
        assert(i == j);
        j += 1;
    }
}

DRJIT_TEST_ALL(test21_mask_assign) {
    T x = arange<T>();
    x[x > Value(0)] = x + Value(1);
    if (Size >= 2) {
        assert(x.entry(0) == Value(0));
        assert(x.entry(1) == Value(2));
    }
    x[x > Value(0)] = Value(-1);
    if (Size >= 2) {
        assert(x.entry(0) == Value(0));
        assert(x.entry(1) == Value(-1));
    }
}

DRJIT_TEST_FLOAT(test22_copysign) {
    auto sample = test::sample_values<Value>(false);

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return drjit::copysign(a, b); },
        [](Value a, Value b) -> Value { return std::copysign(a, b); }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return drjit::copysign_neg(a, b); },
        [](Value a, Value b) -> Value { return std::copysign(a, -b); }
    );
}

DRJIT_TEST_FLOAT(test23_mulsign) {
    auto sample = test::sample_values<Value>(false);

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return drjit::mulsign(a, b); },
        [](Value a, Value b) -> Value { return a*std::copysign(Value(1), b); }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { return drjit::mulsign_neg(a, b); },
        [](Value a, Value b) -> Value { return a*std::copysign(Value(1), -b); }
    );
}

DRJIT_TEST_ALL(test25_rorl_array) {
    auto a = arange<T>();
    auto b = rotate_left<2>(a);
    auto c = rotate_right<2>(b);
    assert(a == c);
}

DRJIT_TEST_ALL(test26_mask_op) {
    auto sample = test::sample_values<Value>();

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T result(1); result[a <= b] = T(0); return result; },
        [](Value a, Value b) -> Value { return Value(a <= b ? 0 : 1); }
    );

    test::validate_binary<T>(sample,
        [](const T &a, const T &b) -> T { T result(1); result[a <= b] -= T(1); return result; },
        [](Value a, Value b) -> Value { return Value(a <= b ? 0 : 1); }
    );
}

#if defined(DRJIT_X86_SSE42)
DRJIT_TEST(test27_flush_denormals) {
    bool prev = flush_denormals();
    set_flush_denormals(false);
    assert(!flush_denormals());
    set_flush_denormals(true);
    assert(flush_denormals());
    set_flush_denormals(prev);
}
#endif

DRJIT_TEST_ALL(test28_mask_from_int) {
    using Mask = mask_t<T>;
    // Mask construction, select
    Mask m1(true), m2(true | true), m3(true & false);
    assert(all_nested(m1) && all_nested(m2) && none_nested(m3));
    assert(all_nested(eq(m1, m2)) && none_nested(eq(m2, m3)));
    assert(all(eq(T(1), select(m1, T(1), T(0)))));
    assert(all(eq(T(1), select(m2, T(1), T(0)))));
    assert(all(eq(T(0), select(m3, T(1), T(0)))));
    // Masked assignment
    T zero(0);
    T one(1);
    T val                          = zero;
    masked(val, true)              = one;    assert(all(eq(val, one)));
    val                            = zero;
    masked(val, Mask(true & true)) = one;    assert(all(eq(val, one)));
    val                            = zero;
    masked(val, true & true)       = one;    assert(all(eq(val, one)));
    val                            = zero;
    masked(val, true & false)      = one;    assert(all(eq(val, zero)));
}
