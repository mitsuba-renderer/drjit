/*
    enoki/matrix.h -- Square matrix data structure

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/


#pragma once

#include <enoki/array.h>

NAMESPACE_BEGIN(enoki)

template <typename Value_, size_t Size_>
struct Matrix : StaticArrayImpl<Array<Value_, Size_>, Size_, false,
                                Matrix<Value_, Size_>> {
    using Column = Array<Value_, Size_>;
    using Base = StaticArrayImpl<Column, Size_, false, Matrix<Value_, Size_>>;
    using Base::entry;
    using Base::Size;
    ENOKI_ARRAY_DEFAULTS(Matrix)


    static constexpr bool IsMatrix = true;
    static constexpr bool IsSpecial = true;
    static constexpr bool IsVector = false;

    using ArrayType = Matrix;
    using MaskType = Mask<mask_t<Column>, Size_>;

    template <typename T> using ReplaceValue = Matrix<value_t<T>, Size_>;

    Matrix() = default;

    template <typename T, enable_if_t<is_matrix_v<T> || array_depth_v<T> == Base::Depth> = 0>
    ENOKI_INLINE Matrix(T&& m) {
        constexpr size_t Size2 = array_size_v<T>;
        if constexpr (Size2 >= Size) {
            /// Other matrix is equal or bigger -- retain the top left part
            for (size_t i = 0; i < Size; ++i)
                entry(i) = head<Size>(m.entry(i));
        } else {
            /// Other matrix is smaller -- copy the top left part and set remainder to identity
            using Remainder = Array<Value_, Size - Size2>;
            for (size_t i = 0; i < Size2; ++i)
                entry(i) = concat(m.entry(i), zero<Remainder>());
            for (size_t i = Size2; i < Size; ++i) {
                Column col = zero<Column>();
                col.entry(i) = 1;
                entry(i) = col;
            }
        }
    }

    template <typename T, enable_if_t<!is_matrix_v<T> && array_depth_v<T> != Base::Depth> = 0>
    ENOKI_INLINE Matrix(T&& v) : Base(zero<Value_>()) {
        for (size_t i = 0; i < Size; ++i)
            entry(i, i) = v;
    }

    /// Initialize the matrix from a list of columns
    template <typename... Args, enable_if_t<sizeof...(Args) == Size_ &&
              std::conjunction_v<std::is_constructible<Column, Args>...>> = 0>
    ENOKI_INLINE Matrix(const Args&... args) : Base(args...) { }

    /// Initialize the matrix from a list of entries in row-major order
    template <typename... Args, enable_if_t<sizeof...(Args) == Size_ * Size_ &&
              std::conjunction_v<std::is_constructible<Value_, Args>...>> = 0>
    ENOKI_INLINE Matrix(const Args&... args) {
        Value_ values[sizeof...(Args)] = { Value_(args)... };
        for (size_t j = 0; j < Size; ++j)
            for (size_t i = 0; i < Size; ++i)
                entry(j, i) = values[i * Size + j];
    }

    /// Return a reference to the (i, j) element
    ENOKI_INLINE Value_ operator()(size_t i, size_t j) { return entry(j, i); }

    /// Return a reference to the (i, j) element (const)
    ENOKI_INLINE const Value_ &operator()(size_t i, size_t j) const { return entry(j, i); }

};

template <typename T0, typename T1, size_t Size>
Matrix<expr_t<T0, T1>, Size> operator*(const Matrix<T0, Size> &m0,
                                       const Matrix<T1, Size> &m1) {
    using Result = Matrix<expr_t<T0, T1>, Size>;
    Result result;

    for (size_t j = 0; j < Size; ++j) {
        using Column = value_t<Result>;
        Column col = m0.entry(0) * full<Column>(m1(0, j));
        for (size_t i = 1; i < Size; ++i)
            col = fmadd(m0.entry(i), full<Column>(m1(i, j), 1), col);
        result.entry(j) = col;
    }

    return result;
}

template <typename T0, typename T1, size_t Size,
          enable_if_t<!is_matrix_v<T1>> = 0>
ENOKI_INLINE auto operator*(const Matrix<T0, Size> &m0, const T1 &a1) {
    if constexpr (is_vector_v<T1> && array_size_v<T1> == Size) {
        using Result = Array<expr_t<T0, value_t<T1>>, Size>;
        Result result = m0.entry(0) * full<Result>(a1.entry(0));
        for (size_t i = 1; i < Size; ++i)
            result = fmadd(m0.entry(i), full<Result>(a1.entry(i)), result);
        return result;
    } else {
        using Value = expr_t<T0, T1>;
        using Result = Matrix<Value, Size>;
        using AsArray = Array<Array<Value, Size>, Size>;
        return Result(AsArray(m0) * full<AsArray>(Value(a1)));
    }
}

template <typename T0, typename T1, size_t Size,
          enable_if_t<!is_matrix_v<T0>> = 0>
ENOKI_INLINE auto operator*(const T0 &a0, const Matrix<T1, Size> &m1) {
    using Value = expr_t<T0, T1>;
    using Result = Matrix<Value, Size>;
    using AsArray = Array<Array<Value, Size>, Size>;
    return Result(full<AsArray>(Value(a0)) * AsArray(m1));
}

template <typename T0, typename T1, size_t Size,
          enable_if_t<!is_matrix_v<T1>> = 0>
ENOKI_INLINE auto operator/(const Matrix<T0, Size> &m0, const T1 &a1) {
    using Value = expr_t<T0, T1>;
    using Result = Matrix<Value, Size>;
    using AsArray = Array<Array<Value, Size>, Size>;
    return Result(AsArray(m0) * full<AsArray>(rcp(Value(a1))));
}


NAMESPACE_END(enoki)
