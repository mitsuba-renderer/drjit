/*
    drjit/matrix.h -- Square matrix data structure

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/


#pragma once

#include <drjit/packet.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_, size_t Size_>
struct Matrix : StaticArrayImpl<Array<Value_, Size_>, Size_, false,
                                Matrix<Value_, Size_>> {
    using Column = Array<Value_, Size_>;
    using Base = StaticArrayImpl<Column, Size_, false, Matrix<Value_, Size_>>;
    using Base::entry;
    using Base::Size;
    DRJIT_ARRAY_DEFAULTS(Matrix)


    static constexpr bool IsMatrix = true;
    static constexpr bool IsSpecial = true;
    static constexpr bool IsVector = false;

    using ArrayType = Matrix;
    using PlainArrayType = Array<Array<Value_, Size>, Size>;
    using MaskType = Mask<mask_t<Column>, Size_>;
    using Entry = Value_;

    template <typename T> using ReplaceValue = Matrix<value_t<T>, Size_>;

    Matrix() = default;

    template <typename T, enable_if_t<is_matrix_v<T> || array_depth_v<T> == Base::Depth> = 0>
    DRJIT_INLINE Matrix(T&& m) {
        constexpr size_t ArgSize = array_size_v<T>;
        if constexpr (ArgSize >= Size) {
            /// Other matrix is equal or bigger -- retain the top left part
            for (size_t i = 0; i < Size; ++i)
                entry(i) = head<Size>(m.entry(i));
        } else {
            /// Other matrix is smaller -- copy the top left part and set remainder to identity
            using Remainder = Array<Value_, Size - ArgSize>;
            for (size_t i = 0; i < ArgSize; ++i)
                entry(i) = concat(m.entry(i), zeros<Remainder>());
            for (size_t i = ArgSize; i < Size; ++i) {
                Column col = zeros<Column>();
                col.entry(i) = 1;
                entry(i) = col;
            }
        }
    }

    template <typename T, enable_if_t<!is_matrix_v<T> && array_depth_v<T> != Base::Depth> = 0>
    DRJIT_INLINE Matrix(T&& v) : Base(zeros<Value_>()) {
        for (size_t i = 0; i < Size; ++i)
            entry(i, i) = v;
    }

    /// Initialize the matrix from a list of columns
    template <typename... Args, enable_if_t<sizeof...(Args) == Size_ &&
              std::conjunction_v<std::is_constructible<Column, Args>...>> = 0>
    DRJIT_INLINE Matrix(const Args&... args) : Base(args...) { }

    /// Initialize the matrix from a list of entries in row-major order
    template <typename... Args, enable_if_t<sizeof...(Args) == Size_ * Size_ &&
              std::conjunction_v<std::is_constructible<Value_, Args>...>> = 0>
    DRJIT_INLINE Matrix(const Args&... args) {
        Value_ values[sizeof...(Args)] = { Value_(args)... };
        for (size_t j = 0; j < Size; ++j)
            for (size_t i = 0; i < Size; ++i)
                entry(j, i) = values[i * Size + j];
    }

    /// Return a reference to the (i, j) element
    DRJIT_INLINE Value_& operator()(size_t i, size_t j) { return entry(j, i); }

    /// Return a reference to the (i, j) element (const)
    DRJIT_INLINE const Value_ &operator()(size_t i, size_t j) const { return entry(j, i); }

};

template <typename T, enable_if_matrix_t<T> = 0>
T identity(size_t size = 1) {
    using Entry = value_t<value_t<T>>;
    Entry o = identity<Entry>(size);
    T result = zeros<T>(size);
    for (size_t i = 0; i < T::Size; ++i)
        result.entry(i, i) = o;
    return result;
}

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
auto operator*(const Matrix<T0, Size> &m0, const T1 &a1) {
    if constexpr (is_vector_v<T1> && array_size_v<T1> == Size) {
        using Result = Array<expr_t<T0, value_t<T1>>, Size>;
        Result result = m0.entry(0) * full<Result>(a1.entry(0));
        for (size_t i = 1; i < Size; ++i)
            result = fmadd(m0.entry(i), full<Result>(a1.entry(i)), result);
        return result;
    } else {
        using Value = expr_t<T0, T1>;
        using Result = Matrix<Value, Size>;
        using PlainArrayType = plain_t<Result>;
        return Result(PlainArrayType(m0) * full<PlainArrayType>(Value(a1)));
    }
}

template <typename T0, typename T1, size_t Size,
          enable_if_t<!is_matrix_v<T0>> = 0>
auto operator*(const T0 &a0, const Matrix<T1, Size> &m1) {
    using Value = expr_t<T0, T1>;
    using Result = Matrix<Value, Size>;
    using PlainArrayType = plain_t<Result>;
    return Result(full<PlainArrayType>(Value(a0)) * PlainArrayType(m1));
}

template <typename T0, typename T1, size_t Size,
          enable_if_t<!is_matrix_v<T1>> = 0>
auto operator/(const Matrix<T0, Size> &m0, const T1 &a1) {
    using Value = expr_t<T0, T1>;
    using Result = Matrix<Value, Size>;
    using PlainArrayType = plain_t<Result>;
    return Result(PlainArrayType(m0) * full<PlainArrayType>(rcp(Value(a1))));
}

template <typename Value, size_t Size>
Value trace(const Matrix<Value, Size> &m) {
    Value result = m.entry(0, 0);
    for (size_t i = 1; i < Size; ++i)
        result += m.entry(i, i);
    return result;
}

template <typename Value, size_t Size>
Value frob(const Matrix<Value, Size> &m) {
    Array<Value, Size> result = sqr(m.entry(0));
    for (size_t i = 1; i < Size; ++i)
        result = fmadd(m.entry(i), m.entry(i), result);
    return sum(result);
}

template <typename Value, size_t Size>
Array<Value, Size> diag(const Matrix<Value, Size> &m) {
    Array<Value, Size> result;
    for (size_t i = 0; i < Size; ++i)
        result.set_entry(i, m.entry(i, i));
    return result;
}

template <typename Array, enable_if_t<!is_matrix_v<Array>> = 0>
Matrix<value_t<Array>, Array::Size> diag(const Array &v) {
    using Result = Matrix<value_t<Array>, Array::Size>;

    Result result = zeros<Result>();
    for (size_t i = 0; i < Array::Size; ++i)
        result.entry(i, i) = v.entry(i);
    return result;
}

template <typename Array, enable_if_array_t<Array> = 0>
Array transpose(const Array &a) {
    using Column = value_t<Array>;
    constexpr size_t Size = array_size_v<Array>;

    static_assert(array_depth_v<Array> >= 2 && Size == array_size_v<Column>,
                  "Array must be a square matrix!");

    if constexpr (Column::IsPacked) {
        #if defined(DRJIT_X86_SSE42)
            if constexpr (std::is_same_v<value_t<Column>, float> && Size == 3) {
                __m128 c0 = a.entry(0).m, c1 = a.entry(1).m, c2 = a.entry(2).m;

                __m128 t0 = _mm_unpacklo_ps(c0, c1);
                __m128 t1 = _mm_unpacklo_ps(c2, c2);
                __m128 t2 = _mm_unpackhi_ps(c0, c1);
                __m128 t3 = _mm_unpackhi_ps(c2, c2);

                return Array(
                    _mm_movelh_ps(t0, t1),
                    _mm_movehl_ps(t1, t0),
                    _mm_movelh_ps(t2, t3)
                );
            } else if constexpr (std::is_same_v<value_t<Column>, float> && Size == 4) {
                __m128 c0 = a.entry(0).m, c1 = a.entry(1).m,
                       c2 = a.entry(2).m, c3 = a.entry(3).m;

                __m128 t0 = _mm_unpacklo_ps(c0, c1);
                __m128 t1 = _mm_unpacklo_ps(c2, c3);
                __m128 t2 = _mm_unpackhi_ps(c0, c1);
                __m128 t3 = _mm_unpackhi_ps(c2, c3);

                return Array(
                    _mm_movelh_ps(t0, t1),
                    _mm_movehl_ps(t1, t0),
                    _mm_movelh_ps(t2, t3),
                    _mm_movehl_ps(t3, t2)
                );
            }
        #endif

        #if defined(DRJIT_X86_AVX)
            if constexpr (std::is_same_v<value_t<Column>, double> && Size == 3) {
                __m256d c0 = a.entry(0).m, c1 = a.entry(1).m, c2 = a.entry(2).m;

                __m256d t3 = _mm256_shuffle_pd(c2, c2, 0b0000),
                        t2 = _mm256_shuffle_pd(c2, c2, 0b1111),
                        t1 = _mm256_shuffle_pd(c0, c1, 0b0000),
                        t0 = _mm256_shuffle_pd(c0, c1, 0b1111);

                return Array(
                    _mm256_permute2f128_pd(t1, t3, 0b0010'0000),
                    _mm256_permute2f128_pd(t0, t2, 0b0010'0000),
                    _mm256_permute2f128_pd(t1, t3, 0b0011'0001)
                );
            } else if constexpr (std::is_same_v<value_t<Column>, double> && Size == 4) {
                __m256d c0 = a.entry(0).m, c1 = a.entry(1).m,
                        c2 = a.entry(2).m, c3 = a.entry(3).m;

                __m256d t3 = _mm256_shuffle_pd(c2, c3, 0b0000),
                        t2 = _mm256_shuffle_pd(c2, c3, 0b1111),
                        t1 = _mm256_shuffle_pd(c0, c1, 0b0000),
                        t0 = _mm256_shuffle_pd(c0, c1, 0b1111);

                return Array(
                    _mm256_permute2f128_pd(t1, t3, 0b0010'0000),
                    _mm256_permute2f128_pd(t0, t2, 0b0010'0000),
                    _mm256_permute2f128_pd(t1, t3, 0b0011'0001),
                    _mm256_permute2f128_pd(t0, t2, 0b0011'0001)
                );
            }
        #endif

        #if defined(DRJIT_ARM_NEON)
            if constexpr (std::is_same_v<value_t<Column>, float> && Size == 3) {
                float32x4x2_t v01 = vtrnq_f32(a.entry(0).m, a.entry(1).m);
                float32x4x2_t v23 = vtrnq_f32(a.entry(2).m, a.entry(2).m);

                return Array(
                    vcombine_f32(vget_low_f32 (v01.val[0]), vget_low_f32 (v23.val[0])),
                    vcombine_f32(vget_low_f32 (v01.val[1]), vget_low_f32 (v23.val[1])),
                    vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]))
                );
            } else if constexpr (std::is_same_v<value_t<Column>, float> && Size == 4) {
                float32x4x2_t v01 = vtrnq_f32(a.entry(0).m, a.entry(1).m);
                float32x4x2_t v23 = vtrnq_f32(a.entry(2).m, a.entry(3).m);

                return Array(
                    vcombine_f32(vget_low_f32 (v01.val[0]), vget_low_f32 (v23.val[0])),
                    vcombine_f32(vget_low_f32 (v01.val[1]), vget_low_f32 (v23.val[1])),
                    vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0])),
                    vcombine_f32(vget_high_f32(v01.val[1]), vget_high_f32(v23.val[1]))
                );
            }
        #endif
    }

    Array result;
    for (size_t i = 0; i < Size; ++i)
        for (size_t j = 0; j < Size; ++j)
            result.entry(i, j) = a.entry(j, i);

    return result;
}

template <typename T> T det(const Matrix<T, 1> &m) {
    return m(0, 0);
}

template <typename T> Matrix<T, 1> inverse(const Matrix<T, 1> &m) {
    return rcp(m(0, 0));
}

template <typename T> Matrix<T, 1> inverse_transpose(const Matrix<T, 1> &m) {
    return rcp(m(0, 0));
}

template <typename T> T det(const Matrix<T, 2> &m) {
    return fmsub(m(0, 0), m(1, 1), m(0, 1) * m(1, 0));
}

template <typename T> Matrix<T, 2> inverse(const Matrix<T, 2> &m) {
    T inv_det = rcp(det(m));
    return Matrix<T, 2>(
        m(1, 1) * inv_det, -m(0, 1) * inv_det,
       -m(1, 0) * inv_det,  m(0, 0) * inv_det
    );
}

template <typename T> Matrix<T, 2> inverse_transpose(const Matrix<T, 2> &m) {
    T inv_det = rcp(det(m));
    return Matrix<T, 2>(
        m(1, 1) * inv_det, -m(1, 0) * inv_det,
       -m(0, 1) * inv_det,  m(0, 0) * inv_det
    );
}

template <typename T> T det(const Matrix<T, 3> &m) {
    return dot(m.entry(0), cross(m.entry(1), m.entry(2)));
}

template <typename T> Matrix<T, 3> inverse_transpose(const Matrix<T, 3> &m) {
    using Vector = Array<T, 3>;

    Vector col0 = m.entry(0),
           col1 = m.entry(1),
           col2 = m.entry(2);

    Vector row0 = cross(col1, col2),
           row1 = cross(col2, col0),
           row2 = cross(col0, col1);

    T inv_det = rcp(dot(col0, row0));
    return Matrix<T, 3>(
        row0 * inv_det,
        row1 * inv_det,
        row2 * inv_det
    );
}

template <typename T> Matrix<T, 3> inverse(const Matrix<T, 3> &m) {
    return transpose(inverse_transpose(m));
}

template <typename T> T det(const Matrix<T, 4> &m) {
    using Vector = Array<T, 4>;

    Vector col0 = m.entry(0), col1 = m.entry(1),
           col2 = m.entry(2), col3 = m.entry(3);

    col1 = shuffle<2, 3, 0, 1>(col1);
    col3 = shuffle<2, 3, 0, 1>(col3);

    Vector temp, row0;

    temp = shuffle<1, 0, 3, 2>(col2 * col3);
    row0 = col1 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    row0 = fmsub(col1, temp, row0);

    temp = shuffle<1, 0, 3, 2>(col1 * col2);
    row0 = fmadd(col3, temp, row0);
    temp = shuffle<2, 3, 0, 1>(temp);
    row0 = fnmadd(col3, temp, row0);

    col1 = shuffle<2, 3, 0, 1>(col1);
    col2 = shuffle<2, 3, 0, 1>(col2);
    temp = shuffle<1, 0, 3, 2>(col1 * col3);
    row0 = fmadd(col2, temp, row0);
    temp = shuffle<2, 3, 0, 1>(temp);
    row0 = fnmadd(col2, temp, row0);

    return dot(col0, row0);
}


template <typename T> Matrix<T, 4> inverse_transpose(const Matrix<T, 4> &m) {
    using Vector = Array<T, 4>;

    Vector col0 = m.entry(0), col1 = m.entry(1),
           col2 = m.entry(2), col3 = m.entry(3);

    col1 = shuffle<2, 3, 0, 1>(col1);
    col3 = shuffle<2, 3, 0, 1>(col3);

    Vector temp, row0, row1, row2, row3;

    temp = shuffle<1, 0, 3, 2>(col2 * col3);
    row0 = col1 * temp;
    row1 = col0 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    row0 = fmsub(col1, temp, row0);
    row1 = shuffle<2, 3, 0, 1>(fmsub(col0, temp, row1));

    temp = shuffle<1, 0, 3, 2>(col1 * col2);
    row0 = fmadd(col3, temp, row0);
    row3 = col0 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    row0 = fnmadd(col3, temp, row0);
    row3 = shuffle<2, 3, 0, 1>(fmsub(col0, temp, row3));

    temp = shuffle<1, 0, 3, 2>(shuffle<2, 3, 0, 1>(col1) * col3);
    col2 = shuffle<2, 3, 0, 1>(col2);
    row0 = fmadd(col2, temp, row0);
    row2 = col0 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    row0 = fnmadd(col2, temp, row0);
    row2 = shuffle<2, 3, 0, 1>(fmsub(col0, temp, row2));

    temp = shuffle<1, 0, 3, 2>(col0 * col1);
    row2 = fmadd(col3, temp, row2);
    row3 = fmsub(col2, temp, row3);
    temp = shuffle<2, 3, 0, 1>(temp);
    row2 = fmsub(col3, temp, row2);
    row3 = fnmadd(col2, temp, row3);

    temp = shuffle<1, 0, 3, 2>(col0 * col3);
    row1 = fnmadd(col2, temp, row1);
    row2 = fmadd(col1, temp, row2);
    temp = shuffle<2, 3, 0, 1>(temp);
    row1 = fmadd(col2, temp, row1);
    row2 = fnmadd(col1, temp, row2);

    temp = shuffle<1, 0, 3, 2>(col0 * col2);
    row1 = fmadd(col3, temp, row1);
    row3 = fnmadd(col1, temp, row3);
    temp = shuffle<2, 3, 0, 1>(temp);
    row1 = fnmadd(col3, temp, row1);
    row3 = fmadd(col1, temp, row3);

    T inv_det = rcp(dot(col0, row0));

    return Matrix<T, 4>(
        row0 * inv_det, row1 * inv_det,
        row2 * inv_det, row3 * inv_det
    );
}

template <typename T> Matrix<T, 4> inverse(const Matrix<T, 4> &m) {
    return transpose(inverse_transpose(m));
}

template <typename T, size_t Size> std::pair<Matrix<T, Size>, Matrix<T, Size>>
polar_decomp(const Matrix<T, Size> &A, size_t it = 10) {
    using PlainArrayType = plain_t<Matrix<T, Size>>;
    Matrix<T, Size> Q = A;
    for (size_t i = 0; i < it; ++i) {
        Matrix<T, Size> Qi = inverse_transpose(Q);
        T gamma = sqrt(frob(Qi) / frob(Q));
        Q = fmadd(PlainArrayType(Q), gamma * 0.5f,
                  PlainArrayType(Qi) * (rcp(gamma) * 0.5f));
    }
    return { Q, transpose(Q) * A };
}

template <typename T> using entry_t = typename T::Entry;

NAMESPACE_END(drjit)
