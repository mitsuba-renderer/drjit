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

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning (disable: 4702) // unreachable code
#endif

NAMESPACE_BEGIN(drjit)

template <typename Value_, size_t Size_>
struct Matrix : StaticArrayImpl<Array<Value_, Size_>, Size_, false,
                                Matrix<Value_, Size_>> {
    using Row = Array<Value_, Size_>;
    using Base = StaticArrayImpl<Row, Size_, false, Matrix<Value_, Size_>>;
    using Base::entry;
    using Base::Size;
    DRJIT_ARRAY_DEFAULTS(Matrix)

    static constexpr bool IsMatrix = true;
    static constexpr bool IsSpecial = true;
    static constexpr bool IsVector = false;

    using ArrayType = Matrix;
    using PlainArrayType = Array<Array<Value_, Size>, Size>;
    using MaskType = Mask<mask_t<Row>, Size_>;
    using Entry = Value_;

    template <typename T> using ReplaceValue = Matrix<value_t<T>, Size_>;

    Matrix() = default;

    template <typename T, enable_if_t<is_matrix_v<T> || depth_v<T> == Base::Depth> = 0>
    DRJIT_INLINE Matrix(T&& m) {
        constexpr size_t ArgSize = size_v<T>;
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
                Row col = zeros<Row>();
                col.entry(i) = 1;
                entry(i) = col;
            }
        }
    }

    template <typename T, enable_if_t<!is_matrix_v<T> && depth_v<T> != Base::Depth> = 0>
    DRJIT_INLINE Matrix(T&& v) : Base(zeros<Value_>()) {
        for (size_t i = 0; i < Size; ++i)
            entry(i, i) = v;
    }

    /// Initialize the matrix from a list of rows
    template <typename... Args, enable_if_t<sizeof...(Args) == Size_ &&
              std::conjunction_v<std::is_constructible<Row, Args>...>> = 0>
    DRJIT_INLINE Matrix(const Args&... args) : Base(args...) { }

    /// Initialize the matrix from a list of entries in row-major order
    template <typename... Args, enable_if_t<sizeof...(Args) == Size_ * Size_ &&
              std::conjunction_v<std::is_constructible<Value_, Args>...>> = 0>
    DRJIT_INLINE Matrix(const Args&... args) {
        Value_ values[sizeof...(Args)] = { Value_(args)... };
        for (size_t i = 0; i < Size; ++i)
            for (size_t j = 0; j < Size; ++j)
                entry(i, j) = values[i * Size + j];
    }

    /// Return a reference to the (i, j) element
    DRJIT_INLINE Value_& operator()(size_t i, size_t j) { return entry(i, j); }

    /// Return a reference to the (i, j) element (const)
    DRJIT_INLINE const Value_ &operator()(size_t i, size_t j) const { return entry(i, j); }
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
    using Row = value_t<Result>;

    Result result;

    for (size_t i = 0; i < Size; ++i) {
        Row row = m0(i, 0) * m1.entry(0);
        for (size_t j = 1; j < Size; ++j)
            row = fmadd(m0(i, j), m1.entry(j), row);
        result.entry(i) = row;
    }

    return result;
}

template <typename T0, typename T1, size_t Size,
          enable_if_t<!is_matrix_v<T1>> = 0>
auto operator*(const Matrix<T0, Size> &m0, const T1 &a1) {
    if constexpr (is_vector_v<T1> && size_v<T1> == Size) {
        using Result = Array<expr_t<T0, value_t<T1>>, Size>;

        if constexpr (Size == 4 && is_packed_array_v<T1> &&
                      std::is_same_v<T0, float> &&
                      std::is_same_v<value_t<T1>, float>) {
            #if defined(DRJIT_X86_SSE42)
                __m128 v = a1.m,
                       r0 = _mm_mul_ps(m0.entry(0).m, v),
                       r1 = _mm_mul_ps(m0.entry(1).m, v),
                       r2 = _mm_mul_ps(m0.entry(2).m, v),
                       r3 = _mm_mul_ps(m0.entry(3).m, v),
                       s01 = _mm_hadd_ps(r0, r1),
                       s23 = _mm_hadd_ps(r2, r3),
                       s = _mm_hadd_ps(s01, s23);
                return Result(s);
            #endif
        }

        Matrix<T0, Size> t = transpose(m0);
        Result result = t.entry(0) * full<Result>(a1.entry(0));
        for (size_t i = 1; i < Size; ++i)
            result = fmadd(t.entry(i), full<Result>(a1.entry(i)), result);

        return result;
    } else {
        using Value = expr_t<T0, T1>;
        using Result = Matrix<Value, Size>;
        using PlainArrayType = plain_t<Result>;
        return Result(PlainArrayType(m0) * full<PlainArrayType>(Value(a1)));
    }
}

template <typename T0, typename T1, typename T2, size_t Size>
Matrix<expr_t<T0, T1, T2>, Size> fmadd(const Matrix<T0, Size> &m0,
                                       const Matrix<T1, Size> &m1,
                                       const Matrix<T2, Size> &m2) {
    using Result = Matrix<expr_t<T0, T1, T2>, Size>;
    using Row = value_t<Result>;

    Result result;

    for (size_t i = 0; i < Size; ++i) {
        Row row = m2.entry(i);
        for (size_t j = 0; j < Size; ++j)
            row = fmadd(m0(i, j), m1.entry(j), row);
        result.entry(i) = row;
    }

    return result;
}

template <typename T0, typename T1, typename T2, size_t Size>
auto fmadd(const Matrix<T0, Size> &m0, const T1 &a1, const T2 &a2) {
    if constexpr (is_vector_v<T1> && size_v<T1> == Size && is_vector_v<T2> && size_v<T2> == Size) {
        Matrix<T0, Size> t = transpose(m0);
        Array<expr_t<T0, value_t<T1>, value_t<T2>>, Size> result = a2;
        for (size_t i = 0; i < Size; ++i)
            result = fmadd(t.entry(i), a1.entry(i), result);
        return result;

    } else {
        return m0 * a1 + a2;
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
    Array<Value, Size> result = square(m.entry(0));
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
    using Row = value_t<Array>;
    constexpr size_t Size = size_v<Array>;

    static_assert(depth_v<Array> >= 2 && Size == size_v<Row>,
                  "Array must be a square matrix!");

    if constexpr (Row::IsPacked) {
        #if defined(DRJIT_X86_SSE42)
            if constexpr (std::is_same_v<value_t<Row>, float> && Size == 3) {
                __m128 r0 = a.entry(0).m, r1 = a.entry(1).m, r2 = a.entry(2).m;

                __m128 t0 = _mm_unpacklo_ps(r0, r1);
                __m128 t1 = _mm_unpacklo_ps(r2, r2);
                __m128 t2 = _mm_unpackhi_ps(r0, r1);
                __m128 t3 = _mm_unpackhi_ps(r2, r2);

                return Array(
                    _mm_movelh_ps(t0, t1),
                    _mm_movehl_ps(t1, t0),
                    _mm_movelh_ps(t2, t3)
                );
            } else if constexpr (std::is_same_v<value_t<Row>, float> && Size == 4) {
                __m128 r0 = a.entry(0).m, r1 = a.entry(1).m,
                       r2 = a.entry(2).m, r3 = a.entry(3).m;

                __m128 t0 = _mm_unpacklo_ps(r0, r1);
                __m128 t1 = _mm_unpacklo_ps(r2, r3);
                __m128 t2 = _mm_unpackhi_ps(r0, r1);
                __m128 t3 = _mm_unpackhi_ps(r2, r3);

                return Array(
                    _mm_movelh_ps(t0, t1),
                    _mm_movehl_ps(t1, t0),
                    _mm_movelh_ps(t2, t3),
                    _mm_movehl_ps(t3, t2)
                );
            }
        #endif

        #if defined(DRJIT_X86_AVX)
            if constexpr (std::is_same_v<value_t<Row>, double> && Size == 3) {
                __m256d r0 = a.entry(0).m, r1 = a.entry(1).m, r2 = a.entry(2).m;

                __m256d t3 = _mm256_shuffle_pd(r2, r2, 0b0000),
                        t2 = _mm256_shuffle_pd(r2, r2, 0b1111),
                        t1 = _mm256_shuffle_pd(r0, r1, 0b0000),
                        t0 = _mm256_shuffle_pd(r0, r1, 0b1111);

                return Array(
                    _mm256_permute2f128_pd(t1, t3, 0b0010'0000),
                    _mm256_permute2f128_pd(t0, t2, 0b0010'0000),
                    _mm256_permute2f128_pd(t1, t3, 0b0011'0001)
                );
            } else if constexpr (std::is_same_v<value_t<Row>, double> && Size == 4) {
                __m256d r0 = a.entry(0).m, r1 = a.entry(1).m,
                        r2 = a.entry(2).m, r3 = a.entry(3).m;

                __m256d t3 = _mm256_shuffle_pd(r2, r3, 0b0000),
                        t2 = _mm256_shuffle_pd(r2, r3, 0b1111),
                        t1 = _mm256_shuffle_pd(r0, r1, 0b0000),
                        t0 = _mm256_shuffle_pd(r0, r1, 0b1111);

                return Array(
                    _mm256_permute2f128_pd(t1, t3, 0b0010'0000),
                    _mm256_permute2f128_pd(t0, t2, 0b0010'0000),
                    _mm256_permute2f128_pd(t1, t3, 0b0011'0001),
                    _mm256_permute2f128_pd(t0, t2, 0b0011'0001)
                );
            }
        #endif

        #if defined(DRJIT_ARM_NEON)
            if constexpr (std::is_same_v<value_t<Row>, float> && Size == 3) {
                float32x4x2_t v01 = vtrnq_f32(a.entry(0).m, a.entry(1).m);
                float32x4x2_t v23 = vtrnq_f32(a.entry(2).m, a.entry(2).m);

                return Array(
                    vcombine_f32(vget_low_f32 (v01.val[0]), vget_low_f32 (v23.val[0])),
                    vcombine_f32(vget_low_f32 (v01.val[1]), vget_low_f32 (v23.val[1])),
                    vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]))
                );
            } else if constexpr (std::is_same_v<value_t<Row>, float> && Size == 4) {
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

    Vector row0 = m.entry(0),
           row1 = m.entry(1),
           row2 = m.entry(2);

    Vector col0 = cross(row1, row2),
           col1 = cross(row2, row0),
           col2 = cross(row0, row1);

    T inv_det = rcp(dot(row0, col0));
    return Matrix<T, 3>(
        col0 * inv_det,
        col1 * inv_det,
        col2 * inv_det
    );
}

template <typename T> Matrix<T, 3> inverse(const Matrix<T, 3> &m) {
    return transpose(inverse_transpose(m));
}

template <typename T> T det(const Matrix<T, 4> &m) {
    using Vector = Array<T, 4>;

    Vector row0 = m.entry(0), row1 = m.entry(1),
           row2 = m.entry(2), row3 = m.entry(3);

    row1 = shuffle<2, 3, 0, 1>(row1);
    row3 = shuffle<2, 3, 0, 1>(row3);

    Vector temp, col0;

    temp = shuffle<1, 0, 3, 2>(row2 * row3);
    col0 = row1 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    col0 = fmsub(row1, temp, col0);

    temp = shuffle<1, 0, 3, 2>(row1 * row2);
    col0 = fmadd(row3, temp, col0);
    temp = shuffle<2, 3, 0, 1>(temp);
    col0 = fnmadd(row3, temp, col0);

    row1 = shuffle<2, 3, 0, 1>(row1);
    row2 = shuffle<2, 3, 0, 1>(row2);
    temp = shuffle<1, 0, 3, 2>(row1 * row3);
    col0 = fmadd(row2, temp, col0);
    temp = shuffle<2, 3, 0, 1>(temp);
    col0 = fnmadd(row2, temp, col0);

    return dot(row0, col0);
}


template <typename T> Matrix<T, 4> inverse_transpose(const Matrix<T, 4> &m) {
    using Vector = Array<T, 4>;

    Vector row0 = m.entry(0), row1 = m.entry(1),
           row2 = m.entry(2), row3 = m.entry(3);

    row1 = shuffle<2, 3, 0, 1>(row1);
    row3 = shuffle<2, 3, 0, 1>(row3);

    Vector temp, col0, col1, col2, col3;

    temp = shuffle<1, 0, 3, 2>(row2 * row3);
    col0 = row1 * temp;
    col1 = row0 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    col0 = fmsub(row1, temp, col0);
    col1 = shuffle<2, 3, 0, 1>(fmsub(row0, temp, col1));

    temp = shuffle<1, 0, 3, 2>(row1 * row2);
    col0 = fmadd(row3, temp, col0);
    col3 = row0 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    col0 = fnmadd(row3, temp, col0);
    col3 = shuffle<2, 3, 0, 1>(fmsub(row0, temp, col3));

    temp = shuffle<1, 0, 3, 2>(shuffle<2, 3, 0, 1>(row1) * row3);
    row2 = shuffle<2, 3, 0, 1>(row2);
    col0 = fmadd(row2, temp, col0);
    col2 = row0 * temp;
    temp = shuffle<2, 3, 0, 1>(temp);
    col0 = fnmadd(row2, temp, col0);
    col2 = shuffle<2, 3, 0, 1>(fmsub(row0, temp, col2));

    temp = shuffle<1, 0, 3, 2>(row0 * row1);
    col2 = fmadd(row3, temp, col2);
    col3 = fmsub(row2, temp, col3);
    temp = shuffle<2, 3, 0, 1>(temp);
    col2 = fmsub(row3, temp, col2);
    col3 = fnmadd(row2, temp, col3);

    temp = shuffle<1, 0, 3, 2>(row0 * row3);
    col1 = fnmadd(row2, temp, col1);
    col2 = fmadd(row1, temp, col2);
    temp = shuffle<2, 3, 0, 1>(temp);
    col1 = fmadd(row2, temp, col1);
    col2 = fnmadd(row1, temp, col2);

    temp = shuffle<1, 0, 3, 2>(row0 * row2);
    col1 = fmadd(row3, temp, col1);
    col3 = fnmadd(row1, temp, col3);
    temp = shuffle<2, 3, 0, 1>(temp);
    col1 = fnmadd(row3, temp, col1);
    col3 = fmadd(row1, temp, col3);

    T inv_det = rcp(dot(row0, col0));

    return Matrix<T, 4>(
        col0 * inv_det, col1 * inv_det,
        col2 * inv_det, col3 * inv_det
    );
}

template <typename T> Matrix<T, 4> inverse(const Matrix<T, 4> &m) {
    return transpose(inverse_transpose(m));
}

template <typename T, size_t Size> Matrix<T, Size> rcp(const Matrix<T, Size> &m) {
    return inverse(m);
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

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
