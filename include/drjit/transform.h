/*
    drjit/transform.h -- 3D homogeneous coordinate transformations

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/matrix.h>
#include <drjit/quaternion.h>
#include <tuple>

NAMESPACE_BEGIN(drjit)

template <typename Matrix>
Matrix translate(const Array<entry_t<Matrix>, array_size_v<Matrix> - 1> &v) {
    Matrix trafo = identity<Matrix>();
    trafo.entry(array_size_v<Matrix> - 1) = concat(v, Array<entry_t<Matrix>, 1>(1));
    return trafo;
}

template <typename Matrix>
Matrix scale(const Array<entry_t<Matrix>, array_size_v<Matrix> - 1> &v) {
    return diag(concat(v, Array<entry_t<Matrix>, 1>(1)));
}

template <typename Matrix,
          enable_if_t<is_matrix_v<Matrix> && array_size_v<Matrix> == 3> = 0>
Matrix rotate(const entry_t<Matrix> &angle) {
    entry_t<Matrix> z(0.f), o(1.f);
    auto [s, c] = sincos(angle);
    return Matrix(c, -s, z, s, c, z, z, z, o);
}

template <typename Matrix,
          enable_if_t<is_matrix_v<Matrix> && array_size_v<Matrix> == 4> = 0>
Matrix rotate(const Array<entry_t<Matrix>, 3> &axis,
              const entry_t<Matrix> &angle) {
    using Value = entry_t<Matrix>;
    using Vector3 = Array<Value, 3>;
    using Vector4 = Array<Value, 4>;

    auto [sin_theta, cos_theta] = sincos(angle);
    Value cos_theta_m = 1.f - cos_theta;

    Vector3 shuf1 = shuffle<1, 2, 0>(axis),
            shuf2 = shuffle<2, 0, 1>(axis),
            tmp0  = fmadd(axis * axis, cos_theta_m, cos_theta),
            tmp1  = fmadd(axis * shuf1, cos_theta_m, shuf2 * sin_theta),
            tmp2  = fmsub(axis * shuf2, cos_theta_m, shuf1 * sin_theta);

    return Matrix(
        Vector4(tmp0.x(), tmp1.x(), tmp2.x(), 0.f),
        Vector4(tmp2.y(), tmp0.y(), tmp1.y(), 0.f),
        Vector4(tmp1.z(), tmp2.z(), tmp0.z(), 0.f),
        Vector4(0.f, 0.f, 0.f, 1.f)
    );
}

template <typename Matrix>
Matrix perspective(const entry_t<Matrix> &fov,
                   const entry_t<Matrix> &near_,
                   const entry_t<Matrix> &far_,
                   const entry_t<Matrix> &aspect = 1.f) {
    using Value = entry_t<Matrix>;

    static_assert(
        array_size_v<Matrix> == 4,
        "Matrix::perspective(): implementation assumes 4x4 matrix output");

    Value recip = rcp(near_ - far_);
    Value c = cot(.5f * fov);

    Matrix trafo = diag<Matrix>(
        value_t<Matrix>(c / aspect, c, (near_ + far_) * recip, 0.f));

    trafo(2, 3) = 2.f * near_ * far_ * recip;
    trafo(3, 2) = -1.f;

    return trafo;
}

template <typename Matrix>
Matrix frustum(const entry_t<Matrix> &left,
               const entry_t<Matrix> &right,
               const entry_t<Matrix> &bottom,
               const entry_t<Matrix> &top,
               const entry_t<Matrix> &near_,
               const entry_t<Matrix> &far_) {
    using Value = entry_t<Matrix>;

    static_assert(
        is_matrix_v<Matrix> && array_size_v<Matrix> == 4,
        "Matrix::frustum(): template argument must be of type Matrix<T, 4>");

    Value rl = rcp(right - left),
          tb = rcp(top - bottom),
          fn = rcp(far_ - near_);

    Matrix trafo = zeros<Matrix>();
    trafo(0, 0) = (2.f * near_) * rl;
    trafo(1, 1) = (2.f * near_) * tb;
    trafo(0, 2) = (right + left) * rl;
    trafo(1, 2) = (top + bottom) * tb;
    trafo(2, 2) = -(far_ + near_) * fn;
    trafo(3, 2) = -1.f;
    trafo(2, 3) = -2.f * far_ * near_ * fn;

    return trafo;
}

template <typename Matrix>
Matrix ortho(const entry_t<Matrix> &left,
             const entry_t<Matrix> &right,
             const entry_t<Matrix> &bottom,
             const entry_t<Matrix> &top,
             const entry_t<Matrix> &near_,
             const entry_t<Matrix> &far_) {
    using Value = entry_t<Matrix>;

    static_assert(
        is_matrix_v<Matrix> && array_size_v<Matrix> == 4,
        "Matrix::ortho(): template argument must be of type Matrix<T, 4>");

    Value rl = rcp(right - left),
          tb = rcp(top - bottom),
          fn = rcp(far_ - near_);

    Matrix trafo = zeros<Matrix>();

    trafo(0, 0) = 2.f * rl;
    trafo(1, 1) = 2.f * tb;
    trafo(2, 2) = -2.f * fn;
    trafo(3, 3) = 1.f;
    trafo(0, 3) = -(right + left) * rl;
    trafo(1, 3) = -(top + bottom) * tb;
    trafo(2, 3) = -(far_ + near_) * fn;

    return trafo;
}

template <typename Matrix>
Matrix look_at(const Array<entry_t<Matrix>, 3> &origin,
               const Array<entry_t<Matrix>, 3> &target,
               const Array<entry_t<Matrix>, 3> &up) {
    static_assert(
        is_matrix_v<Matrix> && array_size_v<Matrix> == 4,
        "Matrix::look_at(): template argument must be of type Matrix<T, 4>");

    using Value = entry_t<Matrix>;
    using Vector3 = Array<Value, 3>;
    using Vector4 = Array<Value, 4>;

    Vector3 dir = normalize(target - origin);
    Vector3 left = normalize(cross(dir, up));
    Vector3 new_up = cross(left, dir);

    Array<Value, 1> z(0);

    return Matrix(
        concat(left, z),
        concat(new_up, z),
        concat(-dir, z),
        Vector4(
            -dot(left, origin),
            -dot(new_up, origin),
             dot(dir, origin),
             1.f
        )
    );
}

template <typename Value>
std::tuple<Matrix<Value, 3>, Quaternion<Value>, Array<Value, 3>>
transform_decompose(const Matrix<Value, 4> &a, size_t it = 10) {
    auto [Q, P] = polar_decomp(Matrix<Value, 3>(a), it);
    Q[isnan(Q(0, 0))] = identity<Matrix<Value, 3>>();

    Value sign_q = det(Q);
    Q = mulsign(Q, sign_q);
    P = mulsign(P, sign_q);

    return std::make_tuple(P, matrix_to_quat(Q), head<3>(a.entry(3)));
}

template <typename Matrix4>
Matrix4 transform_compose(const Matrix<entry_t<Matrix4>, 3> &s,
                          const Quaternion<entry_t<Matrix4>> &q,
                          const Array<entry_t<Matrix4>, 3> &t) {
    static_assert(
        is_matrix_v<Matrix4> && array_size_v<Matrix4> == 4,
        "Matrix::transform_compose(): template argument must be of type Matrix<T, 4>");

    using Value = entry_t<Matrix4>;
    Matrix4 result(quat_to_matrix<Matrix<Value, 3>>(q) * s);
    result.entry(3) = concat(t, Array<Value, 1>(1));
    return result;
}

template <typename Matrix4>
Matrix4 transform_compose_inverse(const Matrix<entry_t<Matrix4>, 3> &s,
                                  const Quaternion<entry_t<Matrix4>> &q,
                                  const Array<entry_t<Matrix4>, 3> &t) {
    static_assert(
        is_matrix_v<Matrix4> && array_size_v<Matrix4> == 4,
        "Matrix::transform_compose_inverse(): template argument must be of type Matrix<T, 4>");

    using Value = entry_t<Matrix4>;
    Matrix<Value, 3> inv_m = inverse(quat_to_matrix<Matrix<Value, 3>>(q) * s);
    Matrix4 result(inv_m);
    result.entry(3) = concat(inv_m * -t, Array<Value, 1>(1));
    return result;
}

NAMESPACE_END(drjit)
