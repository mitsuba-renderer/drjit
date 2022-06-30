/*
    drjit/quaternion.h -- Quaternion data structure

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/complex.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_>
struct Quaternion : StaticArrayImpl<Value_, 4, false, Quaternion<Value_>> {
    using Base = StaticArrayImpl<Value_, 4, false, Quaternion<Value_>>;
    DRJIT_ARRAY_DEFAULTS(Quaternion)

    static constexpr bool IsQuaternion = true;
    static constexpr bool IsVector = false;

    using ArrayType = Quaternion;
    using PlainArrayType = Array<Value_, 4>;
    using MaskType = Mask<Value_, 4>;

    template <typename T> using ReplaceValue = Quaternion<T>;

    Quaternion() = default;

    template <typename T, enable_if_t<is_quaternion_v<T> || array_depth_v<T> == Base::Depth> = 0>
    DRJIT_INLINE Quaternion(T&& q) : Base(std::forward<T>(q)) { }

    template <typename T, enable_if_t<!is_quaternion_v<T> && array_depth_v<T> != Base::Depth &&
                                       (is_array_v<T> || std::is_scalar_v<std::decay_t<T>>)> = 0>
    DRJIT_INLINE Quaternion(T&& v) : Base(zeros<Value_>(), zeros<Value_>(), zeros<Value_>(), std::forward<T>(v)) { }

    template <typename T, enable_if_t<!is_array_v<T> && !std::is_scalar_v<std::decay_t<T>>> = 0> // __m128, __m256d
    DRJIT_INLINE Quaternion(T&& v) : Base(v) { }

    DRJIT_INLINE Quaternion(const Value_ &vi, const Value_ &vj, const Value_ &vk, const Value_ &vr)
        : Base(vi, vj, vk, vr) { }

    DRJIT_INLINE Quaternion(Value_ &&vi, Value_ &&vj, Value_ &&vk, Value_ &&vr)
        : Base(std::move(vi), std::move(vj), std::move(vk), std::move(vr)) { }

    template <typename Im, typename Re, enable_if_t<array_size_v<Im> == 3> = 0>
    DRJIT_INLINE Quaternion(const Im &im, const Re &re)
        : Base(im.x(), im.y(), im.z(), re) { }

    template <typename T1, typename T2, typename T = Quaternion, enable_if_t<
              array_depth_v<T1> == array_depth_v<T> && array_size_v<T1> == 2 &&
              array_depth_v<T2> == array_depth_v<T> && array_size_v<T2> == 2> = 0>
    Quaternion(const T1 &a1, const T2 &a2)
        : Base(a1, a2) { }
};

template <typename T, enable_if_quaternion_t<T> = 0> T identity(size_t size = 1) {
    using Value = value_t<T>;
    Value z = zeros<Value>(size),
          o = identity<Value>(size);
    return T(z, z, z, o);
}

template <typename T> T real(const Quaternion<T> &q) { return q.entry(3); }
template <typename T> Array<T, 3> imag(const Quaternion<T> &q) { return head<3>(q); }

template <typename T> Quaternion<T> conj(const Quaternion<T> &q) {
    if constexpr (!is_array_v<T>)
        return q ^ Quaternion<T>(-0.f, -0.f, -0.f, 0.f);
    else
        return { -q.x(), -q.y(), -q.z(), q.w() };
}

template <typename T0, typename T1, typename T = expr_t<T0, T1>>
T dot(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    using Base = Array<T, 4>;
    return dot(Base(q0), Base(q1));
}

template <typename T> T squared_norm(const Quaternion<T> &q) {
    return dot(q, q);
}

template <typename T0, typename T1,
          typename Value = expr_t<T0, T1>, typename Result = Quaternion<Value>>
Result operator*(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    using Base = Array<Value, 4>;
    Base a0 = q0, a1 = q1;

    Base t1 = fmadd(shuffle<0, 1, 2, 0>(a0), shuffle<3, 3, 3, 0>(a1),
                    shuffle<1, 2, 0, 1>(a0) * shuffle<2, 0, 1, 1>(a1));
    Base t2 = fmsub(shuffle<3, 3, 3, 3>(a0), a1,
                    shuffle<2, 0, 1, 2>(a0) * shuffle<1, 2, 0, 2>(a1));

    if constexpr (!is_array_v<Value>)
        t1 ^= Base(0.f, 0.f, 0.f, -0.f);
    else
        t1.w() = -t1.w();

    return t1 + t2;
}

template <typename T0, typename T1>
Quaternion<expr_t<T0, T1>> operator*(const Quaternion<T0> &q0, const T1 &v1) {
    return Array<T0, 4>(q0) * v1;
}

template <typename T0, typename T1>
Quaternion<expr_t<T0, T1>> operator*(const T0 &v0, const Quaternion<T1> &q1) {
    return v0 * Array<T1, 4>(q1);
}

template <typename T> Quaternion<T> rcp(const Quaternion<T> &q) {
    return conj(q) * rcp(squared_norm(q));
}

template <typename T0, typename T1>
Quaternion<expr_t<T0, T1>> operator/(const Quaternion<T0> &q0, const Quaternion<T1> &q1) {
    return q0 * rcp(q1);
}

template <typename T0, typename T1>
Quaternion<expr_t<T0, T1>> operator/(const Quaternion<T0> &q0, const T1 &v1) {
    return Array<T0, 4>(q0) / v1;
}

template <typename T> T abs(const Quaternion<T> &z) {
    return norm(z);
}

template <typename T> Quaternion<T> exp(const Quaternion<T> &q) {
    Array<T, 3> qi = imag(q);

    T ri    = norm(qi),
      exp_w = exp(real(q));

    auto [s, c] = sincos(ri);

    return { qi * (s * exp_w / ri), c * exp_w };
}

template <typename T> Quaternion<T> log(const Quaternion<T> &q) {
    Array<T, 3> qi_n = normalize(imag(q));

    T rq      = norm(q),
      acos_rq = acos(real(q) / rq),
      log_rq  = log(rq);

    return { qi_n * acos_rq, log_rq };
}

template <typename T0, typename T1>
Quaternion<expr_t<T0, T1>> pow(const Quaternion<T0> &q0,
                               const Quaternion<T1> &q1) {
    return exp(log(q0) * q1);
}

template <typename T> Quaternion<T> sqrt(const Quaternion<T> &q) {
    T ri = norm(imag(q));
    Complex<T> cs = sqrt(Complex<T>(real(q), ri));
    return { imag(q) * (rcp(ri) * imag(cs)), real(cs) };
}

template <typename Matrix, typename T>
Matrix quat_to_matrix(const Quaternion<T> &q_) {
    Quaternion<T> q = q_ * SqrtTwo<T>;

    T xx = q.x() * q.x(), yy = q.y() * q.y(), zz = q.z() * q.z(),
      xy = q.x() * q.y(), xz = q.x() * q.z(), yz = q.y() * q.z(),
      xw = q.x() * q.w(), yw = q.y() * q.w(), zw = q.z() * q.w();

    if constexpr (Matrix::Size == 4) {
        return Matrix(
             1.f - (yy + zz), xy - zw, xz + yw, 0.f,
             xy + zw, 1.f - (xx + zz), yz - xw, 0.f,
             xz - yw, yz + xw, 1.f - (xx + yy), 0.f,
             0.f, 0.f, 0.f, 1.f
        );
    } else if constexpr (Matrix::Size == 3) {
        return Matrix(
             1.f - (yy + zz), xy - zw, xz + yw,
             xy + zw, 1.f - (xx + zz), yz - xw,
             xz - yw,  yz + xw, 1.f - (xx + yy)
        );
    } else {
        static_assert(detail::false_v<Matrix>, "Invalid matrix size!");
    }
}

template <typename Value, size_t Size>
Quaternion<Value> matrix_to_quat(const Matrix<Value, Size> &m) {
    static_assert(Size == 3 || Size == 4, "Invalid matrix size!");
    using Mask = mask_t<Value>;
    using Quat = Quaternion<Value>;

    /* Converting a Rotation Matrix to a Quaternion
       - Mike Day, Insomniac Games */
    Value o = 1.f;
    Value t0(o + m(0, 0) - m(1, 1) - m(2, 2));
    Quat q0(t0, m(1, 0) + m(0, 1), m(0, 2) + m(2, 0), m(2, 1) - m(1, 2));

    Value t1(o - m(0, 0) + m(1, 1) - m(2, 2));
    Quat q1(m(1, 0) + m(0, 1), t1, m(2, 1) + m(1, 2), m(0, 2) - m(2, 0));

    Value t2(o - m(0, 0) - m(1, 1) + m(2, 2));
    Quat q2(m(0, 2) + m(2, 0), m(2, 1) + m(1, 2), t2, m(1, 0) - m(0, 1));

    Value t3(o + m(0, 0) + m(1, 1) + m(2, 2));
    Quat q3(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1), t3);

    Mask mask0 = m(0, 0) > m(1, 1);
    Value t01 = select(mask0, t0, t1);
    Quat q01 = select(mask0, q0, q1);

    Mask mask1 = m(0, 0) < -m(1, 1);
    Value t23 = select(mask1, t2, t3);
    Quat q23 = select(mask1, q2, q3);

    Mask mask2 = m(2, 2) < 0.f;
    Value t0123 = select(mask2, t01, t23);
    Quat q0123 = select(mask2, q01, q23);

    return q0123 * (rsqrt(t0123) * .5f);
}


template <typename Value>
Array<Value, 3> quat_to_euler(const Quaternion<Value> &q) {
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    // Clamp the result to stay in the valid range for asin
    Value sinp = clamp(2 * fmsub(q.w(), q.y(), q.z() * q.x()), -1.0, 1.0);
    Mask gimbal_lock = abs(sinp) > (1.0f - 5e-8f);

    // roll (x-axis rotation)
    Value q_y_2 = sqr(q.y());
    Value sinr_cosp = 2 * fmadd(q.w(), q.x(), q.y() * q.z());
    Value cosr_cosp = fnmadd(2, fmadd(q.x(), q.x(), q_y_2), 1);
    Value roll = select(gimbal_lock, 2.0f * atan2(q.x(), q.w()), atan2(sinr_cosp, cosr_cosp));

    // pitch (y-axis rotation)
    Value pitch = select(gimbal_lock, copysign(0.5f * Pi<Value>, sinp), asin(sinp));

    // yaw (z-axis rotation)
    Value siny_cosp = 2 * fmadd(q.w(), q.z(), q.x() * q.y());
    Value cosy_cosp = fnmadd(2, fmadd(q.z(), q.z(), q_y_2), 1);
    Value yaw = select(gimbal_lock, 0.f, atan2(siny_cosp, cosy_cosp));

    return Array<Value, 3>(roll, pitch, yaw);
}

template <typename Value>
Quaternion<Value> euler_to_quat(const Array<Value, 3> &a) {
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Array<Value, 3> angles = a / 2.0f;
    auto [sr, cr] = sincos(angles.x);
    auto [sp, cp] = sincos(angles.y);
    auto [sy, cy] = sincos(angles.z);

    Value w = cr*cp*cy + sr*sp*sy;
    Value x = sr*cp*cy - cr*sp*sy;
    Value y = cr*sp*cy + sr*cp*sy;
    Value z = cr*cp*sy - sr*sp*cy;
    return Quaternion<Value>(x, y, z, w);
}

template <typename Value>
Quaternion<Value> slerp(const Quaternion<Value> &q0,
                        const Quaternion<Value> &q1_,
                        const Value &t) {
    using Base = Array<Value, 4>;

    Value cos_theta = dot(q0, q1_);
    Quaternion<Value> q1 = mulsign(Base(q1_), Base(cos_theta));
    cos_theta = mulsign(cos_theta, cos_theta);

    Value theta = acos(cos_theta);
    auto [s, c] = sincos(theta * t);

    Quaternion<Value> qperp  = normalize(q1 - q0 * cos_theta);

    return select(
        cos_theta > 0.9995f,
        normalize(q0 * (1.f - t) + q1 * t),
        q0 * c + qperp * s
    );
}

template <typename Quat, typename Vector3, enable_if_quaternion_t<Quat> = 0>
Quat rotate(const Vector3 &axis, const value_t<Quat> &angle) {
    auto [s, c] = sincos(angle * .5f);
    return Quat(axis * s, c);
}

template <typename T, typename Stream>
DRJIT_NOINLINE Stream &operator<<(Stream &os, const Quaternion<T> &q) {
    if constexpr (is_array_v<T>) {
        os << "[";
        size_t size = q.x().size();
        for (size_t i = 0; i < size; ++i) {
            os << Quaternion<typename T::Value>(q.x().entry(i), q.y().entry(i),
                                                q.z().entry(i), q.w().entry(i));
            if (i + 1 < size)
                os << ",\n ";
        }
        os << "]";
    } else {
        os << q.w();
        os << (q.x() < 0 ? " - " : " + ") << abs(q.x()) << "i";
        os << (q.y() < 0 ? " - " : " + ") << abs(q.y()) << "j";
        os << (q.z() < 0 ? " - " : " + ") << abs(q.z()) << "k";
    }
    return os;
}

NAMESPACE_END(drjit)
