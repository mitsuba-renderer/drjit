/*
    drjit/sphere.h -- Utility functions for 3D spherical geometry

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/array.h>

#pragma once

NAMESPACE_BEGIN(drjit)

//// Convert radians to degrees
template <typename Value> Value rad_to_deg(const Value &a) {
    return a * scalar_t<Value>(180.0 / Pi<double>);
}

/// Convert degrees to radians
template <typename Value> Value deg_to_rad(const Value &a) {
    return a * scalar_t<Value>(Pi<double> / 180.0);
}

/// Spherical coordinate parameterization of the unit sphere
template <typename Value>
Array<Value, 3> sphdir(const Value &theta, const Value &phi) {
    auto [sin_theta, cos_theta] = sincos(theta);
    auto [sin_phi,   cos_phi]   = sincos(phi);

    return {
        cos_phi * sin_theta,
        sin_phi * sin_theta,
        cos_theta
    };
}

/**
 * \brief Numerically well-behaved routine for computing the angle between two
 * normalized 3D direction vectors
 *
 * This should be used wherever one is tempted to compute the angle via
 * ``acos(dot(a, b))``. It yields significantly more accurate results when the
 * angle is close to zero.
 *
 * By Don Hatch at http://www.plunk.org/~hatch/rightway.php
 */
template <typename Vector>
value_t<Vector> unit_angle(const Vector &a, const Vector &b) {
    static_assert(Vector::Size == 3, "unit_angle_z(): input is not a 3D vector");
    using Value = value_t<Vector>;

    Value dot_uv = dot(a, b),
          temp   = 2.f * asin(.5f * norm(b - mulsign(a, dot_uv)));
    return select(dot_uv >= 0, temp, Pi<Value> - temp);
}


/**
 * \brief Numerically well-behaved routine for computing the angle between the
 * normalized 3D direction vector ``v`` and the z-axis ``(0, 0, 1)``
 *
 * This should be used wherever one is tempted to compute the angle via
 * ``acos(v.z())``. It yields significantly more accurate results when the
 * angle is close to zero.
 *
 * By Don Hatch at http://www.plunk.org/~hatch/rightway.php
 */
template <typename Vector> value_t<Vector> unit_angle_z(const Vector &v) {
    static_assert(Vector::Size == 3, "unit_angle_z(): input is not a 3D vector");

    using Value = value_t<Vector>;

    Value temp = asin(.5f * norm(Vector(v.x(), v.y(),
                                        v.z() - mulsign(Value(1.f), v.z())))) * 2.f;

    return select(v.z() >= 0, temp, Pi<Value> - temp);
}

NAMESPACE_END(drjit)
