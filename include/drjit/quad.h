/*
    drjit/quad.h -- Numerical quadrature rules

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>
#include <drjit/math.h>
#include <memory>
#include <utility>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(quad)
NAMESPACE_BEGIN(detail)

/// Evaluate the Legendre polynomial P_l(x) and its derivative
inline std::pair<double, double> legendre_pd(int l, double x) {
    if (l == 0)
        return { 1.0, 0.0 };
    else if (l == 1)
        return { x, 1.0 };

    double l_p_pred = 1.0, l_pred = x, l_cur = 0.0,
           d_p_pred = 0.0, d_pred = 1.0, d_cur = 0.0,
           k0 = 3.0, k1 = 2.0, k2 = 1.0;

    for (int ki = 2; ki <= l; ++ki) {
        l_cur = (k0 * x * l_pred - k2 * l_p_pred) / k1;
        d_cur = d_p_pred + k0 * l_pred;
        l_p_pred = l_pred; l_pred = l_cur;
        d_p_pred = d_pred; d_pred = d_cur;
        k2 = k1; k0 += 2.0; k1 += 1.0;
    }

    return { l_cur, d_cur };
}

/// Evaluate the function P_{l+1}(x) - P_{l-1}(x) and its derivative
inline std::pair<double, double> legendre_pd_diff(int l, double x) {
    if (l == 1)
        return { 0.5 * (3.0 * x * x - 1.0) - 1.0, 3.0 * x };

    double l_p_pred = 1.0, l_pred = x, l_cur = 0.0,
           d_p_pred = 0.0, d_pred = 1.0, d_cur = 0.0,
           k0 = 3.0, k1 = 2.0, k2 = 1.0;

    for (int ki = 2; ki <= l; ++ki) {
        l_cur = (k0 * x * l_pred - k2 * l_p_pred) / k1;
        d_cur = d_p_pred + k0 * l_pred;
        l_p_pred = l_pred; l_pred = l_cur;
        d_p_pred = d_pred; d_pred = d_cur;
        k2 = k1; k0 += 2.0; k1 += 1.0;
    }

    double l_next = (k0 * x * l_pred - k2 * l_p_pred) / k1,
           d_next = d_p_pred + k0 * l_pred;

    return { l_next - l_p_pred, d_next - d_p_pred };
}

/// Convert a double precision host buffer into a Dr.Jit array
template <typename Float> Float load_double(const double *src, size_t n) {
    static_assert(is_dynamic_v<Float> && depth_v<Float> == 1,
                  "Template type must be a dynamic 1D array!");
    using Scalar = scalar_t<Float>;

    if constexpr (std::is_same_v<Scalar, double>) {
        return load<Float>(src, n);
    } else {
        std::unique_ptr<Scalar[]> tmp(new Scalar[n]);
        for (size_t i = 0; i < n; ++i)
            tmp[i] = (Scalar) src[i];
        return load<Float>(tmp.get(), n);
    }
}

NAMESPACE_END(detail)

/**
 * \brief Computes the nodes and weights of a Gauss-Legendre quadrature rule
 * with ``n`` evaluations.
 *
 * Integration is over the interval [-1, 1]. Gauss-Legendre quadrature
 * maximizes the order of exactly integrable polynomials and achieves this up
 * to degree 2n-1.
 *
 * The method is numerically well-behaved until about n=200 and then becomes
 * progressively less accurate. A composite or adaptive scheme is preferable
 * for larger n.
 *
 * The output arrays ``nodes`` and ``weights`` must provide space for ``n``
 * entries each.
 */
inline void gauss_legendre(int n, double *nodes, double *weights) {
    if (n < 1)
        drjit_raise("gauss_legendre(): n must be >= 1");

    if (n == 1) {
        nodes[0] = 0.0;
        weights[0] = 2.0;
        return;
    }

    n--;

    int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        // Initial guess for this root using that of a Chebyshev polynomial
        double x = -cos((double) (2 * i + 1) / (double) (2 * n + 2) * Pi<double>);
        int it = 0;

        while (true) {
            if (++it > 20)
                drjit_raise("gauss_legendre(%i): did not converge after 20 iterations!", n + 1);

            // Search for the interior roots of P_{n+1}(x) using Newton's method.
            std::pair<double, double> L = detail::legendre_pd(n + 1, x);
            double step = L.first / L.second;
            x -= step;

            if (abs(step) <= 4 * abs(x) * Epsilon<double>)
                break;
        }

        std::pair<double, double> L = detail::legendre_pd(n + 1, x);
        weights[i] = weights[n - i] = 2 / ((1 - x * x) * (L.second * L.second));
        nodes[i] = x;
        nodes[n - i] = -x;
    }

    if ((n % 2) == 0) {
        std::pair<double, double> L = detail::legendre_pd(n + 1, 0.0);
        weights[n / 2] = 2.0 / (L.second * L.second);
        nodes[n / 2] = 0.0;
    }
}

/**
 * \brief Computes the nodes and weights of a Gauss-Lobatto quadrature rule
 * with ``n`` evaluations.
 *
 * Integration is over the interval [-1, 1]. Gauss-Lobatto quadrature is
 * preferable to Gauss-Legendre quadrature whenever the endpoints of the
 * integration domain should explicitly be included. It maximizes the order of
 * exactly integrable polynomials subject to this constraint and achieves this
 * up to degree 2n-3.
 *
 * The method is numerically well-behaved until about n=200 and then becomes
 * progressively less accurate. A composite or adaptive scheme is preferable
 * for larger n.
 *
 * The output arrays ``nodes`` and ``weights`` must provide space for ``n``
 * entries each.
 */
inline void gauss_lobatto(int n, double *nodes, double *weights) {
    if (n < 2)
        drjit_raise("gauss_lobatto(): n must be >= 2");

    n--;
    nodes[0] = -1.0;
    nodes[n] = 1.0;
    weights[0] = weights[n] = 2.0 / (double) (n * (n + 1));

    int m = (n + 1) / 2;
    for (int i = 1; i < m; ++i) {
        // Initial guess for this root -- see "On the Legendre-Gauss-Lobatto Points
        // and Weights" by Seymor V. Parter, Journal of Sci. Comp., Vol. 14, 4, 1999
        double x = -cos((i + 0.25) * Pi<double> / n -
                        3 / (8 * n * Pi<double> * (i + 0.25)));
        int it = 0;

        while (true) {
            if (++it > 20)
                drjit_raise("gauss_lobatto(%i): did not converge after 20 iterations!", n + 1);

            // Search for the interior roots of P_n'(x) using Newton's method. The same
            // roots are also shared by P_{n+1}-P_{n-1}, which is nicer to evaluate.
            std::pair<double, double> Q = detail::legendre_pd_diff(n, x);
            double step = Q.first / Q.second;
            x -= step;

            if (abs(step) <= 4 * abs(x) * Epsilon<double>)
                break;
        }

        double l_n = detail::legendre_pd(n, x).first;
        weights[i] = weights[n - i] = 2.0 / ((n * (n + 1)) * l_n * l_n);
        nodes[i] = x;
        nodes[n - i] = -x;
    }

    if ((n % 2) == 0) {
        double l_n = detail::legendre_pd(n, 0.0).first;
        weights[n / 2] = 2.0 / ((n * (n + 1)) * l_n * l_n);
        nodes[n / 2] = 0.0;
    }
}

/**
 * \brief Computes the nodes and weights of a composite Simpson quadrature
 * rule with ``n`` evaluations.
 *
 * Integration is over the interval [-1, 1], which will be split into
 * (n-1)/2 sub-intervals with overlapping endpoints. A 3-point Simpson rule
 * is applied per interval, which is exact for polynomials of degree three or
 * less. The value ``n`` must be odd and at least 3.
 *
 * The output arrays ``nodes`` and ``weights`` must provide space for ``n``
 * entries each.
 */
inline void composite_simpson(int n, double *nodes, double *weights) {
    if (n % 2 != 1 || n < 3)
        drjit_raise("composite_simpson(): n must be >= 3 and odd");

    n = (n - 1) / 2;

    double h = 2.0 / (double) (2 * n),
           weight = h * (1.0 / 3.0);

    for (int i = 0; i < n; ++i) {
        double x = -1 + h * (2 * i);
        nodes[2 * i] = x;
        nodes[2 * i + 1] = x + h;
        weights[2 * i] = (i == 0 ? 1 : 2) * weight;
        weights[2 * i + 1] = 4 * weight;
    }

    nodes[2 * n] = 1.0;
    weights[2 * n] = weight;
}

/**
 * \brief Computes the nodes and weights of a composite Simpson 3/8 quadrature
 * rule with ``n`` evaluations.
 *
 * Integration is over the interval [-1, 1], which will be split into
 * (n-1)/3 sub-intervals with overlapping endpoints. A 4-point Simpson rule
 * is applied per interval, which is exact for polynomials of degree four or
 * less. The value ``n-1`` must be divisible by 3, and ``n`` must be at least 4.
 *
 * The output arrays ``nodes`` and ``weights`` must provide space for ``n``
 * entries each.
 */
inline void composite_simpson_38(int n, double *nodes, double *weights) {
    if ((n - 1) % 3 != 0 || n < 4)
        drjit_raise("composite_simpson_38(): n must be >= 4 and n-1 must be divisible by 3");

    n = (n - 1) / 3;

    double h = 2.0 / (double) (3 * n),
           weight = h * (3.0 / 8.0);

    for (int i = 0; i < n; ++i) {
        double x = -1 + h * (3 * i);
        nodes[3 * i] = x;
        nodes[3 * i + 1] = x + h;
        nodes[3 * i + 2] = x + 2 * h;
        weights[3 * i] = (i == 0 ? 1 : 2) * weight;
        weights[3 * i + 1] = 3 * weight;
        weights[3 * i + 2] = 3 * weight;
    }

    nodes[3 * n] = 1.0;
    weights[3 * n] = weight;
}

/**
 * \brief Computes the ``n`` Chebyshev nodes, i.e., the roots of the Chebyshev
 * polynomial of the first kind of degree ``n``.
 *
 * The output array ``nodes`` must provide space for ``n`` entries. It receives
 * positions on the interval [-1, 1] in increasing order.
 */
inline void chebyshev(int n, double *nodes) {
    if (n < 1)
        drjit_raise("chebyshev(): n must be >= 1");

    for (int i = 0; i < n; ++i)
        nodes[i] = -cos((2 * i + 1) / (double) (2 * n) * Pi<double>);
}

/// Array-valued variant of \ref gauss_legendre()
template <typename Float> std::pair<Float, Float> gauss_legendre(int n) {
    size_t size = (size_t) (n > 0 ? n : 0);
    std::unique_ptr<double[]> nodes(new double[size]), weights(new double[size]);
    gauss_legendre(n, nodes.get(), weights.get());
    return { detail::load_double<Float>(nodes.get(), size),
             detail::load_double<Float>(weights.get(), size) };
}

/// Array-valued variant of \ref gauss_lobatto()
template <typename Float> std::pair<Float, Float> gauss_lobatto(int n) {
    size_t size = (size_t) (n > 0 ? n : 0);
    std::unique_ptr<double[]> nodes(new double[size]), weights(new double[size]);
    gauss_lobatto(n, nodes.get(), weights.get());
    return { detail::load_double<Float>(nodes.get(), size),
             detail::load_double<Float>(weights.get(), size) };
}

/// Array-valued variant of \ref composite_simpson()
template <typename Float> std::pair<Float, Float> composite_simpson(int n) {
    size_t size = (size_t) (n > 0 ? n : 0);
    std::unique_ptr<double[]> nodes(new double[size]), weights(new double[size]);
    composite_simpson(n, nodes.get(), weights.get());
    return { detail::load_double<Float>(nodes.get(), size),
             detail::load_double<Float>(weights.get(), size) };
}

/// Array-valued variant of \ref composite_simpson_38()
template <typename Float> std::pair<Float, Float> composite_simpson_38(int n) {
    size_t size = (size_t) (n > 0 ? n : 0);
    std::unique_ptr<double[]> nodes(new double[size]), weights(new double[size]);
    composite_simpson_38(n, nodes.get(), weights.get());
    return { detail::load_double<Float>(nodes.get(), size),
             detail::load_double<Float>(weights.get(), size) };
}

/// Array-valued variant of \ref chebyshev()
template <typename Float> Float chebyshev(int n) {
    size_t size = (size_t) (n > 0 ? n : 0);
    std::unique_ptr<double[]> nodes(new double[size]);
    chebyshev(n, nodes.get());
    return detail::load_double<Float>(nodes.get(), size);
}

NAMESPACE_END(quad)
NAMESPACE_END(drjit)
