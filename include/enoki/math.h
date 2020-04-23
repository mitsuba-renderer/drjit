/*
    enoki/math.h -- Mathematical support library

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <enoki/array.h>

#pragma once

NAMESPACE_BEGIN(enoki)

// -----------------------------------------------------------------------
//! @{ \name Polynomial evaluation with short dependency chains and
//           fused multply-adds based on Estrin's scheme
// -----------------------------------------------------------------------

template <typename T>
ENOKI_INLINE T poly2(const T &x, double c0, double c1, double c2) {
    using S = scalar_t<T>;
    T x2 = sqr(x);
    return fmadd(x2, S(c2), fmadd(x, S(c1), S(c0)));
}

template <typename T>
ENOKI_INLINE T poly3(const T &x, double c0, double c1, double c2, double c3) {
    using S = scalar_t<T>;
    T x2 = sqr(x);
    return fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)));
}

template <typename T>
ENOKI_INLINE T poly4(const T &x, double c0, double c1, double c2, double c3,
                     double c4) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2);
    return fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)) + S(c4) * x4);
}

template <typename T>
ENOKI_INLINE T poly5(const T &x, double c0, double c1, double c2, double c3,
                     double c4, double c5) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2);
    return fmadd(x2, fmadd(x, S(c3), S(c2)),
                     fmadd(x4, fmadd(x, S(c5), S(c4)), fmadd(x, S(c1), S(c0))));
}

template <typename T>
ENOKI_INLINE T poly6(const T &x, double c0, double c1, double c2, double c3,
                     double c4, double c5, double c6) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2);
    return fmadd(x4, fmadd(x2, S(c6), fmadd(x, S(c5), S(c4))),
                     fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0))));
}

template <typename T>
ENOKI_INLINE T poly7(const T &x, double c0, double c1, double c2, double c3,
                     double c4, double c5, double c6, double c7) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2);
    return fmadd(x4, fmadd(x2, fmadd(x, S(c7), S(c6)), fmadd(x, S(c5), S(c4))),
                     fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0))));
}

template <typename T>
ENOKI_INLINE T poly8(const T &x, double c0, double c1, double c2, double c3,
                     double c4, double c5, double c6, double c7, double c8) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2), x8 = sqr(x4);
    return fmadd(x4, fmadd(x2, fmadd(x, S(c7), S(c6)), fmadd(x, S(c5), S(c4))),
                     fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)) + S(c8) * x8));
}

template <typename T>
ENOKI_INLINE T poly9(const T &x, double c0, double c1, double c2, double c3,
                     double c4, double c5, double c6, double c7, double c8,
                     double c9) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2), x8 = sqr(x4);
    return fmadd(x8, fmadd(x, S(c9), S(c8)),
                     fmadd(x4, fmadd(x2, fmadd(x, S(c7), S(c6)), fmadd(x, S(c5), S(c4))),
                               fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)))));
}

template <typename T>
ENOKI_INLINE T poly10(const T &x, double c0, double c1, double c2, double c3,
                      double c4, double c5, double c6, double c7, double c8,
                      double c9, double c10) {
    using S = scalar_t<T>;
    T x2 = sqr(x), x4 = sqr(x2), x8 = sqr(x4);
    return fmadd(x8, fmadd(x2, S(c10), fmadd(x, S(c9), S(c8))),
                     fmadd(x4, fmadd(x2, fmadd(x, S(c7), S(c6)), fmadd(x, S(c5), S(c4))),
                               fmadd(x2, fmadd(x, S(c3), S(c2)), fmadd(x, S(c1), S(c0)))));
}

//! @}
// -----------------------------------------------------------------------

namespace detail {
    template <bool Sin, bool Cos, typename Value>
    ENOKI_INLINE void sincos(const Value &x, Value *s_out, Value *c_out) {
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        using IntArray = int_array_t<Value>;
        using Int = scalar_t<IntArray>;
        using Mask = mask_t<Value>;
        ENOKI_MARK_USED(s_out);
        ENOKI_MARK_USED(c_out);

        /* Joint sine & cosine function approximation based on CEPHES.
           Excellent accuracy in the domain |x| < 8192

           Redistributed under a BSD license with permission of the author, see
           https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

         - sin (in [-8192, 8192]):
           * avg abs. err = 6.61896e-09
           * avg rel. err = 1.37888e-08
              -> in ULPs  = 0.166492
           * max abs. err = 5.96046e-08
             (at x=-8191.31)
           * max rel. err = 1.76826e-06
             -> in ULPs   = 19
             (at x=-6374.29)

         - cos (in [-8192, 8192]):
           * avg abs. err = 6.59965e-09
           * avg rel. err = 1.37432e-08
              -> in ULPs  = 0.166141
           * max abs. err = 5.96046e-08
             (at x=-8191.05)
           * max rel. err = 3.13993e-06
             -> in ULPs   = 47
             (at x=-6199.93)
        */

        Value xa = abs(x);

        // Scale by 4/Pi and get the integer part
        IntArray j(xa * Scalar(1.2732395447351626862));

        // Map zeros to origin; if (j & 1) j += 1
        j = (j + Int(1)) & Int(~1u);

        // Cast back to a floating point value
        Value y(j);

        // Determine sign of result
        Value sign_sin, sign_cos;
        constexpr size_t Shift = sizeof(Scalar) * 8 - 3;

        if constexpr (Sin)
            sign_sin = detail::xor_(reinterpret_array<Value>(sl<Shift>(j)), x);

        if constexpr (Cos)
            sign_cos = reinterpret_array<Value>(sl<Shift>(~(j - Int(2))));

        // Extended precision modular arithmetic
        if constexpr (Single) {
            y = xa - y * Scalar(0.78515625)
                   - y * Scalar(2.4187564849853515625e-4)
                   - y * Scalar(3.77489497744594108e-8);
        } else {
            y = xa - y * Scalar(7.85398125648498535156e-1)
                   - y * Scalar(3.77489470793079817668e-8)
                   - y * Scalar(2.69515142907905952645e-15);
        }

        Value z = sqr(y), s, c;
        z = detail::or_(z, eq(xa, Infinity<Value>));

        if constexpr (Single) {
            s = poly2(z, -1.6666654611e-1,
                          8.3321608736e-3,
                         -1.9515295891e-4) * z;

            c = poly2(z,  4.166664568298827e-2,
                         -1.388731625493765e-3,
                          2.443315711809948e-5) * z;
        } else {
            s = poly5(z, -1.66666666666666307295e-1,
                          8.33333333332211858878e-3,
                         -1.98412698295895385996e-4,
                          2.75573136213857245213e-6,
                         -2.50507477628578072866e-8,
                          1.58962301576546568060e-10) * z;

            c = poly5(z,  4.16666666666665929218e-2,
                         -1.38888888888730564116e-3,
                          2.48015872888517045348e-5,
                         -2.75573141792967388112e-7,
                          2.08757008419747316778e-9,
                         -1.13585365213876817300e-11) * z;
        }

        s = fmadd(s, y, y);
        c = fmadd(c, z, fmadd(z, Scalar(-0.5), Scalar(1)));

        Mask polymask = eq(j & Int(2), zero<IntArray>());

        if constexpr (Sin)
            *s_out = mulsign(select(polymask, s, c), sign_sin);

        if constexpr (Cos)
            *c_out = mulsign(select(polymask, c, s), sign_cos);
    }
};

namespace detail {
    #define ENOKI_DETECTOR(name)                                          \
        template <typename T>                                             \
        using has_##name = decltype(std::declval<T>().name##_());         \
        template <typename T>                                             \
        constexpr bool has_##name##_v = is_detected_v<has_##name, T>;

    ENOKI_DETECTOR(sin)
    ENOKI_DETECTOR(cos)
    ENOKI_DETECTOR(sincos)
}

template <typename Value> ENOKI_INLINE Value sin(const Value &value) {
    if constexpr (detail::has_sin_v<Value>) {
        return value.sin_();
    } else {
        Value result;
        detail::sincos<true, false>(value, &result, (Value *) nullptr);
        return result;
    }
}

template <typename Value> ENOKI_INLINE Value cos(const Value &value) {
    if constexpr (detail::has_cos_v<Value>) {
        return value.cos_();
    } else {
        Value result;
        detail::sincos<false, true>(value, (Value *) nullptr, &result);
        return result;
    }
}

template <typename Value> ENOKI_INLINE std::pair<Value, Value> sincos(const Value &value) {
    if constexpr (detail::has_sincos_v<Value>) {
        return value.sincos_();
    } else if constexpr (detail::has_sin_v<Value> && detail::has_cos_v<Value>) {
        return { value.sin_(), value.cos_()} ;
    } else {
        Value result_s, result_c;
        detail::sincos<true, true>(value, &result_s, &result_c);
        return { result_s, result_c };
    }
}

NAMESPACE_END(enoki)
