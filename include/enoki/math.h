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

    template <bool Tan, typename Value>
    ENOKI_INLINE auto tancot(const Value &x) {
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        using IntArray = int_array_t<Value>;
        using Int = scalar_t<IntArray>;

        /*
         - tan (in [-8192, 8192]):
           * avg abs. err = 4.63693e-06
           * avg rel. err = 3.60191e-08
              -> in ULPs  = 0.435442
           * max abs. err = 0.8125
             (at x=-6199.93)
           * max rel. err = 3.12284e-06
             -> in ULPs   = 30
             (at x=-7406.3)
        */

        Value xa = abs(x);

        // Scale by 4/Pi and get the integer part
        IntArray j(xa * Scalar(1.2732395447351626862));

        // Map zeros to origin; if (j & 1) j += 1
        j = (j + Int(1)) & Int(~1u);

        // Cast back to a floating point value
        Value y(j);

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

        Value z = y * y;
        z |= eq(xa, Infinity<Scalar>);

        Value r;
        if constexpr (Single) {
            r = poly5(z, 3.33331568548e-1,
                         1.33387994085e-1,
                         5.34112807005e-2,
                         2.44301354525e-2,
                         3.11992232697e-3,
                         9.38540185543e-3);
        } else {
            r = poly2(z, -1.79565251976484877988e7,
                          1.15351664838587416140e6,
                         -1.30936939181383777646e4) /
                poly4(z, -5.38695755929454629881e7,
                          2.50083801823357915839e7,
                         -1.32089234440210967447e6,
                          1.36812963470692954678e4,
                          1.00000000000000000000e0);
        }

        r = fmadd(r, z * y, y);

        auto recip_mask = Tan ? neq(j & Int(2), Int(0)) :
                                 eq(j & Int(2), Int(0));
        masked(r, xa < Scalar(1e-4)) = y;
        masked(r, recip_mask) = rcp(r);

        Value sign =
            reinterpret_array<Value>(sl<sizeof(Scalar) * 8 - 2>(j)) ^ x;

        return mulsign(r, sign);
    }
};

namespace detail {
    #define ENOKI_DETECTOR(name)                                          \
        template <typename T>                                             \
        using has_##name = decltype(std::declval<T>().name##_());

    ENOKI_DETECTOR(sin)
    ENOKI_DETECTOR(cos)
    ENOKI_DETECTOR(sincos)
    ENOKI_DETECTOR(exp)
    ENOKI_DETECTOR(log)
}

template <typename Value> ENOKI_INLINE Value sin(const Value &x) {
    if constexpr (is_detected_v<detail::has_sin, Value>) {
        return x.sin_();
    } else {
        Value result;
        detail::sincos<true, false>(x, &result, (Value *) nullptr);
        return result;
    }
}

template <typename Value> ENOKI_INLINE Value cos(const Value &x) {
    if constexpr (is_detected_v<detail::has_cos, Value>) {
        return x.cos_();
    } else {
        Value result;
        detail::sincos<false, true>(x, (Value *) nullptr, &result);
        return result;
    }
}

template <typename Value> ENOKI_INLINE std::pair<Value, Value> sincos(const Value &x) {
    if constexpr (is_detected_v<detail::has_sincos, Value>) {
        return x.sincos_();
    } else {
        Value result_s, result_c;
        detail::sincos<true, true>(x, &result_s, &result_c);
        return { result_s, result_c };
    }
}

template <typename Value> Value csc(const Value &x) { return rcp(sin(x)); }
template <typename Value> Value sec(const Value &x) { return rcp(cos(x)); }

template <typename Value> ENOKI_INLINE Value tan(const Value &x) {
    return detail::tancot<true>(x);
}

template <typename Value> ENOKI_INLINE Value cot(const Value &x) {
    return detail::tancot<false>(x);
}

template <typename Value> ENOKI_INLINE Value asin(const Value &x) {
    /*
       Arc sine function approximation based on CEPHES.

     - asin (in [-1, 1]):
       * avg abs. err = 2.25422e-08
       * avg rel. err = 2.85777e-08
          -> in ULPs  = 0.331032
       * max abs. err = 1.19209e-07
         (at x=-0.999998)
       * max rel. err = 2.27663e-07
         -> in ULPs   = 2
         (at x=-0.841416)
    */

    using Scalar = scalar_t<Value>;
    using Mask = mask_t<Value>;
    constexpr bool Single = std::is_same_v<Scalar, float>;

    Value xa          = abs(x),
          x2          = sqr(x),
          r;

    if constexpr (Single) {
        Mask mask_big = xa > Scalar(0.5);

        Value x1 = Scalar(0.5) * (Scalar(1) - xa);
        Value x3 = select(mask_big, x1, x2);
        Value x4 = select(mask_big, sqrt(x1), xa);

        Value z1 = poly4(x3, 1.6666752422e-1f,
                             7.4953002686e-2f,
                             4.5470025998e-2f,
                             2.4181311049e-2f,
                             4.2163199048e-2f);

        z1 = fmadd(z1, x3*x4, x4);

        r = select(mask_big, Scalar(.5f * Pi<Scalar>) - (z1 + z1), z1);
    } else {
        Mask mask_big = xa > Scalar(0.625);

        if (any_nested_or<true>(mask_big)) {
            const Scalar pio4 = Scalar(0.78539816339744830962);
            const Scalar more_bits = Scalar(6.123233995736765886130e-17);

            /* arcsin(1-x) = pi/2 - sqrt(2x)(1+R(x))  */
            Value zz = Scalar(1) - xa;
            Value p = poly4(zz, 2.853665548261061424989e1,
                               -2.556901049652824852289e1,
                                6.968710824104713396794e0,
                               -5.634242780008963776856e-1,
                                2.967721961301243206100e-3) /
                      poly4(zz, 3.424398657913078477438e2,
                               -3.838770957603691357202e2,
                                1.470656354026814941758e2,
                               -2.194779531642920639778e1,
                                1.000000000000000000000e0) * zz;
            zz = sqrt(zz + zz);
            Value z = pio4 - zz;
            masked(r, mask_big) = z - fmsub(zz, p, more_bits) + pio4;
        }

        if (!all_nested_or<false>(mask_big)) {
            Value z = poly5(x2, -8.198089802484824371615e0,
                                 1.956261983317594739197e1,
                                -1.626247967210700244449e1,
                                 5.444622390564711410273e0,
                                -6.019598008014123785661e-1,
                                 4.253011369004428248960e-3) /
                      poly5(x2, -4.918853881490881290097e1,
                                 1.395105614657485689735e2,
                                -1.471791292232726029859e2,
                                 7.049610280856842141659e1,
                                -1.474091372988853791896e1,
                                 1.000000000000000000000e0) * x2;
            z = fmadd(xa, z, xa);
            z = select(xa < Scalar(1e-8), xa, z);
            masked(r, ~mask_big) = z;
        }
    }

    return copysign(r, x);
}

template <typename Value> ENOKI_INLINE Value acos(const Value &x) {
    /*
       Arc cosine function approximation based on CEPHES.

     - acos (in [-1, 1]):
       * avg abs. err = 4.72002e-08
       * avg rel. err = 2.85612e-08
          -> in ULPs  = 0.33034
       * max abs. err = 2.38419e-07
         (at x=-0.99999)
       * max rel. err = 1.19209e-07
         -> in ULPs   = 1
         (at x=-0.99999)
    */

    using Scalar = scalar_t<Value>;
    using Mask = mask_t<Value>;
    constexpr bool Single = std::is_same_v<Scalar, float>;

    if constexpr (Single) {
        Value xa = abs(x), x2 = sqr(x);

        Mask mask_big = xa > Scalar(0.5);

        Value x1 = Scalar(0.5) * (Scalar(1) - xa);
        Value x3 = select(mask_big, x1, x2);
        Value x4 = select(mask_big, sqrt(x1), xa);

        Value z1 = poly4(x3, 1.666675242e-1f,
                             7.4953002686e-2f,
                             4.5470025998e-2f,
                             2.4181311049e-2f,
                             4.2163199048e-2f);

        z1 = fmadd(z1, x3 * x4, x4);
        Value z2 = z1 + z1;
        z2 = select(x < Scalar(0), Scalar(Pi<Scalar>) - z2, z2);

        Value z3 = Scalar(Pi<Scalar> * .5f) - copysign(z1, x);
        return select(mask_big, z2, z3);
    } else {
        const Scalar pio4 = Scalar(0.78539816339744830962);
        const Scalar more_bits = Scalar(6.123233995736765886130e-17);
        const Scalar h = Scalar(0.5);

        Mask mask = x > h;

        Value y = asin(select(mask, sqrt(fnmadd(h, x, h)), x));
        return select(mask, y + y, pio4 - y + more_bits + pio4);
    }
}

template <typename Y, typename X> ENOKI_INLINE auto atan2(const Y &y, const X &x) {
    if constexpr (!std::is_same_v<X, Y>) {
        using E = expr_t<X, Y>;
        return atan2(static_cast<ref_cast_t<Y, E>>(y),
                     static_cast<ref_cast_t<X, E>>(x));
    } else {
        /*
           MiniMax fit by Wenzel Jakob, May 2016

         - atan2() tested via atan() (in [-1, 1]):
           * avg abs. err = 1.81543e-07
           * avg rel. err = 4.15224e-07
              -> in ULPs  = 4.9197
           * max abs. err = 5.96046e-07
             (at x=-0.976062)
           * max rel. err = 7.73931e-07
             -> in ULPs   = 12
             (at x=-0.015445)
        */

        using Value = X;
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        Value x_ = y,
              y_ = x,
              abs_x      = abs(x_),
              abs_y      = abs(y_),
              min_val    = min(abs_y, abs_x),
              max_val    = max(abs_x, abs_y),
              scaled_min = min_val / max_val,
              z          = sqr(scaled_min);

        // How to find these:
        // f[x_] = MiniMaxApproximation[ArcTan[Sqrt[x]]/Sqrt[x],
        //         {x, {1/10000, 1}, 6, 0}, WorkingPrecision->20][[2, 1]]

        Value t;
        if constexpr (Single) {
            t = poly6(z, 0.99999934166683966009,
                        -0.33326497518773606976,
                        +0.19881342388439013552,
                        -0.13486708938456973185,
                        +0.083863120428809689910,
                        -0.037006525670417265220,
                         0.0078613793713198150252);
        } else {
            t = poly6(z, 9.9999999999999999419e-1,
                         2.50554429737833465113e0,
                         2.28289058385464073556e0,
                         9.20960512187107069075e-1,
                         1.59189681028889623410e-1,
                         9.35911604785115940726e-3,
                         8.07005540507283419124e-5) /
                poly6(z, 1.00000000000000000000e0,
                         2.83887763071166519407e0,
                         3.02918312742541450749e0,
                         1.50576983803701596773e0,
                         3.49719171130492192607e-1,
                         3.29968942624402204199e-2,
                         8.26619391703564168942e-4);
        }

        t = t * scaled_min;

        t = select(abs_y > abs_x, Pi<Scalar> * Scalar(.5f) - t, t);
        t = select(x_ < zero<Value>(), Pi<Scalar> - t, t);
        Value r = select(y_ < zero<Value>(), -t, t);
        r &= neq(max_val, Scalar(0));
        return r;
    }
}

template <typename Value> ENOKI_INLINE Value atan(const Value &x) {
    return atan2(x, Value(1));
}

NAMESPACE_END(enoki)
