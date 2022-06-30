/*
    drjit/math.h -- Mathematical support library

    This file contains a number of ported implementations of transcendental
    functions based on the excellent CEPHES library by Stephen Mosher. They are
    redistributed under a BSD license with permission of the author, see
    https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/array.h>

#pragma once

NAMESPACE_BEGIN(drjit)

NAMESPACE_BEGIN(detail)

template <typename Value, size_t n>
DRJIT_INLINE Value estrin_impl(const Value &x, const Value (&coeff)[n]) {
    constexpr size_t n_rec = (n - 1) / 2, n_fma = n / 2;

    Value coeff_rec[n_rec + 1];
    DRJIT_UNROLL for (size_t i = 0; i < n_fma; ++i)
        coeff_rec[i] = fmadd(x, coeff[2 * i + 1], coeff[2 * i]);

    if constexpr (n_rec == n_fma) // odd case
        coeff_rec[n_rec] = coeff[n - 1];

    if constexpr (n_rec == 0)
        return coeff_rec[0];
    else
        return estrin_impl(sqr(x), coeff_rec);
}

template <typename Value, size_t n>
DRJIT_INLINE Value horner_impl(const Value &x, const Value (&coeff)[n]) {
    Value accum = coeff[n - 1];
    DRJIT_UNROLL for (size_t i = 1; i < n; ++i)
        accum = fmadd(x, accum, coeff[n - 1 - i]);
    return accum;
}

NAMESPACE_END(detail)

// Estrin's scheme for polynomial evaluation
template <typename Value, typename... Ts>
DRJIT_INLINE Value estrin(const Value &x, Ts... ts) {
    Value coeffs[] { scalar_t<Value>(ts)... };
    return detail::estrin_impl(x, coeffs);
}

// Horner's scheme for polynomial evaluation
template <typename Value, typename... Ts>
DRJIT_INLINE Value horner(const Value &x, Ts... ts) {
    Value coeffs[] { scalar_t<Value>(ts)... };
    return detail::horner_impl(x, coeffs);
}

//! @
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Trigonometric and inverse trigonometric functions
// -----------------------------------------------------------------------

namespace detail {
    template <bool Sin, bool Cos, typename Value>
    DRJIT_INLINE void sincos(const Value &x, Value *s_out, Value *c_out) {
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        using IntArray = int_array_t<Value>;
        using Int = scalar_t<IntArray>;
        using Mask = mask_t<Value>;
        DRJIT_MARK_USED(s_out);
        DRJIT_MARK_USED(c_out);
        static_assert(!is_special_v<Value>,
                      "sincos(): requires a regular scalar/array argument!");
        static_assert(std::is_floating_point_v<Scalar>,
                      "sin()/cos(): function requires a floating point argument!");

        /* Joint sine & cosine function approximation based on CEPHES.
           Excellent accuracy in the domain |x| < 8192

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
        IntArray j = IntArray(xa * Scalar(1.2732395447351626862));

        // Map zeros to origin; if (j & 1) j += 1
        j = (j + Int(1)) & Int(~1u);

        // Cast back to a floating point value
        Value y = Value(j);

        // Determine sign of result
        Value sign_sin, sign_cos;
        constexpr size_t Shift = sizeof(Scalar) * 8 - 3;

        if constexpr (Sin)
            sign_sin = detail::xor_(reinterpret_array<Value>(sl<Shift>(j)), x);

        if constexpr (Cos)
            sign_cos = reinterpret_array<Value>(sl<Shift>(~(j - Int(2))));

        DRJIT_MARK_USED(sign_sin);
        DRJIT_MARK_USED(sign_cos);

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
            s = estrin(z, -1.6666654611e-1,
                           8.3321608736e-3,
                          -1.9515295891e-4) * z;

            c = estrin(z,  4.166664568298827e-2,
                          -1.388731625493765e-3,
                           2.443315711809948e-5) * z;
        } else {
            s = estrin(z, -1.66666666666666307295e-1,
                           8.33333333332211858878e-3,
                          -1.98412698295895385996e-4,
                           2.75573136213857245213e-6,
                          -2.50507477628578072866e-8,
                           1.58962301576546568060e-10) * z;

            c = estrin(z,  4.16666666666665929218e-2,
                          -1.38888888888730564116e-3,
                           2.48015872888517045348e-5,
                          -2.75573141792967388112e-7,
                           2.08757008419747316778e-9,
                          -1.13585365213876817300e-11) * z;
        }

        s = fmadd(s, y, y);
        c = fmadd(c, z, fmadd(z, Scalar(-0.5), Scalar(1)));

        Mask polymask = eq(j & Int(2), zeros<IntArray>());

        if constexpr (Sin)
            *s_out = mulsign(select(polymask, s, c), sign_sin);

        if constexpr (Cos)
            *c_out = mulsign(select(polymask, c, s), sign_cos);
    }

    template <bool Tan, typename Value>
    DRJIT_INLINE auto tancot(const Value &x) {
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        using IntArray = int_array_t<Value>;
        using Int = scalar_t<IntArray>;
        static_assert(std::is_floating_point_v<Scalar>,
                      "tan()/cot(): function requires a floating point argument!");
        static_assert(!is_special_v<Value>,
                      "tancot(): requires a regular scalar/array argument!");

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
        IntArray j = IntArray(xa * Scalar(1.2732395447351626862));

        // Map zeros to origin; if (j & 1) j += 1
        j = (j + Int(1)) & Int(~1u);

        // Cast back to a floating point value
        Value y = Value(j);

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
        z = detail::or_(z, eq(xa, Infinity<Scalar>));

        Value r;
        if constexpr (Single) {
            r = estrin(z, 3.33331568548e-1,
                          1.33387994085e-1,
                          5.34112807005e-2,
                          2.44301354525e-2,
                          3.11992232697e-3,
                          9.38540185543e-3);
        } else {
            r = estrin(z, -1.79565251976484877988e7,
                           1.15351664838587416140e6,
                          -1.30936939181383777646e4) /
                estrin(z, -5.38695755929454629881e7,
                           2.50083801823357915839e7,
                          -1.32089234440210967447e6,
                           1.36812963470692954678e4,
                           1.00000000000000000000e0);
        }

        r = fmadd(r, z * y, y);

        auto recip_mask = Tan ? neq(j & Int(2), Int(0)) :
                                 eq(j & Int(2), Int(0));
        drjit::masked(r, xa < Scalar(1e-4)) = y;
        drjit::masked(r, recip_mask) = rcp(r);

        Value sign = detail::xor_(
            reinterpret_array<Value>(sl<sizeof(Scalar) * 8 - 2>(j)), x);

        return mulsign(r, sign);
    }
};

namespace detail {
    #define DRJIT_DETECTOR(name)                                               \
        template <typename T> using has_##name = decltype(T().name##_());
    #define DRJIT_DETECTOR_ARG(name)                                           \
        template <typename T> using has_##name = decltype(T().name##_(T()));

    DRJIT_DETECTOR(sin)
    DRJIT_DETECTOR(cos)
    DRJIT_DETECTOR(sincos)
    DRJIT_DETECTOR(csc)
    DRJIT_DETECTOR(sec)
    DRJIT_DETECTOR(tan)
    DRJIT_DETECTOR(cot)
    DRJIT_DETECTOR(asin)
    DRJIT_DETECTOR(acos)
    DRJIT_DETECTOR(atan)
    DRJIT_DETECTOR_ARG(atan2)

    DRJIT_DETECTOR(frexp)
    DRJIT_DETECTOR_ARG(ldexp)
    DRJIT_DETECTOR(exp)
    DRJIT_DETECTOR(exp2)
    DRJIT_DETECTOR(log)
    DRJIT_DETECTOR(log2)
    DRJIT_DETECTOR_ARG(pow)

    DRJIT_DETECTOR(sinh)
    DRJIT_DETECTOR(cosh)
    DRJIT_DETECTOR(sincosh)
    DRJIT_DETECTOR(tanh)

    DRJIT_DETECTOR(asinh)
    DRJIT_DETECTOR(acosh)
    DRJIT_DETECTOR(atanh)

    DRJIT_DETECTOR(cbrt)
    DRJIT_DETECTOR(erf)
}

template <typename Value> Value sin(const Value &x) {
    if constexpr (is_detected_v<detail::has_sin, Value>) {
        return x.sin_();
    } else {
        Value result;
        detail::sincos<true, false>(x, &result, (Value *) nullptr);
        return result;
    }
}

template <typename Value> Value cos(const Value &x) {
    if constexpr (is_detected_v<detail::has_cos, Value>) {
        return x.cos_();
    } else {
        Value result;
        detail::sincos<false, true>(x, (Value *) nullptr, &result);
        return result;
    }
}

template <typename Value> std::pair<Value, Value> sincos(const Value &x) {
    if constexpr (is_detected_v<detail::has_sincos, Value>) {
        return x.sincos_();
    } else {
        Value result_s, result_c;
        detail::sincos<true, true>(x, &result_s, &result_c);
        return { result_s, result_c };
    }
}

template <typename Value> Value csc(const Value &x) {
    if constexpr (is_detected_v<detail::has_csc, Value>)
        return x.csc_();
    else
        return rcp(sin(x));
}

template <typename Value> Value sec(const Value &x) {
    if constexpr (is_detected_v<detail::has_sec, Value>)
        return x.sec_();
    else
        return rcp(cos(x));

}

template <typename Value> Value tan(const Value &x) {
    if constexpr (is_detected_v<detail::has_tan, Value>)
        return x.tan_();
    else
        return detail::tancot<true>(x);
}

template <typename Value> Value cot(const Value &x) {
    if constexpr (is_detected_v<detail::has_cot, Value>)
        return x.cot_();
    else
        return detail::tancot<false>(x);
}

template <typename Value> Value asin(const Value &x) {
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

    if constexpr (is_detected_v<detail::has_asin, Value>) {
        return x.asin_();
    } else {
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        static_assert(std::is_floating_point_v<Scalar>,
                      "asin(): function requires a floating point argument!");
        static_assert(!is_special_v<Value>,
                      "asin(): requires a regular scalar/array argument!");

        Value xa = abs(x),
              x2 = sqr(x),
              r;

        if constexpr (Single) {
            Mask mask_big = xa > Scalar(0.5);

            Value x1 = Scalar(0.5) * (Scalar(1) - xa);
            Value x3 = select(mask_big, x1, x2);
            Value x4 = select(mask_big, sqrt(x1), xa);

            Value z1 = estrin(x3, 1.6666752422e-1f,
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
                Value p = estrin(zz, 2.853665548261061424989e1,
                                    -2.556901049652824852289e1,
                                     6.968710824104713396794e0,
                                    -5.634242780008963776856e-1,
                                     2.967721961301243206100e-3) /
                          estrin(zz, 3.424398657913078477438e2,
                                    -3.838770957603691357202e2,
                                     1.470656354026814941758e2,
                                    -2.194779531642920639778e1,
                                     1.000000000000000000000e0) * zz;
                zz = sqrt(zz + zz);
                Value z = pio4 - zz;
                r = z - fmsub(zz, p, more_bits) + pio4;
            }

            if (!all_nested_or<false>(mask_big)) {
                Value z = estrin(x2, -8.198089802484824371615e0,
                                      1.956261983317594739197e1,
                                     -1.626247967210700244449e1,
                                      5.444622390564711410273e0,
                                     -6.019598008014123785661e-1,
                                      4.253011369004428248960e-3) /
                          estrin(x2, -4.918853881490881290097e1,
                                      1.395105614657485689735e2,
                                     -1.471791292232726029859e2,
                                      7.049610280856842141659e1,
                                     -1.474091372988853791896e1,
                                      1.000000000000000000000e0) * x2;
                z = fmadd(xa, z, xa);
                z = select(xa < Scalar(1e-8), xa, z);
                masked(r, !mask_big) = z;
            }
        }

        return copysign(r, x);
    }
}

template <typename Value> Value acos(const Value &x) {
    if constexpr (is_detected_v<detail::has_acos, Value>) {
        return x.acos_();
    } else {
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
        static_assert(std::is_floating_point_v<Scalar>,
                      "acos(): function requires a floating point argument!");
        static_assert(!is_special_v<Value>,
                      "acos(): requires a regular scalar/array argument!");

        if constexpr (Single) {
            Value xa = abs(x), x2 = sqr(x);

            Mask mask_big = xa > Scalar(0.5);

            Value x1 = Scalar(0.5) * (Scalar(1) - xa);
            Value x3 = select(mask_big, x1, x2);
            Value x4 = select(mask_big, sqrt(x1), xa);

            Value z1 = estrin(x3, 1.666675242e-1f,
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
}

template <typename Y, typename X> expr_t<Y, X> atan2(const Y &y, const X &x) {
    if constexpr (!std::is_same_v<X, Y>) {
        using E = expr_t<X, Y>;
        return atan2(static_cast<ref_cast_t<Y, E>>(y),
                     static_cast<ref_cast_t<X, E>>(x));
    } else if constexpr (is_detected_v<detail::has_atan2, Y>) {
        return y.atan2_(x);
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
        static_assert(std::is_floating_point_v<Scalar>,
                      "atan2(): function requires a floating point argument!");
        static_assert(!is_special_v<Value>,
                      "atan2(): requires a regular scalar/array argument!");

        Value abs_x      = abs(x),
              abs_y      = abs(y),
              min_val    = minimum(abs_y, abs_x),
              max_val    = maximum(abs_x, abs_y),
              scaled_min = min_val / max_val,
              z          = sqr(scaled_min);

        // How to find these:
        // f[x_] = MiniMaxApproximation[ArcTan[Sqrt[x]]/Sqrt[x],
        //         {x, {1/10000, 1}, 6, 0}, WorkingPrecision->20][[2, 1]]

        Value t;
        if constexpr (Single) {
            t = estrin(z, 0.99999934166683966009,
                         -0.33326497518773606976,
                         +0.19881342388439013552,
                         -0.13486708938456973185,
                         +0.083863120428809689910,
                         -0.037006525670417265220,
                          0.0078613793713198150252);
        } else {
            t = estrin(z, 9.9999999999999999419e-1,
                          2.50554429737833465113e0,
                          2.28289058385464073556e0,
                          9.20960512187107069075e-1,
                          1.59189681028889623410e-1,
                          9.35911604785115940726e-3,
                          8.07005540507283419124e-5) /
                estrin(z, 1.00000000000000000000e0,
                          2.83887763071166519407e0,
                          3.02918312742541450749e0,
                          1.50576983803701596773e0,
                          3.49719171130492192607e-1,
                          3.29968942624402204199e-2,
                          8.26619391703564168942e-4);
        }

        t = t * scaled_min;

        t = select(abs_y > abs_x, Pi<Scalar> * Scalar(.5f) - t, t);
        t = select(x < zeros<Value>(), Pi<Scalar> - t, t);
        Value r = select(y < zeros<Value>(), -t, t);
        r = detail::and_(r, neq(max_val, Scalar(0)));
        return r;
    }
}

template <typename Value> Value atan(const Value &x) {
    if constexpr (is_detected_v<detail::has_atan, Value>)
        return x.atan_();
    else
        return atan2(x, Value(1));
}

//! @
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Exponential function, logarithm, power
// -----------------------------------------------------------------------

template <typename X, typename Y> expr_t<X, Y> ldexp(const X &x, const Y &y) {
    if constexpr (!std::is_same_v<X, Y>) {
        using E = expr_t<X, Y>;
        return ldexp(static_cast<ref_cast_t<X, E>>(x),
                     static_cast<ref_cast_t<Y, E>>(y));
    } else if constexpr (is_detected_v<detail::has_ldexp, X>) {
        return x.ldexp_(y);
    } else {
        using Scalar = scalar_t<X>;
        static_assert(std::is_floating_point_v<Scalar>,
                      "ldexp(): function requires a floating point argument!");
        static_assert(!is_special_v<X>,
                      "ldexp(): requires a regular scalar/array argument!");
        constexpr bool Single = std::is_same_v<Scalar, float>;
        return x * reinterpret_array<X>(
            sl<Single ? 23 : 52>(int_array_t<X>(int32_array_t<X>(y) + (Single ? 0x7f : 0x3ff))));
    }
}

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4310) // cast truncates constant value
#endif

template <typename Value> std::pair<Value, Value> frexp(const Value &x) {
    if constexpr (is_detected_v<detail::has_frexp, Value>) {
        return x.frexp_();
    } else {
        using Scalar = scalar_t<Value>;
        using IntArray = int_array_t<Value>;
        using Int32Array = int32_array_t<Value>;
        using Int = scalar_t<IntArray>;
        using IntMask = mask_t<IntArray>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        static_assert(std::is_floating_point_v<Scalar>,
                      "frexp(): function requires a floating point argument!");
        static_assert(!is_special_v<Value>,
                      "frexp(): requires a regular scalar/array argument!");

        const IntArray
            exponent_mask(Int(Single ? 0x7f800000ull : 0x7ff0000000000000ull)),
            mantissa_sign_mask(Int(Single ? ~0x7f800000ull : ~0x7ff0000000000000ull)),
            bias(Int(Single ? 0x7f : 0x3ff));

        IntArray xi = reinterpret_array<IntArray>(x);
        IntArray exponent_bits = xi & exponent_mask;

        // Detect zero/inf/NaN
        IntMask is_normal =
            IntMask(neq(x, zeros<Value>())) & neq(exponent_bits, exponent_mask);

        IntArray exponent_i = detail::and_(
            (sr < Single ? 23 : 52 > (exponent_bits)) - bias, is_normal);

        IntArray mantissa = (xi & mantissa_sign_mask) |
                             IntArray(memcpy_cast<Int>(Scalar(.5f)));

        Value exponent;
        if constexpr (Single)
            exponent = Value(exponent_i);
        else
            exponent = Value(Int32Array(exponent_i));

        return std::make_pair(
            reinterpret_array<Value>(select(is_normal, mantissa, xi)),
            exponent
        );
    }
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

template <typename Value> Value exp(const Value &x) {
    if constexpr (is_detected_v<detail::has_exp, Value>) {
        return x.exp_();
    } else {
        /* Exponential function approximation based on CEPHES

         - exp (in [-20, 30]):
           * avg abs. err = 7155.01
           * avg rel. err = 2.35929e-08
              -> in ULPs  = 0.273524
           * max abs. err = 1.04858e+06
             (at x=29.8057)
           * max rel. err = 1.192e-07
             -> in ULPs   = 1
             (at x=-19.9999)
        */

        static_assert(!is_special_v<Value>,
                      "exp(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        using Mask = mask_t<Value>;

        const Scalar range =
            Scalar(Single ? 88.3762588501 : 709.43613930310391424428);

        Mask mask_overflow  = x >  range,
             mask_underflow = x < -range;

        /* Express e^x = e^g 2^n
             = e^g e^(n log(2))
             = e^(g + n log(2)) */
        Value n = floor(fmadd(InvLogTwo<Scalar>, x, Scalar(0.5)));

        // -log(2), most significant bits & remaining bits
        const Scalar nlog2_hi = -Scalar(Single ? 0.693359375 : 0.693145751953125),
                     nlog2_lo = -Scalar(Single ? -2.12194440e-4 : 1.42860682030941723212e-6);

        Value y = x;
        y = fmadd(n, Scalar(nlog2_hi), y);
        y = fmadd(n, Scalar(nlog2_lo), y);

        Value z = sqr(y);

        if constexpr (Single) {
            z = estrin(y, 5.0000001201e-1, 1.6666665459e-1,
                          4.1665795894e-2, 8.3334519073e-3,
                          1.3981999507e-3, 1.9875691500e-4);
            z = fmadd(z, sqr(y), y + Scalar(1));
        } else {
            /* Rational approximation for exponential
               of the fractional part:
                  e^x = 1 + 2y P(y^2) / (Q(y^2) - P(y^2)) */
            Value p = estrin(z, 9.99999999999999999910e-1,
                                3.02994407707441961300e-2,
                                1.26177193074810590878e-4) * y;

            Value q = estrin(z, 2.00000000000000000009e0,
                                2.27265548208155028766e-1,
                                2.52448340349684104192e-3,
                                3.00198505138664455042e-6);

            z = p / (q - p);
            z = z + z + Scalar(1);
        }

        return select(mask_overflow, Infinity<Value>,
                      select(mask_underflow, zeros<Value>(), ldexp(z, n)));
    }
}

template <typename Value> Value exp2(const Value &x) {
    if constexpr (is_detected_v<detail::has_exp2, Value>) {
        return x.exp2_();
    } else {
        /* Base-2 exponential function approximation based on CEPHES

         - exp2 (in [-20, 30]):
           * avg abs. err = 7155.01
           * avg rel. err = 2.35929e-08
              -> in ULPs  = 0.273524
           * max abs. err = 1.04858e+06
             (at x=29.8057)
           * max rel. err = 1.192e-07
             -> in ULPs   = 1
             (at x=-19.9999)
        */

        static_assert(!is_special_v<Value>,
                      "exp2(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;
        using Mask = mask_t<Value>;

        const Scalar inf   = Infinity<Scalar>,
                     range = Scalar(Single ? 127 : 1024);

        Mask mask_overflow  = x >  range,
             mask_underflow = x < -range;

        Value n, y, z;
        if constexpr (Single) {
            // Separate into integer and fractional parts
            n = floor(x);
            y = x - n;

            Mask gth = y > .5f;
            masked(n, gth) += 1.f;
            masked(y, gth) -= 1.f;

            z = estrin(y, 6.931472028550421e-1, 2.402264791363012e-1,
                          5.550332471162809e-2, 9.618437357674640e-3,
                          1.339887440266574e-3, 1.535336188319500e-4);
            z = fmadd(y, z, 1.f);
        } else {
            // Separate into integer and fractional parts
            n = floor(x + .5f);
            y = x - n;

            /* Rational approximation for exponential
               of the fractional part:
                  e^x = 1 + 2y P(y^2) / (Q(y^2) - P(y^2)) */
            Value y2 = sqr(y);
            Value p = estrin(y2, 1.51390680115615096133e+3,
                                 2.02020656693165307700e+1,
                                 2.30933477057345225087e-2) * y;
            Value q = estrin(y2, 4.36821166879210612817e+3,
                                 2.33184211722314911771e+2,
                                 1.00000000000000000000e+0);

            z = p / (q - p);
            z = z + z + 1.f;
        }

        return select(mask_overflow, inf,
                      select(mask_underflow, zeros<Value>(), ldexp(z, n)));
    }
}

template <typename Value> Value log(const Value &x) {
    if constexpr (is_detected_v<detail::has_log, Value>) {
        return x.log_();
    } else {
        /* Logarithm function approximation based on CEPHES

         - log (in [1e-20, 1000]):
           * avg abs. err = 8.8672e-09
           * avg rel. err = 1.57541e-09
              -> in ULPs  = 0.020038
           * max abs. err = 4.76837e-07
             (at x=54.7661)
           * max rel. err = 1.19194e-07
             -> in ULPs   = 1
             (at x=0.021)
        */

        static_assert(!is_special_v<Value>,
                      "log(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        // Catch negative and NaN values
        Mask valid_mask = x >= Scalar(0);

        // Note: does not handle denormalized numbers on some target
        auto [xm, e] = frexp(x);

        Mask mask_ge_inv_sqrt2 = xm >= InvSqrtTwo<Scalar>;

        masked(e, mask_ge_inv_sqrt2) += Scalar(1);
        xm += detail::andnot_(xm, mask_ge_inv_sqrt2) - Scalar(1);

        // Logarithm using log(1+x) = x - .5x**2 + x**3 P(x)
        Value y;
        if constexpr (Single) {
            y = estrin(xm, 3.3333331174e-1, -2.4999993993e-1,
                           2.0000714765e-1, -1.6668057665e-1,
                           1.4249322787e-1, -1.2420140846e-1,
                           1.1676998740e-1, -1.1514610310e-1,
                           7.0376836292e-2);
        } else {
            /// Use a more accurate rational polynomial fit
            y = estrin(xm, 7.70838733755885391666e0,
                           1.79368678507819816313e1,
                           1.44989225341610930846e1,
                           4.70579119878881725854e0,
                           4.97494994976747001425e-1,
                           1.01875663804580931796e-4) /
                estrin(xm, 2.31251620126765340583e1,
                           7.11544750618563894466e1,
                           8.29875266912776603211e1,
                           4.52279145837532221105e1,
                           1.12873587189167450590e1,
                           1.00000000000000000000e0);
        }

        Value z = sqr(xm);
        y *= xm * z;
        y = fmadd(e, Scalar(-2.121944400546905827679e-4), y);

        Value r = xm + fmadd(Scalar(-.5), z, y);
        r = fmadd(e, Scalar(0.693359375), r);


        // Explicit handling of special cases
        const Scalar n_inf(-Infinity<Scalar>),
                     p_inf( Infinity<Scalar>);

        masked(r, eq(x, p_inf)) = p_inf;
        masked(r, eq(x, Scalar(0))) = n_inf;

        return detail::or_(r, !valid_mask);
    }
}

template <typename Value> Value log2(const Value &x) {
    if constexpr (is_detected_v<detail::has_log2, Value>) {
        return x.log2_();
    } else {
        /* Logarithm function approximation based on CEPHES

         - log (in [1e-20, 1000]):
           * avg abs. err = 8.8672e-09
           * avg rel. err = 1.57541e-09
              -> in ULPs  = 0.020038
           * max abs. err = 4.76837e-07
             (at x=54.7661)
           * max rel. err = 1.19194e-07
             -> in ULPs   = 1
             (at x=0.021)
        */

        static_assert(!is_special_v<Value>,
                      "log2(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        // Catch negative and NaN values
        Mask valid_mask = x >= Scalar(0);

        // Note: does not handle denormalized numbers on some target
        auto [xm, e] = frexp(x);

        Mask mask_ge_inv_sqrt2 = xm >= InvSqrtTwo<Scalar>;

        masked(e, mask_ge_inv_sqrt2) += Scalar(1);
        xm += detail::andnot_(xm, mask_ge_inv_sqrt2) - Scalar(1);

        // Logarithm using log(1+x) = x - .5x**2 + x**3 P(x)
        Value y;
        if constexpr (Single) {
            y = estrin(xm, 3.3333331174e-1, -2.4999993993e-1,
                           2.0000714765e-1, -1.6668057665e-1,
                           1.4249322787e-1, -1.2420140846e-1,
                           1.1676998740e-1, -1.1514610310e-1,
                           7.0376836292e-2);
        } else {
            /// Use a more accurate rational polynomial fit
            y = estrin(xm, 7.70838733755885391666e0,
                           1.79368678507819816313e1,
                           1.44989225341610930846e1,
                           4.70579119878881725854e0,
                           4.97494994976747001425e-1,
                           1.01875663804580931796e-4) /
                estrin(xm, 2.31251620126765340583e1,
                           7.11544750618563894466e1,
                           8.29875266912776603211e1,
                           4.52279145837532221105e1,
                           1.12873587189167450590e1,
                           1.00000000000000000000e0);
        }

        Value z = sqr(xm);
        y *= xm * z;

        Value r = xm + fmadd(Scalar(-.5), z, y);
        r = fmadd(r, InvLogTwo<Scalar>, e);

        // Explicit handling of special cases
        const Scalar n_inf(-Infinity<Scalar>),
                     p_inf( Infinity<Scalar>);

        masked(r, eq(x, p_inf)) = p_inf;
        masked(r, eq(x, Scalar(0))) = n_inf;

        return detail::or_(r, !valid_mask);
    }
}

namespace detail {
    template <typename Value> DRJIT_INLINE Value powi(const Value &x_, int y) {
        uint32_t n = (uint32_t) abs(y);
        Value result(1), x(x_);

        while (n) {
            if (n & 1)
                result *= x;
            x *= x;
            n >>= 1;
        }

        if constexpr (std::is_floating_point_v<scalar_t<Value>>)
            return (y >= 0) ? result : rcp(result);
        else
            return result;
    }
}

template <typename X, typename Y> expr_t<X, Y> pow(const X &x, const Y &y) {
    static_assert(!is_special_v<X> && !is_special_v<Y>,
                  "pow(): requires a regular scalar/array argument!");
    if constexpr ((is_dynamic_v<X> && std::is_scalar_v<Y>) || std::is_integral_v<expr_t<X, Y>>) {
        if constexpr (std::is_floating_point_v<Y>) {
            if (detail::round_(y) == y)
                return detail::powi(x, (int) y);
            else
                return pow(x, X(y));
        } else {
            return detail::powi(x, (int) y);
        }
    } else if constexpr (!std::is_same_v<X, Y>) {
        using E = expr_t<X, Y>;
        return pow(static_cast<ref_cast_t<X, E>>(x),
                   static_cast<ref_cast_t<Y, E>>(y));
    } else if constexpr (is_detected_v<detail::has_pow, X>) {
        return x.pow_(y);
    } else {
        return exp2(log2(x) * y);
    }
}

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Hyperbolic and inverse hyperbolic functions
// -----------------------------------------------------------------------

template <typename Value> Value sinh(const Value &x) {
    if constexpr (is_detected_v<detail::has_sinh, Value>) {
        return x.sinh_();
    } else {
        /*
         - sinh (in [-10, 10]):
           * avg abs. err = 2.92524e-05
           * avg rel. err = 2.80831e-08
              -> in ULPs  = 0.336485
           * max abs. err = 0.00195312
             (at x=-9.99894)
           * max rel. err = 2.36862e-07
             -> in ULPs   = 3
             (at x=-9.69866)
        */

        static_assert(!is_special_v<Value>,
                      "sinh(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;


        Value xa = abs(x),
              r_small, r_big;

        Mask mask_big = xa > Scalar(1);

        if (any_nested_or<true>(mask_big)) {
            Value exp0 = exp(x),
                  exp1 = rcp(exp0);

            r_big = (exp0 - exp1) * Scalar(0.5);
        }

        if (!all_nested_or<false>(mask_big)) {
            Value x2 = sqr(x), y;

            if constexpr (Single) {
                y = estrin(x2, 1.66667160211e-1,
                               8.33028376239e-3,
                               2.03721912945e-4);
            } else {
                y = estrin(x2, -3.51754964808151394800e5,
                               -1.15614435765005216044e4,
                               -1.63725857525983828727e2,
                               -7.89474443963537015605e-1) /
                    estrin(x2, -2.11052978884890840399e6,
                                3.61578279834431989373e4,
                               -2.77711081420602794433e2,
                                1.00000000000000000000e0);
            }

            r_small = fmadd(y, x2 * x, x);
        }

        return select(mask_big, r_big, r_small);
    }
}

template <typename Value> Value cosh(const Value &x) {
    if constexpr (is_detected_v<detail::has_cosh, Value>) {
        return x.cosh_();
    } else {
        /*
         - cosh (in [-10, 10]):
           * avg abs. err = 4.17738e-05
           * avg rel. err = 3.15608e-08
              -> in ULPs  = 0.376252
           * max abs. err = 0.00195312
             (at x=-9.99894)
           * max rel. err = 2.38001e-07
             -> in ULPs   = 3
             (at x=-9.70164)
        */
        static_assert(!is_special_v<Value>,
                      "cosh(): requires a regular scalar/array argument!");

        Value exp0 = exp(x),
              exp1 = rcp(exp0);

        return (exp0 + exp1) * scalar_t<Value>(.5f);
    }
}

template <typename Value> std::pair<Value, Value> sincosh(const Value &x) {
    if constexpr (is_detected_v<detail::has_sincosh, Value>) {
        return x.sincosh_();
    } else {
        static_assert(!is_special_v<Value>,
                      "sincosh(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;


        Value xa = abs(x),
              exp0 = exp(x),
              exp1 = rcp(exp0),
              r_small, r_big;

        Mask mask_big = xa > Scalar(1);

        if (any_nested_or<true>(mask_big))
            r_big = (exp0 - exp1) * Scalar(0.5);

        if (!all_nested_or<false>(mask_big)) {
            Value x2 = sqr(x), y;

            if constexpr (Single) {
                y = estrin(x2, 1.66667160211e-1,
                               8.33028376239e-3,
                               2.03721912945e-4);
            } else {
                y = estrin(x2, -3.51754964808151394800e5,
                               -1.15614435765005216044e4,
                               -1.63725857525983828727e2,
                               -7.89474443963537015605e-1) /
                    estrin(x2, -2.11052978884890840399e6,
                                3.61578279834431989373e4,
                               -2.77711081420602794433e2,
                                1.00000000000000000000e0);
            }

            r_small = fmadd(y, x2 * x, x);
        }

        return { select(mask_big, r_big, r_small),
                 (exp0 + exp1) * Scalar(.5f) };
    }
}

template <typename Value> Value tanh(const Value &x) {
    if constexpr (is_detected_v<detail::has_tanh, Value>) {
        return x.tanh_();
    } else {
        /*
           Hyperbolic tangent function approximation based on CEPHES.

         - tanh (in [-10, 10]):
           * avg abs. err = 4.44655e-08
           * avg rel. err = 4.58074e-08
              -> in ULPs  = 0.698044
           * max abs. err = 3.57628e-07
             (at x=-2.12867)
           * max rel. err = 4.1006e-07
             -> in ULPs   = 6
             (at x=-2.12867)
        */

        static_assert(!is_special_v<Value>,
                      "tanh(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        Value r_big, r_small;

        Mask mask_big = abs(x) >= Scalar(0.625);

        if (any_nested_or<true>(mask_big)) {
            Value e  = exp(x + x),
                  e2 = rcp(e + Scalar(1));
            r_big = Scalar(1) - (e2 + e2);
        }

        if (!all_nested_or<false>(mask_big)) {
            Value x2 = sqr(x);

            if constexpr (Single) {
                r_small = estrin(x2, -3.33332819422e-1,
                                      1.33314422036e-1,
                                     -5.37397155531e-2,
                                      2.06390887954e-2,
                                     -5.70498872745e-3);
            } else {
                r_small = estrin(x2, -1.61468768441708447952e3,
                                     -9.92877231001918586564e1,
                                     -9.64399179425052238628e-1) /
                          estrin(x2,  4.84406305325125486048e3,
                                      2.23548839060100448583e3,
                                      1.12811678491632931402e2,
                                      1.00000000000000000000e0);
            }

            r_small = fmadd(r_small, x2 * x, x);
        }

        return select(mask_big, r_big, r_small);
    }
}

template <typename Value> Value csch(const Value &a) { return rcp(sinh(a)); }
template <typename Value> Value sech(const Value &a) { return rcp(cosh(a)); }
template <typename Value> Value coth(const Value &a) { return rcp(tanh(a)); }

template <typename Value> Value asinh(const Value &x) {
    if constexpr (is_detected_v<detail::has_asinh, Value>) {
        return x.asinh_();
    } else {
        /*
           Hyperbolic arc sine function approximation based on CEPHES.

         - asinh (in [-10, 10]):
           * avg abs. err = 2.75626e-08
           * avg rel. err = 1.51762e-08
              -> in ULPs  = 0.178341
           * max abs. err = 2.38419e-07
             (at x=-10)
           * max rel. err = 1.71857e-07
             -> in ULPs   = 2
             (at x=-1.17457)
        */

        static_assert(!is_special_v<Value>,
                      "asinh(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        Value x2 = sqr(x), xa = abs(x), r_big, r_small;

        Mask mask_big  = xa >= Scalar(Single ? 0.51 : 0.533),
             mask_huge = xa >= Scalar(Single ? 1e10 : 1e20);

        if (!all_nested_or<false>(mask_big)) {
            if constexpr (Single) {
                r_small = estrin(x2, -1.6666288134e-1,
                                      7.4847586088e-2,
                                     -4.2699340972e-2,
                                      2.0122003309e-2);
            } else {
                r_small = estrin(x2, -5.56682227230859640450e0,
                                     -9.09030533308377316566e0,
                                     -4.37390226194356683570e0,
                                     -5.91750212056387121207e-1,
                                     -4.33231683752342103572e-3) /
                          estrin(x2,  3.34009336338516356383e1,
                                      6.95722521337257608734e1,
                                      4.86042483805291788324e1,
                                      1.28757002067426453537e1,
                                      1.00000000000000000000e0);
            }
            r_small = fmadd(r_small, x2 * x, x);
        }

        if (any_nested_or<true>(mask_big)) {
            r_big = log(xa + detail::and_(sqrt(x2 + Scalar(1)), !mask_huge));
            masked(r_big, mask_huge) += Scalar(LogTwo<Scalar>);
            r_big = copysign(r_big, x);
        }

        return select(mask_big, r_big, r_small);
    }
}

template <typename Value> Value acosh(const Value &x) {
    if constexpr (is_detected_v<detail::has_acosh, Value>) {
        return x.acosh_();
    } else {
        /*
           Hyperbolic arc cosine function approximation based on CEPHES.

         - acosh (in [-10, 10]):
           * avg abs. err = 2.8897e-08
           * avg rel. err = 1.49658e-08
              -> in ULPs  = 0.175817
           * max abs. err = 2.38419e-07
             (at x=3.76221)
           * max rel. err = 2.35024e-07
             -> in ULPs   = 3
             (at x=1.02974)
        */

        static_assert(!is_special_v<Value>,
                      "acosh(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        Value x1 = x - Scalar(1), r_big, r_small;

        Mask mask_big  = x1 >= Scalar(0.49),
             mask_huge = x1 >= Scalar(1e10);

        if (!all_nested_or<false>(mask_big)) {
            if constexpr (Single) {
                r_small = estrin(x1, 1.4142135263e+0,
                                    -1.1784741703e-1,
                                     2.6454905019e-2,
                                    -7.5272886713e-3,
                                     1.7596881071e-3);
            } else {
                r_small = estrin(x1, 1.10855947270161294369E5,
                                     1.08102874834699867335E5,
                                     3.43989375926195455866E4,
                                     3.94726656571334401102E3,
                                     1.18801130533544501356E2) /
                          estrin(x1, 7.83869920495893927727E4,
                                     8.29725251988426222434E4,
                                     2.97683430363289370382E4,
                                     4.15352677227719831579E3,
                                     1.86145380837903397292E2,
                                     1.00000000000000000000E0);
            }

            r_small *= sqrt(x1);
            r_small = detail::or_(r_small, x1 < Scalar(0));
        }

        if (any_nested_or<true>(mask_big)) {
            r_big = log(x + detail::and_(sqrt(fmsub(x, x, Scalar(1))), !mask_huge));
            masked(r_big, mask_huge) += Scalar(LogTwo<Scalar>);
        }

        return select(mask_big, r_big, r_small);
    }
}

template <typename Value> Value atanh(const Value &x) {
    if constexpr (is_detected_v<detail::has_atanh, Value>) {
        return x.atanh_();
    } else {
        /*
           Hyperbolic arc tangent function approximation based on CEPHES.

         - acosh (in [-10, 10]):
           * avg abs. err = 9.87529e-09
           * avg rel. err = 1.52741e-08
              -> in ULPs  = 0.183879
           * max abs. err = 2.38419e-07
             (at x=-0.998962)
           * max rel. err = 1.19209e-07
             -> in ULPs   = 1
             (at x=-0.998962)
        */

        static_assert(!is_special_v<Value>,
                      "atanh(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        using Mask = mask_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        Value xa = abs(x), r_big, r_small;

        Mask mask_big = xa >= Scalar(0.5);

        if (!all_nested_or<false>(mask_big)) {
            Value x2 = sqr(x);

            if constexpr (Single) {
                r_small = estrin(x2, 3.33337300303e-1,
                                     1.99782164500e-1,
                                     1.46691431730e-1,
                                     8.24370301058e-2,
                                     1.81740078349e-1);
            } else {
                r_small = estrin(x2, -3.09092539379866942570e1,
                                      6.54566728676544377376e1,
                                     -4.61252884198732692637e1,
                                      1.20426861384072379242e1,
                                     -8.54074331929669305196e-1) /
                          estrin(x2, -9.27277618139601130017e1,
                                      2.52006675691344555838e2,
                                     -2.49839401325893582852e2,
                                      1.08938092147140262656e2,
                                     -1.95638849376911654834e1,
                                      1.00000000000000000000e0);
            }
            r_small = fmadd(r_small, x2*x, x);
        }

        if (any_nested_or<true>(mask_big)) {
            r_big = log((Scalar(1) + xa) / (Scalar(1) - xa)) * Scalar(0.5);
            r_big = copysign(r_big, x);
        }

        return select(mask_big, r_big, r_small);
    }
}

//! @}
// -----------------------------------------------------------------------

template <typename Value> Value cbrt(const Value &x) {
    if constexpr (is_detected_v<detail::has_cbrt, Value>) {
        return x.cbrt_();
    } else {
        /* Cubic root approximation based on CEPHES

         - cbrt (in [-10, 10]):
           * avg abs. err = 2.91027e-17
           * avg rel. err = 1.79292e-17
              -> in ULPs  = 0.118351
           * max abs. err = 4.44089e-16
             (at x=-9.99994)
           * max rel. err = 2.22044e-16
             -> in ULPs   = 1
             (at x=-9.99994)
        */

        static_assert(!is_special_v<Value>,
                      "cbrt(): requires a regular scalar/array argument!");
        using Scalar = scalar_t<Value>;
        constexpr bool Single = std::is_same_v<Scalar, float>;

        const Scalar cbrt2 = Scalar(1.25992104989487316477),
                     cbrt4 = Scalar(1.58740105196819947475),
                     third = Scalar(1.0 / 3.0);

        Value xa = abs(x);

        auto [xm, xe] = frexp(xa);
        xe += Scalar(1);

        Value xea = abs(xe),
              xea1 = floor(xea * third),
              rem = fnmadd(xea1, Scalar(3), xea);

        /* Approximate cube root of number between .5 and 1,
           peak relative error = 9.2e-6 */
        xm = estrin(xm, 0.40238979564544752126924,
                        1.1399983354717293273738,
                       -0.95438224771509446525043,
                        0.54664601366395524503440,
                       -0.13466110473359520655053);

        Value f1 = select(xe >= 0.f, cbrt2, 1.f / cbrt2),
              f2 = select(xe >= 0.f, cbrt4, 1.f / cbrt4),
              f  = select(eq(rem, 1.f), f1, f2);

        masked(xm, neq(rem, 0.f)) *= f;

        Value r = ldexp(xm, mulsign(xea1, xe));
        r = mulsign(r, x);

        // Newton iteration
        r -= (r - (x / sqr(r))) * third;

        if constexpr (!Single)
            r -= (r - (x / sqr(r))) * third;

        return select(isfinite(x), r, x);
    }
}

template <typename Value> Value erf(const Value &x) {
    if constexpr (is_detected_v<detail::has_erf, Value>) {
        return x.erf_();
    } else {
        // Fits computed using 'resources/remez.cpp'
        // Total error on [-6, 6]:
        //  - single precision: avg = 0.231 ulp, max = 3.80 ulp
        //  - double precision: avg = 0.306 ulp, max = 3.59 ulp

        Value xa = abs(x), x2 = sqr(x), c0, c1;

        if constexpr (std::is_same_v<Value, float>) {
            // max = 1.98 ulp, avg = 0.495 ulp on [0, 1]
            c0 = estrin(x2,
                0x1.20dd74p+0, -0x1.812672p-2,
                0x1.ce0934p-4, -0x1.b5a334p-6,
                0x1.4246b4p-8, -0x1.273facp-11
            );

            // max = 2.60 ulp, avg = 0.783 ulp on [1, 4]
            c1 = estrin(xa,
                -0x1.a0d71ap+0, -0x1.d51e3ap-1,
                -0x1.3a904cp-3,  0x1.1c395cp-5,
                -0x1.6856bep-8,  0x1.180f1ep-11,
                -0x1.8ca9f6p-16);
        } else {
            // max = 2.0 ulp, avg = 0.482 ulp on [0, 1]
            c0 = estrin(x2,
                0x1.20dd750429b6dp+0,  -0x1.812746b0379bcp-2,
                0x1.ce2f21a040d12p-4,  -0x1.b82ce311fa924p-6,
                0x1.565bccf92b298p-8,  -0x1.c02db03dd71b8p-11,
                0x1.f9a2baa8fee07p-14, -0x1.f4ca4d6f3e31bp-17,
                0x1.b97fd3d992af4p-20, -0x1.5c0726f04e805p-23,
                0x1.d71b0f1b15b0ap-27, -0x1.abae491c540bp-31
            );

            /// max = 4.0 ulp, avg = 0.605 ulp on [1, 6]
            c1 = estrin(xa,
                 -0x1.a0be83b09c3d7p+0, -0x1.8bb29648c7afep+1,
                 -0x1.639eb89a5975p+1,  -0x1.7b48b8cd14d9fp+0,
                 -0x1.fb25a03ddc781p-2, -0x1.9cdb7dcacdfb3p-4,
                 -0x1.64f7fbe544f07p-7, -0x1.9a3c3874b3919p-12
            ) / estrin(xa,
                  0x1p+0,                 0x1.55b5d06f1c2dep+0,
                  0x1.b998ffae5528ep-1,   0x1.46884d56cb49bp-2,
                  0x1.18e8848a2cc38p-4,   0x1.ee7f90e8d480cp-8,
                  0x1.1c6a194029df4p-12, -0x1.03d1306b29028p-31);
        }

        Value xb = 1.f - exp2(c1 * xa);
        return select(
            xa < 1, x * c0,
            copysign(select(isfinite(xb), xb, 1.f), x));
    }
}

// Inverse real error function approximation based on on "Approximating the
// erfinv function" by Mark Giles
template <typename Value> Value erfinv(const Value &x) {
    Value w = -log((Value(1.f) - x) * (Value(1.f) + x));

    Value w1 = w - 2.5f;
    Value w2 = sqrt(w) - 3.f;

    Value p1 = estrin(w1,
         1.50140941,     0.246640727,
        -0.00417768164, -0.00125372503,
         0.00021858087, -4.39150654e-06,
        -3.5233877e-06,  3.43273939e-07,
         2.81022636e-08);

    Value p2 = estrin(w2,
         2.83297682,     1.00167406,
         0.00943887047, -0.0076224613,
         0.00573950773, -0.00367342844,
         0.00134934322,  0.000100950558,
        -0.000200214257);

    return select(w < 5.f, p1, p2) * x;
}

/// Natural logarithm of the Gamma function
template <typename Value> Value lgamma(Value x_) {
    using Mask = mask_t<Value>;
    using Scalar = scalar_t<Value>;

    // 'g' and 'n' parameters of the Lanczos approximation
    // See mrob.com/pub/ries/lanczos-gamma.html
    const int n = 6;
    const Scalar g = 5.0f;
    const Scalar log_sqrt2pi = Scalar(0.91893853320467274178);
    const Scalar coeff[n + 1] = { (Scalar)  1.000000000190015, (Scalar) 76.18009172947146,
                                  (Scalar) -86.50532032941677, (Scalar) 24.01409824083091,
                                  (Scalar) -1.231739572450155, (Scalar) 0.1208650973866179e-2,
                                  (Scalar) -0.5395239384953e-5 };

    // potentially reflect using gamma(x) = pi / (sin(pi*x) * gamma(1-x))
    Mask reflect = x_ < .5f;

    Value x = select(reflect, -x_, x_ - 1.f),
          b = x + g + .5f; // base

    Value sum = 0;
    for (int i = n; i >= 1; --i)
        sum += coeff[i] / (x + Scalar(i));
    sum += coeff[0];

    // gamma(x) = sqrt(2*pi) * sum * b^(x + .5) / exp(b)
    Value result = ((log_sqrt2pi + log(sum)) - b) + log(b) * (x + .5f);

    if (any_nested_or<true>(reflect)) {
        masked(result, reflect) = log(abs(Scalar(Pi<Scalar>) / sin(Scalar(Pi<Scalar>) * x_))) - result;
        masked(result, reflect && eq(x_, round(x_))) = Infinity<Scalar>;
    }

    return result;
}

/// Gamma function
template <typename Value> Value tgamma(Value x) { return exp(lgamma(x)); }

NAMESPACE_END(drjit)
