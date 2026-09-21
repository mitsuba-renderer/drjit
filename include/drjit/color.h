/*
    drjit/color.h -- Color space transformations (sRGB, Oklab, HSV, HSL)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/math.h>

NAMESPACE_BEGIN(drjit)

template <typename Value> Value linear_to_srgb(const Value &x) {
    using Mask = mask_t<Value>;
    using Scalar = scalar_t<Value>;
    constexpr bool Single = std::is_same_v<Scalar, float>;

    Value r = Scalar(12.92);
    Mask large_mask = x > Scalar(0.0031308);

    if (DRJIT_LIKELY(any_nested_or<true>(large_mask))) {
        Value y = sqrt(x), p, q;

        if constexpr (Single) {
            p = estrin(y, -0.0016829072605308378, 0.03453868659826638,
                      0.7642611304733891, 2.0041169284241644,
                      0.7551545191665577, -0.016202083165206348);
            q = estrin(y, 4.178892964897981e-7, -0.00004375359692957097,
                      0.03467195408529984, 0.6085338522168684,
                      1.8970238036421054, 1.);
        } else {
            p = estrin(y, -3.7113872202050023e-6, -0.00021805827098915798,
                       0.002531335520959116, 0.2263810267005674,
                       3.0477578489880823, 15.374469584296442,
                       32.44669922192121, 27.901125077137042, 8.450947414259522,
                       0.5838023820686707, -0.0031151377052754843);
            q = estrin(y, 2.2380622409188757e-11, -8.387527630781522e-9,
                       0.00007045228641004039, 0.007244514696840552,
                       0.21749170309546628, 2.575446652731678,
                       13.297981743005433, 30.50364355650628, 29.70548706952188,
                       10.723011300050162, 1.);
        }

        masked(r, large_mask) = p / q;
    }

    return r * x;
}

template <typename Value> Value srgb_to_linear(const Value &x) {
    using Mask = mask_t<Value>;
    using Scalar = scalar_t<Value>;
    constexpr bool Single = std::is_same_v<Scalar, float>;

    Value r = Scalar(1.0 / 12.92);
    Mask large_mask = x > Scalar(0.04045);

    if (DRJIT_LIKELY(any_nested_or<true>(large_mask))) {
        Value p, q;

        if constexpr (Single) {
            p = estrin(x, -0.0163933279112946, -0.7386328024653209,
                      -11.199318357635072, -47.46726633009393,
                      -36.04572663838034);
            q = estrin(x, -0.004261480793199332, -19.140923959601675,
                      -59.096406619244426, -18.225745396846637, 1.);
        } else {
            p = estrin(x, -0.008042950896814532, -0.5489744177844188,
                      -14.786385491859248, -200.19589605282445,
                      -1446.951694673217, -5548.704065887224,
                      -10782.158977031822, -9735.250875334352,
                      -3483.4445569178347, -342.62884098034357);
            q = estrin(x, -2.2132610916769585e-8, -9.646075249097724,
                      -237.47722999429413, -2013.8039726540235,
                      -7349.477378676199, -11916.470977597566,
                      -8059.219012060384, -1884.7738197074218,
                      -84.8098437770271, 1.);
        }

        masked(r, large_mask) = p / q;
    }

    return r * x;
}

NAMESPACE_BEGIN(detail)

/// Build a color of the same type as 'ref', passing a fourth (alpha) channel through
template <typename Value, typename... Args>
DRJIT_INLINE Value make_color(const Value &ref, const Args &...args) {
    static_assert(size_v<Value> == 3 || size_v<Value> == 4,
                  "Color conversions require arrays with 3 or 4 channels.");
    if constexpr (size_v<Value> == 4)
        return Value(args..., ref.w());
    else
        return Value(args...);
}

template <typename Entry, typename Scalar>
DRJIT_INLINE Entry color_dot3(Scalar a, Scalar b, Scalar c, const Entry &x,
                              const Entry &y, const Entry &z) {
    return fmadd(Entry(a), x, fmadd(Entry(b), y, Entry(c) * z));
}

/**
 * Hue in [0, 1) of an RGB color, along with the maximum and minimum channel
 * values that HSV and HSL both need.
 */
template <typename Value>
DRJIT_INLINE value_t<Value> rgb_to_hue(const Value &c, value_t<Value> &cmax,
                                       value_t<Value> &cmin) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry r = c.x(), g = c.y(), b = c.z();
    cmax = maximum(maximum(r, g), b);
    cmin = minimum(minimum(r, g), b);

    Entry delta = cmax - cmin,
          delta_safe = select(delta > Scalar(0), delta, Entry(Scalar(1)));

    Entry h = select(cmax == r, (g - b) / delta_safe,
              select(cmax == g, Scalar(2) + (b - r) / delta_safe,
                                Scalar(4) + (r - g) / delta_safe));

    h = select(delta > Scalar(0), h * Scalar(1.0 / 6.0), Entry(Scalar(0)));
    return h - floor(h);
}

/// One channel of the fully saturated, maximally bright color with hue 'h'
template <typename Entry>
DRJIT_INLINE Entry hue_to_rgb_channel(const Entry &h, scalar_t<Entry> offset) {
    using Scalar = scalar_t<Entry>;
    Entry x = h + offset;
    x = x - floor(x);
    return clip(abs(fmadd(x, Scalar(6), Scalar(-3))) - Scalar(1), Scalar(0), Scalar(1));
}

NAMESPACE_END(detail)

/**
 * \brief Convert colors from linear sRGB to the Oklab color space
 *
 * See https://bottosson.github.io/posts/oklab/ for details. The function
 * expects an array or custom color type with 3 channels, or 4 channels when
 * the last one holds an alpha value that should pass through unchanged.
 */
template <typename Value> Value linear_srgb_to_oklab(const Value &c) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry r = c.x(), g = c.y(), b = c.z();

    Entry l = cbrt(detail::color_dot3(Scalar(0.4122214708), Scalar(0.5363325363), Scalar(0.0514459929), r, g, b)),
          m = cbrt(detail::color_dot3(Scalar(0.2119034982), Scalar(0.6806995451), Scalar(0.1073969566), r, g, b)),
          s = cbrt(detail::color_dot3(Scalar(0.0883024619), Scalar(0.2817188376), Scalar(0.6299787005), r, g, b));

    return detail::make_color(c,
        detail::color_dot3(Scalar(0.2104542553), Scalar( 0.7936177850), Scalar(-0.0040720468), l, m, s),
        detail::color_dot3(Scalar(1.9779984951), Scalar(-2.4285922050), Scalar( 0.4505937099), l, m, s),
        detail::color_dot3(Scalar(0.0259040371), Scalar( 0.7827717662), Scalar(-0.8086757660), l, m, s));
}

/// Convert colors from the Oklab color space to linear sRGB
template <typename Value> Value oklab_to_linear_srgb(const Value &c) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry L = c.x(), a = c.y(), b = c.z();

    Entry l = detail::color_dot3(Scalar(1), Scalar( 0.3963377774), Scalar( 0.2158037573), L, a, b),
          m = detail::color_dot3(Scalar(1), Scalar(-0.1055613458), Scalar(-0.0638541728), L, a, b),
          s = detail::color_dot3(Scalar(1), Scalar(-0.0894841775), Scalar(-1.2914855480), L, a, b);

    l = l * l * l;
    m = m * m * m;
    s = s * s * s;

    return detail::make_color(c,
        detail::color_dot3(Scalar( 4.0767416621), Scalar(-3.3077115913), Scalar( 0.2309699292), l, m, s),
        detail::color_dot3(Scalar(-1.2684380046), Scalar( 2.6097574011), Scalar(-0.3413193965), l, m, s),
        detail::color_dot3(Scalar(-0.0041960863), Scalar(-0.7034186147), Scalar( 1.7076147010), l, m, s));
}

/**
 * \brief Convert colors from RGB to HSV (hue, saturation, value)
 *
 * All three output components lie in [0, 1] for in-gamut input. The hue wraps
 * around so that 0 and 1 both correspond to red, and grays have a hue of 0.
 */
template <typename Value> Value rgb_to_hsv(const Value &c) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry cmax, cmin;
    Entry h = detail::rgb_to_hue(c, cmax, cmin);
    auto nonzero = cmax > Scalar(0);
    Entry s = select(nonzero, (cmax - cmin) / select(nonzero, cmax, Entry(Scalar(1))),
                     Entry(Scalar(0)));

    return detail::make_color(c, h, s, cmax);
}

/// Convert colors from HSV (hue, saturation, value) to RGB. The hue may take any value.
template <typename Value> Value hsv_to_rgb(const Value &c) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry h = c.x(), s = c.y(), v = c.z();

    Entry r = detail::hue_to_rgb_channel(h, Scalar(0)),
          g = detail::hue_to_rgb_channel(h, Scalar(2.0 / 3.0)),
          b = detail::hue_to_rgb_channel(h, Scalar(1.0 / 3.0));

    return detail::make_color(c,
        v * lerp(Entry(Scalar(1)), r, s),
        v * lerp(Entry(Scalar(1)), g, s),
        v * lerp(Entry(Scalar(1)), b, s));
}

/**
 * \brief Convert colors from RGB to HSL (hue, saturation, lightness)
 *
 * All three output components lie in [0, 1] for in-gamut input. The hue wraps
 * around so that 0 and 1 both correspond to red, and grays have a hue of 0.
 */
template <typename Value> Value rgb_to_hsl(const Value &c) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry cmax, cmin;
    Entry h = detail::rgb_to_hue(c, cmax, cmin);
    Entry l = (cmax + cmin) * Scalar(0.5),
          denom = Scalar(1) - abs(cmax + cmin - Scalar(1));
    auto nonzero = denom > Scalar(0);
    Entry s = select(nonzero, (cmax - cmin) / select(nonzero, denom, Entry(Scalar(1))),
                     Entry(Scalar(0)));

    return detail::make_color(c, h, s, l);
}

/// Convert colors from HSL (hue, saturation, lightness) to RGB. The hue may take any value.
template <typename Value> Value hsl_to_rgb(const Value &c) {
    using Entry = value_t<Value>;
    using Scalar = scalar_t<Value>;

    Entry h = c.x(), s = c.y(), l = c.z();
    Entry chroma = (Scalar(1) - abs(fmadd(l, Scalar(2), Scalar(-1)))) * s;

    Entry r = detail::hue_to_rgb_channel(h, Scalar(0)),
          g = detail::hue_to_rgb_channel(h, Scalar(2.0 / 3.0)),
          b = detail::hue_to_rgb_channel(h, Scalar(1.0 / 3.0));

    return detail::make_color(c,
        fmadd(r - Scalar(0.5), chroma, l),
        fmadd(g - Scalar(0.5), chroma, l),
        fmadd(b - Scalar(0.5), chroma, l));
}

NAMESPACE_END(drjit)
