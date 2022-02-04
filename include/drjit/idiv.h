/*
    drjit/array_idiv.h -- fast precomputed integer division by constants based
    on libdivide (https://github.com/ridiculousfish/libdivide)

    Copyright (C) 2010 ridiculous_fish

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.

    libdivide@ridiculousfish.com

*/

#pragma once

#include <drjit/array.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

// -----------------------------------------------------------------------
//! @{ \name Precomputation for division by integer constants
// -----------------------------------------------------------------------

inline std::pair<uint32_t, uint32_t> div_wide(uint32_t u1, uint32_t u0,
                                              uint32_t v) {
    uint64_t u = (((uint64_t) u1) << 32) | u0;

    return { (uint32_t) (u / v),
             (uint32_t) (u % v) };
}

inline std::pair<uint64_t, uint64_t> div_wide(uint64_t u1, uint64_t u0,
                                              uint64_t d) {
#if defined(__SIZEOF_INT128__)
    __uint128_t n = (((__uint128_t) u1) << 64) | u0;
    return {
        (uint64_t) (n / d),
        (uint64_t) (n % d)
    };
#else
    // Code taken from Hacker's Delight:
    // http://www.hackersdelight.org/HDcode/divlu.c.
    // License permits inclusion here per:
    // http://www.hackersdelight.org/permissions.htm

    const uint64_t b = (1ULL << 32); // Number base (16 bits).
    uint64_t un1, un0,  // Norm. dividend LSD's.
    vn1, vn0,           // Norm. divisor digits.
    q1, q0,             // Quotient digits.
    un64, un21, un10,   // Dividend digit pairs.
    rhat;               // A remainder.
    int s;              // Shift amount for norm.

    if (u1 >= d) // overflow
        return { (uint64_t) -1, (uint64_t) -1 };

    // count leading zeros
    s = (int) (63 - log2i(d)); // 0 <= s <= 63.
    if (s > 0) {
        d = d << s;         // Normalize divisor.
        un64 = (u1 << s) | ((u0 >> (64 - s)) & uint64_t(-s >> 31));
        un10 = u0 << s;     // Shift dividend left.
    } else {
        // Avoid undefined behavior.
        un64 = u1 | u0;
        un10 = u0;
    }

    vn1 = d >> 32;            // Break divisor up into
    vn0 = d & 0xFFFFFFFF;     // two 32-bit digits.

    un1 = un10 >> 32;         // Break right half of
    un0 = un10 & 0xFFFFFFFF;  // dividend into two digits.

    q1 = un64/vn1;            // Compute the first
    rhat = un64 - q1*vn1;     // quotient digit, q1.

again1:
    if (q1 >= b || q1*vn0 > b*rhat + un1) {
        q1 = q1 - 1;
        rhat = rhat + vn1;
        if (rhat < b)
            goto again1;
    }

    un21 = un64*b + un1 - q1*d;  // Multiply and subtract.

    q0 = un21/vn1;            // Compute the second
    rhat = un21 - q0*vn1;     // quotient digit, q0.

again2:
    if (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0 = q0 - 1;
        rhat = rhat + vn1;
        if (rhat < b)
            goto again2;
    }

    return {
        q1*b + q0,
        (un21*b + un0 - q0*d) >> s
    };
#endif
}

//! @}
// -----------------------------------------------------------------------

NAMESPACE_END(detail)

template <typename T, typename = int> struct divisor;

template <typename T> struct divisor<T, enable_if_t<std::is_unsigned_v<T>>> {
    T div;
    T multiplier;
    uint8_t shift;

    divisor() = default;

    divisor(T div) : div(div) {
        shift = (uint8_t) log2i(div);

        if ((div & (div - 1)) == 0) {
            // Power of two
            multiplier = 0;
            shift--;
        } else {
            // General case
            auto [m, rem] = detail::div_wide(T(1) << shift, T(0), div);
            multiplier = m * 2 + 1;

            T rem2 = rem * 2;
            if (rem2 >= div || rem2 < rem)
                multiplier += 1;
        }
    }

    template <typename Value>
    DRJIT_INLINE Value operator()(const Value &value) const {
        /* Division by +/-1 is not supported by the
           precomputation-based approach */
        if (div == 1)
            return value;

        if constexpr (is_dynamic_v<Value>) {
            if (multiplier == 0)
                return value >> (shift + 1);
        }

        Value q = mulhi(multiplier, value);
        Value t = sr<1>(value - q) + q;
        return t >> shift;
    }
} DRJIT_PACK;

template <typename T>
struct divisor<T, enable_if_t<std::is_signed_v<T>>> {
    using U = std::make_unsigned_t<T>;

    T div;
    T multiplier;
    uint8_t shift;

    divisor() = default;

    divisor(T div) : div(div) {
        U ad = div < 0 ? (U) -div : (U) div;
        shift = (uint8_t) log2i(ad);

        if ((ad & (ad - 1)) == 0) {
            // Power of two
            multiplier = 0;
        } else {
            // General case
            auto [m, rem] =
                detail::div_wide(U(1) << (shift - 1), U(0), ad);
            multiplier = T(m * 2 + 1);

            U rem2 = rem * 2;
            if (rem2 >= ad || rem2 < rem)
                multiplier += 1;
        }
    }

    template <typename Value> DRJIT_INLINE Value operator()(const Value &value) const {
        /* Division by +/-1 is not supported by the
           precomputation-based approach */
        if (div == 1)
            return value;

        Value q = mulhi(multiplier, value) + value;
        Value q_sign = sr<sizeof(T) * 8 - 1>(q);
        q = q + (q_sign & ((T(1) << shift) - (multiplier == 0 ? 1 : 0)));
        Value sign = div < 0 ? -1 : 0;
        return ((q >> shift) ^ sign) - sign;
    }
} DRJIT_PACK;

template <typename Value> DRJIT_INLINE Value idiv(const Value &a, const divisor<scalar_t<Value>> &div) {
    static_assert(std::is_integral_v<scalar_t<Value>>, "idiv(): requires integral operands!");
    return div(a);
}

template <typename Value> DRJIT_INLINE Value imod(const Value &a, const divisor<scalar_t<Value>> &div) {
    static_assert(std::is_integral_v<scalar_t<Value>>, "imod(): requires integral operands!");

    if constexpr (is_dynamic_v<Value> && std::is_unsigned_v<scalar_t<Value>>) {
        if (div.multiplier == 0)
            return a & (div.div - 1);
    }

    return a - div(a) * div.div;
}

template <typename Value> DRJIT_INLINE std::pair<Value, Value> idivmod(const Value &a, const divisor<scalar_t<Value>> &div) {
    static_assert(std::is_integral_v<scalar_t<Value>>, "idivmod(): requires integral operands!");
    Value d = div(a);

    if constexpr (is_dynamic_v<Value> && std::is_unsigned_v<scalar_t<Value>>) {
        if (div.multiplier == 0)
            return { d, a & (div.div - 1) };
    }

    return { d, a - d*div.div };
}

NAMESPACE_END(drjit)
