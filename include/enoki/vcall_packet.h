/*
    enoki/vcall.h -- Vectorized method call support -- packet types

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <typename Mask> ENOKI_INLINE Mask get_mask() {
    return Mask(true);
}

template <typename Mask, typename Arg, typename... Args>
ENOKI_INLINE Mask get_mask(const Arg &arg, const Args &... args) {
    if constexpr (is_mask_v<Arg>)
        return Mask(arg);
    else
        return get_mask<Mask>(args...);
}

template <typename Arg, typename Mask>
ENOKI_INLINE auto &replace_mask(Arg &arg, const Mask &mask) {
    if constexpr (is_mask_v<Arg>)
        return mask;
    else
        return arg;
}


template <typename Result, typename Func, typename Self, typename... Args>
ENOKI_INLINE Result dispatch_packet(Func func, const Self &self, const Args&... args) {
    using Class = scalar_t<Self>;
    using Mask = mask_t<Self>;
    Mask mask = get_mask<Mask>(args...);
    mask &= neq(self, nullptr);

    if constexpr (!std::is_void_v<Result>) {
        Result result = zero<Result>(self.size());
        while (any(mask)) {
            Class instance         = extract(self, mask);
            Mask active            = mask & eq(self, instance);
            mask                   = andnot(mask, active);
            masked(result, active) = func(instance, replace_mask(args, active)...);
        }
        return result;
    } else {
        while (any(mask)) {
            Class instance    = extract(self, mask);
            Mask active       = mask & eq(self, instance);
            mask              = andnot(mask, active);
            func(instance, replace_mask(args, active)...);
        }
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
