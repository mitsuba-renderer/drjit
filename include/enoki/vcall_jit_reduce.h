/*
    enoki/vcall.h -- Vectorized method call support, via horiz. reduction

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <typename T, typename UInt32>
ENOKI_INLINE decltype(auto) gather_helper(const T& value, const UInt32 &perm) {
    if constexpr (is_jit_array_v<T>) {
        return gather<T, true>(value, perm);
    } else if constexpr (is_enoki_struct_v<T>) {
        T result = value;
        struct_support_t<T>::apply_1(
            result, [&perm](auto &x) { x = gather_helper(x, perm); });
        return result;
    } else {
        ENOKI_MARK_USED(perm);
        return value;
    }
}

template <typename Result, typename Func, typename Self, typename... Args>
ENOKI_INLINE Result vcall_jit_reduce(Func func, const Self &self, const Args&... args) {
    using UInt32 = uint32_array_t<Self>;
    using Class = scalar_t<Self>;

    if constexpr (!std::is_void_v<Result>) {
        Result result;

        if (self.size() == 1) {
            Class ptr = (Class) self.entry(0);
            if (ptr)
                result = func(ptr, args...);
            else
                result = zero<Result>();
        } else {
            schedule(args...);
            auto [buckets, size] = self.vcall_();

            if (size > 0) {
                result = empty<Result>(self.size());
                for (size_t i = 0; i < size; ++i) {
                    UInt32 perm = UInt32::borrow(buckets[i].index);

                    if (buckets[i].ptr) {
                        using OrigResult = decltype(func((Class) nullptr, args...));
                        scatter<true>(
                            result,
                            ref_cast_t<OrigResult, Result>(func(
                                (Class) buckets[i].ptr,
                                detail::gather_helper(args, perm)...)),
                            perm);
                    } else {
                        scatter<true>(result, zero<Result>(), perm);
                    }
                }
                schedule(result);
            } else {
                result = zero<Result>(self.size());
            }
        }
        return result;
    } else {
        if (self.size() == 1) {
            Class ptr = (Class) self.entry(0);
            if (ptr)
                func(ptr, args...);
        } else {
            auto [buckets, size] = self.vcall_();
            for (size_t i = 0; i < size; ++i) {
                if (!buckets[i].ptr)
                    continue;
                UInt32 perm = UInt32::borrow(buckets[i].index);
                func((Class) buckets[i].ptr,
                     detail::gather_helper(args, perm)...);
            }
        }
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
