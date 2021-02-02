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


template <size_t I, size_t N, typename T, typename UInt32>
ENOKI_INLINE decltype(auto) gather_helper(const T& value, const UInt32 &perm) {
    if constexpr (is_mask_v<T> && I == N - 1) {
        return true;
    } else if constexpr (is_jit_array_v<T>) {
        return gather<T, true>(value, perm);
    } else if constexpr (is_enoki_struct_v<T>) {
        T result = value;
        struct_support_t<T>::apply_1(
            result, [&perm](auto &x) { x = gather_helper<1, 1>(x, perm); });
        return result;
    } else {
        ENOKI_MARK_USED(perm);
        return value;
    }
}

template <typename Mask>
struct MaskScope {
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;
    MaskScope(const Mask &mask) {
        jit_var_mask_push(Backend, mask.index(), 0);
    }
    ~MaskScope() { jit_var_mask_pop(Backend); }
};

template <typename Result, typename Func, typename Self, size_t... Is,
          typename... Args>
Result vcall_jit_reduce_impl(Func func, const Self &self,
                             std::index_sequence<Is...>, const Args &... args) {
    using UInt32 = uint32_array_t<Self>;
    using Class = scalar_t<Self>;
    using Mask = mask_t<UInt32>;
    constexpr size_t N = sizeof...(Args);
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;

    schedule(args...);
    auto [buckets, n_inst] = self.vcall_();

    size_t self_size = self.size();

    Result result;
    if (n_inst > 0 && self_size > 0) {
        Mask mask = extract_mask<Mask>(args...);
        mask &= Mask::steal(jit_var_mask_default(Backend));

        if (jit_var_mask_size(Backend))
            mask &= Mask::steal(jit_var_mask_peek(Backend));

        result = empty<Result>(self_size);
        for (size_t i = 0; i < n_inst ; ++i) {
            UInt32 perm = UInt32::borrow(buckets[i].index);
            MaskScope<Mask> scope(gather<Mask>(mask, perm));
            if (buckets[i].ptr) {
                if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
                    using OrigResult = decltype(func((Class) nullptr, args...));
                    scatter<true>(
                        result,
                        ref_cast_t<OrigResult, Result>(func(
                            (Class) buckets[i].ptr,
                            gather_helper<Is, N>(args, perm)...)),
                        perm);
                } else {
                    func((Class) buckets[i].ptr, gather_helper<Is, N>(args, perm)...);
                }
            } else {
                if constexpr (!std::is_same_v<Result, std::nullptr_t>)
                    scatter<true>(result, zero<Result>(), perm);
            }
        }
        schedule(result);
    } else {
        result = zero<Result>(self_size);
    }

    return result;
}

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_reduce(const Func &func, const Self &self,
                        const Args &... args) {
    return vcall_jit_reduce_impl<Result>(
        func, self, std::make_index_sequence<sizeof...(Args)>(), args...);
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
