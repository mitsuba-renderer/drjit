/*
    drjit/vcall.h -- Vectorized method call support, via horiz. reduction

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)


template <size_t I, size_t N, typename T, typename UInt32>
DRJIT_INLINE decltype(auto) gather_helper(const T& value, const UInt32 &perm) {
    if constexpr (is_mask_v<T> && I == N - 1) {
        return true;
    } else if constexpr (is_jit_v<T>) {
        return gather<T, true>(value, perm);
    } else if constexpr (is_drjit_struct_v<T>) {
        T result = value;
        struct_support_t<T>::apply_1(
            result, [&perm](auto &x) { x = gather_helper<1, 1>(x, perm); });
        return result;
    } else {
        DRJIT_MARK_USED(perm);
        return value;
    }
}

template <typename Mask>
struct MaskScope {
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;
    MaskScope(const Mask &mask) {
        jit_var_mask_push(Backend, mask.index(), 0);
    }
    ~MaskScope() {
        jit_var_mask_pop(Backend);
    }
};

template <typename Result, typename Func, typename Self, size_t... Is,
          typename... Args>
Result vcall_jit_reduce_impl(Func func, const Self &self_,
                             std::index_sequence<Is...>, const Args &... args) {
    using UInt32 = uint32_array_t<Self>;
    using Class = scalar_t<Self>;
    using Mask = mask_t<UInt32>;
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;
    constexpr size_t N = sizeof...(Args);
    DRJIT_MARK_USED(N);

    schedule(args...);

    size_t self_size = self_.size();
    if (self_size == 1) {
        auto self = self_.entry(0);
        if (self)
            return func(self, args...);
        else
            return zeros<Result>();
    }

    Mask mask = extract_mask<Mask>(args...);
    if (jit_var_mask_size(Backend))
        mask &= Mask::steal(jit_var_mask_peek(Backend));

    MaskScope<Mask> scope(Mask::steal(jit_var_mask_default(Backend)));

    struct SetSelfHelper {
        void set(uint32_t value, uint32_t index) {
            jit_vcall_set_self(detached_t<Mask>::Backend, value, index);
        }

        ~SetSelfHelper() {
            jit_vcall_set_self(detached_t<Mask>::Backend, 0, 0);
        }
    };

    Self self = self_ & mask;
    auto [buckets, n_inst] = self.vcall_();

    Result result;
    SetSelfHelper self_helper;
    if (n_inst > 0 && self_size > 0) {
        result = empty<Result>(self_size);
        size_t last_size = 0;

        for (size_t i = 0; i < n_inst ; ++i) {
            UInt32 perm        = UInt32::borrow(buckets[i].index),
                   instance_id = gather<UInt32>(self, perm);

            size_t wavefront_size = perm.size();

            // Avoid merging multiple vcall launches if size repeats..
            if (wavefront_size != last_size)
                last_size = wavefront_size;
            else
                eval(result);

            if (buckets[i].ptr) {
                self_helper.set(buckets[i].id, instance_id.index());

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
                    scatter<true>(result, zeros<Result>(), perm);
            }
        }
        schedule(result);
    } else {
        result = zeros<Result>(self_size);
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
NAMESPACE_END(drjit)
