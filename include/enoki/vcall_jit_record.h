/*
    enoki/vcall.h -- Vectorized method call support, via jump table

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

template <typename T>
void collect_indices(ek_index_vector &indices, const T &value) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            collect_indices(indices, value.derived().entry(i));
    } else if constexpr (is_diff_array_v<T>) {
        collect_indices(indices, value.detach_());
    } else if constexpr (is_jit_array_v<T>) {
        uint32_t index = value.index();
        if (!index)
            enoki_raise("enoki::detail::collect_indices(): encountered an "
                        "uninitialized function argument while recording a "
                        "virtual function call!");
        indices.push_back(index);
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](const auto &x) { collect_indices(indices, x); });
    }
}

template <typename T>
void write_indices(ek_vector<uint32_t> &indices, T &value, uint32_t &offset) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            write_indices(indices, value.derived().entry(i), offset);
    } else if constexpr (is_diff_array_v<T>) {
        write_indices(indices, value.detach_(), offset);
    } else if constexpr (is_jit_array_v<T>) {
        value = T::steal(indices[offset++]);
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](auto &x) { write_indices(indices, x, offset); });
    }
}

inline bool extract_mask() { return true; }
template <typename T> decltype(auto) extract_mask(const T &v) {
    if constexpr (is_mask_v<T>)
        return v;
    else
        return true;
}

template <typename T, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
decltype(auto) extract_mask(const T &/*v*/, const Ts &... vs) {
    return extract_mask(vs...);
}

template <size_t I, size_t N, typename T>
decltype(auto) set_mask_true(const T &v) {
    if constexpr (is_mask_v<T> && I == N - 1)
        return true;
    else
        return v;
}

template <typename Result, typename Func, JitBackend Backend,
          typename Base, typename... Args, size_t... Is>
Result vcall_impl(const char *name, uint32_t n_inst, const Func &func,
                  const JitArray<Backend, Base *> &self,
                  const JitArray<Backend, bool> &mask,
                  std::index_sequence<Is...>, const Args &... args) {
    constexpr size_t N = sizeof...(Args);
    char label[128];
    Result result;
    using Self = JitArray<Backend, Base *>;

    ek_index_vector indices_in, indices_out_all;
    ek_vector<uint32_t> se_count(n_inst + 1, 0);

    (collect_indices(indices_in, args), ...);
    se_count[0] = jit_side_effects_scheduled(Backend);

    for (uint32_t i = 1; i <= n_inst; ++i) {
        snprintf(label, sizeof(label), "VCall: %s::%s() [instance %u]",
                 Base::Domain, name, i);
        Base *base = (Base *) jit_registry_get_ptr(Base::Domain, i);

        jit_prefix_push(Backend, label);
        int flag_before = jit_flag(JitFlag::PostponeSideEffects);
        try {
            jit_set_flag(JitFlag::PostponeSideEffects, 1);
            if constexpr (std::is_same_v<Result, std::nullptr_t>) {
                func(base, (detail::set_mask_true<Is, N>(args))...);
            } else {
                collect_indices(indices_out_all, func(base, args...));
            }
        } catch (...) {
            jit_prefix_pop(Backend);
            jit_side_effects_rollback(Backend, se_count[0]);
            jit_set_flag(JitFlag::PostponeSideEffects, flag_before);
            throw;
        }
        jit_set_flag(JitFlag::PostponeSideEffects, flag_before);
        jit_prefix_pop(Backend);
        se_count[i] = jit_side_effects_scheduled(Backend);
    }

    ek_vector<uint32_t> indices_out(indices_out_all.size() / n_inst, 0);

    JitArray<Backend, Base *> self_masked =
        self &
        (JitArray<Backend, bool>::steal(jit_var_mask_peek(Backend)) & mask);

    snprintf(label, sizeof(label), "%s::%s()", Base::Domain, name);

    jit_var_vcall(label, self_masked.index(), n_inst, indices_in.size(),
                  indices_in.data(), indices_out_all.size(),
                  indices_out_all.data(), se_count.data(), indices_out.data());

    if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
        uint32_t offset = 0;
        write_indices(indices_out, result, offset);
    }

    return result;
}

template <typename Result, typename Func, JitBackend Backend, typename Base, typename... Args>
Result vcall_jit_record(const char *name, const Func &func,
                        const JitArray<Backend, Base *> &self,
                        const Args &... args) {
    constexpr bool IsVoid = std::is_void_v<Result>;
    using Result_2 = std::conditional_t<IsVoid, std::nullptr_t, Result>;
    using Self = JitArray<Backend, Base *>;
    using Bool = JitArray<Backend, bool>;

    uint32_t n_inst = jit_registry_get_max(Base::Domain);

    Result_2 result;
    if (n_inst == 0) {
        result = zero<Result_2>(width(self));
    } else if (n_inst == 1) {
        uint32_t i = 1;
        Base *inst = nullptr;
        do {
            inst = (Base *) jit_registry_get_ptr(Base::Domain, i++);
        } while (!inst);

        if constexpr (IsVoid)
            func(inst, args...);
        else
            result = select(neq(self, nullptr), func(inst, args...),
                            zero<Result_2>());
    } else {
        result = vcall_impl<Result_2>(
            name, n_inst, func, self,
            Bool(detail::extract_mask(args...)),
            std::make_index_sequence<sizeof...(Args)>(),
            placeholder(args)...);
    }
    return result;
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
