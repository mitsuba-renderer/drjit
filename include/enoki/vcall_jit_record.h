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

template <typename Result, typename Func, typename Self, typename Mask,
          size_t... Is, typename... Args>
Result vcall_jit_record_impl(const char *name, uint32_t n_inst_max,
                             uint32_t n_inst_actual, const Func &func,
                             const Self &self, const Mask &mask,
                             std::index_sequence<Is...>, const Args &... args) {
    using Base = std::remove_const_t<std::remove_pointer_t<value_t<Self>>>;
    constexpr size_t N = sizeof...(Args);
    char label[128];
    Result result;

    ek_index_vector indices_in, indices_out_all;
    ek_vector<uint32_t> se_count(n_inst_actual + 1, 0);

    (collect_indices(indices_in, args), ...);
    se_count[0] = jit_side_effects_scheduled(Self::Backend);

    for (uint32_t i = 1, j = 1; i <= n_inst_max; ++i) {
        snprintf(label, sizeof(label), "VCall: %s::%s() [instance %u]",
                 Base::Domain, name, j);
        Base *base = (Base *) jit_registry_get_ptr(Base::Domain, i);
        if (!base)
            continue;

        jit_prefix_push(Self::Backend, label);
        int flag_before = jit_flag(JitFlag::PostponeSideEffects);
        try {
            jit_set_flag(JitFlag::PostponeSideEffects, 1);
            if constexpr (std::is_same_v<Result, std::nullptr_t>) {
                func(base, (set_mask_true<Is, N>(args))...);
            } else {
                collect_indices(indices_out_all, func(base, args...));
            }
        } catch (...) {
            jit_prefix_pop(Self::Backend);
            jit_side_effects_rollback(Self::Backend, se_count[0]);
            jit_set_flag(JitFlag::PostponeSideEffects, flag_before);
            throw;
        }
        jit_set_flag(JitFlag::PostponeSideEffects, flag_before);
        jit_prefix_pop(Self::Backend);
        se_count[j] = jit_side_effects_scheduled(Self::Backend);
        ++j;
    }

    ek_vector<uint32_t> indices_out(indices_out_all.size() / n_inst_actual, 0);

    Self self_masked = self & (Mask::steal(jit_var_mask_peek(Self::Backend)) & mask);

    snprintf(label, sizeof(label), "%s::%s()", Base::Domain, name);

    jit_var_vcall(label, self_masked.index(), n_inst_actual, indices_in.size(),
                  indices_in.data(), indices_out_all.size(),
                  indices_out_all.data(), se_count.data(), indices_out.data());

    if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
        uint32_t offset = 0;
        write_indices(indices_out, result, offset);
    }

    return result;
}

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_record(const char *name, const Func &func, Self &self,
                        const Args &... args) {
    using Base = std::remove_const_t<std::remove_pointer_t<value_t<Self>>>;
    uint32_t n_inst_max = jit_registry_get_max(Base::Domain),
             n_inst_actual = 0;

    Base *inst = nullptr;

    for (uint32_t i = 1; i <= n_inst_max; ++i) {
        Base *base = (Base *) jit_registry_get_ptr(Base::Domain, i);
        if (!base)
            continue;
        inst = base;
        n_inst_actual++;
    }

    Result result;
    if (n_inst_actual == 0) {
        result = zero<Result>(width(self));
    } else if (n_inst_actual == 1) {
        if constexpr (std::is_same_v<Result, std::nullptr_t>)
            func(inst, args...);
        else
            result = select(neq(self, nullptr), func(inst, args...),
                            zero<Result>());
    } else {
        result = vcall_jit_record_impl<Result>(
            name, n_inst_max, n_inst_actual, func, self,
            extract_mask<mask_t<Self>>(args...),
            std::make_index_sequence<sizeof...(Args)>(),
            placeholder(args)...);
    }
    return result;
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
