/*
    enoki/vcall.h -- Vectorized method call support, via jump table

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/containers.h>

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

template <typename Mask> struct VCallRAIIGuard {
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;

    VCallRAIIGuard(const char *label, uint32_t self) {
        self_before = jit_vcall_self(Backend);
        jit_vcall_set_self(Backend, self);

        flag_before = jit_flag(JitFlag::Recording);
        jit_set_flag(JitFlag::Recording, 1);

        ENOKI_MARK_USED(label);
#if defined(ENOKI_VCALL_DEBUG)
        jit_prefix_push(Backend, label);
#endif

        Mask vcall_mask;
        if constexpr (Backend == JitBackend::LLVM) {
            vcall_mask = Mask::steal(jit_var_new_stmt(
                Backend, VarType::Bool,
                "$r0 = or <$w x i1> %mask, zeroinitializer", 1, 0,
                nullptr));
        } else {
            vcall_mask = true;
        }
        jit_var_mask_push(Backend, vcall_mask.index(), 0);
    }

    ~VCallRAIIGuard() {
        jit_vcall_set_self(Backend, self_before);
        jit_var_mask_pop(Backend);
#if defined(ENOKI_VCALL_DEBUG)
        jit_prefix_pop(Backend);
#endif
        jit_set_flag(JitFlag::Recording, flag_before);
    }

    int flag_before;
    uint32_t self_before;
};

template <typename Mask> struct MaskRAIIGuard {
    static constexpr JitBackend Backend = detached_t<Mask>::Backend;

    MaskRAIIGuard(const Mask &mask) {
        jit_var_mask_push(Backend, mask.index(), 1);
    }

    ~MaskRAIIGuard() {
        jit_var_mask_pop(Backend);
    }
};

template <typename Result, typename Base, typename Func, typename Self,
          typename Mask, size_t... Is, typename... Args>
Result vcall_jit_record_impl(const char *name, uint32_t n_inst,
                             const Func &func, const Self &self,
                             const Mask &mask, std::index_sequence<Is...>,
                             const Args &... args) {
    static constexpr JitBackend Backend = detached_t<Self>::Backend;
    constexpr size_t N = sizeof...(Args);
    ENOKI_MARK_USED(N);

    char label[128];

    ek_index_vector indices_in, indices_out_all;
    ek_vector<uint32_t> se_count(n_inst + 1, 0);
    ek_vector<uint32_t> inst_id(n_inst, 0);

    (collect_indices(indices_in, args), ...);
    se_count[0] = jit_side_effects_scheduled(Backend);

    uint32_t n_inst_max = jit_registry_get_max(Backend, Base::Domain);
    try {
        for (uint32_t i = 1, j = 1; i <= n_inst_max; ++i) {
            snprintf(label, sizeof(label), "VCall: %s::%s() [instance %u]",
                     Base::Domain, name, j);
            Base *base = (Base *) jit_registry_get_ptr(Backend, Base::Domain, i);
            if (!base)
                continue;

            VCallRAIIGuard<Mask> guard(label, i);

            if constexpr (std::is_same_v<Result, std::nullptr_t>) {
                func(base, (set_mask_true<Is, N>(args))...);
            } else {
                // The following assignment converts scalar return values
                Result tmp = func(base, set_mask_true<Is, N>(args)...);
                collect_indices(indices_out_all, tmp);
            }
            inst_id[j - 1] = i;
            se_count[j++] = jit_side_effects_scheduled(Backend);
        }
    } catch (...) {
        jit_side_effects_rollback(Backend, se_count[0]);
        throw;
    }

    ek_vector<uint32_t> indices_out(indices_out_all.size() / n_inst, 0);

    snprintf(label, sizeof(label), "%s::%s()", Base::Domain, name);

    jit_var_vcall(label, self.index(), mask.index(), n_inst, inst_id.data(),
                  (uint32_t) indices_in.size(), indices_in.data(),
                  (uint32_t) indices_out_all.size(), indices_out_all.data(),
                  se_count.data(), indices_out.data());

    if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
        Result result;
        uint32_t offset = 0;
        write_indices(indices_out, result, offset);
        return result;
    } else {
        return nullptr;
    }
}

template <typename Result, typename Base, typename Func, typename Mask,
          size_t... Is, typename... Args>
Result
vcall_jit_record_impl_scalar(Base *inst, const Func &func, const Mask &mask,
                             std::index_sequence<Is...>, const Args &... args) {
    constexpr size_t N = sizeof...(Args);
    ENOKI_MARK_USED(N);

    // Evaluate the single instance with mask = true, mask side effects
    MaskRAIIGuard<Mask> guard(mask);

    if constexpr (is_enoki_struct_v<Result> || is_array_v<Result>) {
        // Return zero for masked results
        return select(mask, func(inst, (set_mask_true<Is, N>(args))...),
                      zero<Result>());
    } else {
        return func(inst, (set_mask_true<Is, N>(args))...);
    }
}

inline std::pair<void *, uint32_t> vcall_registry_get(JitBackend Backend,
                                                      const char *domain) {
    uint32_t n = jit_registry_get_max(Backend, domain), n_inst = 0;
    void *inst = nullptr;

    for (uint32_t i = 1; i <= n; ++i) {
        void *ptr = jit_registry_get_ptr(Backend, domain, i);
        if (ptr) {
            inst = ptr;
            n_inst++;
        }
    }

    return { inst, n_inst };
}

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_record(const char *name, const Func &func, Self &self,
                        const Args &... args) {
    using Base = std::remove_const_t<std::remove_pointer_t<value_t<Self>>>;
    static constexpr JitBackend Backend = detached_t<Self>::Backend;
    using Mask = mask_t<Self>;

    auto [inst, n_inst] = vcall_registry_get(Backend, Base::Domain);

    size_t self_size = width(self, args...);
    Mask mask = extract_mask<Mask>(args...) && neq(self, nullptr);
    bool masked = mask.is_literal() && mask[0] == false;

    if (n_inst == 0 || self_size == 0 || masked) {
        jit_log(::LogLevel::InfoSym,
                "jit_var_vcall(self=r%u): call (\"%s::%s()\") not performed (%s)",
                self.index(), Base::Domain, name,
                n_inst == 0 ? "no instances"
                            : (masked ? "masked" : "self.size == 0"));
        return zero<Result>(self_size);
    } else if (n_inst == 1) {
        jit_log(::LogLevel::InfoSym,
                "jit_var_vcall(self=r%u): call (\"%s::%s()\") inlined (only 1 "
                "instance exists.)", self.index(), Base::Domain, name);
        return vcall_jit_record_impl_scalar<Result, Base>(
            (Base *) inst, func, mask,
            std::make_index_sequence<sizeof...(Args)>(), args...);
    } else {
        // Also check the mask stack to constrain side effects in recorded computation
        Mask mask_combined = mask && Mask::steal(jit_var_mask_peek(Backend));
        return vcall_jit_record_impl<Result, Base>(
            name, n_inst, func, self, mask_combined,
            std::make_index_sequence<sizeof...(Args)>(),
            placeholder(args, /* preserve_size = */ false,
                        /* propagate_literals = */ true)...);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
