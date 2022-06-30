/*
    drjit/vcall.h -- Vectorized method call support, via jump table

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/containers.h>
#include <drjit-core/state.h>

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

template <typename T>
void collect_indices(dr_index_vector &indices, const T &value) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            collect_indices(indices, value.derived().entry(i));
    } else if constexpr (is_diff_v<T>) {
        collect_indices(indices, value.detach_());
    } else if constexpr (is_jit_v<T>) {
        uint32_t index = value.index();
        if (!index)
            drjit_raise("drjit::detail::collect_indices(): encountered an "
                        "uninitialized function argument while recording a "
                        "virtual function call!");
        indices.push_back(index);
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](const auto &x) { collect_indices(indices, x); });
    }
}

template <typename T>
void write_indices(dr_vector<uint32_t> &indices, T &value, uint32_t &offset) {
    if constexpr (array_depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            write_indices(indices, value.derived().entry(i), offset);
    } else if constexpr (is_diff_v<T>) {
        write_indices(indices, value.detach_(), offset);
    } else if constexpr (is_jit_v<T>) {
        value = T::steal(indices[offset++]);
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](auto &x) { write_indices(indices, x, offset); });
    }
}

template <typename T> DRJIT_INLINE auto wrap_vcall(const T &value) {
    if constexpr (array_depth_v<T> > 1) {
        T result;
        for (size_t i = 0; i < value.derived().size(); ++i)
            result.derived().entry(i) = wrap_vcall(value.derived().entry(i));
        return result;
    } else if constexpr (is_diff_v<T>) {
        return wrap_vcall(value.detach_());
    } else if constexpr (is_jit_v<T>) {
        return T::steal(jit_var_wrap_vcall(value.index()));
    } else if constexpr (is_drjit_struct_v<T>) {
        T result;
        struct_support_t<T>::apply_2(
            result, value,
            [](auto &x, const auto &y) DRJIT_INLINE_LAMBDA {
                x = wrap_vcall(y);
            });
        return result;
    } else {
        return (const T &) value;
    }
}

template <typename Result, typename Base, typename Func, typename Self,
          typename Mask, size_t... Is, typename... Args>
Result vcall_jit_record_impl(const char *name, uint32_t n_inst,
                             const Func &func, const Self &self,
                             const Mask &mask, std::index_sequence<Is...>,
                             const Args &... args) {
    static constexpr JitBackend Backend = backend_v<Self>;
    constexpr size_t N = sizeof...(Args);
    DRJIT_MARK_USED(N);

    char label[128];

    dr_index_vector indices_in, indices_out_all;
    dr_vector<uint32_t> state(n_inst + 1, 0);
    dr_vector<uint32_t> inst_id(n_inst, 0);

    (collect_indices(indices_in, args), ...);

    detail::JitState<Backend> jit_state;
    jit_state.begin_recording();
    jit_state.new_scope();

    state[0] = jit_record_checkpoint(Backend);

    uint32_t n_inst_max = jit_registry_get_max(Backend, Base::Domain);
    for (uint32_t i = 1, j = 1; i <= n_inst_max; ++i) {
        snprintf(label, sizeof(label), "VCall: %s::%s() [instance %u]",
                 Base::Domain, name, j);

        Base *base = (Base *) jit_registry_get_ptr(Backend, Base::Domain, i);
        if (!base)
            continue;

#if defined(DRJIT_VCALL_DEBUG)
        jit_state.set_prefix(label);
#endif
        jit_state.set_self(i);

        Mask vcall_mask = true;
        if constexpr (Backend == JitBackend::LLVM) {
            // no-op to copy the mask into a local parameter
            vcall_mask = Mask::steal(jit_var_new_stmt(
                Backend, VarType::Bool,
                "$r0 = bitcast <$w x i1> %mask to <$w x i1>", 1, 0,
                nullptr));
        }

        jit_state.set_mask(vcall_mask.index(), false);

        if constexpr (std::is_same_v<Result, std::nullptr_t>) {
            func(base, (set_mask_true<Is, N>(args))...);
        } else {
            // The following assignment converts scalar return values
            Result tmp = func(base, set_mask_true<Is, N>(args)...);
            collect_indices(indices_out_all, tmp);
        }

        jit_state.clear_mask();

#if defined(DRJIT_VCALL_DEBUG)
        jit_state.clear_prefix();
#endif

        state[j] = jit_record_checkpoint(Backend);
        inst_id[j - 1] = i;
        j++;
    }

    dr_vector<uint32_t> indices_out((uint32_t) indices_out_all.size() / n_inst, 0);

    snprintf(label, sizeof(label), "%s::%s()", Base::Domain, name);

    uint32_t se = jit_var_vcall(
        label, self.index(), mask.index(), n_inst, inst_id.data(),
        (uint32_t) indices_in.size(), indices_in.data(),
        (uint32_t) indices_out_all.size(), indices_out_all.data(), state.data(),
        indices_out.data());

    jit_state.end_recording();
    jit_var_mark_side_effect(se);

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
    static constexpr JitBackend Backend = backend_v<Mask>;
    constexpr size_t N = sizeof...(Args);
    DRJIT_MARK_USED(N);

    // Evaluate the single instance with mask = true, mask side effects
    detail::JitState<Backend> jit_state;
    jit_state.set_mask(mask.index());

    if constexpr (is_drjit_struct_v<Result> || is_array_v<Result>) {
        // Return zero for masked results
        return select(mask, func(inst, (set_mask_true<Is, N>(args))...),
                      zeros<Result>());
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
    using DiffType = leaf_array_t<Result, Args...>;
    static constexpr JitBackend Backend = detached_t<Self>::Backend;
    using Mask = mask_t<Self>;

    auto [inst, n_inst] = vcall_registry_get(Backend, Base::Domain);

    size_t self_size = width(self, args...);
    Mask mask = extract_mask<Mask>(args...) && neq(self, nullptr);
    bool masked = mask.is_literal() && mask[0] == false,
         vcall_inline = jit_flag(JitFlag::VCallInline);

    if (n_inst == 0 || self_size == 0 || masked) {
        jit_log(::LogLevel::InfoSym,
                "jit_var_vcall(self=r%u): call (\"%s::%s()\") not performed (%s)",
                self.index(), Base::Domain, name,
                n_inst == 0 ? "no instances"
                            : (masked ? "masked" : "self.size == 0"));
        return zeros<Result>(self_size);
    } else if (n_inst == 1 && vcall_inline) {
        jit_log(::LogLevel::InfoSym,
                "jit_var_vcall(self=r%u): call (\"%s::%s()\") inlined (only 1 "
                "instance exists.)", self.index(), Base::Domain, name);
        return vcall_jit_record_impl_scalar<Result, Base>(
            (Base *) inst, func, mask,
            std::make_index_sequence<sizeof...(Args)>(), args...);
    } else {
        // Also check the mask stack to constrain side effects in recorded computation
        Mask mask_combined = mask && Mask::steal(jit_var_mask_peek(Backend));
        isolate_grad<DiffType> guard;

        return vcall_jit_record_impl<Result, Base>(
            name, n_inst, func, self, mask_combined,
            std::make_index_sequence<sizeof...(Args)>(),
            wrap_vcall(args)...);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
