/*
    enoki/vcall.h -- Vectorized method call support, autodiff part

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

// #define ENOKI_VCALL_DEBUG

#include <enoki/custom.h>
#include <enoki/struct.h>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

inline void ad_copy() { }

template <typename T, typename... Ts> void ad_copy(T &value, Ts&...values) {
    ENOKI_MARK_USED(value);
    if constexpr (is_diff_array_v<T>) {
        if constexpr (array_depth_v<T> > 1) {
            for (size_t i = 0; i < value.size(); ++i)
                ad_copy(value.entry(i));
        } else {
            if (value.index_ad())
                value = value.derived().copy();
        }
    } else if constexpr (is_enoki_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [](auto &x1) ENOKI_INLINE_LAMBDA { ad_copy(x1); });
    }

    if constexpr (sizeof...(Ts) > 0)
        ad_copy(values...);
}

using ConstStr = const char *;

template <typename Type> struct ADProcessPostponedGuard {
    ~ADProcessPostponedGuard() {
        if (detail::ad_enqueue_postponed<Type>())
            detail::ad_traverse<Type>(ADMode::Reverse, true);
    }
};


template <typename DiffType, typename Self, typename Result, typename Func,
          typename... Args>
struct DiffVCall : CustomOp<DiffType, Result, ConstStr, Self, Func, Args...> {
    using Base = CustomOp<DiffType, Result, ConstStr, Self, Func, Args...>;
    using Type = typename DiffType::Type;
    using Base::m_implicit_in;

    static constexpr bool ClearPrimal = false;

    Result eval(const ConstStr &name, const Self &self, const Func &func,
                const Args &... args) override {
        ADProcessPostponedGuard<Type> guard;

        using Class = std::decay_t<std::remove_pointer_t<scalar_t<Self>>>;
        m_name_static = name;
        snprintf(m_name_long, sizeof(m_name_long), "VCall: %s::%s()",
                 Class::Domain, m_name_static);

        // Perform the function call
        size_t implicit_snapshot = ad_implicit<Type>();
        Result result = vcall_jit_record<Result>(name, func, self, args...);

        /// Capture implicit dependencies of the operation
        m_implicit_in = ek_vector<int32_t>(ad_implicit<Type>() - implicit_snapshot, 0);
        ad_extract_implicit<Type>(implicit_snapshot, m_implicit_in.data());
        for (size_t i = 0; i < m_implicit_in.size(); ++i)
            detail::ad_inc_ref<Type>(m_implicit_in[i]);

        return result;
    }

    void forward() override {
        forward_impl(std::make_index_sequence<sizeof...(Args)>());
    }

    template <size_t... Is> void forward_impl(std::index_sequence<Is...>) {
        const Self &self = Base::template value_in<1>();
        const Func &func = Base::template value_in<2>();

        auto func_fwd = [func](auto *self2, auto... value_grad_pair) {
            ad_copy(value_grad_pair.first...);
            enable_grad(value_grad_pair.first...);
            size_t implicit_snapshot = ad_implicit<Type>();
            Result result = func(self2, value_grad_pair.first...);
            ad_copy(result);
            (set_grad(value_grad_pair.first, value_grad_pair.second), ...);

#if defined(ENOKI_VCALL_DEBUG)
            ek_tuple args_t(value_grad_pair.first...);
            set_label(args_t, "args");
            set_label(result, "result");
            fprintf(stderr, "%s\n", ad_graphviz<Type>());
#endif

            enqueue(ADMode::Forward, value_grad_pair.first...);

            ad_enqueue_implicit<Type>(implicit_snapshot);
            traverse<DiffType>(ADMode::Forward, true);
            ad_dequeue_implicit<Type>(implicit_snapshot);

            return grad<false>(result);
        };

        size_t name_size = strlen(m_name_static) + 8;
        ek_unique_ptr<char[]> name(new char[name_size]);
        snprintf(name.get(), name_size, "%s_ad_fwd", m_name_static);

        Result grad_out = vcall_jit_record<Result>(
            name.get(), func_fwd, self,
            std::make_pair(Base::template value_in<3 + Is>(),
                           Base::template grad_in<3 + Is>())...);

        Base::set_grad_out(grad_out);
    }

    template <size_t... Is>
    void backward_impl(std::index_sequence<Is...>) {
        const Self &self = Base::template value_in<1>();
        const Func &func = Base::template value_in<2>();
        using Inputs = ek_tuple<Args...>;
        ADProcessPostponedGuard<Type> guard;

        auto func_rev = [func](auto *self2, auto &grad_out,
                               auto... args) -> Inputs {
            ad_copy(args...);
            enable_grad(args...);
            Result result = func(self2, args...);
            ad_copy(result);
            set_grad(result, grad_out);

#if defined(ENOKI_VCALL_DEBUG)
            ek_tuple args_t(args...);
            set_label(args_t, "args");
            set_label(result, "result");
            fprintf(stderr, "%s\n", ad_graphviz<Type>());
#endif

            enqueue(ADMode::Reverse, result);
            traverse<DiffType>(ADMode::Reverse, true);
            return Inputs(grad<false>(args)...);
        };

        size_t name_size = strlen(m_name_static) + 8;
        ek_unique_ptr<char[]> name(new char[name_size]);
        snprintf(name.get(), name_size, "%s_ad_rev", m_name_static);

        Inputs grad_in = vcall_jit_record<Inputs>(
            name.get(), func_rev, self, Base::grad_out(),
            Base::template value_in<3 + Is>()...);

        ENOKI_MARK_USED(grad_in);
        (Base::template set_grad_in<3 + Is>(grad_in.template get<Is>()), ...);
    }

    void backward() override {
        backward_impl(std::make_index_sequence<sizeof...(Args)>());
    }

    const char *name() const override { return m_name_long; }

private:
    const char *m_name_static = nullptr;
    char m_name_long[128];
};

inline std::pair<void *, uint32_t> vcall_registry_get(JitBackend Backend,
                                                      const char *domain);

template <typename Result, typename Func, typename Self, typename... Args>
ENOKI_INLINE Result vcall_autodiff(const char *name, const Func &func,
                                   const Self &self, const Args &... args) {
    using Type = leaf_array_t<Result, Args...>;
    using Base = std::remove_const_t<std::remove_pointer_t<value_t<Self>>>;

    /* Only perform a differentiable vcall if there is a differentiable
       float type somewhere within the argument or return values */
    if constexpr (is_diff_array_v<Type> && std::is_floating_point_v<scalar_t<Type>>) {
        auto [base, n_inst] = vcall_registry_get(backend_v<Self>, Base::Domain);

        // Complex approach in this header file only needed if > 1 instance
        if (n_inst > 1)
            return custom<DiffVCall<Type, Self, Result, Func, Args...>>(
                name, self, func, args...);
    }

    return vcall_jit_record<Result>(name, func, self, args...);
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
