/*
    enoki/vcall.h -- Vectorized method call support, autodiff part

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/custom.h>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

using ConstStr = const char *;

template <typename Type, typename Self, typename Result, typename Func,
          typename... Args>
struct DiffVCall : CustomOp<Type, Result, ConstStr, Self, Func, Args...> {
    using Base = CustomOp<Type, Result, ConstStr, Self, Func, Args...>;

    static constexpr bool ClearPrimal   = false;
    static constexpr bool ForceCreation = false;

    Result eval(const ConstStr &name, const Self &self, const Func &func,
                const Args &... args) override {
        using Class = std::decay_t<std::remove_pointer_t<scalar_t<Self>>>;
        m_name_static = name;
        snprintf(m_name_long, sizeof(m_name_long), "VCall: %s::%s()",
                 Class::Domain, m_name_static);
        return vcall_jit_record<Result>(name, func, self, args...);
    }

    void forward() override {
        forward_impl(std::make_index_sequence<sizeof...(Args)>());
    }

    template <size_t... Is> void forward_impl(std::index_sequence<Is...>) {
        const Self &self = Base::template value_in<1>();
        const Func &func = Base::template value_in<2>();

        auto func_fwd = [func](auto *self2, auto... value_grad_pair) {
            enoki::enable_grad(value_grad_pair.first...);
            Result result = func(self2, value_grad_pair.first...);
            (enoki::set_grad(value_grad_pair.first, value_grad_pair.second), ...);
            enoki::enqueue(value_grad_pair.first...);
            enoki::traverse<Type>(false, true);
            return enoki::grad<false>(result);
        };

        size_t name_size = strlen(m_name_static) + 11;
        ek_unique_ptr<char[]> name(new char[name_size]);
        snprintf(name.get(), name_size, "%s [AD, fwd]", m_name_static);

        Result grad_out = vcall_jit_record<Result>(
            name.get(), func_fwd, self,
            std::make_pair(Base::template value_in<3 + Is>(),
                           Base::template grad_in<3 + Is>())...);

        Base::set_grad_out(grad_out);
    }

#if 0
    template <size_t... Is>
    void backward_impl(std::index_sequence<Is...>) {
        const Self &self = Base::m_grad_input->template get<1>();
        const FuncRev &func_rev = Base::m_grad_input->template get<4>();
        using ResultRev = ek_tuple<Args...>;

        size_t name_size = strlen(m_name_static) + 11;
        ek_unique_ptr<char[]> name(new char[name_size]);
        snprintf(name.get(), name_size, "%s [AD, rev]", m_name_static);

        ResultRev grad_in = vcall_jit_record<ResultRev>(
            name.get(), func_rev, self, Base::grad_out(),
            Base::template value_in<5 + Is>()...);

        (Base::template set_grad_in<5 + Is>(grad_in.template get<Is>()), ...);
    }
#endif

    void backward() override { }
#if 0
    void backward() override {
        backward_impl(std::make_index_sequence<sizeof...(Args)>());
    }
#endif

    const char *name() const override { return m_name_long; }

private:
    const char *m_name_static = nullptr;
    char m_name_long[128];
};

template <typename Result, typename Func, typename Self, typename... Args>
ENOKI_INLINE Result vcall_autodiff(const char *name, const Func &func,
                                   const Self &self, const Args &... args) {
    using Type = leaf_array_t<Result, Args...>;

    /* Only perform a differentiable vcall if there is a differentiable
       float type somewhere within the argument or return values */
    if constexpr (is_diff_array_v<Type> && std::is_floating_point_v<scalar_t<Type>>) {
        return custom<DiffVCall<Type, Self, Result, Func, Args...>>(
            name, self, func, args...);
    } else {
        return vcall_jit_record<Result>(name, func, self, args...);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
