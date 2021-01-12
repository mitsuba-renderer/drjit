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
          typename FuncFwd, typename FuncRev, typename... Args>
struct DiffVCall
    : CustomOp<Type, Result, ConstStr, Self, Func, FuncFwd, FuncRev, Args...> {
    using Base = CustomOp<Type, Result, ConstStr, Self, Func, FuncFwd, FuncRev, Args...>;

    static constexpr bool ClearPrimal   = false;
    static constexpr bool ForceCreation = true;

    ~DiffVCall() { free(m_name); }

    Result eval(const ConstStr &name,
                const Self &self,
                const Func &func,
                const FuncFwd &,
                const FuncRev &,
                const Args &... args) override {
        if (m_name)
            free(m_name);
        m_name = strdup(name);
        return vcall_jit_record<Result>(name, func, self, args...);
    }

    template <size_t... Is>
    void forward_impl(std::index_sequence<Is...>) {
        const Self &self = Base::m_grad_input->template get<1>();
        const FuncFwd &func_fwd = Base::m_grad_input->template get<3>();

        size_t name_size = strlen(m_name) + 9;
        ek_unique_ptr<char[]> name(new char[name_size]);
        snprintf(name.get(), name_size, "ad_fwd[%s]", m_name);

        Result grad_out = vcall_jit_record<Result>(
            name.get(), func_fwd, self,
            ek_tuple(Base::template grad_in<5 + Is>()...),
            Base::template value_in<5 + Is>()...);

        Base::set_grad_out(grad_out);
    }

    template <size_t... Is>
    void backward_impl(std::index_sequence<Is...>) {
        const Self &self = Base::m_grad_input->template get<1>();
        const FuncRev &func_rev = Base::m_grad_input->template get<4>();
        using ResultRev = ek_tuple<Args...>;

        size_t name_size = strlen(m_name) + 9;
        ek_unique_ptr<char[]> name(new char[name_size]);
        snprintf(name.get(), name_size, "ad_rev[%s]", m_name);

        ResultRev grad_in = vcall_jit_record<ResultRev>(
            name.get(), func_rev, self, Base::grad_out(),
            Base::template value_in<5 + Is>()...);

        (Base::template set_grad_in<5 + Is>(grad_in.template get<Is>()), ...);
    }

    void forward() override {
        forward_impl(std::make_index_sequence<sizeof...(Args)>());
    }

    void backward() override {
        backward_impl(std::make_index_sequence<sizeof...(Args)>());
    }

    const char *name() const override {
        return m_name;
    }

private:
    char *m_name = nullptr;
};

template <typename Result, typename Func, typename FuncFwd, typename FuncRev,
          typename Self, typename... Args>
ENOKI_INLINE Result vcall_autodiff(const char *name,
                                   const Func &func, const FuncFwd &func_fwd,
                                   const FuncRev &func_rev, const Self &self,
                                   const Args &... args) {
    using Type = leaf_array_t<Result, Args...>;
    if constexpr (is_diff_array_v<Type> && std::is_floating_point_v<scalar_t<Type>>) {
        return custom<DiffVCall<Type, Self, Result, Func, FuncFwd, FuncRev, Args...>>(
            name, self, func, func_fwd, func_rev, args...);
    } else {
        return vcall_jit_record<Result>(name, func, self, args...);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(enoki)
