/*
    drjit/vcall.h -- Vectorized method call support. This header file provides
    the logic to capture a call to ``Jit/DiffArray<T*>().foo()`` and dispatch
    it to ``T::foo()``.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/autodiff.h>
#include <drjit/struct.h>

#define DRJIT_VCALL_BEGIN(Name)                                                \
    namespace drjit {                                                          \
        template <typename Self>                                               \
        struct call_support<Name, Self> {                                      \
            using Class = Name;                                                \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Self &self) : self(self) { }                    \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_VCALL_TEMPLATE_BEGIN(Name)                                       \
    namespace drjit {                                                          \
        template <typename Self, typename... Ts>                               \
        struct call_support<Name<Ts...>, Self> {                               \
            using Class = Name<Ts...>;                                         \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Self &self) : self(self) { }                    \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_VCALL_END(Name)                                                  \
        private:                                                               \
            const Self &self;                                                  \
        };                                                                     \
    }

#define DRJIT_VCALL_METHOD(Name)                                               \
public:                                                                        \
    template <typename... Args> auto Name(const Args &...args) const {         \
        return drjit_impl_##Name(std::make_index_sequence<sizeof...(Args)>(),  \
                                 args...);                                     \
    }                                                                          \
                                                                               \
private:                                                                       \
    template <typename... Args, size_t... Is>                                  \
    auto drjit_impl_##Name(std::index_sequence<Is...>, const Args &...args)    \
        const {                                                                \
        using Ret = decltype(std::declval<Class &>().Name(args...));           \
        using Ret2 = std::conditional_t<std::is_void_v<Ret>, void_t, Ret>;     \
        using VCallStateT = detail::VCallState<Ret2, Args...>;                 \
        ad_vcall_callback callback = [](void *state_p, void *self,             \
                                        const dr_vector<uint64_t> &args_i,     \
                                        dr_vector<uint64_t> &rv_i) {           \
            VCallStateT *state = (VCallStateT *) state_p;                      \
            state->update_args(args_i);                                        \
            if constexpr (std::is_same_v<Ret, void>) {                         \
                ((Class *) self)->Name(state->args.template get<Is>()...);     \
            } else {                                                           \
                state->rv =                                                    \
                    ((Class *) self)->Name(state->args.template get<Is>()...); \
                state->collect_rv(rv_i);                                       \
            }                                                                  \
        };                                                                     \
        return detail::vcall<Self, Ret, Ret2, Args...>(                        \
            self, Domain, #Name "()", callback, args...);                      \
    }

NAMESPACE_BEGIN(drjit)

struct void_t { };

template <bool IncRef, typename T>
void collect_indices(const T &value, dr_vector<uint64_t> &indices) {
    if constexpr (depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            collect_indices<IncRef>(value.derived().entry(i), indices);
    } else if constexpr (is_tensor_v<T>) {
        collect_indices<IncRef>(value.array(), indices);
    } else if constexpr (is_jit_v<T>) {
        uint64_t index = value.index_combined();
        if constexpr (IncRef)
            ad_var_inc_ref(index);
        indices.push_back(index);
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](const auto &x) { collect_indices<IncRef>(x, indices); });
    }
}

template <typename T>
void update_indices(T &value, const dr_vector<uint64_t> &indices, size_t &pos) {
    if constexpr (depth_v<T> > 1) {
        for (size_t i = 0; i < value.derived().size(); ++i)
            update_indices(value.derived().entry(i), indices, pos);
    } else if constexpr (is_tensor_v<T>) {
        update_indices(value.array(), indices, pos);
    } else if constexpr (is_jit_v<T>) {
        value = T::borrow((typename T::Index) indices[pos++]);
    } else if constexpr (is_drjit_struct_v<T>) {
        struct_support_t<T>::apply_1(
            value, [&](auto &x) { update_indices(x, indices, pos); });
    }
}

template <typename T> void update_indices(T &value, const dr_vector<uint64_t> &indices) {
    size_t pos = 0;
    update_indices(value, indices, pos);
#if !defined(NDEBUG)
    if (pos != indices.size())
        throw std::runtime_error("update_indices(): did not consume the expected number of indices!");
#endif
}

NAMESPACE_BEGIN(detail)

template <typename Mask, typename ... Args>
Mask extract_mask(dr_tuple<Args...> &t) {
    constexpr size_t N = sizeof...(Args);
    Mask result = true;

    if constexpr (N > 0) {
        auto &last = t.template get<N-1>();
        if constexpr (is_mask_v<decltype(last)>)
            std::swap(result, last);
    }

    return result;
}

template <typename Ret, typename... Args> struct VCallState {
    dr_tuple<Args...> args;
    Ret rv;

    VCallState(const Args &...arg) : args(arg...) { }

    static void cleanup(void *p) {
        delete (VCallState *) p;
    }

    void update_args(const dr_vector<uint64_t> &indices) {
        update_indices(args, indices);
    }

    void collect_rv(dr_vector<uint64_t> &indices) const {
        collect_indices<false>(rv, indices);
    }
};

struct dr_index_vector : dr_vector<uint64_t> {
    using Base = dr_vector<uint64_t>;
    using Base::Base;
    ~dr_index_vector() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
    }
};

template <typename Self, typename Ret, typename Ret2, typename... Args>
Ret vcall(const Self &self, const char *domain, const char *name,
             ad_vcall_callback callback, const Args &...args) {
    using Mask = mask_t<Self>;
    using VCallStateT = VCallState<Ret2, Args...>;
    VCallStateT *state = new VCallStateT(args...);

    Mask mask = extract_mask<Mask>(state->args);

    dr_index_vector args_i, rv_i;
    collect_indices<true>(state->args, args_i);
    bool done =
        ad_vcall(Self::Backend, domain, 0, name, self.index(), mask.index(),
                 args_i, rv_i, state, callback, &VCallStateT::cleanup, true);

    if constexpr (!std::is_same_v<Ret, void>) {
        Ret2 result(std::move(state->rv));
        update_indices(result, rv_i);

        if (done)
            VCallStateT::cleanup(state);

        return result;
    } else {
        if (done)
            VCallStateT::cleanup(state);
    }
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
