/*
    drjit/call.h -- Vectorized method call support. This header file provides
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

NAMESPACE_BEGIN(drjit)

#define DRJIT_CALL_BEGIN(Name)                                                 \
    namespace drjit {                                                          \
        template <typename Self>                                               \
        struct call_support<Name, Self> {                                      \
            using Base_ = void;                                                \
            using Class_ = Name;                                               \
            using Mask_ = mask_t<Self>;                                        \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Self &self) : self(self) { }                    \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_CALL_TEMPLATE_BEGIN(Name)                                        \
    namespace drjit {                                                          \
        template <typename Self, typename... Ts>                               \
        struct call_support<Name<Ts...>, Self> {                               \
            using Base_ = void;                                                \
            using Class_ = Name<Ts...>;                                        \
            using Mask_ = mask_t<Self>;                                        \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Self &self) : self(self) { }                    \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_CALL_TEMPLATE_INHERITED_BEGIN(Name, Parent)                      \
    namespace drjit {                                                          \
        template <typename Self, typename... Ts>                               \
        struct call_support<Name<Ts...>, Self>                                 \
                : call_support<Parent<Ts...>, Self> {                          \
            using Base_ = call_support<Parent<Ts...>, Self>;                   \
            using Base_::self;                                                 \
            using Base_::Domain;                                               \
            using Class_ = Name<Ts...>;                                        \
            using Mask_ = mask_t<Self>;                                        \
            call_support(const Self &self) : Base_(self) { }                   \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_CALL_INHERITED_END(Name)                                         \
        };                                                                     \
    }

#define DRJIT_CALL_END(Name)                                                   \
        protected:                                                             \
            const Self &self;                                                  \
        };                                                                     \
    }

#define DRJIT_CALL_METHOD(Name)                                                \
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
        using Ret = decltype(std::declval<Class_ &>().Name(args...));          \
        using Ret2 = std::conditional_t<std::is_void_v<Ret>, std::nullptr_t,   \
                                        vectorize_rv_t<Self, Ret>>;            \
        using CallStateT = detail::CallState<Ret2, Args...>;                   \
                                                                               \
        ad_call_func callback = [](void *state_p, void *self_,                 \
                                   const vector<uint64_t> &args_i,             \
                                   vector<uint64_t> &rv_i) {                   \
            CallStateT *state = (CallStateT *) state_p;                        \
            state->update_args(args_i);                                        \
            if constexpr (std::is_same_v<Ret, void>) {                         \
                if (detail::is_valid_call_ptr<Class_, Base_>(self_))           \
                    ((Class_ *) self_)->Name(drjit::get<Is>(state->args)...);  \
            } else {                                                           \
                if (detail::is_valid_call_ptr<Class_, Base_>(self_))           \
                    state->rv = ((Class_ *) self_)                             \
                                    ->Name(drjit::get<Is>(state->args)...);    \
                else                                                           \
                    state->rv = zeros<Ret2>();                                 \
                state->collect_rv(rv_i);                                       \
            }                                                                  \
        };                                                                     \
                                                                               \
        return detail::call<Self, Ret, Ret2, Args...>(                         \
            self, Domain, #Name "()", false, callback, args...);               \
    }

#define DRJIT_CALL_GETTER(Name)                                                \
public:                                                                        \
    auto Name(Mask_ mask = true) const {                                       \
        using Ret =                                                            \
            vectorize_rv_t<Self, decltype(std::declval<Class_ &>().Name())>;   \
        using CallStateT = detail::CallState<Ret, Mask_>;                      \
                                                                               \
        ad_call_func callback = [](void *state_p, void *self_,                 \
                                   const vector<uint64_t> &,                   \
                                   vector<uint64_t> &rv_i) {                   \
            CallStateT *state = (CallStateT *) state_p;                        \
            if (detail::is_valid_call_ptr<Class_, Base_>(self_))               \
                state->rv = ((Class_ *) self_)->Name();                        \
            else                                                               \
                state->rv = zeros<Ret>();                                      \
            state->collect_rv(rv_i);                                           \
        };                                                                     \
                                                                               \
        return detail::call<Self, Ret, Ret, Mask_>(self, Domain, #Name "()",   \
                                                  true, callback, mask);       \
    }
template <typename Guide, typename T>
using vectorize_rv_t =
    std::conditional_t<std::is_scalar_v<T>, replace_scalar_t<Guide, T>, T>;

NAMESPACE_BEGIN(detail)

/**
 * Since instances of derived classes are part of the DrJit
 * registry domain of their base class, CallSupport may
 * receive pointers to instances of type other than Class.
 * We use a `dynamic_cast` to skip calls on these invalid pointers.
 */
template <typename ChildClass, typename Parent>
bool is_valid_call_ptr(void *ptr) {
    if constexpr (std::is_same_v<Parent, void>) {
        return ptr != nullptr;
    } else {
        return dynamic_cast<ChildClass *>((typename Parent::Class_ *) ptr);
    }
}

template <typename Mask, typename ... Args>
Mask extract_mask(drjit::tuple<Args...> &t) {
    constexpr size_t N = sizeof...(Args);
    Mask result = true;

    if constexpr (N > 0) {
        auto &last = drjit::get<N-1>(t);
        if constexpr (is_mask_v<decltype(last)>)
            std::swap(result, last);
    }

    return result;
}

template <typename Ret, typename... Args> struct CallState {
    drjit::tuple<Args...> args;
    Ret rv;

    CallState(const Args &...arg) : args(arg...) { }

    static void cleanup(void *p) {
        delete (CallState *) p;
    }

    void update_args(const vector<uint64_t> &indices) {
        update_indices(args, indices);
    }

    void collect_rv(vector<uint64_t> &indices) const {
        collect_indices<false>(rv, indices);
    }
};

template <typename Self, typename Ret, typename Ret2, typename... Args>
Ret call(const Self &self, const char *domain, const char *name,
         bool is_getter, ad_call_func callback, const Args &...args) {
    using Mask = mask_t<Self>;
    using CallStateT = CallState<Ret2, Args...>;
    CallStateT *state = new CallStateT(args...);

    Mask mask = extract_mask<Mask>(state->args);

    index64_vector args_i, rv_i;
    collect_indices<true>(state->args, args_i);
    bool done = ad_call(Self::Backend, domain, -1, 0, name, is_getter,
                        self.index(), mask.index(), args_i, rv_i, state,
                        callback, &CallStateT::cleanup, true);

    if constexpr (!std::is_same_v<Ret, void>) {
        Ret2 result(std::move(state->rv));
        if (!rv_i.empty())
            update_indices(result, rv_i);
        else
            result = zeros<Ret2>();

        if (done)
            CallStateT::cleanup(state);

        return result;
    } else {
        if (done)
            CallStateT::cleanup(state);
    }
}

template <typename Self, typename Func, typename... Args, size_t... Is>
auto dispatch_impl(std::index_sequence<Is...>, const Self &self, const Func &func, const Args &... args) {
    using Ptr = value_t<Self>;
    using Ret = decltype(func(std::declval<Ptr>(), args...));
    using Ret2 = std::conditional_t<std::is_void_v<Ret>, std::nullptr_t,
                                   vectorize_rv_t<Self, Ret>>;
    using CallStateT = detail::CallState<Ret2, Func, Args...>;

    ad_call_func callback = [](void *state_p, void *self,
                               const vector<uint64_t> &args_i,
                               vector<uint64_t> &rv_i) {
        CallStateT *state = (CallStateT *) state_p;
        state->update_args(args_i);
        const Func &func = drjit::get<0>(state->args);

        if constexpr (std::is_same_v<Ret, void>) {
            if (self)
                func((Ptr) self, drjit::get<1 + Is>(state->args)...);
        } else {
            if (self)
                state->rv = func((Ptr) self, drjit::get<1 + Is>(state->args)...);
            else
                state->rv = zeros<Ret2>();
            state->collect_rv(rv_i);
        }
    };

    return detail::call<Self, Ret, Ret2, Func, Args...>(
        self, Self::CallSupport::Domain, "drjit::dispatch()", false, callback,
        func, args...);
}

NAMESPACE_END(detail)

template <typename Self, typename Func, typename... Args>
auto dispatch(const Self &self, const Func &func, const Args &... args) {
    return detail::dispatch_impl(std::make_index_sequence<sizeof...(Args)>(),
                                 self, func, args...);
}

NAMESPACE_END(drjit)
