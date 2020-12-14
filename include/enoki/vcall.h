/*
    enoki/vcall.h -- Vectorized method call support

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#define ENOKI_VCALL_H

#include <enoki/array.h>
#include <enoki/vcall_packet.h>

extern "C" {
    extern ENOKI_IMPORT uint32_t jitc_registry_put(const char *domain, void *ptr);
    extern ENOKI_IMPORT void jitc_registry_remove(void *ptr);
    extern ENOKI_IMPORT uint32_t jitc_registry_get_id(const void *ptr);
    extern ENOKI_IMPORT void jitc_registry_set_attr(void *ptr, const char *name,
                                                    const void *value, size_t size);
    extern ENOKI_IMPORT const void *jitc_registry_attr_data(const char *domain,
                                                            const char *name);
    extern ENOKI_IMPORT uint32_t jitc_flags();
};

NAMESPACE_BEGIN(enoki)

namespace detail {
    template <typename T>
    ENOKI_INLINE decltype(auto) copy_diff(const T& value) {
        if constexpr (is_jit_array_v<T> && is_diff_array_v<T> &&
                      std::is_floating_point_v<scalar_t<T>>) {
            T result;
            if constexpr (array_depth_v<T> == 1) {
                result = value.copy();
            } else {
                for (size_t i = 0; i < value.derived().size(); ++i)
                    result.entry(i) = copy_diff(value.entry(i));
            }
            return result;
        } else if constexpr (is_enoki_struct_v<T>) {
            T result = value;
            struct_support_t<T>::apply_1(result,
                                         [](auto &x) { x = copy_diff(x); });
            return result;
        } else {
            return value;
        }
    }

    template <typename Guide, typename Type, typename = int> struct vectorize_type {
        using type = Type;
    };

    template <typename Guide, typename Type>
    struct vectorize_type<Guide, Type, enable_if_t<std::is_scalar_v<Type> && !std::is_same_v<Type, std::nullptr_t>>> {
        using type = replace_scalar_t<Guide, Type>;
    };

    template <typename Result, typename Func, typename Self, typename... Args>
    ENOKI_INLINE Result dispatch_jit_symbolic(Func func, const Self &self, const Args&... args);

    template <typename Result, typename Func, typename Self, typename... Args>
    ENOKI_INLINE Result dispatch_jit_reduce(Func func, const Self &self, const Args&... args);

    template <typename Result, typename Func, typename FuncFwd, typename FuncRev, typename Self,
              typename... Args>
    ENOKI_INLINE Result dispatch_autodiff(const Func &func,
                                          const FuncFwd &func_fwd,
                                          const FuncRev &func_rev,
                                          const Self &self,
                                          const Args &... args);

    template <typename Class, typename Func, typename FuncFwd, typename FuncRev,
              typename Self, typename... Args>
    auto dispatch(const Func &func, const FuncFwd &func_fwd, const FuncRev func_rev,
                  const Self &self, const Args &... args) {
        using Result =
            typename vectorize_type<Self, decltype(func((Class *) nullptr, args...))>::type;

        if constexpr (is_jit_array_v<Self>) {
            if ((jitc_flags() & 2) == 0 || is_llvm_array_v<Self>) {
                return detail::dispatch_jit_reduce<Result>(func, self, copy_diff(args)...);
            } else {
                if constexpr (is_diff_array_v<Self>)
                    return detail::dispatch_autodiff<Result>(func, func_fwd, func_rev, self, args...);
                else
                    return detail::dispatch_jit_symbolic<Result>(func, self, args...);
            }
        } else {
            return detail::dispatch_packet<Result>(func, self, args...);
        }
    }
}

template <typename Class, typename Value>
void set_attr(Class *self, const char *name, const Value &value) {
    if constexpr (Class::Registered) {
        if constexpr (std::is_pointer_v<Value> &&
                      std::is_class_v<std::remove_pointer_t<Value>>) {
            set_attr(self, name, jitc_registry_get_id(value));
        } else {
            jitc_registry_set_attr(self, name, &value, sizeof(Value));
        }
    }
}

NAMESPACE_END(enoki)

#define ENOKI_VCALL_REGISTER_IF(Class, Cond)                                   \
    static constexpr const char *Domain = #Class;                              \
    static constexpr bool Registered = Cond;                                   \
    void *operator new(size_t size) {                                          \
        void *ptr = ::operator new(size);                                      \
        if constexpr (Registered)                                              \
            jitc_registry_put(#Class, ptr);                                    \
        return ptr;                                                            \
    }                                                                          \
    void *operator new(size_t size, std::align_val_t align) {                  \
        void *ptr = ::operator new(size, align);                               \
        if constexpr (Registered)                                              \
            jitc_registry_put(#Class, ptr);                                    \
        return ptr;                                                            \
    }                                                                          \
    void operator delete(void *ptr) {                                          \
        if constexpr (Registered)                                              \
            jitc_registry_remove(ptr);                                         \
        ::operator delete(ptr);                                                \
    }                                                                          \
    void operator delete(void *ptr, std::align_val_t align) {                  \
        if constexpr (Registered)                                              \
            jitc_registry_remove(ptr);                                         \
        ::operator delete(ptr, align);                                         \
    }

#define ENOKI_VCALL_REGISTER(Class)                                            \
    ENOKI_VCALL_REGISTER_IF(Class, true)

#define ENOKI_VCALL_METHOD(name)                                               \
    template <typename... Args> auto name(const Args &... args_) const {       \
        return detail::dispatch<Class>(                                        \
            [](auto self, const auto &... args) ENOKI_INLINE_LAMBDA {          \
                using Result = decltype(self->name(args...));                  \
                if constexpr (std::is_same_v<Result, void>) {                  \
                    self->name(args...);                                       \
                    return nullptr;                                            \
                } else {                                                       \
                    return self->name(args...);                                \
                }                                                              \
            },                                                                 \
            [](auto self, const auto &grad_in, auto ... args)                  \
                ENOKI_INLINE_LAMBDA {                                          \
                    enoki::enable_grad(args...);                               \
                    using Result = decltype(self->name(args...));              \
                    if constexpr (!std::is_same_v<Result, void>) {             \
                        Result result = self->name(args...);                   \
                        detail::tuple args_tuple{ args... };                   \
                        enoki::set_grad(args_tuple, grad_in);                  \
                        enoki::enqueue(args_tuple);                            \
                        enoki::traverse<decltype(result),                      \
                                        decltype(args)...>(false, true);       \
                        return enoki::grad(result);                            \
                    } else {                                                   \
                        self->name(args...);                                   \
                        detail::tuple args_tuple{ args... };                   \
                        enoki::set_grad(args_tuple, grad_in);                  \
                        enoki::enqueue(args_tuple);                            \
                        enoki::traverse<decltype(args)...>(false, true);       \
                        return nullptr;                                        \
                    }                                                          \
                },                                                             \
            [](auto self, const auto &grad_out, auto ... args)                 \
                ENOKI_INLINE_LAMBDA {                                          \
                    enoki::enable_grad(args...);                               \
                    using Result = decltype(self->name(args...));              \
                    if constexpr (!std::is_same_v<Result, void>) {             \
                        Result result = self->name(args...);                   \
                        enoki::set_grad(result, grad_out);                     \
                        enoki::enqueue(result);                                \
                        enoki::traverse<decltype(result),                      \
                                        decltype(args)...>(true, true);        \
                        return detail::tuple{ enoki::grad(args)... };          \
                    } else {                                                   \
                        self->name(args...);                                   \
                    }                                                          \
                    return detail::tuple{ enoki::grad(args)... };              \
                }, array, args_...);                                           \
    }

#define ENOKI_VCALL_GETTER(name, type)                                         \
    auto name(const mask_t<Array> &mask = true) const {                        \
        if constexpr (is_jit_array_v<Array>) {                                 \
            using Result = replace_scalar_t<Array, type>;                      \
            using UInt32 = uint32_array_t<Array>;                              \
            return enoki::gather<Result>(                                      \
                jitc_registry_attr_data(Domain, #name),                        \
                reinterpret_cast<const UInt32 &>(array), mask);                \
        } else {                                                               \
            return detail::dispatch<Class>(                                    \
                [](Class *ptr)                                                 \
                    ENOKI_INLINE_LAMBDA { return ptr->name(); },               \
                nullptr, array & mask);                                        \
        }                                                                      \
    }

#define ENOKI_VCALL_BEGIN(Name)                                                \
    namespace enoki {                                                          \
        template <typename Array>                                              \
        struct call_support<Name, Array> {                                     \
            using Class = Name;                                                \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Array &array) : array(array) { }                \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define ENOKI_VCALL_TEMPLATE_BEGIN(Name)                                       \
    namespace enoki {                                                          \
        template <typename Array, typename... Ts>                              \
        struct call_support<Name<Ts...>, Array> {                              \
            using Class = Name<Ts...>;                                         \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Array &array) : array(array) { }                \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define ENOKI_VCALL_END(Name)                                                  \
        private:                                                               \
            const Array &array;                                                \
        };                                                                     \
    }

#define ENOKI_VCALL_TEMPLATE_END(Name)                                         \
    ENOKI_VCALL_END(Name)

#if defined(ENOKI_CUDA_H) || defined(ENOKI_LLVM_H)
#  include <enoki/vcall_jit_reduce.h>
#  include <enoki/vcall_jit_symbolic.h>
#endif

#if defined(ENOKI_AUTODIFF_H)
#  include <enoki/vcall_autodiff.h>
#endif

