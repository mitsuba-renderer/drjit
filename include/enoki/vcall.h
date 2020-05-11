/*
    enoki/vcall.h -- Vectorized method call support

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>

NAMESPACE_BEGIN(enoki)

namespace detail {
    template <typename T, typename UInt32>
    ENOKI_INLINE decltype(auto) gather_helper(T& v, const UInt32 &perm) {
        ENOKI_MARK_USED(perm);
        using DT = std::decay_t<T>;
        if constexpr (!is_jit_array_v<DT> && !has_struct_support_v<DT>)
            return v;
        else
            return gather<DT, true>(v, perm);
    }


    template <typename Guide, typename Type, typename = int> struct vectorize_type {
        using type = Type;
    };

    template <typename Guide, typename Type>
    struct vectorize_type<
        Guide, Type, enable_if_t<!std::is_void_v<Type> && !is_array_v<Type>>> {
        using type = replace_scalar_t<Guide, Type>;
    };

    template <typename Guide, typename Type>
    using vectorize_type_t = typename vectorize_type<Guide, Type>::type;

    template <typename Mask> ENOKI_INLINE Mask get_mask() {
        return Mask(true);
    }

    template <typename Mask, typename Arg, typename...Args> ENOKI_INLINE Mask get_mask(const Arg &arg, const Args&... args) {
        if constexpr (is_mask_v<Arg>)
            return Mask(arg);
        else
            return get_mask<Mask>(args...);
    }

    template <typename Arg, typename Mask> ENOKI_INLINE auto& replace_mask(Arg &arg, const Mask &mask) {
        if constexpr (is_mask_v<Arg>)
            return mask;
        else
            return arg;
    }

    template <typename Func, typename Array, typename... Args>
    auto dispatch(Func func, const Array &self, Args&&... args) {
        using FuncRV = decltype(func(nullptr, args...));
        using Mask = mask_t<Array>;

        if constexpr (is_jit_array_v<Array>) {
            enoki::schedule(args...);

            if constexpr (!std::is_void_v<FuncRV>) {
                using Result = detail::vectorize_type_t<Array, FuncRV>;
                Result result;

                if (self.size() == 1) {
                    result = func(self.entry(0), args...);
                } else {
                    result = zero<Result>(self.size());
                    for (auto const &kv : self.vcall_()) {
                        scatter<true>(
                            result,
                            ref_cast_t<FuncRV, Result>(func(
                                kv.first, detail::gather_helper(args, kv.second)...)),
                            kv.second);
                    }
                    enoki::schedule(result);
                }
                return result;
            } else {
                if (self.size() == 1) {
                    func(self.entry(0), args...);
                } else {
                    for (auto const &kv : self.vcall_())
                        func(kv.first, detail::gather_helper(args, kv.second)...);
                }
            }
        } else {
            using Instance = scalar_t<Array>;
            Mask mask = get_mask<Mask>(args...);
            mask &= neq(self, nullptr);

            if constexpr (!std::is_void_v<FuncRV>) {
                using Result = detail::vectorize_type_t<Array, FuncRV>;
                Result result = zero<Result>(self.size());
                while (any(mask)) {
                    Instance instance      = extract(self, mask);
                    Mask active            = mask & eq(self, instance);
                    mask                   = andnot(mask, active);
                    masked(result, active) = func(instance, replace_mask(args, active)...);
                }
                return result;
            } else {
                while (any(mask)) {
                    Instance instance = extract(self, mask);
                    Mask active       = mask & eq(self, instance);
                    mask              = andnot(mask, active);
                    func(instance, replace_mask(args, active)...);
                }
            }
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

extern "C" {
    extern ENOKI_IMPORT uint32_t jitc_registry_put(const char *domain, void *ptr);
    extern ENOKI_IMPORT void jitc_registry_remove(void *ptr);
    extern ENOKI_IMPORT uint32_t jitc_registry_get_id(const void *ptr);
    extern ENOKI_IMPORT void jitc_registry_set_attr(void *ptr, const char *name,
                                                    const void *value, size_t size);
    extern ENOKI_IMPORT const void *jitc_registry_attr_data(const char *domain,
                                                            const char *name);
};

#define ENOKI_VCALL_REGISTER_IF(Class, Cond)                                   \
    static constexpr bool Registered = Cond;                                   \
    void *operator new(size_t size) {                                          \
        void *ptr = ::operator new(size);                                      \
        if (Registered)                                                        \
            jitc_registry_put(#Class, ptr);                                    \
        return ptr;                                                            \
    }                                                                          \
    void *operator new(size_t size, std::align_val_t align) {                  \
        void *ptr = ::operator new(size, align);                               \
        if (Registered)                                                        \
            jitc_registry_put(#Class, ptr);                                    \
        return ptr;                                                            \
    }                                                                          \
    void operator delete(void *ptr) {                                          \
        if (Registered)                                                        \
            jitc_registry_remove(ptr);                                         \
        ::operator delete(ptr);                                                \
    }                                                                          \
    void operator delete(void *ptr, std::align_val_t align) {                  \
        if (Registered)                                                        \
            jitc_registry_remove(ptr);                                         \
        ::operator delete(ptr, align);                                         \
    }

#define ENOKI_VCALL_REGISTER(Class)                                            \
    ENOKI_VCALL_REGISTER_IF(Class, true)

#define ENOKI_VCALL_METHOD(name)                                               \
    template <typename... Args> auto name(Args &&... args) const {             \
        return detail::dispatch(                                               \
            [](void *ptr, auto &&... args2) ENOKI_INLINE_LAMBDA {              \
                return ((Base *) ptr)->name(args2...);                         \
            },                                                                 \
            array, std::forward<Args>(args)...);                               \
    }

#define ENOKI_VCALL_GETTER(name)                                               \
    auto get_field(const mask_t<Array> &mask = true) const {                   \
        if constexpr (is_jit_array_v<Array>) {                                 \
            using Result = replace_scalar_t<Array, float>;                     \
            return gather<Result>(jitc_registry_attr_data(Domain, #name),      \
                                  reinterpret_cast<const UInt32 &>(array),     \
                                  mask);                                       \
        } else {                                                               \
            return detail::dispatch(                                           \
                [](void *ptr)                                                  \
                    ENOKI_INLINE_LAMBDA { return ((Base *) ptr)->name(); },    \
                array & mask);                                                 \
        }                                                                      \
    }

#define ENOKI_VCALL_BEGIN(Name)                                                \
    namespace enoki {                                                          \
        template <typename Array> struct call_support<Name, Array> {           \
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
