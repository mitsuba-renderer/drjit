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
    ENOKI_INLINE decltype(auto) gather_helper(const T& value, const UInt32 &perm) {
        if constexpr (is_jit_array_v<T>) {
            return enoki::gather<T, true>(value, perm);
        } else if constexpr (is_enoki_struct_v<T>) {
            T result = value;
            struct_support_t<T>::apply_1(
                result, [&perm](auto &x) { x = gather_helper(x, perm); });
            return result;
        } else {
            ENOKI_MARK_USED(perm);
            return value;
        }
    }

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
    struct vectorize_type<
        Guide, Type, enable_if_t<!std::is_void_v<Type> && std::is_scalar_v<Type>>> {
        using type = replace_scalar_t<Guide, Type>;
    };

    template <typename Guide, typename Type>
    using vectorize_type_t = typename vectorize_type<Guide, Type>::type;

    template <typename Mask> ENOKI_INLINE Mask get_mask() {
        return Mask(true);
    }

    template <typename Mask, typename Arg, typename... Args>
    ENOKI_INLINE Mask get_mask(const Arg &arg, const Args &... args) {
        if constexpr (is_mask_v<Arg>)
            return Mask(arg);
        else
            return get_mask<Mask>(args...);
    }

    template <typename Arg, typename Mask>
    ENOKI_INLINE auto &replace_mask(Arg &arg, const Mask &mask) {
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
            using UInt32 = uint32_array_t<Array>;

            if constexpr (!std::is_void_v<FuncRV>) {
                using Result = detail::vectorize_type_t<Array, FuncRV>;
                Result result = enoki::empty<Result>(self.size());

                if (self.size() == 1) {
                    void* ptr = (void*)self.entry(0);
                    if (ptr)
                        result = func(ptr, args...);
                    else
                        result = zero<Result>();
                } else {
                    enoki::schedule(args...);
                    auto [buckets, size] = self.vcall_();
                    for (size_t i = 0; i < size; ++i) {
                        UInt32 perm = UInt32::borrow(buckets[i].index);

                        if (buckets[i].ptr) {
                            enoki::scatter<true>(
                                result,
                                ref_cast_t<FuncRV, Result>(func(
                                    buckets[i].ptr, detail::gather_helper(args, perm)...)),
                                perm);
                        } else {
                            enoki::scatter<true>(result, zero<Result>(), perm);
                        }
                    }
                    enoki::schedule(result);
                }
                return result;
            } else {
                if (self.size() == 1) {
                    void* ptr = (void*)self.entry(0);
                    if (ptr)
                        func(ptr, args...);
                } else {
                    auto [buckets, size] = self.vcall_();
                    for (size_t i = 0; i < size; ++i) {
                        if (!buckets[i].ptr)
                            continue;
                        UInt32 perm = UInt32::borrow(buckets[i].index);
                        func(buckets[i].ptr, detail::gather_helper(args, perm)...);
                    }
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
    template <typename... Args> auto name(const Args &... args) const {        \
        return detail::dispatch(                                               \
            [](void *ptr, const auto &... args2) ENOKI_INLINE_LAMBDA {         \
                return ((Class *) ptr)->name(args2...);                        \
            }, array, detail::copy_diff(args)...);                             \
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
            return detail::dispatch(                                           \
                [](void *ptr)                                                  \
                    ENOKI_INLINE_LAMBDA { return ((Class *) ptr)->name(); },   \
                array & mask);                                                 \
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

#define ENOKI_VCALL_END(Name)                                                  \
        private:                                                               \
            const Array &array;                                                \
        };                                                                     \
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

#define ENOKI_VCALL_TEMPLATE_END(Name)                                         \
    ENOKI_VCALL_END(Name)
