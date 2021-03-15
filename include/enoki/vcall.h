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
extern ENOKI_IMPORT uint32_t jit_registry_put(JitBackend backend,
                                              const char *domain, void *ptr);
extern ENOKI_IMPORT void jit_registry_remove(JitBackend backend, void *ptr);
extern ENOKI_IMPORT uint32_t jit_registry_get_id(JitBackend backend,
                                                 const void *ptr);
extern ENOKI_IMPORT void jit_registry_set_attr(JitBackend backend, void *ptr,
                                               const char *name,
                                               const void *value, size_t size);
extern ENOKI_IMPORT uint32_t jit_var_registry_attr(JitBackend backend,
                                                   VarType type,
                                                   const char *domain,
                                                   const char *name);
extern ENOKI_IMPORT uint32_t jit_flags();
};

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

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
        struct_support_t<T>::apply_1(result, [](auto &x) { x = copy_diff(x); });
        return result;
    } else {
        return value;
    }
}

template <typename Mask> Mask extract_mask() { return true; }

template <typename Mask, typename T, typename... Ts>
Mask extract_mask(const T &v, const Ts &... vs) {
    ENOKI_MARK_USED(v);
    if constexpr (sizeof...(Ts) != 0)
        return extract_mask<Mask>(vs...);
    else if constexpr (is_mask_v<T>)
        return v;
    else
        return true;
}

template <size_t I, size_t N, typename T>
decltype(auto) set_mask_true(const T &v) {
    if constexpr (is_mask_v<T> && I == N - 1)
        return T(true);
    else
        return (const T &) v;
}

template <typename Guide, typename Type, typename = int> struct vectorize_type {
    using type = Type;
};

template <typename Guide, typename Type>
struct vectorize_type<Guide, Type, enable_if_t<std::is_scalar_v<Type> && !std::is_same_v<Type, std::nullptr_t>>> {
    using type = replace_scalar_t<Guide, Type>;
};

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_reduce(const Func &func, const Self &self,
                        const Args &... args);

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_jit_record(const char *name, const Func &func, Self &self,
                        const Args &... args);

template <typename Result, typename Func, typename Self, typename... Args>
Result vcall_autodiff(const char *name, const Func &func, const Self &self,
                      const Args &... args);

template <typename Class, typename Func, typename Self, typename... Args>
auto vcall(const char *name, const Func &func, const Self &self,
           const Args &... args) {
    using Output = decltype(func((Class *) nullptr, args...));
    using Result = typename vectorize_type<Self, Output>::type;

    ENOKI_MARK_USED(name);
    if constexpr (is_jit_array_v<Self>) {
        if ((jit_flags() & 4 /* RecordVCall */) == 0) {
            return detail::vcall_jit_reduce<Result>(func, self, copy_diff(args)...);
        } else {
            if constexpr (is_diff_array_v<Self>)
                return detail::vcall_autodiff<Result>(name, func, self, args...);
            else
                return detail::vcall_jit_record<Result>(name, func, self, args...);
        }
    } else {
        return detail::vcall_packet<Result>(func, self, args...);
    }
}

NAMESPACE_END(detail)

template <typename Class, typename Value>
void set_attr(Class *self, const char *name, const Value &value) {
    ENOKI_MARK_USED(self);
    ENOKI_MARK_USED(name);
    ENOKI_MARK_USED(value);
    if constexpr (Class::Registered) {
        if constexpr (std::is_pointer_v<Value> &&
                      std::is_class_v<std::remove_pointer_t<Value>>) {
            set_attr(self, name, jit_registry_get_id(Class::Backend, value));
        } else {
            jit_registry_set_attr(Class::Backend, self, name, &value,
                                  sizeof(Value));
        }
    }
}

NAMESPACE_END(enoki)

#define ENOKI_VCALL_REGISTER(Array, Class)                                     \
    static constexpr const char *Domain = #Class;                              \
    static constexpr bool Registered = enoki::is_jit_array_v<Array>;           \
    static constexpr JitBackend Backend = enoki::backend_v<Array>;             \
    void *operator new(size_t size) {                                          \
        void *ptr = ::operator new(size);                                      \
        if constexpr (Registered)                                              \
            jit_registry_put(Backend, #Class, ptr);                            \
        return ptr;                                                            \
    }                                                                          \
    void *operator new(size_t size, std::align_val_t align) {                  \
        void *ptr = ::operator new(size, align);                               \
        if constexpr (Registered)                                              \
            jit_registry_put(Backend, #Class, ptr);                            \
        return ptr;                                                            \
    }                                                                          \
    void operator delete(void *ptr) {                                          \
        if constexpr (Registered)                                              \
            jit_registry_remove(Backend, ptr);                                 \
        ::operator delete(ptr);                                                \
    }                                                                          \
    void operator delete(void *ptr, std::align_val_t align) {                  \
        if constexpr (Registered)                                              \
            jit_registry_remove(Backend, ptr);                                 \
        ::operator delete(ptr, align);                                         \
    }

#define ENOKI_VCALL_METHOD(name)                                               \
    template <typename... Args> auto name(const Args &... args_) const {       \
        return detail::vcall<Class>(                                           \
            #name,                                                             \
            [](auto self, const auto &... args) ENOKI_INLINE_LAMBDA {          \
                using Result = decltype(self->name(args...));                  \
                if constexpr (std::is_same_v<Result, void>) {                  \
                    self->name(args...);                                       \
                    return nullptr;                                            \
                } else {                                                       \
                    return self->name(args...);                                \
                }                                                              \
            },                                                                 \
            array, args_...);                                                  \
    }

#define ENOKI_VCALL_GETTER(name, type)                                         \
    auto name(const mask_t<Array> &mask = true) const {                        \
        if constexpr (is_jit_array_v<Array>) {                                 \
            using Result = replace_scalar_t<Array, type>;                      \
            using UInt32 = uint32_array_t<Array>;                              \
            uint32_t attr_id = jit_var_registry_attr(                          \
                detached_t<Result>::Backend, detached_t<Result>::Type,         \
                Domain, #name);                                                \
            if (attr_id == 0)                                                  \
                return zero<Result>();                                         \
            else                                                               \
                return enoki::gather<Result>(Result::steal(attr_id),           \
                            UInt32::borrow(array.index()),                     \
                            mask && enoki::neq(array, nullptr));               \
        } else {                                                               \
            return detail::vcall<Class>(                                       \
                #name, [](auto self)                                           \
                    ENOKI_INLINE_LAMBDA { return self->name(); },              \
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

#if defined(ENOKI_JIT_H)
#  include <enoki/vcall_jit_reduce.h>
#  include <enoki/vcall_jit_record.h>
#endif

#if defined(ENOKI_AUTODIFF_H)
#  include <enoki/vcall_autodiff.h>
#endif

