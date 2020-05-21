/*
    enoki/struct.h -- Infrastructure for working with general data structures

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <enoki/map.h>

#pragma once

#define ENOKI_STRUCT_SCATTER(x) \
    enoki::scatter<Perm>(dst.x, src.x, index, mask);

#define ENOKI_STRUCT_GATHER(x) \
    result.x = enoki::gather<decltype(src.x), Perm>(src.x, index, mask);

#define ENOKI_STRUCT_ZERO(x) \
    result.x = enoki::zero<decltype(Class::x)>(size);

#define ENOKI_STRUCT_EMPTY(x) \
    result.x = enoki::empty<decltype(Class::x)>(size);

#define ENOKI_STRUCT_SET_GRAD_ENABLED(x) \
    enoki::set_grad_enabled(v.x, value);

#define ENOKI_STRUCT_DETACHED(x) \
    result.x = enoki::detach(v.x);

#define ENOKI_STRUCT_GRADIENT(x) \
    result.x = enoki::grad(v.x);

#define ENOKI_STRUCT_MASKED(x) \
    result.x = enoki::masked(v.x, mask);

#define ENOKI_STRUCT_CONSTR_COPY(x) \
    x(v.x)

#define ENOKI_STRUCT_CONSTR_MOVE(x) \
    x(std::move(v.x))

#define ENOKI_STRUCT_ASSIGN_COPY(x) \
    x = v.x;

#define ENOKI_STRUCT_ASSIGN_MOVE(x) \
    x = std::move(v.x);

#define ENOKI_STRUCT_WIDTH(x) \
    enoki::width(v.x)

#define ENOKI_STRUCT_ITEMS(x) \
    v.x

#define ENOKI_STRUCT_SET_LABEL(x) \
    snprintf(tmp, sizeof(tmp), "%s_%s", label, #x); \
    enoki::set_label(v.x, tmp);

#define ENOKI_STRUCT(Name, ...)                                                \
    Name() = default;                                                          \
    Name(const Name &) = default;                                              \
    Name(Name &&) = default;                                                   \
    Name &operator=(const Name &) = default;                                   \
    Name &operator=(Name &&) = default;                                        \
    template <typename... Ts>                                                  \
    Name(const Name<Ts...> &v)                                                 \
        : ENOKI_MAPC(ENOKI_STRUCT_CONSTR_COPY, __VA_ARGS__) { }                \
    template <typename... Ts>                                                  \
    Name(Name<Ts...> &&v)                                                      \
        : ENOKI_MAPC(ENOKI_STRUCT_CONSTR_MOVE, __VA_ARGS__) { }                \
    template <typename... Ts> Name &operator=(Name<Ts...> &&v) {               \
        ENOKI_MAP(ENOKI_STRUCT_ASSIGN_MOVE, __VA_ARGS__)                       \
        return *this;                                                          \
    }                                                                          \
    template <typename... Ts> Name &operator=(const Name<Ts...> &v) {          \
        ENOKI_MAP(ENOKI_STRUCT_ASSIGN_COPY, __VA_ARGS__)                       \
        return *this;                                                          \
    }

#define ENOKI_STRUCT_SUPPORT(Name, ...)                                        \
    namespace enoki {                                                          \
        template <typename... Args> struct struct_support<Name<Args...>> {     \
            using Class = Name<Args...>;                                       \
            static constexpr bool Defined = true;                              \
            static Class empty(size_t size) {                                  \
                Class result;                                                  \
                ENOKI_MAP(ENOKI_STRUCT_EMPTY, __VA_ARGS__)                     \
                return result;                                                 \
            }                                                                  \
            static Class zero(size_t size) {                                   \
                Class result;                                                  \
                ENOKI_MAP(ENOKI_STRUCT_ZERO, __VA_ARGS__)                      \
                return result;                                                 \
            }                                                                  \
            template <bool Perm, typename Index, typename Mask>                \
            static void scatter(Class &dst, const Class &src,                  \
                                const Index &index, const Mask &mask) {        \
                ENOKI_MAP(ENOKI_STRUCT_SCATTER, __VA_ARGS__)                   \
            }                                                                  \
            template <bool Perm, typename Index, typename Mask>                \
            static Class gather(const Class &src, const Index &index,          \
                                const Mask &mask) {                            \
                Class result;                                                  \
                ENOKI_MAP(ENOKI_STRUCT_GATHER, __VA_ARGS__)                    \
                return result;                                                 \
            }                                                                  \
            static bool grad_enabled(const Class &v) {                         \
                return enoki::grad_enabled(                                    \
                    ENOKI_MAPC(ENOKI_STRUCT_ITEMS, __VA_ARGS__ ));             \
            }                                                                  \
            static void set_grad_enabled(Class &v, bool value) {               \
                ENOKI_MAP(ENOKI_STRUCT_SET_GRAD_ENABLED, __VA_ARGS__)          \
            }                                                                  \
            static void enqueue(const Class &v) {                              \
                enoki::enqueue(                                                \
                    ENOKI_MAPC(ENOKI_STRUCT_ITEMS, __VA_ARGS__) );             \
            }                                                                  \
            static auto detach(const Class &v) {                               \
                using Result =                                                 \
                    Name<decltype(enoki::detach(std::declval<Args &>()))...>;  \
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_DETACHED, __VA_ARGS__)                  \
                return result;                                                 \
            }                                                                  \
            static auto grad(const Class &v) {                                 \
                using Result =                                                 \
                    Name<decltype(enoki::grad(std::declval<Args &>()))...>;    \
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_GRADIENT, __VA_ARGS__)                  \
                return result;                                                 \
            }                                                                  \
            template <typename Mask>                                           \
            static auto masked(Class &v, const Mask &mask) {                   \
                using Result = Name<decltype(                                  \
                    enoki::masked(std::declval<Args &>(), mask))...>;          \
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_MASKED, __VA_ARGS__)                    \
                return result;                                                 \
            }                                                                  \
            static bool schedule(const Class &v) {                             \
                return enoki::schedule(                                        \
                    ENOKI_MAPC(ENOKI_STRUCT_ITEMS, __VA_ARGS__) );             \
            }                                                                  \
            static size_t width(const Class &v) {                              \
                size_t widths[] = {                                            \
                    ENOKI_MAPC(ENOKI_STRUCT_WIDTH, __VA_ARGS__) };             \
                size_t width = 0;                                              \
                for (size_t w: widths)                                         \
                    width = w > width ? w : width;                             \
                return width;                                                  \
            }                                                                  \
            static void set_label(Class &v, const char *label) {               \
                char tmp[256];                                                 \
                ENOKI_MAP(ENOKI_STRUCT_SET_LABEL, __VA_ARGS__)                 \
            }                                                                  \
        };                                                                     \
    }

NAMESPACE_BEGIN(enoki)

template <typename T1, typename T2> struct struct_support<std::pair<T1, T2>> {
    using Class = std::pair<T1, T2>;
    static constexpr bool Defined = true;
    static Class empty(size_t size) {
        return Class(enoki::empty<T1>(size), enoki::empty<T2>(size));
    }
    static Class zero(size_t size) {
        return Class(enoki::zero<T1>(size), enoki::zero<T2>(size));
    }
    template <bool Perm, typename Index, typename Mask>
    static void scatter(Class &dst, const Class &src, const Index &index,
                        const Mask &mask) {
        enoki::scatter<Perm>(dst.first, src.first, index, mask);
        enoki::scatter<Perm>(dst.second, src.second, index, mask);
    }
    template <bool Perm, typename Index, typename Mask>
    static Class gather(const Class &src, const Index &index,
                        const Mask &mask) {
        Class result;
        result.first  = enoki::gather<T1, Perm>(src.first,  index, mask);
        result.second = enoki::gather<T2, Perm>(src.second, index, mask);
        return result;
    }
    static bool grad_enabled(const Class &v) {
        return enoki::grad_enabled(v.first, v.second);
    }
    static void set_grad_enabled(Class &v, bool value) {
        enoki::set_grad_enabled(v.first, value);
        enoki::set_grad_enabled(v.second, value);
    }
    static void enqueue(const Class &v) { enoki::enqueue(v.first, v.second); }
    static auto detach(const Class &v) {
        using Result = std::pair<
            decltype(enoki::detach(std::declval<T1 &>())),
            decltype(enoki::detach(std::declval<T2 &>()))
        >;
        Result result;
        result.first = enoki::detach(v.first);
        result.second = enoki::detach(v.second);
        return result;
    }
    static auto grad(const Class &v) {
        using Result = std::pair<
            decltype(enoki::grad(std::declval<T1 &>())),
            decltype(enoki::grad(std::declval<T2 &>()))
        >;
        Result result;
        result.first = enoki::grad(v.first);
        result.second = enoki::grad(v.second);
        return result;
    }
    template <typename Mask> static auto masked(Class &v, const Mask &mask) {
        using Result = std::pair<
            decltype(enoki::masked(std::declval<T1 &>(), mask)),
            decltype(enoki::masked(std::declval<T2 &>(), mask))
        >;
        Result result;
        result.first = enoki::masked(v.first, mask);
        result.second = enoki::masked(v.second, mask);
        return result;
    }
    static bool schedule(const Class &v) {
        return enoki::schedule(v.first, v.second);
    }
    static size_t width(const Class &v) {
        size_t w1 = enoki::width(v.first),
               w2 = enoki::width(v.second);
        return w1 >= w2 ? w1 : w2;
    }
    static void set_label(Class &v, const char *label) {
        char tmp[256];
        snprintf(tmp, sizeof(tmp), "%s_%s", label, "first");
        enoki::set_label(v.first, tmp);
        snprintf(tmp, sizeof(tmp), "%s_%s", label, "second");
        enoki::set_label(v.second, tmp);
    }
};

NAMESPACE_END(enoki)
