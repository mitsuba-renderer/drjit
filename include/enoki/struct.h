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
    result.x = enoki::gather<Perm>(src.x, index, mask);

#define ENOKI_STRUCT_ZERO(x) \
    result.x = enoki::zero<decltype(Class::x)>(size);

#define ENOKI_STRUCT_EMPTY(x) \
    result.x = enoki::empty<decltype(Class::x)>(size);

#define ENOKI_STRUCT_ATTACH(x) \
    enoki::detach(v.x);

#define ENOKI_STRUCT_DETACH(x) \
    enoki::detach(v.x);

#define ENOKI_STRUCT_DETACHED(x) \
    result.x = enoki::detached(v.x);

#define ENOKI_STRUCT_GRADIENT(x) \
    result.x = enoki::gradient(v.x);

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

#define ENOKI_STRUCT_SLICES(x) \
    enoki::slices(v.x)

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
    template <typename... Ts> Test &operator=(Test<Ts...> &&v) {               \
        ENOKI_MAP(ENOKI_STRUCT_ASSIGN_MOVE, __VA_ARGS__)                       \
        return *this;                                                          \
    }                                                                          \
    template <typename... Ts> Test &operator=(const Test<Ts...> &v) {          \
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
            static void scatter(Class &dst, Class &src, const Index &index,    \
                                const Mask &mask) {                            \
                ENOKI_MAP(ENOKI_STRUCT_SCATTER, __VA_ARGS__)                   \
            }                                                                  \
            template <bool Perm, typename Index, typename Mask>                \
            static Class gather(const Class &src, const Index &index,          \
                                const Mask &mask) {                            \
                Class result;                                                  \
                ENOKI_MAP(ENOKI_STRUCT_GATHER, __VA_ARGS__)                    \
                return result;                                                 \
            }                                                                  \
            static void attach(Class &class) {                                 \
                ENOKI_MAP(ENOKI_STRUCT_ATTACH, __VA_ARGS__);                   \
            }                                                                  \
            static void detach(Class &class) {                                 \
                ENOKI_MAP(ENOKI_STRUCT_DETACH, __VA_ARGS__);                   \
            }                                                                  \
            static auto detached(const Class &v) {                             \
                using Result =                                                 \
                    Name<decltype(enoki::detached(std::declval<Args &>()))...>;\
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_DETACHED, __VA_ARGS__)                  \
                return result;                                                 \
            }                                                                  \
            static auto gradient(const Class &v) {                             \
                using Result =                                                 \
                    Name<decltype(enoki::gradient(std::declval<Args &>()))...>;\
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
                    ENOKI_MAPC(ENOKI_STRUCT_SLICES, __VA_ARGS__) };            \
                size_t width = 0;                                              \
                for (size_t w: widths)                                         \
                    width = w > width ? w : width;                             \
                return width;                                                  \
            }                                                                  \
            static void set_label(Class &v, const char *label) {               \
                char tmp[256];                                                 \
                ENOKI_MAP(ENOKI_STRUCT_SET_LABEL, __VA_ARGS__);                \
            }                                                                  \
        };                                                                     \
    }
