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

#define ENOKI_STRUCT_DETACH(x) \
    result.x = enoki::detach(value.x);

#define ENOKI_STRUCT_GRAD(x) \
    result.x = enoki::grad(value.x);

#define ENOKI_STRUCT_MASKED(x) \
    result.x = enoki::masked(value.x, mask);

#define ENOKI_STRUCT_CONSTR_COPY(x) \
    x(v.x)

#define ENOKI_STRUCT_CONSTR_MOVE(x) \
    x(std::move(v.x))

#define ENOKI_STRUCT_ASSIGN_COPY(x) \
    x = v.x;

#define ENOKI_STRUCT_ASSIGN_MOVE(x) \
    x = std::move(v.x);

#define ENOKI_STRUCT_SLICES(x) \
    enoki::slices(value.x)

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
            static auto detach(const Class &value) {                           \
                using Result =                                                 \
                    Name<decltype(enoki::detach(std::declval<Args &>()))...>;  \
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_DETACH, __VA_ARGS__)                    \
                return result;                                                 \
            }                                                                  \
            static auto grad(const Class &value) {                             \
                using Result =                                                 \
                    Name<decltype(enoki::grad(std::declval<Args &>()))...>;    \
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_GRAD, __VA_ARGS__)                      \
                return result;                                                 \
            }                                                                  \
            template <typename Mask>                                           \
            static auto masked(Class &value, const Mask &mask) {               \
                using Result = Name<decltype(                                  \
                    enoki::masked(std::declval<Args &>(), mask))...>;          \
                Result result;                                                 \
                ENOKI_MAP(ENOKI_STRUCT_MASKED, __VA_ARGS__)                    \
                return result;                                                 \
            }                                                                  \
            static size_t slices(const Class &value) {                         \
                size_t sizes[] = {                                             \
                    ENOKI_MAPC(ENOKI_STRUCT_SLICES, __VA_ARGS__) };            \
                size_t size = 0;                                               \
                for (size_t s: sizes)                                          \
                    size = s > size ? s : size;                                \
                return size;                                                   \
            }                                                                  \
        };                                                                     \
    }
