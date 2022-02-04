/*
    drjit/struct.h -- Infrastructure for working with general data structures

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/array.h>
#include <drjit/map.h>
#include <drjit-core/containers.h>

#pragma once

#define DRJIT_STRUCT_ASSIGN_COPY(x)   x = v.x;
#define DRJIT_STRUCT_ASSIGN_MOVE(x)   x = std::move(v.x);
#define DRJIT_STRUCT_APPLY_1(x)       func(v.x);
#define DRJIT_STRUCT_APPLY_2(x)       func(v1.x, v2.x);
#define DRJIT_STRUCT_APPLY_3(x)       func(v1.x, v2.x, v3.x);
#define DRJIT_STRUCT_APPLY_LABEL(x)   func(#x, v.x);

#define DRJIT_STRUCT(Name, ...)                                                \
    static constexpr bool IsDrJitStruct = true;                                \
    Name() = default;                                                          \
    Name(const Name &) = default;                                              \
    Name(Name &&) = default;                                                   \
    Name &operator=(const Name &) = default;                                   \
    Name &operator=(Name &&) = default;                                        \
    template <typename... Ts> Name(const Name<Ts...> &v) {                     \
        DRJIT_MAP(DRJIT_STRUCT_ASSIGN_COPY, __VA_ARGS__)                       \
    }                                                                          \
    template <typename... Ts> Name(Name<Ts...> &&v) {                          \
        DRJIT_MAP(DRJIT_STRUCT_ASSIGN_MOVE, __VA_ARGS__)                       \
    }                                                                          \
    template <typename... Ts> Name &operator=(Name<Ts...> &&v) {               \
        DRJIT_MAP(DRJIT_STRUCT_ASSIGN_MOVE, __VA_ARGS__)                       \
        return *this;                                                          \
    }                                                                          \
    template <typename... Ts> Name &operator=(const Name<Ts...> &v) {          \
        DRJIT_MAP(DRJIT_STRUCT_ASSIGN_COPY, __VA_ARGS__)                       \
        return *this;                                                          \
    }                                                                          \
    template <typename T_, typename Func_>                                     \
    static DRJIT_INLINE void apply_1(T_ &v, Func_ func) {                      \
        DRJIT_MAP(DRJIT_STRUCT_APPLY_1, __VA_ARGS__)                           \
    }                                                                          \
    template <typename T1_, typename T2_, typename Func_>                      \
    static DRJIT_INLINE void apply_2(T1_ &v1, T2_ &v2, Func_ func) {           \
        DRJIT_MAP(DRJIT_STRUCT_APPLY_2, __VA_ARGS__)                           \
    }                                                                          \
    template <typename T1_, typename T2_, typename T3_, typename Func_>        \
    static DRJIT_INLINE void apply_3(T1_ &v1, T2_ &v2, T3_ &v3, Func_ func) {  \
        DRJIT_MAP(DRJIT_STRUCT_APPLY_3, __VA_ARGS__)                           \
    }                                                                          \
    template <typename T_, typename Func_>                                     \
    static DRJIT_INLINE void apply_label(T_ &v, Func_ func) {                  \
        DRJIT_MAP(DRJIT_STRUCT_APPLY_LABEL, __VA_ARGS__)                       \
    }                                                                          \
    template <typename Array, drjit::enable_if_mask_t<Array> = 0>              \
    auto operator[](const Array &array) {                                      \
        return drjit::masked(*this, array);                                    \
    }


NAMESPACE_BEGIN(drjit)

template <typename T1_, typename T2_> struct struct_support<std::pair<T1_, T2_>> {
    static constexpr bool Defined = true;
    using type = struct_support;

    template <typename T, typename Func>
    static DRJIT_INLINE void apply_1(T &v, Func func) {
        func(v.first);
        func(v.second);
    }
    template <typename T1, typename T2, typename Func>
    static DRJIT_INLINE void apply_2(T1 &v1, T2 &v2, Func func) {
        func(v1.first,  v2.first);
        func(v1.second, v2.second);
    }
    template <typename T1, typename T2, typename T3, typename Func>
    static DRJIT_INLINE void apply_3(T1 &v1, T2 &v2, T3 &v3, Func func) {
        func(v1.first,  v2.first,  v3.first);
        func(v1.second, v2.second, v3.second);
    }
    template <typename T, typename Func>
    static DRJIT_INLINE void apply_label(T &v, Func func) {
        func("0", v.first);
        func("1", v.second);
    }
};

NAMESPACE_BEGIN(detail)

template <unsigned Value, unsigned... D>
struct itoa : itoa<Value / 10, Value % 10, D...> { };

template <unsigned... D> struct itoa<0, D...> {
    static constexpr char value[] = { (char) ('0' + D)..., '\0' };
};

NAMESPACE_END(detail)

template <typename... Ts> struct struct_support<std::tuple<Ts...>> {
    static constexpr bool Defined = true;
    using type = struct_support;

    template <typename T, typename Func>
    static DRJIT_INLINE void apply_1(T &v, Func func) {
        apply_1(v, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_1(T &v, Func func, std::index_sequence<Is...>) {
        (func(std::get<Is>(v)), ...);
    }
    template <typename T1, typename T2, typename Func>
    static DRJIT_INLINE void apply_2(T1 &v1, T2 &v2, Func func) {
        apply_2(v1, v2, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T1, typename T2, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_2(T1 &v1, T2 &v2, Func func, std::index_sequence<Is...>) {
        (func(std::get<Is>(v1), std::get<Is>(v2)), ...);
    }
    template <typename T1, typename T2, typename T3, typename Func>
    static DRJIT_INLINE void apply_3(T1 &v1, T2 &v2, T3 &v3, Func func) {
        apply_3(v1, v2, v3, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T1, typename T2, typename T3, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_3(T1 &v1, T2 &v2, T3 &v3, Func func, std::index_sequence<Is...>) {
        (func(std::get<Is>(v1), std::get<Is>(v2), std::get<Is>(v3)), ...);
    }
    template <typename T, typename Func>
    static DRJIT_INLINE void apply_label(T &v, Func func) {
        apply_label(v, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_label(T &v, Func func, std::index_sequence<Is...>) {
        (func(detail::itoa<Is>::value, std::get<Is>(v)), ...);
    }
};

template <typename... Ts> struct struct_support<dr_tuple<Ts...>> {
    static constexpr bool Defined = true;
    using type = struct_support;

    template <typename T, typename Func>
    static DRJIT_INLINE void apply_1(T &v, Func func) {
        apply_1(v, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_1(T &v, Func func, std::index_sequence<Is...>) {
        DRJIT_MARK_USED(func);
        (func(v.template get<Is>()), ...);
    }
    template <typename T1, typename T2, typename Func>
    static DRJIT_INLINE void apply_2(T1 &v1, T2 &v2, Func func) {
        apply_2(v1, v2, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T1, typename T2, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_2(T1 &v1, T2 &v2, Func func, std::index_sequence<Is...>) {
        DRJIT_MARK_USED(func);
        (func(v1.template get<Is>(), v2.template get<Is>()), ...);
    }
    template <typename T1, typename T2, typename T3, typename Func>
    static DRJIT_INLINE void apply_3(T1 &v1, T2 &v2, T3 &v3, Func func) {
        apply_3(v1, v2, v3, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T1, typename T2, typename T3, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_3(T1 &v1, T2 &v2, T3 &v3, Func func, std::index_sequence<Is...>) {
        DRJIT_MARK_USED(func);
        (func(v1.template get<Is>(), v2.template get<Is>(), v3.template get<Is>()), ...);
    }
    template <typename T, typename Func>
    static DRJIT_INLINE void apply_label(T &v, Func func) {
        apply_label(v, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    template <typename T, typename Func, size_t... Is>
    static DRJIT_INLINE void apply_label(T &v, Func func, std::index_sequence<Is...>) {
        (func(detail::itoa<Is>::value, v.template get<Is>()), ...);
    }
};

NAMESPACE_END(drjit)
