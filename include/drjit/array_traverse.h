/*
    drjit/struct.h -- Infrastructure for working with general data structures

    (This file isn't meant to be included as-is. Please use 'drjit/array.h',
     which bundles all the 'array_*' headers in the right order.)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#define DRJIT_STRUCT_NODEF(Name, ...)                                          \
    Name(const Name &) = default;                                              \
    Name(Name &&) = default;                                                   \
    Name &operator=(const Name &) = default;                                   \
    Name &operator=(Name &&) = default;                                        \
    DRJIT_INLINE auto fields_() { return drjit::tie(__VA_ARGS__); }            \
    DRJIT_INLINE auto fields_() const { return drjit::tie(__VA_ARGS__); }      \
    const char *name_() const { return #Name; }                                \
    auto labels_() const {                                                     \
        return drjit::detail::unpack_labels(                                   \
            #__VA_ARGS__,                                                      \
            std::make_index_sequence<decltype(fields_())::Size>());            \
    }

#define DRJIT_STRUCT(Name, ...)                                                \
    Name() = default;                                                          \
    DRJIT_STRUCT_NODEF(Name, __VA_ARGS__)

NAMESPACE_BEGIN(drjit)

template <typename T1, typename F>
DRJIT_INLINE void traverse_1(T1 &&v1, F &&f) {
    if constexpr (std::decay_t<T1>::Size != 0) {
        f(v1.value);
        traverse_1(v1.base(), f);
    }
}

template <typename T1, typename T2, typename F>
DRJIT_INLINE void traverse_2(T1 &&v1, T2 &&v2, F &&f) {
    if constexpr (std::decay_t<T1>::Size != 0) {
        f(v1.value, v2.value);
        traverse_2(v1.base(), v2.base(), f);
    }
}

template <typename T1, typename T2, typename T3, typename F>
DRJIT_INLINE void traverse_3(T1 &&v1, T2 &&v2, T3 &&v3, F &&f) {
    if constexpr (std::decay_t<T1>::Size != 0) {
        f(v1.value, v2.value, v3.value);
        traverse_3(v1.base(), v2.base(), v3.base(), f);
    }
}

namespace detail {
    // Traversal helper for objects that cannot be traversed
    template <typename T, typename SFINAE = int> struct traversable {
        static constexpr bool value = false;
        template <typename Tv> static tuple<> fields(Tv&) { return { }; }
        template <typename Tv> static tuple<> labels(const Tv&) { return { }; }
    };

    // Traversal helper for DRJIT_STRUCT(..) instances
    template <typename T> struct traversable<T, enable_if_drjit_struct_t<T>> {
        static constexpr bool value = true;
        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) { return v.fields_(); }
        template <typename Tv> static auto labels(const Tv &v) { return v.labels_(); }
    };

    // Traversal helper for drjit::tuple<...> and std::tuple<...>
    template <typename T> struct traversable<T, enable_if_t<(std::tuple_size<T>::value > 0)>> {
        static constexpr bool value = true;
        static constexpr size_t Size = std::tuple_size<T>::value;

        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) {
            return fields_impl<Tv>(v, std::make_index_sequence<Size>());
        }

        template <typename Tv> static auto labels(const Tv &) {
            return labels_impl(std::make_index_sequence<Size>());
        }

        template <typename Tv, size_t... Is>
        static DRJIT_INLINE auto fields_impl(Tv &v, std::index_sequence<Is...>) {
            using namespace std;
            using namespace drjit;
            return drjit::tie(get<Is>(v)...);
        }

        template <size_t... Is> static auto labels_impl(std::index_sequence<Is...>) {
            return make_tuple(drjit::string(Is)...);
        }
    };

    // Traversal helper for static arrays
    template <typename T> struct traversable<T, enable_if_static_array_t<T>> {
        static constexpr bool value = true;

        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) {
            return fields_impl<Tv>(v, std::make_index_sequence<size_v<T>>());
        }

        template <typename Tv> static auto labels(const Tv &) {
            return labels_impl(std::make_index_sequence<size_v<T>>());
        }

        template <typename Tv, size_t... Is>
        static DRJIT_INLINE auto fields_impl(Tv &v, std::index_sequence<Is...>) {
            return drjit::tie(v.entry(Is)...);
        }

        template <size_t... Is> static auto labels_impl(std::index_sequence<Is...>) {
            return make_tuple(drjit::string(Is)...);
        }
    };

    // Traversal helper for tensors
    template <typename T> struct traversable<T, enable_if_tensor_t<T>> {
        static constexpr bool value = true;
        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) {
            return drjit::tie(v.array());
        }
        template <typename Tv> static auto labels(const Tv &) {
            return make_tuple(drjit::string("array"));
        }
    };

    template <typename T>
    using det_traverse_1_cb_ro =
        decltype(T(nullptr)->traverse_1_cb_ro(nullptr, nullptr));

    template <typename T>
    using det_traverse_1_cb_rw =
        decltype(T(nullptr)->traverse_1_cb_rw(nullptr, nullptr));

    inline drjit::string get_label(const char *s, size_t i) {
        auto skip = [](char c) {
            return c == ' ' || c == '\r' || c == '\n' || c == '\t' || c == ',';
        };

        const char *start = nullptr, *end = nullptr;

        for (size_t j = 0; j <= i; ++j) {
            while (skip(*s))
                s++;
            start = s;
            while (!skip(*s) && *s != '\0')
                s++;
            end = s;
        }

        return drjit::string(start, end - start);
    }

    template <size_t... Is> auto unpack_labels(const char *s, std::index_sequence<Is...>) {
        return drjit::make_tuple(get_label(s, Is)...);
    }
};

template <typename T> using traversable_t = detail::traversable<std::decay_t<T>>;
template <typename T> static constexpr bool is_traversable_v = traversable_t<T>::value;
template <typename T> using enable_if_traversable_t = enable_if_t<is_traversable_v<T>>;

template <typename T> static constexpr bool is_dynamic_traversable_v = 
    is_jit_v<T> && is_dynamic_array_v<T> && is_vector_v<T> && !is_tensor_v<T>;

template <typename T> DRJIT_INLINE auto fields(T &&v) {
    return traversable_t<T>::fields(v);
}

template <typename T> auto labels(const T &v) {
    return traversable_t<T>::labels(v);
}

template <typename Value>
void traverse_1_fn_ro(const Value &value, void *payload, void (*fn)(void *, uint64_t)) {
    (void) payload; (void) fn;
    if constexpr (is_jit_v<Value> && depth_v<Value> == 1) {
        fn(payload, value.index_combined());
    } else if constexpr (is_traversable_v<Value>) {
        traverse_1(fields(value), [payload, fn](auto &x) {
            traverse_1_fn_ro(x, payload, fn);
        });
    } else if constexpr (is_dynamic_traversable_v<Value>) {
        for (size_t i = 0; i < value.size(); ++i) {
            traverse_1(drjit::tie(value.entry(i)), [payload, fn](auto &x) {
                traverse_1_fn_ro(x, payload, fn);
            });
        }
    } else if constexpr (std::is_pointer_v<Value> &&
                         is_detected_v<detail::det_traverse_1_cb_ro, Value>) {
        if (value)
            value->traverse_1_cb_ro(payload, fn);
    }
}

template <typename Value>
void traverse_1_fn_rw(Value &value, void *payload, uint64_t (*fn)(void *, uint64_t)) {
    (void) payload; (void) fn;
    if constexpr (is_jit_v<Value> && depth_v<Value> == 1) {
        value = Value::borrow((typename Value::Index) fn(payload, value.index_combined()));
    } else if constexpr (is_traversable_v<Value>) {
        traverse_1(fields(value), [payload, fn](auto &x) {
            traverse_1_fn_rw(x, payload, fn);
        });
    } else if constexpr (is_dynamic_traversable_v<Value>) {
        for (size_t i = 0; i < value.size(); ++i) {
            traverse_1(drjit::tie(value.entry(i)), [payload, fn](auto &x) {
                traverse_1_fn_rw(x, payload, fn);
            });
        }
    } else if constexpr (std::is_pointer_v<Value> &&
                         is_detected_v<detail::det_traverse_1_cb_rw, Value>) {
        if (value)
            value->traverse_1_cb_rw(payload, fn);
    }
}

NAMESPACE_END(drjit)
