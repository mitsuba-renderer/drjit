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

#define DRJIT_STRUCT(...)                                                      \
    auto fields_() { return drjit::tie(__VA_ARGS__); }                         \
    DRJIT_INLINE auto fields_() const { return drjit::tie(__VA_ARGS__); }      \
    static constexpr auto labels_ = drjit::detail::make_labels(#__VA_ARGS__);

NAMESPACE_BEGIN(drjit)

template <typename T1, typename F>
DRJIT_INLINE void traverse_1(T1 &&v1, F &&f) {
    if constexpr (std::decay_t<T1>::Size) {
        f(v1.value);
        traverse_1(v1.base(), f);
    }
}

template <typename T1, typename T2, typename F>
DRJIT_INLINE void traverse_2(T1 &&v1, T2 &&v2, F &&f) {
    if constexpr (std::decay_t<T1>::Size) {
        f(v1.value, v2.value);
        traverse_2(v1.base(), v2.base(), f);
    }
}

template <typename T1, typename T2, typename T3, typename F>
DRJIT_INLINE void traverse_3(T1 &&v1, T2 &&v2, T3 &&v3, F &&f) {
    if constexpr (std::decay_t<T1>::Size) {
        f(v1.value, v2.value, v3.value);
        traverse_3(v1.base(), v2.base(), v3.base(), f);
    }
}

namespace detail {
    // Helper class to generate numeric strings at compile time
    template <unsigned Value, unsigned... D>
    struct itoa : itoa<Value / 10, Value % 10, D...> { };
    template <unsigned... D> struct itoa<0, D...> {
        static constexpr char value[] = { (char) ('0' + D)..., '\0' };
    };
    template <> struct itoa<0> : itoa<0, 0> { };

    // Traversal helper for objects that cannot be traversed
    template <typename T, typename SFINAE = int> struct traversable {
        static constexpr bool value = false;
        template <typename Tv> static tuple<> fields(Tv&) { return { }; }
        static tuple<> labels() { return { }; }
    };

    // Traversal helper for DRJIT_STRUCT(..) instances
    template <typename T> struct traversable<T, enable_if_drjit_struct_t<T>> {
        static constexpr bool value = true;
        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) { return v.fields_(); }
        static auto labels() {
            constexpr size_t size = decltype(std::declval<T>().fields_())::Size;
            return labels_impl(std::make_index_sequence<size>());
        }
        template <size_t... Is> static auto labels_impl(std::index_sequence<Is...>) {
            return make_tuple(T::labels_[Is]...);
        }
    };

    // Traversal helper for tuple<...> and std::tuple<...>
    template <typename T> struct traversable<T, enable_if_t<(std::tuple_size<T>::value > 0)>> {
        static constexpr bool value = true;
        static constexpr size_t Size = std::tuple_size<T>::value;

        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) {
            return fields_impl<Tv>(v, std::make_index_sequence<Size>());
        }

        static auto labels() {
            return labels_impl(std::make_index_sequence<Size>());
        }

        template <typename Tv, size_t... Is>
        static DRJIT_INLINE auto fields_impl(Tv &v, std::index_sequence<Is...>) {
            using namespace std;
            return tie(get<Is>(v)...);
        }

        template <size_t... Is> static auto labels_impl(std::index_sequence<Is...>) {
            return make_tuple(detail::itoa<Is>::value...);
        }
    };

    // Traversal helper for static arrays
    template <typename T> struct traversable<T, enable_if_static_array_t<T>> {
        static constexpr bool value = true;

        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) {
            return fields_impl<Tv>(v, std::make_index_sequence<size_v<T>>());
        }

        static auto labels() {
            return labels_impl(std::make_index_sequence<size_v<T>>());
        }

        template <typename Tv, size_t... Is>
        static DRJIT_INLINE auto fields_impl(Tv &v, std::index_sequence<Is...>) {
            return tie(v.entry(Is)...);
        }

        template <size_t... Is> static auto labels_impl(std::index_sequence<Is...>) {
            return make_tuple(detail::itoa<Is>::value...);
        }
    };

    // Traversal helper for tensors
    template <typename T> struct traversable<T, enable_if_tensor_t<T>> {
        static constexpr bool value = true;
        template <typename Tv> static DRJIT_INLINE auto fields(Tv &v) {
            return tie(v.array());
        }
        static auto labels() { return make_tuple("array"); }
    };

    /// Constexpr string representation for DR_STRUCT field labels
    template <size_t N> struct struct_labels {
        char s[N + 1];

        template <size_t... Is>
        constexpr struct_labels(char const (&s)[N], std::index_sequence<Is...>)
            : s{ (skip(s[Is]) ? '\0' : s[Is])..., '\0' } { }

        static constexpr bool skip(char c) {
            return c == ' ' || c == '\r' || c == '\n' || c == '\t' || c == ',';
        }

        constexpr const char *operator[](size_t i) const {
            for (size_t j = 0, k = 0; j < N; ++j) {
                if (s[j] != '\0' && (j == 0 || s[j - 1] == '\0')) {
                    if (k++ == i)
                        return s + j;
                }
            }

            return nullptr;
        }
    };

    template <size_t N> constexpr struct_labels<N> make_labels(char const (&s)[N]) {
        return { s, std::make_index_sequence<N>() };
    }

    template <typename T>
    using det_traverse_1_cb_ro =
        decltype(T(nullptr)->traverse_1_cb_ro(nullptr, nullptr));

    template <typename T>
    using det_traverse_1_cb_rw =
        decltype(T(nullptr)->traverse_1_cb_rw(nullptr, nullptr));
};

template <typename T> using traversable_t = detail::traversable<std::decay_t<T>>;
template <typename T> static constexpr bool is_traversable_v = traversable_t<T>::value;
template <typename T> using enable_if_traversable_t = enable_if_t<is_traversable_v<T>>;

template <typename T> DRJIT_INLINE auto fields(T &&v) {
    return traversable_t<T>::fields(v);
}

template <typename T> auto labels() {
    return traversable_t<T>::labels();
}

template <typename Value>
void traverse_1_fn_ro(const Value &value, void *payload, void (*fn)(void *, uint64_t)) {
    if constexpr (is_jit_v<Value> && depth_v<Value> == 1) {
        fn(payload, value.index_combined());
    } else if constexpr (is_traversable_v<Value>) {
        traverse_1(fields(value), [payload, fn](auto &x) {
            traverse_1_fn_ro(x, payload, fn);
        });
    } else if constexpr (std::is_pointer_v<Value> &&
                         is_detected_v<detail::det_traverse_1_cb_ro, Value>) {
        if (value)
            value->traverse_1_cb_ro(payload, fn);
    }
}

template <typename Value>
void traverse_1_fn_rw(Value &value, void *payload, uint64_t (*fn)(void *, uint64_t)) {
    if constexpr (is_jit_v<Value> && depth_v<Value> == 1) {
        value = Value::borrow((typename Value::Index) fn(payload, value.index_combined()));
    } else if constexpr (is_traversable_v<Value>) {
        traverse_1(fields(value), [payload, fn](auto &x) {
            traverse_1_fn_rw(x, payload, fn);
        });
    } else if constexpr (std::is_pointer_v<Value> &&
                         is_detected_v<detail::det_traverse_1_cb_rw, Value>) {
        if (value)
            value->traverse_1_cb_rw(payload, fn);
    }
}

NAMESPACE_END(drjit)
