/*
    drjit/array_format.h -- Formatter to convert a Dr.Jit array into a string.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

template <typename T> bool put_shape(const T *array, size_t *shape) {
    DRJIT_MARK_USED(shape); DRJIT_MARK_USED(array);

    if constexpr (is_array_v<T>) {
        size_t cur = *shape,
               size = size_v<T> == Dynamic ? (array ? array->derived().size() : 0) : size_v<T>,
               maxval = cur > size ? cur : size;

        if (maxval != size && size != 1)
            return false; // ragged array

        *shape = maxval;

        if constexpr (is_array_v<value_t<T>>) {
            if (size == 0) {
                return put_shape<value_t<T>>(nullptr, shape + 1);
            } else {
                for (size_t i = 0; i < size; ++i)
                    if (!put_shape(&(array->derived().entry(i)), shape + 1))
                        return false;
            }
        }
    }

    return true;
}
template <typename Buffer, typename Array, size_t Depth, size_t ... Is>
void format_array(Buffer &buf, const Array &v, size_t indent, const size_t *shape, size_t *indices) {
    static constexpr size_t MaxDepth = depth_v<Array>;
    static constexpr bool Last = Depth == MaxDepth - 1;
    // Skip output when there are more than 20 elements.
    static constexpr size_t Threshold = 20; // Must be divisible by 4

    DRJIT_MARK_USED(shape);
    DRJIT_MARK_USED(indices);

    // On vectorized types, iterate over the last dimension first
    size_t i = Depth;
    using Leaf = leaf_array_t<Array>;
    if constexpr (!Leaf::BroadcastOuter || Leaf::IsDynamic) {
        if (Depth == 0)
            i = MaxDepth - 1;
        else
            i -= 1;
    }

    if constexpr (Last && (is_complex_v<Array> || is_quaternion_v<Array>)) {
        // Special handling for complex numbers and quaternions
        bool prev = false;

        for (size_t j = 0; j < size_v<Array>; ++j) {
            indices[i] = j;

            scalar_t<Array> value = v.derived().entry(indices[Is]...);
            if (value == 0)
                continue;

            if (prev || value < 0)
                buf.put(value < 0 ? '-' : '+');
            buf.put(value);
            prev = true;

            if (is_complex_v<Array> && j == 1)
                buf.put('j');
            else if (is_quaternion_v<Array> && j < 3)
                buf.put("ijk"[j]);
        }
        if (!prev)
            buf.put('0');
        return;
    }

    size_t size = shape[i];

    buf.put('[');
    for (size_t j = 0; j < size; ++j) {
        indices[i] = j;

        if (size >= Threshold && j * 4 == Threshold) {
            buf.put("   ", size - Threshold / 2, " skipped ..");
            j = size - Threshold / 4 - 1;
        } else {
            if constexpr (Last)
                buf.put(v.derived().entry(indices[Is]...));
            else
                format_array<Buffer, Array, Depth + 1, Is..., Depth + 1>(buf, v, indent + 1, shape, indices);
        }

        if (j + 1 < size) {
            if (Last) {
                buf.put(", ");
            } else {
                buf.put(",\n");
                buf.indent(indent);
            }
        }
    }

    buf.put(']');
}

template <typename Array> struct formatter<Array, enable_if_array_t<Array>> {
    static size_t bound(size_t indent, const Array &v) {
        dummy_string d;
        format(d, indent, 0, v);
        return d.size();
    }

    template <typename Buffer>
    static void format(Buffer &s, size_t indent, size_t, const Array &v) {
        static constexpr size_t Depth = depth_v<Array> == 0 ? 1 : depth_v<Array>;
        size_t shape[Depth] { }, indices[Depth];
        if (!put_shape(&v, shape))
            s.put_unchecked("[ragged array]", 14);
        if constexpr (is_jit_v<Array>)
            schedule(v);
        format_array<Buffer, Array, 0, 0>(s, v, indent + 1, shape, indices);
    }
};

template <typename T> struct formatter<T, enable_if_t<is_traversable_v<T> && !is_array_v<T>>> {
    static constexpr size_t Size = decltype(drjit::fields(std::declval<T>()))::Size;

    static size_t bound(size_t indent, const T &v) {
        dummy_string d;
        format(d, indent, 0, v);
        return d.size();
    }

    template <typename Buffer>
    static void format(Buffer &s, size_t indent, size_t, const T &v) {
        format_impl(s, indent, v, std::make_index_sequence<Size>());
    }

    template <typename Buffer, size_t... Is>
    static void format_impl(Buffer &s, size_t indent, const T &v,
                            std::index_sequence<Is...>) {
        auto v_fields = fields(v);
        auto v_labels = labels(v);

        s.put(v.name_());
        s.put_unchecked("[\n", 2);
        (format_field(s, indent + 2, drjit::get<Is>(v_labels),
                      drjit::get<Is>(v_fields), Is == Size - 1),
         ...);
        s.put_unchecked("]", 1);
    }

    template <typename Buffer, typename T2>
    static void format_field(Buffer &s, size_t indent, const drjit::string &label,
                             const T2 &value, bool is_last) {
        s.indent(indent);
        s.iput(indent + label.length() + 1, label, "=", value,
               is_last ? "\n" : ",\n");
    }
};

NAMESPACE_END(detail)

template <typename Stream, typename Sentry = typename Stream::sentry,
          typename T, enable_if_t<is_array_v<T> || is_traversable_v<T>> = 0>
Stream &operator<<(Stream &stream, const T &value) {
    stream << drjit::string(value);
    return stream;
}

NAMESPACE_END(drjit)
