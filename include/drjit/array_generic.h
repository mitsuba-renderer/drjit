/*
    drjit/array_generic.h -- Generic array implementation that forwards
    all operations to the underlying data type (usually without making use of
    hardware vectorization)

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array_static.h>
#include <drjit/string.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_, size_t Size_, bool IsMask_, typename Derived_, typename = int>
struct StaticArrayImpl;

namespace detail {
    template <typename Type, size_t Size, typename = int> struct vectorize {
        using Parent = vectorize<Type, detail::lpow2(Size)>;
        static constexpr bool recurse = Parent::recurse || Parent::self;
        static constexpr bool self = false;
    };

    template <typename Type> struct vectorize<Type, 1> {
        static constexpr bool recurse = false;
        static constexpr bool self = false;
    };

    template <typename Type> struct vectorize<Type, 0> {
        static constexpr bool recurse = false;
        static constexpr bool self = false;
    };

    /// Decide whether an array can be handled using a packet implementation
    template <typename Type>
    constexpr bool vectorizable_type_v = std::is_same_v<Type, float> ||
                                         std::is_same_v<Type, double> ||
                                         (is_integral_ext_v<Type> &&
                                          (sizeof(Type) == 4 || sizeof(Type) == 8));

    template <typename Type, size_t Size>
    using vectorize_t = vectorize<Type, Size * sizeof(Type)>;

    template <typename Type, size_t Size>
    using enable_if_generic =
        enable_if_t<Size != 0 &&
                    !(vectorizable_type_v<Type> &&
                      (vectorize_t<Type, Size>::self ||
                       (Size >= 4 && vectorize_t<Type, Size>::recurse)))>;

    template <typename Type, size_t Size>
    using enable_if_recursive =
        enable_if_t<vectorizable_type_v<Type> && (Size >= 4) &&
                    vectorize_t<Type, Size>::recurse>;
};

/**
 * Generic fallback array type. Requires that 'Value_' is default, copy-, and
 * move-constructible, as well as assignable.
 */
template <typename Value_, size_t Size_, bool IsMask_, typename Derived_>
struct StaticArrayImpl<Value_, Size_, IsMask_, Derived_,
                       detail::enable_if_generic<Value_, Size_>>
    : StaticArrayBase<std::conditional_t<IsMask_, mask_t<Value_>, Value_>,
                      Size_, IsMask_, Derived_> {

    static_assert(
        std::is_default_constructible_v<Value_>,
        "Type underlying drjit::Array must be default-constructible!");

    using Base = StaticArrayBase<
        std::conditional_t<IsMask_, mask_t<Value_>, Value_>,
        Size_, IsMask_, Derived_>;

    using typename Base::Derived;
    using typename Base::Value;
    using typename Base::Scalar;
    using typename Base::Array1;
    using typename Base::Array2;

    using Base::Size;
    using Base::derived;
    using Base::entry;

    DRJIT_ARRAY_DEFAULTS(StaticArrayImpl)
    DRJIT_ARRAY_FALLBACK_CONSTRUCTORS(StaticArrayImpl)

    template <typename Value2, typename D2, typename D = Derived_,
              enable_if_t<D::Size != D2::Size || D::Depth != D2::Depth> = 0>
    StaticArrayImpl(const ArrayBaseT<Value2, false, D2> &v) {
        if constexpr (D::Size == D2::Size && D2::BroadcastOuter) {
            static_assert(std::is_constructible_v<Value, value_t<D2>>);
            for (size_t i = 0; i < derived().size(); ++i)
                derived().entry(i) = (Value) v.derived().entry(i);
        } else {
            static_assert(std::is_constructible_v<Value, D2>);
            for (size_t i = 0; i < derived().size(); ++i)
                derived().entry(i) = v.derived();
        }
    }

    template <typename Value2, typename D2, typename D = Derived_,
              enable_if_t<D::Size != D2::Size || D::Depth != D2::Depth> = 0>
    StaticArrayImpl(const ArrayBaseT<Value2, IsMask_, D2> &v,
                    detail::reinterpret_flag) {
        if constexpr (D::Size == D2::Size && D2::BroadcastOuter) {
            static_assert(std::is_constructible_v<Value, value_t<D2>, detail::reinterpret_flag>);
            for (size_t i = 0; i < derived().size(); ++i)
                derived().entry(i) = reinterpret_array<Value>(v.derived().entry(i));
        } else {
            static_assert(std::is_constructible_v<Value, D2, detail::reinterpret_flag>);
            for (size_t i = 0; i < derived().size(); ++i)
                derived().entry(i) = reinterpret_array<Value>(v.derived());
        }
    }

#if defined(NDEBUG)
    StaticArrayImpl() = default;
#else
    template <typename T = Value_, enable_if_t<!drjit::is_scalar_v<T>> = 0>
    StaticArrayImpl() { }
    template <typename T = Value_, enable_if_scalar_t<T> = 0>
    StaticArrayImpl() : StaticArrayImpl(DebugInitialization<Scalar>) { }
#endif

    template <typename T, enable_if_scalar_t<T> = 0>
    StaticArrayImpl(T v) {
        Scalar value = (Scalar) v;
        DRJIT_CHKSCALAR("Constructor (scalar broadcast)");
        for (size_t i = 0; i < Size_; ++i)
            m_data[i] = value;
    }

    template <typename T = Value_, enable_if_t<!std::is_same_v<T, Scalar>> = 0>
    StaticArrayImpl(const Value &v) {
        for (size_t i = 0; i < Size_; ++i)
            m_data[i] = v;
    }

    /// Construct from component values
    template <typename... Ts, detail::enable_if_components_t<Size_, Ts...> = 0>
    DRJIT_INLINE StaticArrayImpl(Ts&&... ts) : m_data{ move_cast_t<Ts, Value>(ts)... } {
        DRJIT_CHKSCALAR("Constructor (component values)");
    }

    /// Construct from sub-arrays
    template <typename T1, typename T2, typename T = StaticArrayImpl, enable_if_t<
              depth_v<T1> == depth_v<T> && size_v<T1> == Base::Size1 &&
              depth_v<T2> == depth_v<T> && size_v<T2> == Base::Size2 &&
              Base::Size2 != 0> = 0>
    StaticArrayImpl(T1 &&a1, T2 &&a2)
        : StaticArrayImpl(a1, a2, std::make_index_sequence<Base::Size1>(),
                                  std::make_index_sequence<Base::Size2>()) { }

private:
    template <typename T1, typename T2, size_t... Is1, size_t... Is2>
    StaticArrayImpl(T1 &&a1, T2 &&a2, std::index_sequence<Is1...>,
                    std::index_sequence<Is2...>)
        : m_data{ a1.entry(Is1)..., a2.entry(Is2)... } { }

public:
    /// Access elements by reference, and without error-checking
    DRJIT_INLINE Value &entry(size_t i) { return m_data[i]; }

    /// Access elements by reference, and without error-checking (const)
    DRJIT_INLINE const Value &entry(size_t i) const { return m_data[i]; }

    /// Pointer to the underlying storage
    Value *data() { return m_data; }

    /// Pointer to the underlying storage (const)
    const Value *data() const { return m_data; }

private:
    Value m_data[Size_];
};

/// Special case for zero-sized arrays
template <typename Value_, bool IsMask_, typename Derived_>
struct StaticArrayImpl<Value_, 0, IsMask_, Derived_>
    : StaticArrayBase<std::conditional_t<IsMask_, mask_t<Value_>, Value_>, 0,
                      IsMask_, Derived_> {
    using Base =
        StaticArrayBase<std::conditional_t<IsMask_, mask_t<Value_>, Value_>, 0,
                        IsMask_, Derived_>;

    using typename Base::Value;
    using Base::entry;

    Value &entry(size_t /*i*/) { return *data(); }
    const Value &entry(size_t /*i*/) const { return *data(); }

    /// Pointer to the underlying storage (returns \c nullptr)
    Value *data() { return nullptr; }

    /// Pointer to the underlying storage (returns \c nullptr, const)
    const Value *data() const { return nullptr; }

    DRJIT_ARRAY_DEFAULTS(StaticArrayImpl)

    StaticArrayImpl() = default;
    template <typename Value2, typename Derived2>
    StaticArrayImpl(const ArrayBaseT<Value2, IsMask_, Derived2> &) { }
    template <typename Value2, typename Derived2>
    StaticArrayImpl(const ArrayBaseT<Value2, IsMask_, Derived2> &, detail::reinterpret_flag) { }
    StaticArrayImpl(const Value &) { }
};

namespace detail {
    template <typename T> bool put_shape(size_t *shape) {
        size_t cur = *shape,
               size = size_v<T> == Dynamic ? 0 : size_v<T>,
               maxval = cur > size ? cur : size;

        if (maxval != size && size != 1)
            return false; // ragged array

        *shape = maxval;

        if constexpr (is_array_v<value_t<T>>) {
            if (!put_shape<value_t<T>>(shape + 1))
                return false;
        }

        return true;
    }

    /// Write the shape of an array to 'shape'
    template <typename T> bool put_shape(const T &array, size_t *shape) {
        DRJIT_MARK_USED(shape); DRJIT_MARK_USED(array);

        if constexpr (is_array_v<T>) {
            size_t cur = *shape, size = array.derived().size(),
                   maxval = cur > size ? cur : size;

            if (maxval != size && size != 1)
                return false; // ragged array

            *shape = maxval;

            if constexpr (is_array_v<value_t<T>>) {
                if (size == 0) {
                    return put_shape<value_t<T>>(shape + 1);
                } else {
                    for (size_t i = 0; i < size; ++i)
                        if (!put_shape(array.derived().entry(i), shape + 1))
                            return false;
                }
            }
        }

        return true;
    }

    template <bool Abbrev, size_t Depth, typename T, size_t ... Is>
    void to_string(StringBuffer &buf, const T &v, const size_t *shape, size_t *indices) {
        constexpr size_t Dimensions = depth_v<T>;
        constexpr bool Last = Depth == Dimensions - 1;
        // Skip output when there are more than 20 elements.
        constexpr size_t Threshold = 20; // Must be divisible by 4

        DRJIT_MARK_USED(shape);
        DRJIT_MARK_USED(indices);

        // On vectorized types, iterate over the last dimension first
        size_t i = Depth;
        using Leaf = leaf_array_t<T>;
        if constexpr (!Leaf::BroadcastOuter || Leaf::IsDynamic) {
            if (Depth == 0)
                i = Dimensions - 1;
            else
                i -= 1;
        }

        if constexpr (Last && (is_complex_v<T> || is_quaternion_v<T>)) {
            // Special handling for complex numbers and quaternions
            bool prev = false;

            for (size_t j = 0; j < size_v<T>; ++j) {
                indices[i] = j;

                scalar_t<T> value = v.derived().entry(indices[Is]...);
                if (value == 0)
                    continue;

                if (prev || value < 0)
                    buf.put(value < 0 ? "-" : "+");
                buf.put(value);
                prev = true;

                if (is_complex_v<T> && j == 1)
                    buf.put('j');
                else if (is_quaternion_v<T> && j < 3)
                    buf.put("ijk"[j]);
            }
            if (!prev)
                buf.put("0");
            return;
        }

        size_t size = shape[i];

        buf.put('[');
        for (size_t j = 0; j < size; ++j) {
            indices[i] = j;

            if (Abbrev && size >= Threshold && j * 4 == Threshold) {
                buf.fmt(".. %zu skipped ..", size - Threshold / 2);
                j = size - Threshold / 4 - 1;
            } else {
                if constexpr (Last)
                    buf.put(v.derived().entry(indices[Is]...));
                else
                    to_string<Abbrev, Depth + 1, T, Is..., Depth + 1>(buf, v, shape, indices);
            }

            if (j + 1 < size) {
                if (Last) {
                    buf.put(", ");
                } else {
                    buf.put(",\n");
                    for (size_t i = 0; i <= Depth; ++i)
                        buf.put(' ');
                }
            }
        }
        buf.put(']');
    }
}

template <typename Array> bool ragged(const Array &a) {
    size_t shape[depth_v<Array> == 0 ? 1 : depth_v<Array>] { };
    return !detail::put_shape(a, shape);
}

template <typename Stream, typename Value, bool IsMask, typename Derived,
          enable_if_not_array_t<Stream> = 0>
DRJIT_NOINLINE Stream &operator<<(Stream &os, const ArrayBaseT<Value, IsMask, Derived> &a) {
    StringBuffer buf;
    buf.put(a);
    os << buf.get();
    return os;
}

NAMESPACE_END(drjit)
