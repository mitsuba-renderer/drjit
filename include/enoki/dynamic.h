/*
    enoki/array_dynamic.h -- Naive dynamic array (extremely inefficient,
    CUDA/LLVM arrays are almost always preferable)

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/array.h>

NAMESPACE_BEGIN(enoki)

template <typename Value_>
struct DynamicArray
    : ArrayBaseT<Value_, is_mask_v<Value_>, DynamicArray<Value_>> {
    static constexpr bool IsMask = is_mask_v<Value_>;
    using Base = ArrayBaseT<Value_, IsMask, DynamicArray<Value_>>;
    using typename Base::Value;
    using typename Base::Scalar;
    using Base::empty;

    /// This is a dynamically allocated array, indicated using a special size flag
    static constexpr size_t Size = Dynamic;
    static constexpr bool IsDynamic = true;

    using ArrayType = DynamicArray<array_t<Value>>;
    using MaskType  = DynamicArray<mask_t<Value>>;
    template <typename T> using ReplaceValue = DynamicArray<T>;

    using Base::Base;
    using Base::entry;

    DynamicArray() = default;

    DynamicArray(const DynamicArray &a)
        : m_size(a.m_size) {
        if (empty())
            return;
        m_data = new Value[m_size];
        memcpy(m_data, a.m_data, m_size * sizeof(Value));
    }

    DynamicArray(DynamicArray &&a)
        : m_data(a.m_data), m_size(a.m_size), m_free(a.m_free) {
        a.m_size = 0;
        a.m_data = nullptr;
        a.m_free = true;
    }

    template <typename Value2, typename Derived2>
    DynamicArray(const ArrayBaseT<Value2, IsMask, Derived2> &v) {
        size_t size = v.derived().size();
        init_(size);
        for (size_t i = 0; i < size; ++i)
            m_data[i] = (Value) v.derived().entry(i);
    }

    template <typename Value2, typename Derived2>
    DynamicArray(const ArrayBaseT<Value2, IsMask, Derived2> &v,
                 detail::reinterpret_flag) {
        size_t size = v.derived().size();
        init_(size);
        for (size_t i = 0; i < size; ++i)
            m_data[i] = reinterpret_array<Value>(v.derived().entry(i));
    }

    DynamicArray(const Value &v) {
        init_(1);
        m_data[0] = v;
    }

    /// Move-construct if possible. Convert values with the wrong type.
    template <typename Src>
    using cast_t = std::conditional_t<
        std::is_same_v<std::decay_t<Src>, Value>,
        std::conditional_t<std::is_reference_v<Src>, Src, Src &&>, Value>;

    /// Construct from component values
    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              (!std::is_same_v<Ts, detail::reinterpret_flag> && ...))> = 0>
    ENOKI_INLINE DynamicArray(Ts&&... ts) {
        ENOKI_CHKSCALAR("Constructor (component values)");
        Value data[] = { cast_t<Ts>(ts)... };
        init_(sizeof...(Ts));
        for (size_t i = 0; i < sizeof...(Ts); ++i)
            m_data[i] = std::move(data[i]);
    }

    ~DynamicArray() {
        if (m_free)
            delete[] m_data;
    }

    DynamicArray &operator=(const DynamicArray &a) {
        if (m_free)
            delete[] m_data;
        m_data = new Value[a.m_size];
        m_size = a.m_size;
        m_free = true;
        memcpy(m_data, a.m_data, m_size * sizeof(Value));
    }

    DynamicArray &operator=(DynamicArray &&a) {
        delete[] m_data;
        m_data = a.m_data;
        m_size = a.m_size;
        m_free = a.m_free;
        a.m_data = nullptr;
        a.m_size = 0;
        a.m_free = true;
        return *this;
    }

    ENOKI_INLINE size_t size() const { return m_size; }

    ENOKI_INLINE Value &entry(size_t i) {
        if (m_size == 1)
            i = 0;
        return m_data[i];
    }

    ENOKI_INLINE const Value &entry(size_t i) const {
        if (m_size == 1)
            i = 0;
        return m_data[i];
    }

    static DynamicArray empty_(size_t size) {
        DynamicArray result;
        result.init_(size);
        return result;
    }

    static DynamicArray zero_(size_t size) {
        DynamicArray result;
        result.init_(size);

        for (size_t i = 0; i < size; ++i)
            result.entry(i) = zero<Value>();

        return result;
    }

    static DynamicArray full_(const Value &v, size_t size) {
        DynamicArray result;
        result.init_(size);

        for (size_t i = 0; i < size; ++i)
            result.entry(i) = v;

        return result;
    }

    static DynamicArray arange_(ssize_t start, ssize_t stop, ssize_t step) {
        size_t size = size_t((stop - start + step - (step > 0 ? 1 : -1)) / step);
        DynamicArray result;
        result.init_(size);

        for (size_t i = 0; i < size; ++i)
            result.entry(i) = (Scalar) ((ssize_t) start + (ssize_t) i * step);

        return result;
    }

    static DynamicArray linspace_(Value min, Value max, size_t size) {
        DynamicArray result;
        result.init_(size);

        Scalar step = (max - min) / Scalar(size - 1);

        for (size_t i = 0; i < size; ++i) {
            if constexpr (std::is_floating_point_v<Scalar>)
                result.entry(i) = fmadd(Scalar(i), step, min);
            else
                result.entry(i) = Scalar(i) * step + min;
        }

        return result;
    }

    void init_(size_t size) {
        if (size == 0)
            return;
        m_data = new Value[size];
        m_size = size;
    }

protected:
    Value *m_data = nullptr;
    size_t m_size = 0;
    bool m_free = true;
};

NAMESPACE_END(enoki)
