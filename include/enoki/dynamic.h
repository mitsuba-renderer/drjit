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
#include <iostream>

NAMESPACE_BEGIN(enoki)

template <typename Value_> struct DynamicArray : ArrayBaseT<Value_, DynamicArray<Value_>> {
    using Base = ArrayBaseT<Value_, DynamicArray<Value_>>;
    using typename Base::Value;
    using Base::empty;

    using ArrayType = DynamicArray<array_t<Value>>;
    using MaskType  = DynamicArray<mask_t<Value>>;
    template <typename T> using ReplaceValue = DynamicArray<T>;

    static constexpr size_t Size = Dynamic;

    using Base::Base;
    using Base::coeff;

    DynamicArray() : m_data(nullptr), m_size(0), m_free(true) { }

    /// Move-construct if possible. Convert values with the wrong type.
    template <typename Src>
    using cast_t = std::conditional_t<
        std::is_same_v<std::decay_t<Src>, Value>,
        std::conditional_t<std::is_reference_v<Src>, Src, Src &&>, Value>;

    /// Construct from component values
    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              (!std::is_same_v<Ts, detail::reinterpret_flag> && ...))> = 0>
    ENOKI_INLINE DynamicArray(Ts&&... ts) : m_data(nullptr), m_size(0), m_free(true) {
        ENOKI_CHKSCALAR("Constructor (component values)");
        Value data[] = { cast_t<Ts>(ts)... };
        init_(sizeof...(Ts));
        for (size_t i = 0; i < sizeof...(Ts); ++i)
            m_data[i] = std::move(data[i]);
    }

    DynamicArray(const DynamicArray &a)
        : m_data(nullptr), m_size(a.m_size), m_free(true) {
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

    ENOKI_INLINE Value &coeff(size_t i) { return m_data[i]; }
    ENOKI_INLINE const Value &coeff(size_t i) const { return m_data[i]; }

    static DynamicArray zero_(size_t size) {
        DynamicArray result;
        result.init_(size);

        for (size_t i = 0; i < size; ++i)
            result.coeff(i) = zero<Value>();

        return result;
    }

    static DynamicArray empty_(size_t size) {
        DynamicArray result;
        result.init_(size);
        return result;
    }

    void init_(size_t size) {
        m_data = size == 0 ? nullptr : new Value[size];
        m_size = size;
        m_free = true;
    }

private:
    Value *m_data;
    size_t m_size;
    bool m_free;
};

NAMESPACE_END(enoki)
