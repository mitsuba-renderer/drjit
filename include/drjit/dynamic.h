/*
    drjit/dynamic.h -- Naive dynamic array (extremely inefficient, CUDA/LLVM
    arrays are almost always preferable)

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/array.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_>
struct DynamicArray
    : ArrayBase<Value_, is_mask_v<Value_>, DynamicArray<Value_>> {
    template <typename Value2_> friend struct DynamicArray;

    static constexpr bool IsMask = is_mask_v<Value_>;
    using Base = ArrayBase<Value_, IsMask, DynamicArray<Value_>>;
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

    DynamicArray(const DynamicArray &a) : m_size(a.m_size) {
        if (!empty()) {
            m_data = new Value[m_size];
            for (size_t i = 0; i < m_size; ++i)
                m_data[i] = a.m_data[i];
        }
    }

    DynamicArray(DynamicArray &&a)
        : m_data(a.m_data), m_size(a.m_size), m_free(a.m_free) {
        a.m_size = 0;
        a.m_data = nullptr;
        a.m_free = true;
    }

    template <typename Value2, bool IsMask2, typename Derived2>
    DynamicArray(const ArrayBase<Value2, IsMask2, Derived2> &v) {
        size_t size = v.derived().size();
        init_(size);
        for (size_t i = 0; i < size; ++i)
            m_data[i] = (Value) v.derived().entry(i);
    }

    template <typename Value2, bool IsMask2, typename Derived2>
    DynamicArray(const ArrayBase<Value2, IsMask2, Derived2> &v,
                 detail::reinterpret_flag) {
        size_t size = v.derived().size();
        init_(size);
        for (size_t i = 0; i < size; ++i)
            m_data[i] = reinterpret_array<Value>(v.derived().entry(i));
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    DynamicArray(T value) {
        init_(1);
        m_data[0] = (Value) value;
    }

    /// Move-construct if possible. Convert values with the wrong type.
    template <typename Src>
    using cast_t = std::conditional_t<
        std::is_same_v<std::decay_t<Src>, Value>,
        std::conditional_t<std::is_reference_v<Src>, Src, Src &&>, Value>;

    /// Construct from component values
    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    DRJIT_INLINE DynamicArray(Ts&&... ts) {
        DRJIT_CHKSCALAR("Constructor (component values)");
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
        for (size_t i = 0; i < m_size; ++i)
            m_data[i] = a.m_data[i];
        return *this;
    }

    DynamicArray &operator=(DynamicArray &&a) {
        std::swap(a.m_data, m_data);
        std::swap(a.m_free, m_free);
        std::swap(a.m_size, m_size);
        return *this;
    }

    DRJIT_INLINE size_t size() const { return m_size; }
    DRJIT_INLINE DynamicArray copy() { return DynamicArray(*this); }

    DRJIT_INLINE Value &entry(size_t i) {
        if (m_size == 1)
            i = 0;
        return m_data[i];
    }

    DRJIT_INLINE const Value &entry(size_t i) const {
        if (m_size == 1)
            i = 0;
        return m_data[i];
    }

    static DynamicArray load_(const void *ptr, size_t size) {
        DynamicArray result;
        result.init_(size);
        memcpy(result.m_data, ptr, sizeof(Value) * size);
        return result;
    }

    void store_(void *ptr) const {
        memcpy(ptr, m_data, sizeof(Value) * m_size);
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
            result.entry(i) = zeros<Value>();

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

    static DynamicArray linspace_(Value min, Value max, size_t size, bool endpoint) {
        DynamicArray result;
        result.init_(size);

        Scalar step = (max - min) / Scalar(size - (endpoint ? 1 : 0));

        for (size_t i = 0; i < size; ++i) {
            if constexpr (std::is_floating_point_v<Scalar>)
                result.entry(i) = fmadd(Scalar(i), step, min);
            else
                result.entry(i) = Scalar(i) * step + min;
        }

        return result;
    }

    DynamicArray<uint32_t> compress_() const {
        if constexpr (!IsMask) {
            drjit_raise("Unsupported argument type!");
        } else {
            DynamicArray<uint32_t> result;
            result.init_(m_size);

            uint32_t accum = 0;
            for (size_t i = 0; i < m_size; ++i) {
                if (m_data[i])
                    result.m_data[accum++] = Value(i);
            }
            result.m_size = accum;
            return result;
        }
    }

    void init_(size_t size) {
        if (size == 0 || size == Dynamic)
            return;
        m_data = new Value[size];
        m_size = size;
        m_free = true;
    }

    const Value *data() const { return m_data; }
    Value *data() { return m_data; }

protected:
    Value *m_data = nullptr;
    size_t m_size = 0;
    bool m_free = true;
};

NAMESPACE_END(drjit)
