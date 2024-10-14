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
#include <limits>

NAMESPACE_BEGIN(drjit)

template <typename Value_>
struct DynamicArray
    : ArrayBaseT<Value_, is_mask_v<Value_>, DynamicArray<Value_>> {
    template <typename Value2_> friend struct DynamicArray;

    static constexpr bool IsMask = is_mask_v<Value_>;
    using Base = ArrayBaseT<Value_, IsMask, DynamicArray<Value_>>;
    using typename Base::Value;
    using typename Base::Scalar;
    using Base::empty;

    /// This is a dynamically allocated array, indicated using a special size flag
    static constexpr size_t Size = Dynamic;
    static constexpr bool IsDynamic = true;
    static constexpr bool IsVector = true;

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
    DynamicArray(const ArrayBaseT<Value2, IsMask2, Derived2> &v) {
        size_t size = v.derived().size();
        init_(size);
        for (size_t i = 0; i < size; ++i)
            m_data[i] = (Value) v.derived().entry(i);
    }

    template <typename Value2, bool IsMask2, typename Derived2>
    DynamicArray(const ArrayBaseT<Value2, IsMask2, Derived2> &v,
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
        Value* new_data = new Value[a.m_size];
        for (size_t i = 0; i < a.m_size; ++i)
            new_data[i] = a.m_data[i];
        if (m_free)
            delete[] m_data;
        m_size = a.m_size;
        m_data = new_data;
        m_free = true;
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

        if constexpr (drjit::detail::is_scalar_v<Value>) {
            memcpy(result.m_data, ptr, sizeof(Value) * size);
        } else {
            for (size_t i = 0; i < size; ++i)
                result.entry(i) =
                    load<Value>(static_cast<const Value *>(ptr) + i);
        }

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
            if constexpr (drjit::is_floating_point_v<Scalar>)
                result.entry(i) = fmadd(Scalar(i), step, min);
            else
                result.entry(i) = Scalar(i) * step + min;
        }

        return result;
    }

    DynamicArray<uint32_t> compress_() const {
        if constexpr (!IsMask) {
            drjit_fail("Unsupported argument type!");
        } else {
            DynamicArray<uint32_t> result;
            result.init_(m_size);

            size_t accum = 0;
            for (size_t i = 0; i < m_size; ++i) {
                if (m_data[i])
                    result.m_data[accum++] = (uint32_t) i;
            }
            result.m_size = accum;
            return result;
        }
    }

    DynamicArray<uint32_t> count_() const {
        if constexpr (!IsMask) {
            drjit_fail("Unsupported argument type!");
        } else {
            uint32_t accum = 0;
            for (size_t i = 0; i < m_size; ++i) {
                if (m_data[i])
                    ++accum;
            }
            return DynamicArray<uint32_t>(accum);
        }
    }

    DynamicArray block_reduce_(ReduceOp op, size_t block_size, int) const {
        if constexpr (IsMask) {
            drjit_fail("Unsupported argument type!");
        } else {
            size_t blocks = (m_size + block_size - 1) / block_size;
            DynamicArray result;
            result.init_(blocks);
            using Intermediate = std::conditional_t<std::is_same_v<Value, half>, float, Value>;

            for (size_t i = 0; i < blocks; ++i) {
                size_t start = i * block_size,
                       end = drjit::minimum(start + block_size, m_size);

                Intermediate value = m_data[start];

                switch (op) {
                    case ReduceOp::Add:
                        for (size_t j = start + 1; j != end; ++j)
                            value += m_data[j];
                    break;

                    case ReduceOp::Mul:
                        for (size_t j = start + 1; j != end; ++j)
                            value *= m_data[j];
                    break;

                    case ReduceOp::Min:
                        for (size_t j = start + 1; j != end; ++j)
                            value = drjit::minimum(value, m_data[j]);
                        break;

                    case ReduceOp::Max:
                        for (size_t j = start + 1; j != end; ++j)
                            value = drjit::maximum(value, m_data[j]);
                        break;

                    case ReduceOp::Or:
                        if constexpr (drjit::is_integral_v<Scalar>) {
                            for (size_t j = start + 1; j != end; ++j)
                                value = value | m_data[j];
                        }
                        break;

                    case ReduceOp::And:
                        if constexpr (drjit::is_integral_v<Scalar>) {
                            for (size_t j = start + 1; j != end; ++j)
                                value = value & m_data[j];
                        }
                        break;

                    default:
                        drjit_fail("Unsupported reduction type!");
                }

                result[i] = (Value) value;
            }

            return result;
        }
    }

    DynamicArray block_prefix_reduce_(ReduceOp op, uint32_t block_size, bool exclusive, bool reverse) const {
        if constexpr (IsMask) {
            drjit_fail("Unsupported argument type!");
        } else {
            size_t blocks = (m_size + block_size - 1) / block_size;
            DynamicArray result;
            result.init_(m_size);
            using Intermediate = std::conditional_t<std::is_same_v<Value, half>, float, Value>;

            for (size_t i = 0; i < blocks; ++i) {
                size_t start = i * block_size,
                       end = drjit::minimum(start + block_size, m_size);
                int step = 1;

                if (reverse) {
                    size_t tmp = start;
                    start = end - 1;
                    end = tmp - 1;
                    step = -1;
                }

                Intermediate value;

                switch (op) {
                    case ReduceOp::Add:
                        value = 0;
                        for (size_t j = start; j != end; j += step) {
                            Intermediate before = value;
                            value += m_data[j];
                            result[j] = Value(exclusive ? before : value);
                        }
                        break;

                    case ReduceOp::Mul:
                        value = 1;
                        for (size_t j = start; j != end; j += step) {
                            Intermediate before = value;
                            value *= m_data[j];
                            result[j] = Value(exclusive ? before : value);
                        }
                        break;

                    case ReduceOp::Min:
                        if constexpr (std::is_floating_point_v<Scalar>)
                            value = std::numeric_limits<Scalar>::infinity();
                        else
                            value = std::numeric_limits<Scalar>::max();

                        for (size_t j = start; j != end; j += step) {
                            Intermediate before = value;
                            value *= m_data[j];
                            result[j] = Value(exclusive ? before : value);
                        }
                        break;

                    case ReduceOp::Max:
                        if constexpr (std::is_floating_point_v<Scalar>)
                            value = -std::numeric_limits<Scalar>::infinity();
                        else
                            value = std::numeric_limits<Scalar>::min();

                        for (size_t j = start; j != end; j += step) {
                            Intermediate before = value;
                            value *= m_data[j];
                            result[j] = exclusive ? before : value;
                        }
                        break;

                    case ReduceOp::And:
                        if constexpr (drjit::is_integral_v<Scalar>) {
                            value = (Scalar) -1;
                            for (size_t j = start; j != end; j += step) {
                                Intermediate before = value;
                                value &= m_data[j];
                                result[j] = Value(exclusive ? before : value);
                            }
                        }
                        break;

                    case ReduceOp::Or:
                        if constexpr (drjit::is_integral_v<Scalar>) {
                            value = 0;
                            for (size_t j = start; j != end; j += step) {
                                Intermediate before = value;
                                value |= m_data[j];
                                result[j] = Value(exclusive ? before : value);
                            }
                        }
                        break;

                    default:
                        drjit_fail("Unsupported reduction type!");
                }
            }

            return result;
        }
    }

    void init_(size_t size) {
        if (size == 0)
            return;
        m_data = new Value[size];
        m_size = size;
        m_free = true;
    }

    static auto counter(size_t size) {
        uint32_array_t<DynamicArray> result;
        result.init_(size);
        for (size_t i = 0; i < size; ++i)
            result.entry(i) = (uint32_t) i;
        return result;
    }

    const Value *data() const { return m_data; }
    Value *data() { return m_data; }

protected:
    Value *m_data = nullptr;
    size_t m_size = 0;
    bool m_free = true;
};

NAMESPACE_END(drjit)
