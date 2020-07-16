/*
    enoki/util.h -- Supplemental utility functions for Enoki arrays

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki/idiv.h>

NAMESPACE_BEGIN(enoki)

template <typename Array> Array tile(const Array &array, size_t count) {
    static_assert(is_array_v<Array> && is_dynamic_v<Array>,
                  "tile(): requires a dynamic Enoki array as input!");

    size_t size = array.size();

    if constexpr (Array::Depth > 1) {
        Array result;
        if (Array::Size == Dynamic)
            result.init_(size);

        for (size_t i = 0; i < size; ++i)
            result.set_entry(i, tile(array.entry(i), count));

        return result;
    } else {
        using UInt = uint32_array_t<Array>;
        UInt index = imod(arange<UInt>((uint32_t) (size * count)), (uint32_t) size);
        return gather<Array>(array, index);
    }
}

template <typename Array> Array repeat(const Array &array, size_t count) {
    static_assert(is_array_v<Array> && is_dynamic_v<Array>,
                  "repeat(): requires a dynamic Enoki array as input!");

    size_t size = array.size();

    if constexpr (Array::Depth > 1) {
        Array result;
        if (Array::Size == Dynamic)
            result.init_(size);

        for (size_t i = 0; i < size; ++i)
            result.set_entry(i, repeat(array.entry(i), count));

        return result;
    } else {
        using UInt = uint32_array_t<Array>;
        UInt index = idiv(arange<UInt>((uint32_t) (size * count)), (uint32_t) count);
        return gather<Array>(array, index);
    }
}

template <typename T> std::pair<T, T> meshgrid(const T &x, const T &y) {
    static_assert(array_depth_v<T> == 1 && is_dynamic_array_v<T>,
                  "meshgrid(): requires two 1D dynamic Enoki arrays as input!");

    uint32_t lx = (uint32_t) x.size(), ly = (uint32_t) y.size();

    if (lx == 1 || ly == 1) {
        return { x, y };
    } else {
        auto [yi, xi] = idivmod(arange<uint32_array_t<T>>(lx*ly), lx);
        return { gather<T>(x, xi), gather<T>(y, yi) };
    }
}

template <typename Index, typename Predicate>
Index binary_search(scalar_t<Index> start_, scalar_t<Index> end_,
                    const Predicate &pred) {
    Index start(start_), end(end_);

    scalar_t<Index> iterations = (start_ < end_) ?
        (log2i(end_ - start_) + 1) : 0;

    for (size_t i = 0; i < iterations; ++i) {
        Index middle = sr<1>(start + end);

        mask_t<Index> cond = pred(middle);

        masked(start,  cond) = min(middle + 1, end);
        masked(end,   !cond) = middle;
    }

    return start;
}

/// Vectorized N-dimensional 'range' iterable with automatic mask computation
template <typename Value> struct range {
    static constexpr size_t Dimension = array_depth_v<Value> == 2 ?
        array_size_v<Value> : 1;

    using Scalar = scalar_t<Value>;
    using Packet =
        std::conditional_t<array_depth_v<Value> == 2, value_t<Value>, Value>;
    using Size   = Array<Scalar, Dimension>;

    struct iterator {
        iterator(size_t index) : index(index) { }
        iterator(size_t index, Size size)
            : index(index), size(size) {
            for (size_t i = 0; i < Dimension - 1; ++i)
                div[i] = size[i];
            if constexpr (!is_dynamic_v<Value>)
                index_p = arange<Packet>();
        }

        bool operator==(const iterator &it) const { return it.index == index; }
        bool operator!=(const iterator &it) const { return it.index != index; }

        iterator &operator++() {
            index += 1;
            if constexpr (!is_dynamic_v<Value>)
                index_p += Scalar(Packet::Size);
            return *this;
        }

        std::pair<Value, mask_t<Packet>> operator*() const {
            if constexpr (array_depth_v<Value> == 1) {
                if constexpr (!is_dynamic_v<Value>)
                    return { index_p, index_p < size[0] };
                else
                    return { arange<Value>(size[0]), true };
            } else {
                Value value;
                if constexpr (!is_dynamic_v<Value>)
                    value[0] = index_p;
                else
                    value[0] = arange<Packet>(hprod(size));
                ENOKI_UNROLL for (size_t i = 0; i < Dimension - 1; ++i)
                    value[i + 1] = div[i](value[i]);
                Packet offset = zero<Packet>();
                ENOKI_UNROLL for (size_t i = Dimension - 2; ; --i) {
                    offset = size[i] * (value[i + 1] + offset);
                    value[i] -= offset;
                    if (i == 0)
                        break;
                }

                return { value, value[Dimension - 1] < size[Dimension - 1] };
            }
        }

    private:
        size_t index;
        Packet index_p;
        Size size;
        divisor<Scalar> div[Dimension > 1 ? (Dimension - 1) : 1];
    };

    template <typename... Args>
    range(Args&&... args) : size(args...) { }

    iterator begin() {
        return iterator(0, size);
    }

    iterator end() {
        if constexpr (is_dynamic_v<Value>)
            return iterator(hprod(size) == 0 ? 0 : 1);
        else
            return iterator((hprod(size) + Packet::Size - 1) / Packet::Size);
    }

private:
    Size size;
};


NAMESPACE_END(enoki)
