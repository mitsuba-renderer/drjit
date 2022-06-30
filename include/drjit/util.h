/*
    drjit/util.h -- Supplemental utility functions for Dr.Jit arrays

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/idiv.h>
#include <drjit/loop.h>

NAMESPACE_BEGIN(drjit)

template <typename Array> Array tile(const Array &array, size_t count) {
    static_assert(is_array_v<Array> && is_dynamic_v<Array>,
                  "tile(): requires a dynamic Dr.Jit array as input!");

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
                  "repeat(): requires a dynamic Dr.Jit array as input!");

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

template <typename Array> auto ravel(const Array &array) {
    if constexpr (array_depth_v<Array> <= 1) {
        return array;
    } else {
        using Result = leaf_array_t<Array>;
        using Index = uint32_array_t<Result>;

        size_t shape[array_depth_v<Array> + 1 /* avoid zero-sized array */ ] { };
        detail::put_shape(array, shape);

        size_t size = shape[0];
        for (size_t i = 1; i < array_depth_v<Array>; ++i)
            size *= shape[i];

        Result result = empty<Result>(size);
        scatter(result, array,
                arange<Index>(shape[array_depth_v<Array> - 1]));

        return result;
    }
}

template <typename Target, typename Source> Target unravel(const Source &source) {
    static_assert(array_depth_v<Source> == 1, "Expected a flat array as input!");
    static_assert(array_depth_v<Target> > 1, "Expected a nested array as output!");

    Target target;
    size_t shape[array_depth_v<Target> + 1 /* avoid zero-sized array */ ] { };
    detail::put_shape(target, shape);

    size_t source_size = source.size(), size = shape[0];
    for (size_t i = 1; i < array_depth_v<Target> - 1; ++i)
        size *= shape[i];

    if (size == 0 || source_size % size != 0)
        drjit_raise("unravel(): input array length not divisible by stride");

    using Index = uint32_array_t<Source>;
    Index indices = arange<Index>(source_size / size);

    return gather<Target>(source, indices);
}

template <typename T> std::pair<T, T> meshgrid(const T &x, const T &y) {
    static_assert(array_depth_v<T> == 1 && is_dynamic_array_v<T>,
                  "meshgrid(): requires two 1D dynamic Dr.Jit arrays as input!");

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
    scalar_t<Index> iterations = (start_ < end_) ?
        (log2i(end_ - start_) + 1) : 0;

    Index start(start_), end(end_);

    using Mask = mask_t<Index>;

    if constexpr (is_jit_v<Index>) {
        // We might be running multiple binary searches in parallel..
        using Index1 = detached_t<std::conditional_t<is_static_array_v<Index>,
                                                     value_t<Index>, Index>>;
        using Mask1 = mask_t<Index1>;

        if (iterations >= 2 && jit_flag(JitFlag::LoopRecord)) {
            char title[80];
            snprintf(title, sizeof(title),
                     "dr::binary_search(size=%zu, iterations=%zu)",
                     (size_t)(end_ - start_), (size_t) iterations);

            Index1 index = zeros<Index1>(width(pred(start)));
            Loop<Mask1> loop(title, start, end, index);

            while (loop(index < iterations)) {
                Index middle = sr<1>(start + end);
                Mask cond = detach(pred(middle));

                start = select(cond, minimum(middle + 1, end), start);
                end   = select(cond, end, middle);

                index++;
            }

            return start;
        }
    }

    for (size_t i = 0; i < iterations; ++i) {
        Index middle = sr<1>(start + end);

        Mask cond = pred(middle);

        masked(start,  cond) = minimum(middle + 1, end);
        masked(end,   !cond) = middle;
    }

    return start;
}

/// Vectorized N-dimensional 'range' iterable with automatic mask computation
template <typename Value> struct range {
    static constexpr bool Recurse =
        !(Value::IsPacked || Value::Size == Dynamic) ||
        array_depth_v<Value> == 2;
    static constexpr size_t Dimension = Recurse ? array_size_v<Value> : 1;

    using Scalar = scalar_t<Value>;
    using Packet = std::conditional_t<Recurse, value_t<Value>, Value>;
    using Size   = Array<Scalar, Dimension>;

    static constexpr size_t PacketSize = array_size_v<Packet>;

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
                index_p += Scalar(PacketSize);
            return *this;
        }

        std::pair<Value, mask_t<Packet>> operator*() const {
            if constexpr (!Recurse) {
                if constexpr (!is_dynamic_v<Value>)
                    return { index_p, index_p < size[0] };
                else
                    return { arange<Value>(size[0]), true };
            } else {
                Value value;
                if constexpr (!is_dynamic_v<Value>)
                    value[0] = index_p;
                else
                    value[0] = arange<Packet>(prod(size));
                DRJIT_UNROLL for (size_t i = 0; i < Dimension - 1; ++i)
                    value[i + 1] = div[i](value[i]);
                Packet offset = zeros<Packet>();
                DRJIT_UNROLL for (size_t i = Dimension - 2; ; --i) {
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
            return iterator(prod(size) == 0 ? 0 : 1);
        else
            return iterator((prod(size) + PacketSize - 1) / PacketSize);
    }

private:
    Size size;
};

NAMESPACE_END(drjit)
