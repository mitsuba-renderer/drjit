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
#include <drjit/while_loop.h>

NAMESPACE_BEGIN(drjit)

template <typename Array> auto ravel(const Array &array) {
    if constexpr (depth_v<Array> <= 1) {
        return array;
    } else if constexpr (is_dynamic_array_v<Array>) {
        using Result = value_t<Array>;
        using Index = uint32_array_t<Result>;
        Result result = empty<Result>(array.size() * width(array));
        Index indices = arange<Index>(width(array)) * (uint32_t) array.size();
        for (uint32_t i = 0; i < (uint32_t) array.size(); i++)
            scatter(result, array[i], indices + i);
        return result;
    } else {
        using Result = leaf_array_t<Array>;
        using Index = uint32_array_t<Result>;

        size_t shape[depth_v<Array> + 1 /* avoid zero-sized array */ ] { };
        detail::put_shape(&array, shape);

        size_t size = shape[0];
        for (size_t i = 1; i < depth_v<Array>; ++i)
            size *= shape[i];

        Result result = empty<Result>(size);
        scatter(result, array,
                arange<Index>(shape[depth_v<Array> - 1]));

        return result;
    }
}

template <typename Target, typename Source> Target unravel(const Source &source) {
    static_assert(depth_v<Source> == 1, "Expected a flat array as input!");
    static_assert(depth_v<Target> > 1, "Expected a nested array as output!");

    Target target;
    size_t shape[depth_v<Target> + 1 /* avoid zero-sized array */ ] { };
    detail::put_shape(&target, shape);

    size_t source_size = source.size(), size = shape[0];
    for (size_t i = 1; i < depth_v<Target> - 1; ++i)
        size *= shape[i];

    if (size == 0 || source_size % size != 0)
        drjit_fail("unravel(): input array length not divisible by stride");

    using Index = uint32_array_t<Source>;
    Index indices = arange<Index>(source_size / size);

    return gather<Target>(source, indices);
}

/**
 * The `index_xy` argument allows switching between Cartesian (default),
 * or matrix indexing.
 */
template <typename T> std::pair<T, T> meshgrid(const T &x, const T &y,
                                               bool index_xy = true) {
    static_assert(depth_v<T> == 1 && is_dynamic_array_v<T>,
                  "meshgrid(): requires two or three 1D dynamic Dr.Jit arrays as input!");

    // Cartesian or matrix indexing, consistent with NumPy
    T rx = (index_xy ? x : y);
    T ry = (index_xy ? y : x);
    uint32_t lx = (uint32_t) rx.size(),
             ly = (uint32_t) ry.size();

    if (lx == 1 || ly == 1) {
        // Nothing to do.
    } else {
        auto [yi, xi] = idivmod(arange<uint32_array_t<T>>(ly * lx), lx);
        rx = gather<T>(rx, xi);
        ry = gather<T>(ry, yi);
    }

    if (index_xy)
        return { rx, ry };
    else
        return { ry, rx };
}

/**
 * The `index_xy` argument allows switching between Cartesian (default),
 * or matrix indexing.
 */
template <typename T> std::tuple<T, T, T> meshgrid(const T &x, const T &y, const T &z,
                                                   bool index_xy = true) {
    static_assert(depth_v<T> == 1 && is_dynamic_array_v<T>,
                  "meshgrid(): requires two or three 1D dynamic Dr.Jit arrays as input!");

    // Cartesian or matrix indexing, consistent with NumPy
    T rx = (index_xy ? x : y);
    T ry = (index_xy ? y : x);
    T rz = z;
    uint32_t lx = (uint32_t) rx.size(),
             ly = (uint32_t) ry.size(),
             lz = (uint32_t) rz.size();

    if ((lx == 1 && ly == 1) || (ly == 1 && lz == 1) || (lx == 1 && lz == 1)) {
        // Nothing do to.
    } else {
        // Note: Cartesian indexing consistent with  NumPy (y, x, z)
        auto [yi, tmp] = idivmod(arange<uint32_array_t<T>>(ly * lx * lz), lx * lz);
        auto [xi, zi] = idivmod(tmp, lz);
        rx = gather<T>(rx, xi);
        ry = gather<T>(ry, yi);
        rz = gather<T>(rz, zi);
    }

    if (index_xy)
        return { rx, ry, rz };
    else
        return { ry, rx, rz };
}

/// Binary search with scalar starting/ending indices
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
            Index1 index = zeros<Index1>(width(pred(start)));

            drjit::tie(start, end, index) = drjit::while_loop(
                drjit::make_tuple(start, end, index),
                [iterations](const Index&, const Index&, const Index1& index) {
                    return index < iterations;
                },
                [pred](Index& start, Index& end, Index1& index) {
                    Index middle = sr<1>(start + end);
                    Mask cond = detach(pred(middle));

                    start = select(cond, minimum(middle + 1, end), start);
                    end   = select(cond, end, middle);

                    index++;
                }
            );

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

/**
 * \brief Binary search with non-scalar starting/ending indices
 *
 * Note: this method forcefully uses recorded loops if called in a recorded
 * context.
 */
template <typename IndexN,
          typename Index1 = std::conditional_t<is_static_array_v<IndexN>,
                                               value_t<IndexN>, IndexN>,
          typename Predicate>
IndexN binary_search(typename std::enable_if_t<is_jit_v<Index1>, Index1> start_,
                     typename std::enable_if_t<is_jit_v<Index1>, Index1> end_,
                     const Predicate &pred) {
    static_assert(drjit::depth_v<Index1> == 1,
                  "Starting/ending indices array must have depth 1!");

    using MaskN = mask_t<IndexN>;
    using Mask1 = mask_t<Index1>;

    Index1 iterations =
        detach(select(start_ < end_, log2i(end_ - start_) + 1, 0));

    IndexN start(start_), end(end_);

    Index1 index = zeros<Index1>(width(pred(start)));

    bool loop_record = jit_flag(JitFlag::LoopRecord);
    if (jit_flag(JitFlag::Recording))
        jit_set_flag(JitFlag::LoopRecord, true);

    drjit::tie(start, end, index) = drjit::while_loop(
        drjit::make_tuple(start, end, index),
        [iterations](const IndexN&, const IndexN&, const Index1& index) {
            return index < iterations;
        },
        [pred](IndexN& start, IndexN& end, Index1& index) {
            IndexN middle = sr<1>(start + end);
            MaskN cond    = detach(pred(middle));

            start = select(cond, minimum(middle + 1, end), start);
            end   = select(cond, end, middle);

            index++;
        },
        "dr::binary_search()");

    jit_set_flag(JitFlag::LoopRecord, loop_record);

    return start;
}

/// Vectorized N-dimensional 'range' iterable with automatic mask computation
template <typename Value> struct range {
    static constexpr bool Recurse =
        !(Value::IsPacked || Value::Size == Dynamic) ||
        depth_v<Value> == 2;
    static constexpr size_t Dimension = Recurse ? size_v<Value> : 1;

    using Scalar = scalar_t<Value>;
    using Packet = std::conditional_t<Recurse, value_t<Value>, Value>;
    using Size   = Array<Scalar, Dimension>;

    static constexpr size_t PacketSize = size_v<Packet>;

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
                for (size_t i = 0; i < Dimension - 1; ++i)
                    value[i + 1] = div[i](value[i]);
                Packet offset = zeros<Packet>();
                for (size_t i = Dimension - 2; ; --i) {
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

/// Special handling of packet gathers when packet size is only known at runtime
template <typename Value, typename Source, typename Index, typename Mask>
void gather_packet_dynamic(size_t packet_size, Source &&source, const Index &index,
    Value* out, const Mask &mask_ = true, ReduceMode mode = ReduceMode::Auto) {

    // Broadcast mask to match shape of Index
    mask_t<plain_t<Index>> mask = mask_;

    auto default_gather = [&]{
        for (size_t i = 0; i < packet_size; ++i)
            out[i] = gather<Value>(source,
                fmadd(index, (uint32_t) packet_size, (uint32_t) i), mask, mode);
    };

    if constexpr (is_jit_v<Value>) {
        if ((packet_size & (packet_size - 1)) == 0 && packet_size > 1) {
            uint64_t *res_indices = (uint64_t *) alloca(
                sizeof(uint64_t) * packet_size);
            ad_var_gather_packet(packet_size,
                source.index_combined(),
                index.index(),
                mask.index(), 
                res_indices,
                mode);

            for (size_t i = 0; i < packet_size; ++i)
                out[i] = Value::steal((typename Value::Index) res_indices[i]);
        } else {
            default_gather();
        }
    } else {
        default_gather();
    }
}

NAMESPACE_END(drjit)
