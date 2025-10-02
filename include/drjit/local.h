/*
    drjit/local.h -- C++ bindings for local memory arrays

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/jit.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_,
          size_t Size_,
          typename Index_ = uint32_array_t<Value_>,
          typename SFINAE = int>
struct Local {
    static constexpr size_t Size = Size_;
    static_assert(Size != Dynamic, "Scalar local arrays are only fixed size. "
                                   "If you meant to instantiate a JIT variant "
                                   "or a DRJIT_STRUCT you may have forgotten to "
                                   "add the Index template parameter.");
    using Value = Value_;
    using Index = Index_;
    using Mask = bool;

    Local() = default;

    Local(const Value &value) {
        for (size_t i = 0; i < Size; ++i)
            m_data[i] = value;
    }

    ~Local() = default;
    Local(const Local &) = delete;
    Local(Local &&l) = default;
    Local &operator=(const Local &) = delete;
    Local &operator=(Local &&l) = default;

    Value read(const Index &offset, const Mask &active = true) const {
        if (active)
            return m_data[offset];
        else
            return Value();
    }

    void write(const Index &offset, const Value &value, const Mask &active = true) {
        if (active)
            m_data[offset] = value;
    }

    size_t size() const { return Size; }

private:
    Value m_data[Size];
};


NAMESPACE_BEGIN(detail)

template<typename T>
void init_impl(const T &value, const size_t size, vector<uint32_t>& arrays) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        uint32_t result;
        if (!value.empty()) {
            uint32_t i1  = value.index();
            size_t width = jit_var_size(i1);
            uint32_t i2  = jit_array_create(
                backend_v<T>, var_type<value_t<T>>::value,
                width, size);
            result = jit_array_init(i2, i1);
            jit_var_dec_ref(i2);
        } else {
            result = jit_array_create(
                backend_v<T>, var_type<value_t<T>>::value,
                1, size);
        }
        arrays.push_back(result);
    } else if constexpr (is_traversable_v<T>) {
        // Recurse and try again if the object is traversable
        traverse_1(fields(value),
                    [&](auto &v) { init_impl(v, size, arrays); });
    }
}

template <typename T>
void read_impl(T &result,
               const uint32_t &offset,
               const uint32_t &active,
               const vector<uint32_t> &arrays,
               size_t &counter) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        if (counter >= arrays.size())
            jit_raise("Local::read(): internal error, ran out of "
                        "variable arrays!");
        result = T::steal(jit_array_read(arrays[counter++], offset, active));
    } else if constexpr (is_traversable_v<T>) {
        // Recurse and try again if the object is traversable
        traverse_1(fields(result), [&](auto &r) {
            read_impl(r, offset, active, arrays, counter);
        });
    }
}

template <typename T>
void write_impl(const uint32_t &offset,
                const T &value,
                const uint32_t &active,
                vector<uint32_t> &arrays,
                size_t &counter) {
    if constexpr (is_jit_v<T> && depth_v<T> == 1) {
        if (counter >= arrays.size())
            jit_raise("Local::write(): internal error, ran out of "
                        "variable arrays!");

        if (value.index_ad())
            jit_raise("Local memory writes are not differentiable. You "
                        "must use 'drjit.detach()' to disable gradient "
                        "tracking of the written value.");

        uint32_t result =
            jit_array_write(arrays[counter], offset, value.index(), active);
        jit_var_dec_ref(arrays[counter]);
        arrays[counter++] = result;

    } else if constexpr (is_traversable_v<T>) {
        // Recurse and try again if the object is traversable
        traverse_1(fields(value),
                        [&](auto &v) { write_impl(offset, v, active, arrays, counter); });
    }
}

NAMESPACE_END(detail)


/**
 * \brief Local memory implemented on top of drjit-core jit_array_*
 * \details The array `value` of static or dynamic width will be used
 * to initialize the entries of local memory with length `Size`.
 * `Size` can be drjit::Dynamic, in which case a call to resize will
 * be required before usage.
 */
template <typename Value_, size_t Size_, typename Index_>
struct Local<Value_, Size_, Index_,
             enable_if_t<is_array_v<Value_> || (is_array_v<Index_> && is_drjit_struct_v<Value_>)>>
{
    static constexpr JitBackend Backend = backend_v<Value_>;
    static constexpr size_t Size = Size_;
    using Value = Value_;
    using Index = Index_;
    using Mask = mask_t<Index>;

    /**
     * \brief Allocate local memory
     * \param value optional inital value (also used when resizing dynamic memory)
     */
    Local(Value value = empty<Value>())
        : m_size(Size == Dynamic ? 1 : Size), m_value(value) {
        detail::init_impl(m_value, m_size, m_arrays);
    }

    ~Local() {
        for (uint32_t index : m_arrays)
            jit_var_dec_ref(index);
    }
    Local(const Local &) = delete;
    Local(Local &&l) = default;

    Local &operator=(const Local &) = delete;
    Local &operator=(Local &&l) {
        for (uint32_t index : m_arrays)
            jit_var_dec_ref(index);
        m_size = std::move(l.m_size);
        m_value = std::move(l.m_value);
        m_arrays = std::move(l.m_arrays);
        return *this;
    }

    Value read(const Index &offset, const Mask &active = true) const {
        Value result;
        size_t counter = 0;
        detail::read_impl(result, offset.index(), active.index(), m_arrays, counter);

        if (counter != m_arrays.size())
            jit_raise(
                "Local::read(): internal error, did not access all variable "
                "arrays!");

        return result;
    }

    void write(const Index &offset, const Value &value, const Mask &active = true) {
        size_t counter = 0;
        detail::write_impl(offset.index(), value, active.index(), m_arrays, counter);

        if (counter != m_arrays.size())
            jit_raise(
                "Local.write(): internal error, did not access all variable "
                "arrays!");
    }

    size_t size() { return m_size; }

    /**
     * Reserve a new array of `length` and discard any current contents
     */
    void resize(size_t size) {
        for (uint32_t index : m_arrays)
            jit_var_dec_ref(index);
        m_arrays.clear();
        m_size = size;
        detail::init_impl(m_value, m_size, m_arrays);
    }

private:
    size_t m_size;
    Value m_value;
    vector<uint32_t> m_arrays;
};

NAMESPACE_END(drjit)
