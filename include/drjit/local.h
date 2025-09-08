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
    Local(Local &&l) {
        for (size_t i = 0; i < Size; ++i)
            m_data[i] = l.m_data[i];
    }
    Local &operator=(const Local &) = delete;
    Local &operator=(Local &&l) {
        for (size_t i = 0; i < Size; ++i)
            m_data[i] = l.m_data[i];
    }

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
        initialize();
    }

    ~Local() {
        for (uint32_t index : m_arrays)
            jit_var_dec_ref(index);
    }
    Local(const Local &) = delete;
    Local(Local &&l) {
        m_arrays.swap(l.m_arrays);
        l.m_arrays.clear();
    }
    Local &operator=(const Local &) = delete;
    Local &operator=(Local &&l) {
        for (uint32_t index : m_arrays)
            jit_var_dec_ref(index);
        m_arrays.swap(l.m_arrays);
        l.m_arrays.clear();
    }

    Value read(const Index &offset, const Mask &active = true) const {
        Value result;
        size_t counter = 0;
        auto callback  = [&](auto &result, auto &&callback) -> void {
            using T = std::decay_t<decltype(result)>;
            if constexpr (is_jit_v<T> && depth_v<T> == 1) {
                if (counter >= m_arrays.size())
                    jit_raise("Local::read(): internal error, ran out of "
                              "variable arrays!");
                result = T::steal(jit_array_read(m_arrays[counter++],
                                                  offset.index(), active.index()));
            } else if constexpr (is_traversable_v<T>) {
                /// Recurse and try again if the object is traversable
                traverse_1(fields(result), [&](auto &result) {
                    callback(result, callback);
                });
            }
        };
        callback(result, callback);

        if (counter != m_arrays.size())
            jit_raise(
                "Local::read(): internal error, did not access all variable "
                "arrays!");

        return result;
    }

    void write(const Index &offset, const Value &value, const Mask &active = true) {
        size_t counter = 0;
        auto callback  = [&](auto &value, auto &&callback) -> void {
            using T = std::decay_t<decltype(value)>;
            if constexpr (is_jit_v<T> && depth_v<T> == 1) {
                if (counter >= m_arrays.size())
                    jit_raise("Local::write(): internal error, ran out of "
                                "variable arrays!");

                if (value.index_ad())
                    jit_raise("Local memory writes are not differentiable. You "
                                "must use 'drjit.detach()' to disable gradient "
                                "tracking of the written value.");

                uint32_t result =
                    jit_array_write(m_arrays[counter], offset.index(),
                                     value.index(), active.index());
                jit_var_dec_ref(m_arrays[counter]);
                m_arrays[counter++] = result;

            } else if constexpr (is_traversable_v<T>) {
                /// Recurse and try again if the object is traversable
                traverse_1(fields(value),
                                [&](auto &value) { callback(value, callback); });
            }
        };
        callback(value, callback);

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
        initialize();
    }

private:
    void initialize() {
        auto callback = [&](auto &value, auto &&callback) -> void {
            using T = std::decay_t<decltype(value)>;
            if constexpr (is_jit_v<T> && depth_v<T> == 1) {
                uint32_t result;
                if (!value.empty()) {
                    uint32_t i1  = value.index();
                    size_t width = jit_var_size(i1);
                    uint32_t i2  = jit_array_create(
                        backend_v<T>, var_type<value_t<T>>::value,
                        width, m_size);
                    result = jit_array_init(i2, i1);
                    jit_var_dec_ref(i2);
                } else {
                    result = jit_array_create(
                        backend_v<T>, var_type<value_t<T>>::value,
                        1, m_size);
                }
                m_arrays.push_back(result);
            } else if constexpr (is_traversable_v<T>) {
                /// Recurse and try again if the object is traversable
                traverse_1(fields(value),
                               [&](auto &value) { callback(value, callback); });
            }
        };
        callback(m_value, callback);
    }

    size_t m_size;
    Value m_value;
    vector<uint32_t> m_arrays;
};

NAMESPACE_END(drjit)
