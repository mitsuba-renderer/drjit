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

template <typename Value, size_t Size, typename SFINAE = int> struct Local {
    using Index = uint32_t;
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
            m_data[i] = l.m_dta[i];
    }
    Local &operator=(const Local &) = delete;
    Local &operator=(Local &&l) {
        for (size_t i = 0; i < Size; ++i)
            m_data[i] = l.m_dta[i];
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

template <typename Value, size_t Size> struct Local<Value, Size, enable_if_array_t<Value>> {
    static constexpr JitBackend Backend = backend_v<Value>;
    using Index = uint32_array_t<Value>;
    using Mask = mask_t<Value>;

    Local() {
        m_index = jit_array_create(Backend, Value::Type, 1, Size);
    }

    Local(const Value &value) {
        uint32_t tmp = jit_array_create(Backend, Value::Type, 1, Size);
        m_index = jit_array_init(tmp, value.index());
        jit_var_dec_ref(tmp);
    }

    ~Local() { jit_var_dec_ref(m_index); }
    Local(const Local &) = delete;
    Local(Local &&l) {
        m_index = l.m_index;
        l.m_index = 0;
    }
    Local &operator=(const Local &) = delete;
    Local &operator=(Local &&l) {
        jit_var_dec_ref(m_index);
        m_index = l.m_index;
        l.m_index = 0;
    }

    Value read(const Index &offset, const Mask &active = true) const {
        return Value::steal(jit_array_read(m_index, offset.index(), active.index()));
    }

    void write(const Index &offset, const Value &value, const Mask &active = true) {
        uint32_t new_index = jit_array_write(m_index, offset.index(),
                                             value.index(), active.index());
        jit_var_dec_ref(m_index);
        m_index = new_index;
    }

    size_t size() const { return Size; }

private:
    uint32_t m_index;
};


NAMESPACE_END(drjit)
