
/*
    drjit/autodiff.h -- Forward/reverse-mode automatic differentiation wrapper

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/jit.h>
#include <drjit/extra.h>

NAMESPACE_BEGIN(drjit)

template <JitBackend Backend_, typename Value_>
struct DRJIT_TRIVIAL_ABI DiffArray
    : ArrayBaseT<Value_, is_mask_v<Value_>, DiffArray<Backend_, Value_>> {
    static_assert(std::is_scalar_v<Value_>,
                  "Differentiable arrays can only be created over scalar types!");

    template <JitBackend Backend2, typename Value2> friend struct DiffArray;

    // -----------------------------------------------------------------------
    //! @{ \name Basic type declarations
    // -----------------------------------------------------------------------

    using Value = Value_;
    using Base = ArrayBaseT<Value_, is_mask_v<Value_>, DiffArray<Backend_, Value_>>;

    static constexpr JitBackend Backend = Backend_;

    static constexpr bool IsDiff = true;
    static constexpr bool IsArray = true;
    static constexpr bool IsDynamic = true;
    static constexpr bool IsJIT = true;
    static constexpr bool IsCUDA = Backend == JitBackend::CUDA;
    static constexpr bool IsLLVM = Backend == JitBackend::LLVM;
    static constexpr bool IsFloat = std::is_floating_point_v<Value_>;
    static constexpr bool IsClass =
        std::is_pointer_v<Value_> &&
        std::is_class_v<std::remove_pointer_t<Value_>>;
    static constexpr size_t Size = Dynamic;

    static constexpr VarType Type =
        IsClass ? VarType::UInt32 : var_type_v<Value>;

    using ActualValue = std::conditional_t<IsClass, uint32_t, Value>;

    using CallSupport =
        call_support<std::decay_t<std::remove_pointer_t<Value_>>, DiffArray>;

    template <typename T> using ReplaceValue = DiffArray<Backend, T>;
    using MaskType = DiffArray<Backend, bool>;
    using ArrayType = DiffArray;

    using Index = std::conditional_t<IsFloat, uint64_t, uint32_t>;
    using Detached = JitArray<Backend, Value>;

    //! @}
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    //! @{ \name Constructors and assignment operators
    // -----------------------------------------------------------------------

    DiffArray() = default;

    ~DiffArray() noexcept {
        if constexpr (IsFloat)
            ad_var_dec_ref(m_index);
        else
            jit_var_dec_ref(m_index);
    }

    DiffArray(const DiffArray &a) {
        if constexpr (IsFloat) {
            m_index = ad_var_inc_ref_maybe(a.m_index);
        } else {
            m_index = a.m_index;
            jit_var_inc_ref(m_index);
        }
    }

    DiffArray(DiffArray &&a) noexcept : m_index(a.m_index) {
        a.m_index = 0;
    }

    template <typename T>
    DiffArray(const DiffArray<Backend, T> &v) {
        if constexpr (IsFloat && std::is_floating_point_v<T>)
            m_index = ad_var_cast(v.m_index, Type);
        else
            m_index = (Index) jit_var_cast((uint32_t) v.m_index, Type, 0);
    }
    template <typename T>
    DiffArray(const DiffArray<Backend, T, DiffArray2> &v, detail::reinterpret_flag) {
        m_index = (Index) jit_var_cast((uint32_t) v.m_index, Type, 1);
    }

    DiffArray(const Detached &v) : m_index(v.m_index) {
        jit_var_inc_ref((uint32_t) m_index);
    }

    template <typename T, enable_if_scalar_t<T> = 0>
    DiffArray(T value) : m_index(Detached(value).release()) { }

    template <typename... Ts, enable_if_t<(sizeof...(Ts) > 1 &&
              detail::and_v<!std::is_same_v<Ts, detail::reinterpret_flag>...>)> = 0>
    DiffArray(Ts&&... ts) : m_index(Detached(ts...).release()) { }

    DiffArray &operator=(const DiffArray &a) {
        if constexpr (IsFloat) {
            Index old_index = m_index;
            m_index = ad_var_inc_ref_maybe(a.m_index);
            ad_var_dec_ref(old_index);
        } else {
            jit_var_inc_ref(a.m_index);
            jit_var_dec_ref(m_index);
            m_index = a.m_index;
        }
        return *this;
    }

    DiffArray &operator=(DiffArray &&a) {
        std::swap(m_index, a.m_index);
        return *this;
    }

    //! @}
    // -----------------------------------------------------------------------

    DiffArray add_(const DiffArray &a) const {
        if constexpr (IsFloat)
            return steal(ad_var_add<Value>(m_index, a.m_index));
        else
            return steal(jit_var_add(m_index, a.m_index));
    }

    //! @}
    // -----------------------------------------------------------------------

    static DiffArray steal(Index index) {
        DiffArray result;
        result.m_index = index;
        return result;
    }

    static DiffArray borrow(Index index) {
        DiffArray result;
        jit_var_inc_ref(index);
        result.m_index = index;
        return result;
    }

    Index release() {
        Index tmp = m_index;
        m_index = 0;
        return tmp;
    }

    // void init_(size_t size) {
    //     *this = empty_(size);
    // }


private:
    Index m_index = 0;
};

NAMESPACE_END(drjit)
