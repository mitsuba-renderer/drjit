/*
    drjit/array.h -- Main header file for the Dr.Jit array class and
    various template specializations

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if __cplusplus < 201703L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)
#  error Dr.Jit requires compilation in C++17 mode!
#endif

/// Clang-specific workaround: don't pull in all of <math.h> when including <stdlib.h>.
#if !defined(_LIBCPP_STDLIB_H)
#  define __need_malloc_and_calloc
#  include <stdlib.h>
#  undef __need_malloc_and_calloc
#  define _LIBCPP_STDLIB_H
#endif

#include <drjit/array_generic.h>
#include <drjit/array_mask.h>

NAMESPACE_BEGIN(drjit)

template <typename Value_, size_t Size_>
struct Array : StaticArrayImpl<Value_, Size_, false, Array<Value_, Size_>> {
    using Base = StaticArrayImpl<Value_, Size_, false, Array<Value_, Size_>>;

    using ArrayType = Array;
    using MaskType = Mask<Value_, Size_>;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = Array<T, Size_>;

    DRJIT_ARRAY_IMPORT(Array, Base)
};

template <typename Value_, size_t Size_>
struct Mask : MaskBase<Value_, Size_, Mask<Value_, Size_>> {
    using Base = MaskBase<Value_, Size_, Mask<Value_, Size_>>;

    using MaskType = Mask;
    using ArrayType = Array<array_t<Value_>, Size_>;
    using Value = Value_;
    using Scalar = scalar_t<Value_>;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = Mask<T, Size_>;

    DRJIT_ARRAY_IMPORT(Mask, Base)
};

template <typename Value_, size_t Size_>
struct Array<detail::MaskedArray<Value_>, Size_>
    : detail::MaskedArray<Array<Value_, Size_>> {
    using Base = detail::MaskedArray<Array<Value_, Size_>>;
    using Base::Base;
    using Base::operator=;
    Array(const Base &b) : Base(b) { }
};

template <typename Value_, size_t Size_>
struct Mask<detail::MaskedArray<Value_>, Size_>
    : detail::MaskedArray<Mask<Value_, Size_>> {
    using Base = detail::MaskedArray<Mask<Value_, Size_>>;
    using Base::Base;
    using Base::operator=;
    Mask(const Base &b) : Base(b) { }
};

NAMESPACE_END(drjit)
