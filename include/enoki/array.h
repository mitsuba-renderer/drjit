/*
    enoki/array.h -- Main header file for the Enoki array class and
    various template specializations

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

/// Clang-specific workaround: don't pull in all of <math.h> when including <stdlib.h>.
#if !defined(_LIBCPP_STDLIB_H)
#  define __need_malloc_and_calloc
#  include <stdlib.h>
#  undef __need_malloc_and_calloc
#  define _LIBCPP_STDLIB_H
#endif

#include <enoki/array_generic.h>
#include <enoki/array_mask.h>
#include <enoki/array_struct.h>

NAMESPACE_BEGIN(enoki)

template <typename Value_, size_t Size_>
struct Array : StaticArrayImpl<Value_, Size_, false, Array<Value_, Size_>> {
    using Base = StaticArrayImpl<Value_, Size_, false, Array<Value_, Size_>>;

    using ArrayType = Array;
    using MaskType = Mask<Value_, Size_>;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = Array<T, Size_>;

    ENOKI_ARRAY_IMPORT(Array, Base)
};

template <typename Value_, size_t Size_>
struct Mask : MaskBase<Value_, Size_, Mask<Value_, Size_>> {
    using Base = MaskBase<Value_, Size_, Mask<Value_, Size_>>;

    using MaskType = Mask;
    using ArrayType = Array<Value_, Size_>;

    /// Type alias for creating a similar-shaped array over a different type
    template <typename T> using ReplaceValue = Mask<T, Size_>;

    ENOKI_ARRAY_IMPORT(Mask, Base)
};

NAMESPACE_END(enoki)
