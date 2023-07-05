/*
    apply.h -- Implementation of the internal apply() function,
    which recursively propagates operations through Dr.Jit arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

enum ApplyMode {
    /// Ordinary unary/binary/ternary operation mapping T+ -> T
    Normal,

    /// In-place variant that modifies the first argument if possible
    InPlace,

    /// Rich comparison, a binary operation mapping T, T -> mask_t<T>
    RichCompare,

    ///  Select, a ternary operation mapping mask_t<T>, T, T -> T
    Select
};

/**
 * A significant portion of Dr.Jit operations pass through the central apply()
 * function below. It performs arithmetic operation (e.g. addition, FMA) by
 *
 * 1.  Casting operands into compatible representations, and
 * 2a. Calling an existing "native" implementation of the operation if
 *     available (see drjit/python.h), or alternatively:
 * 2b. Executing a fallback loop that recursively invokes the operation
 *     on array elements.
 *
 * The ApplyMode template parameter slightly adjusts the functions' operation
 * (see the definition of the ApplyMode enumeration for details).
 */
template <ApplyMode Mode, typename Func, typename... Args, size_t... Is>
PyObject *apply(ArrayOp op, Func func, std::index_sequence<Is...>,
                Args... args) noexcept;

/// Callback operator for 'traverse' below
struct TraverseCallback {
    virtual void operator()(nb::handle h) = 0;
};

/// Invoke the given callback on leaf elements of the pytree 'h'
extern void traverse(const char *op, TraverseCallback &callback, nb::handle h);
