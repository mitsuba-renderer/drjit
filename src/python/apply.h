/*
    apply.h -- Implementation of the internal ``apply()``, ``traverse()``,
    and ``transform()`` functions, which recursively perform operations on
    Dr.Jit arrays and Python object trees ("pytrees")

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

/// Like apply(), but returns a pair of results. Used for dr.sincos, dr.sincosh, dr.frexp
extern nb::object apply_ret_pair(ArrayOp op, const char *name,
                                 nb::handle_t<dr::ArrayBase> h);

/// Callback for the ``traverse()`` operation below
struct TraverseCallback {
    virtual void operator()(nb::handle h) = 0;
};

/// Callback for the ``traverse_pair()`` operation below
struct TraversePairCallback {
    virtual void operator()(nb::handle h1, nb::handle h2) = 0;
};

/// Callback for the ``transform()`` operation below
struct TransformCallback {
    virtual nb::handle transform_type(nb::handle tp) const;
    virtual nb::object transform_unknown(nb::handle h) const;
    virtual void operator()(nb::handle h1, nb::handle h2) = 0;
    virtual void postprocess(nb::handle h1, nb::handle h2);
};

/// Callback for the ``transform_pair()`` operation below
struct TransformPairCallback {
    virtual nb::handle transform_type(nb::handle tp) const;
    virtual void operator()(nb::handle h1, nb::handle h2, nb::handle h3) = 0;
};

/// Invoke the given callback on leaf elements of the pytree 'h'
extern void traverse(const char *op, TraverseCallback &callback,
                     nb::handle h);

/// Parallel traversal of two compatible pytrees 'h1' and 'h2'
extern void traverse_pair(const char *op, TraversePairCallback &callback,
                          nb::handle h1, nb::handle h2, const char *name,
                          bool report_inconsistencies = true);

/// Transform an input pytree 'h' into an output pytree, potentially of a different type
extern nb::object transform(const char *op, TransformCallback &callback, nb::handle h);

/// Transform a pair of input pytrees 'h1' and 'h2' into an output pytree, potentially of a different type
extern nb::object transform_pair(const char *op, TransformPairCallback &callback, nb::handle h1, nb::handle h2);
