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
    Normal,
    InPlace,
    RichCompare,
    Select
};

/// Perform an arithmetic operation (e.g. addition, FMA), potentially via recursion
template <ApplyMode Mode, typename Func, typename... Args, size_t... Is>
PyObject *apply(ArrayOp op, Func func, std::index_sequence<Is...>,
                Args... args) noexcept;

/// Callback operator for 'traverse' below
struct TraverseCallback {
    virtual void operator()(nb::handle h) = 0;
};

/// Invoke the given callback on leaf elements of the pytree 'h'
extern void traverse(const char *op, TraverseCallback &callback, nb::handle h);
