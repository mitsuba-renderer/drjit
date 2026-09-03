/*
    funcenv.h -- Extract the environment that a wrapped callable reads for @dr.freeze

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

/**
 * \brief The environment that a wrapped callable reads
 *
 * This class finds the global and closure variables that a callable to be
 * recorded reads, and it can snapshot their values at call time. It also
 * stores the names of the positional parameters for error reporting.
 */
struct FunctionEnvironment {
    /// The function that is being frozen
    nb::callable func;

    /// Names of its positional parameters
    nb::tuple arg_names;

    /// Names of the global variables that it reads, and the ``__globals__``
    /// dictionary in which their values are looked up
    nb::tuple global_names;
    nb::dict globals;

    /// Names of its closure variables (``co_freevars``) and the matching
    /// cells (``__closure__``, a tuple or ``None``)
    nb::tuple free_names;
    nb::object closure;

    explicit FunctionEnvironment(nb::callable func);

    /// Snapshot the current values of the globals and closure variables
    nb::dict capture() const;

    /// Visit and release the Python references held here (GC support)
    int tp_traverse(visitproc visit, void *arg) const;
    void tp_clear();
};

extern void export_funcenv(nb::module_ &);
