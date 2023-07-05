/*
    base.h -- Bindings of the ArrayBase type underlying
    all Dr.Jit arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

/// Reference to the Python ArrayBase type object
extern nb::handle array_base;

/// Reference to the Dr.Jit core module
extern nb::handle array_module;

/// Reference to the Dr.Jit core module
extern nb::handle array_submodules[5];

/// Create and publish the ArrayBase type object
extern void export_base(nb::module_&);

/// Is 'h' a Dr.Jit array type?
inline bool is_drjit_type(nb::handle h) {
    return PyType_IsSubtype((PyTypeObject *) h.ptr(),
                            (PyTypeObject *) array_base.ptr());
}

/// Is 'type(h)' a Dr.Jit array type?
inline bool is_drjit_array(nb::handle h) { return is_drjit_type(h.type()); }

// Fused multiply-add operation used in a few places in the bindings
extern nb::object fma(nb::handle h0, nb::handle h1, nb::handle h2);

// Tenrnary select/blend operation used in a few places in the bindings
extern nb::object select(nb::handle h0, nb::handle h1, nb::handle h2);
