/*
    detail.h -- Bindings for miscellaneous implementation details

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

/// RAII helper to temporarily stash the reference count of a Dr.Jit variable
struct StashRef {
    StashRef(uint32_t index) : handle(jit_var_stash_ref(index)) { }
    ~StashRef() { jit_var_unstash_ref(handle); }
    StashRef(StashRef &&w) : handle(w.handle) { w.handle = 0; }
    StashRef(const StashRef &) = delete;
    uint64_t handle;
};

// See misc.cpp for documentation of these functions
extern nb::object copy(nb::handle h);
extern void collect_indices(nb::handle, dr::vector<uint64_t> &,
                            bool inc_ref = false);
extern nb::object update_indices(nb::handle, const dr::vector<uint64_t> &);
extern void check_compatibility(nb::handle, nb::handle, bool, const char *);
extern void stash_ref(nb::handle h, dr::vector<StashRef> &);

extern nb::object reset(nb::handle h);
extern void enable_py_tracing();
extern void disable_py_tracing();

extern void export_detail(nb::module_ &);

extern nb::object reduce_identity(nb::type_object_t<dr::ArrayBase> tp, ReduceOp op, uint32_t size);
extern bool can_scatter_reduce(nb::type_object_t<dr::ArrayBase> tp, ReduceOp op);
