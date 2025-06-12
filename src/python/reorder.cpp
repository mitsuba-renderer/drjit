/*
    reorder.cpp -- Bindings for drjit.reorder_threads()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "reorder.h"
#include "detail.h"
#include <drjit/autodiff.h>

nb::object reorder_threads(nb::handle_t<dr::ArrayBase> key, int num_bits,
                           nb::handle value) {
    const ArraySupplement &s_key = supp(key.type());
    if (s_key.ndim != 1 || s_key.type != (uint8_t) VarType::UInt32 ||
        s_key.backend == (uint8_t) JitBackend::None)
        nb::raise("drjit.reorder_threads(): 'key' must be a JIT-compiled 32 "
                  "bit unsigned integer array (e.g., 'drjit.cuda.UInt32' or "
                  "'drjit.llvm.ad.UInt32')");

    dr::vector<uint64_t> value_indices;
    ::collect_indices(value, value_indices);
    if (value_indices.size() == 0)
        nb::raise("drjit.reorder_threads(): 'value' must be a valid PyTree "
                  "containing at least one JIT-compiled type");

    uint32_t n_values = (uint32_t) value_indices.size();

    // Extract JIT indices
    dr::vector<uint32_t> jit_indices(n_values);
    for (size_t i = 0; i < n_values; ++i)
        jit_indices[i] = (uint32_t) value_indices[i];

    // Create updated values with reordering
    dr::detail::index32_vector out_indices(n_values);
    jit_reorder(s_key.index(inst_ptr(key)), num_bits, n_values,
                jit_indices.data(), out_indices.data());

    // Re-combine with AD indices
    dr::vector<uint64_t> new_value_indices(n_values);
    for (size_t i = 0; i < n_values; ++i) {
        uint32_t ad_index = value_indices[i] >> 32;
        new_value_indices[i] = (((uint64_t) ad_index) << 32 | ((uint64_t) out_indices[i]));
    }

    return ::update_indices(value, new_value_indices);
}

void export_reorder(nb::module_ &m) {
    m.def("reorder_threads", &reorder_threads, "key"_a, "num_bits"_a, "value"_a,
          doc_reorder_threads);
}
