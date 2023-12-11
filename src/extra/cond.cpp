/*
    extra/loop.cpp -- Logic to implement conditional through one common
    interface with support for symbolic and evaluated execution styles along
    with automatic differentiation.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <drjit/custom.h>
#include <string>

namespace dr = drjit;
using JitVar = GenericArray<void>;

static void ad_cond_evaluated(JitBackend backend, const char *name,
                              void *payload, uint32_t cond,
                              ad_cond_read read_cb, ad_cond_write write_cb,
                              ad_cond_body body_cb) {
    jit_log(LogLevel::InfoSym,
            "ad_cond_evaluated(\"%s\"): executing conditional expression.",
            name);

    size_t size = jit_var_size(cond);

    JitVar true_mask = JitVar::steal(jit_var_mask_apply(cond, size)),
           neg_mask = JitVar::steal(jit_var_not(cond)),
           false_mask = JitVar::steal(jit_var_mask_apply(neg_mask.index(), size));

    dr_index64_vector true_idx, false_idx, combined_idx;

    {
        scoped_push_mask guard(backend, true_mask.index());
        body_cb(payload, true);
    }

    read_cb(payload, true_idx);

    {
        scoped_push_mask guard(backend, false_mask.index());
        body_cb(payload, false);
    }

    read_cb(payload, false_idx);

    if (true_idx.size() != false_idx.size())
        jit_raise("ad_cond_evaluated(): inconsistent number of outputs!");

    for (size_t i = 0; i < true_idx.size(); ++i) {
        uint64_t i1 = true_idx[i], i2 = false_idx[i];

        combined_idx.push_back_steal(i1 == i2 ? ad_var_inc_ref(i1)
                                              : ad_var_select(cond, i1, i2));
    }

    write_cb(payload, combined_idx);
}

static void ad_cond_symbolic(JitBackend backend, const char *name,
                             void *payload, uint32_t cond,
                             ad_cond_read read_cb, ad_cond_write write_cb,
                             ad_cond_body body_cb) {
    bool symbolic = jit_flag(JitFlag::SymbolicScope);
    scoped_record record_guard(backend);

    JitVar start = JitVar::steal(jit_var_cond_start(name, symbolic, cond));
    dr_index64_vector indices;
    dr::dr_vector<uint32_t> indices32;

    body_cb(payload, true);
    read_cb(payload, indices);

    indices32.reserve(indices.size());
    for (uint64_t index : indices)
        indices32.push_back((uint32_t) index);

    jit_var_cond_append(start.index(), indices32.data(), indices32.size());
    indices.release();
    indices32.clear();

    body_cb(payload, false);
    read_cb(payload, indices);

    for (uint64_t index : indices)
        indices32.push_back((uint32_t) index);

    jit_var_cond_append(start.index(), indices32.data(), indices32.size());
    indices.release();

    jit_var_cond_end(start.index(), indices32.data());

    for (uint32_t index : indices32)
        indices.push_back_steal(index);
    write_cb(payload, indices);
}

bool ad_cond(JitBackend backend, int symbolic, const char *name, void *payload,
             uint32_t cond, ad_cond_read read_cb, ad_cond_write write_cb,
             ad_cond_body body_cb, ad_cond_delete delete_cb, bool ad) {
    try {
        if (name == nullptr)
            name = "unnamed";

        if (strchr(name, '\n') || strchr(name, '\r'))
            jit_raise("'name' may not contain newline characters.");

        if (symbolic == -1)
            symbolic = (int) jit_flag(JitFlag::SymbolicConditionals);

        if (symbolic != 0 && symbolic != 1)
            jit_raise("'symbolic' must equal 0, 1, or -1.");

        if (jit_var_state(cond) == VarState::Literal) {
            jit_log(LogLevel::InfoSym,
                    "ad_cond_evaluated(\"%s\"): removing conditional expression "
                    "with uniform condition.", name);
            body_cb(payload, !jit_var_is_zero_literal(cond));
            return true;
        }

        (void) ad;
        if (symbolic)
            ad_cond_symbolic(backend, name, payload, cond, read_cb, write_cb,
                             body_cb);
        else
            ad_cond_evaluated(backend, name, payload, cond, read_cb, write_cb,
                              body_cb);

        return true; // Caller should directly call delete()
    } catch (...) {
        if (delete_cb)
            delete_cb(payload);
        throw;
    }
}
