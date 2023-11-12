/*
    extra/loop.cpp -- Logic to implement loops through one common interface
    with support for symbolic and evaluated execution styles along with
    automatic differentiation.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include <drjit/custom.h>

namespace dr = drjit;

using JitVar = GenericArray<void>;

/// RAII helper to temporarily push a mask onto the Dr.Jit mask stack
struct scoped_push_mask {
    scoped_push_mask(JitBackend backend, uint32_t index) : backend(backend) {
        jit_var_mask_push(backend, index);
    }

    ~scoped_push_mask() { jit_var_mask_pop(backend); }

    JitBackend backend;
};

/// RAII helper to temporarily record symbolic computation
struct scoped_record {
    scoped_record(JitBackend backend) : backend(backend) {
        checkpoint = jit_record_begin(backend, nullptr);
    }

    ~scoped_record() {
        jit_record_end(backend, checkpoint, cleanup);
    }

    void disarm() { cleanup = false; }

    JitBackend backend;
    uint32_t checkpoint;
    bool cleanup = true;
};

static void ad_loop_symbolic(JitBackend backend, const char *name,
                             void *payload,
                             ad_loop_read read_cb, ad_loop_write write_cb,
                             ad_loop_cond cond_cb, ad_loop_body body_cb) {
    dr_index64_vector indices1, backup;
    dr::dr_vector<uint32_t> indices2;

    // Read the loop state variables
    read_cb(payload, indices1);

    indices2.reserve(indices1.size());
    bool needs_ad = false;
    for (uint64_t i : indices1) {
        backup.push_back_borrow(i);
        indices2.push_back((uint32_t) i);
        needs_ad |= (i >> 32) != 0;
    }

    try {
        scoped_record record_guard(backend);

        // Rewrite the loop state variables
        JitVar loop = JitVar::steal(
            jit_var_loop_start(name, indices2.size(), indices2.data()));

        // Propagate these changes
        indices1.release();
        for (uint32_t i : indices2)
            indices1.push_back_steal(i);
        write_cb(payload, indices1);
        indices1.release();
        indices2.clear();

        do {
            // Evaluate the loop condition
            uint32_t active_initial = cond_cb(payload);

            // Potentially optimize the loop away
            if (jit_var_is_zero_literal(active_initial) && jit_flag(JitFlag::OptimizeLoops)) {
                jit_log(LogLevel::InfoSym,
                        "ad_loop_symbolic(\"%s\"): optimized away (loop "
                        "condition is 'false').", name);
                write_cb(payload, backup);
                break;
            }

            JitVar active = JitVar::steal(
                jit_var_mask_apply(active_initial, jit_var_size(active_initial)));

            JitVar loop_cond = JitVar::steal(
                jit_var_loop_cond(loop.index(), active.index()));

            // Evolve the loop state
            {
                if (backend == JitBackend::CUDA)
                    active = JitVar::steal(jit_var_bool(backend, true));

                scoped_push_mask m(backend, active.index());
                body_cb(payload);
            }

            // Fetch latest version of loop state
            read_cb(payload, indices1);
            for (uint64_t i : indices1) {
                indices2.push_back((uint32_t) i);
                needs_ad |= (i >> 32) != 0;
            }

            int rv = jit_var_loop_end(loop.index(), loop_cond.index(),
                                      indices2.data(), record_guard.checkpoint);

            indices1.release();
            for (uint32_t i : indices2)
                indices1.push_back_steal(i);
            write_cb(payload, indices1);
            indices1.release();
            indices2.clear();

            // Construct the loop object
            if (rv) {
                // All done
                record_guard.disarm();
                break;
            }
        } while (true);
    } catch (...) {
        // Restore all loop state variables to their original state
        try {
            write_cb(payload, backup);
        } catch (...) {
            /* This happens when the user changed a variable type in Python (so
             * writing back the original variable ID isn't possible). The error
             * message of the parent exception already reports this problem, so
             * ignore this duplicated error */
        }
        throw;
    }
}

static void ad_loop_evaluated(JitBackend backend, const char *name,
                              void *payload,
                              ad_loop_read read_cb, ad_loop_write write_cb,
                              ad_loop_cond cond_cb, ad_loop_body body_cb) {
    if (jit_flag(JitFlag::Symbolic))
        jit_raise("Dr.Jit is currently recording symbolic computation and "
                  "cannot execute a loop in *evaluated mode*. You will likely "
                  "want to set the Jit flag dr.JitFlag.SymbolicLoops to True. "
                  "Alternatively, you could also annotate the loop condition "
                  "with dr.hint(.., symbolic=True) if it occurs inside a "
                  "@dr.syntax-annotated function. Please review the Dr.Jit "
                  "documentation of  drjit.JitFlag.SymbolicLoops and "
                  "drjit.while_loop() for general information on symbolic and "
                  "evaluated loops, as well as their limitations.");

    dr_index64_vector indices1, indices2;
    size_t it = 0;

    jit_log(LogLevel::InfoSym,
            "ad_loop_evaluated(\"%s\"): evaluating initial loop state.", name);

    /// Before the loop starts, make the loop state opaque to ensure proper kernel caching
    read_cb(payload, indices1);
    for (uint64_t &index: indices1) {
        int unused = 0;
        uint64_t index_new = ad_var_schedule_force(index, &unused);
        ad_var_dec_ref(index);
        index = index_new;
    }
    write_cb(payload, indices1);

    // Evaluate the condition and merge it into 'active'
    uint32_t active_initial = cond_cb(payload);
    JitVar active = JitVar::steal(
        jit_var_mask_apply(active_initial, jit_var_size(active_initial)));
    active.schedule_force_();

    while (true) {
        // Evaluate the loop state
        jit_eval();

        if (!jit_var_any(active.index()))
            break;

        jit_log(LogLevel::InfoSym,
                "ad_loop_evaluated(\"%s\"): executing loop iteration %zu.", name, ++it);

        // Push the mask onto mask stack and execute the loop body
        {
            scoped_push_mask guard(backend, (uint32_t) active.index());
            body_cb(payload);
        }

        // Capture the state of all variables following execution of the loop body
        read_cb(payload, indices2);

        // Mask disabled lanes and write back
        for (size_t i = 0; i < indices1.size(); ++i) {
            uint64_t i1 = indices1[i], i2 = indices2[i];

            // Skip variables that are unchanged or the target of side effects
            if (i1 == i2 || jit_var_is_dirty((uint32_t) i2))
                continue;

            int unused = 0;
            uint64_t i3 = ad_var_select(active.index(), i2, i1);
            uint64_t i4 = ad_var_schedule_force(i3, &unused);
            indices2[i] = i4;
            ad_var_dec_ref(i2);
            ad_var_dec_ref(i3);
        }

        write_cb(payload, indices2);
        indices1.release();
        indices1.swap(indices2);

        active &= JitVar::borrow(cond_cb(payload));
        active.schedule_force_();
    }

    jit_log(LogLevel::Debug,
            "ad_loop_evaluated(\"%s\"): loop finished after %zu iterations.", name, it);
}

void ad_loop(JitBackend backend, int symbolic, const char *name, void *payload,
             ad_loop_read read_cb, ad_loop_write write_cb, ad_loop_cond cond_cb,
             ad_loop_body body_cb) {
    if (strchr(name, '\n') || strchr(name, '\r'))
        jit_raise("The loop name may not contain newline characters.\n");

    if (symbolic == -1)
        symbolic = (int) jit_flag(JitFlag::SymbolicLoops);

    if (symbolic != 0 && symbolic != 1)
        jit_raise("The 'symbolic' argument must equal 0, 1, or -1");

    if (symbolic)
        ad_loop_symbolic(backend, name, payload, read_cb, write_cb,
                         cond_cb, body_cb);
    else
        ad_loop_evaluated(backend, name, payload, read_cb, write_cb,
                          cond_cb, body_cb);
}
