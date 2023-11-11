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
#include <algorithm>
#include <string>

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

static void ad_loop_evaluated(JitBackend backend, const char *name,
                              void *payload,
                              ad_loop_read read_cb, ad_loop_write write_cb,
                              ad_loop_cond cond_cb, ad_loop_body body_cb) {
    if (jit_flag(JitFlag::Symbolic))
        jit_raise(
            "Dr.Jit is currently recording symbolic computation and cannot execute a\n"
            "loop in *evaluated mode*. You will likely want to set the Jit flag\n"
            "dr.JitFlag.SymbolicLoops to True. Alternatively, you could also annotate\n"
            "the loop condition with dr.hint(.., symbolic=True) if it occurs inside a\n"
            "@dr.syntax-annotated function. Please review the Dr.Jit documentation of\n"
            "drjit.JitFlag.SymbolicLoops and drjit.while_loop() for general information\n"
            "on symbolic and evaluated loops, as well as their limitations.");

    dr_index64_vector indices1, indices2;
    size_t it = 0;

    jit_log(LogLevel::Debug,
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
    JitVar active = JitVar::steal(jit_var_mask_apply(active_initial, jit_var_size(active_initial)));
    active.schedule_force_();

    while (true) {
        // Evaluate the loop state
        jit_eval();

        if (!jit_var_any(active.index()))
            break;

        jit_log(LogLevel::Debug,
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

        active = JitVar::borrow(cond_cb(payload));
        active.schedule_force_();
    }

    jit_log(LogLevel::Debug,
            "ad_loop_evaluated(\"%s\"): loop finished after %zu iterations.", name, it);
}

void ad_loop(JitBackend backend, int symbolic, const char *name, void *payload,
             ad_loop_read read_cb, ad_loop_write write_cb, ad_loop_cond cond_cb,
             ad_loop_body body_cb) {

    if (symbolic == -1)
        symbolic = (int) jit_flag(JitFlag::SymbolicLoops);

    if (symbolic != 0 && symbolic != 1)
        jit_raise("Invalid value of the 'symbolic' argument (must be 0, 1, or -1)");

    ad_loop_evaluated(backend, name, payload, read_cb, write_cb,
                      cond_cb, body_cb);
}

#if 0

/// RAII helper to temporarily record symbolic computation
struct scoped_record {
    scoped_record(JitBackend backend) : backend(backend) {
        checkpoint = jit_record_begin(backend, nullptr);
    }

    void reset() {
        jit_record_end(backend, checkpoint, true);
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
            if (s3 != size && size != 1 && s3 != 1)
                nb::raise("The body of this loop operates on arrays of "
                          "size %zu. Loop state variable '%s' has an "
                          "incompatible size %zu.",
                          size, key.c_str(), s3);

nb::tuple while_loop_symbolic(JitBackend backend, nb::tuple state,
                              nb::handle cond, nb::handle body,
                              const std::vector<std::string> &state_labels,
                              const std::string &name) {
    PySnapshot s1 = capture_state(state, state_labels);
    std::vector<uint32_t> indices;

    using JitVar = drjit::JitArray<JitBackend::None, void>;

    try {
        indices.reserve(s1.size());
        for (auto &[k, v] : s1) {
            if (v.index) {
                jit_var_inc_ref((uint32_t) v.index);
                indices.push_back((uint32_t) v.index);
            }
        }

        scoped_record record_guard(backend);

        JitVar loop =
            JitVar::steal(jit_var_loop_start(name.c_str(), indices.size(), indices.data()));

        // Rewrite the loop state variables
        size_t ctr = 0;
        for (auto &[k, v] : s1) {
            if (v.index)
                steal_and_replace(v.object, indices[ctr++]);
        }

        nb::object active = apply_default_mask(tuple_call(cond, state));
        uint32_t active_index = extract_index(active);

        // Evaluate the loop condition
        JitVar loop_cond = JitVar::steal(
            jit_var_loop_cond(loop.index(), active_index));

        PySnapshot s2;
        do {
            // Evolve the loop state
            {
                if (backend == JitBackend::CUDA)
                    active_index = jit_var_bool(backend, true);
                scoped_set_mask m(backend, active_index);

                if (backend == JitBackend::CUDA)
                    jit_var_dec_ref(active_index);

                state = check_state("body", tuple_call(body, state), state);
            }
            s2 = capture_state(state, state_labels);

            // Ensure that modified loop state remains compatible
            check_sizes(active_index, s1, s2);

            // Re-capture the indices
            indices.clear();
            visit_pairs(
                s1, s2,
                [&](const std::string &, const PyVar &, const PyVar &v2) {
                    indices.push_back(v2.index);
                }
            );

            // Construct the loop object
            if (jit_var_loop_end(loop.index(), loop_cond.index(),
                                 indices.data(), record_guard.checkpoint)) {
                record_guard.disarm();
            } else {
                record_guard.reset();

                // Re-run the loop recording process once more
                ctr = 0;
                for (auto &[k, v] : s2) {
                    if (v.index) {
                        jit_var_inc_ref((uint32_t) indices[ctr]);
                        steal_and_replace(v.object, indices[ctr]);
                        ctr++;
                    }
                }
                s2.clear();
                continue;
            }

            break;
        } while (true);

        // Rewrite the loop state variables
        ctr = 0;
        for (auto &[k, v] : s2) {
            if (v.index)
                steal_and_replace(v.object, indices[ctr++]);
        }

        for (auto &[k, v] : s1)
            jit_var_dec_ref((uint32_t) v.index);

        return state;
    } catch (...) {
        // Restore all loop state variables to their original state
        for (auto &[k, v] : s1) {
            if (!v.index)
                continue;
            steal_and_replace(v.object, v.index);
        }
        throw;
    }
}

#endif

