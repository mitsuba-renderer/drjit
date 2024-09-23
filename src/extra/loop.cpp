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
#include <string>

namespace dr = drjit;

using JitVar = GenericArray<void>;

static bool ad_loop_symbolic(JitBackend backend, const char *name,
                             void *payload,
                             ad_loop_read read_cb, ad_loop_write write_cb,
                             ad_loop_cond cond_cb, ad_loop_body body_cb,
                             index64_vector &backup,
                             dr::vector<uint32_t> &implicit_in,
                             dr::vector<uint32_t> &implicit_out) {
    index64_vector indices1;
    dr::vector<uint32_t> indices2;

    // Read the loop state variables
    read_cb(payload, indices1);

    indices2.reserve(indices1.size());
    bool needs_ad = false;
    for (uint64_t i : indices1) {
        indices2.push_back((uint32_t) i);
        needs_ad |= (i >> 32) != 0;
    }
    bool symbolic = jit_flag(JitFlag::SymbolicScope);

    try {
        /* Postponed operations captured by the isolation scope should only
         * be executed once we've exited the symbolic scope. We therefore
         * need to declare the AD isolation guard before the recording guard. */
        scoped_isolation_boundary isolation_guard(1);
        scoped_record record_guard(backend);

        // Rewrite the loop state variables
        JitVar loop = JitVar::steal(jit_var_loop_start(
            name, symbolic, indices2.size(), indices2.data()));

        // Propagate these changes
        indices1.release();
        for (uint32_t i : indices2)
            indices1.push_back_steal(i);
        write_cb(payload, indices1, false);
        indices1.release();
        indices2.clear();

        // Read back the indices to update the loop state variables
        // This is necessary to catch cases where a variable is added to the
        // loop state twice.
        read_cb(payload, indices1);
        for (uint32_t i : indices1)
            indices2.push_back((uint32_t) i);
        jit_var_loop_update_inner_in(loop.index(), indices2.data());
        indices1.release();
        indices2.clear();

        do {
            // Evaluate the loop condition
            uint32_t active_initial = cond_cb(payload);

            // Potentially optimize the loop away
            if (jit_var_is_zero_literal(active_initial) &&
                jit_flag(JitFlag::OptimizeLoops)) {
                jit_log(LogLevel::InfoSym,
                        "ad_loop_symbolic(\"%s\"): optimized away (loop "
                        "condition is 'false').", name);
                write_cb(payload, backup, false);
                break;
            }

            JitVar active = JitVar::steal(jit_var_mask_apply(
                active_initial, (uint32_t) jit_var_size(active_initial)));

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
            write_cb(payload, indices1, rv == false);
            indices1.release();
            indices2.clear();

            // Construct the loop object
            if (rv) {
                // All done
                record_guard.disarm();
                isolation_guard.disarm();
                ad_copy_implicit_deps(implicit_in, true);
                ad_copy_implicit_deps(implicit_out, false);
                break;
            }

            // Re-record, discard postponed nodes on the AD graph
            isolation_guard.reset();
        } while (true);
    } catch (...) {
        // Restore all loop state variables to their original state
        try {
            write_cb(payload, backup, true);
        } catch (...) {
            /* This happens when the user changed a variable type in Python (so
             * writing back the original variable ID isn't possible). The error
             * message of the parent exception already reports this problem, so
             * ignore this duplicated error */
        }
        throw;
    }

    return needs_ad;
}

// Simple wavefront-style evaluated loop that masks inactive entries
static size_t ad_loop_evaluated_mask(JitBackend backend, const char *name,
                                     void *payload, ad_loop_read read_cb,
                                     ad_loop_write write_cb,
                                     ad_loop_cond cond_cb, ad_loop_body body_cb,
                                     index64_vector indices1,
                                     JitVar active) {
    index64_vector indices2;
    JitVar active_it;
    size_t it = 0;

    while (true) {
        // Evaluate the loop state
        jit_eval();

        if (!jit_var_any(active.index()))
            break;

        jit_log(LogLevel::InfoSym,
                "ad_loop_evaluated(\"%s\"): executing loop iteration %zu.",
                name, ++it);

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

            indices2[i] = ad_var_select(active.index(), i2, i1);
            ad_var_dec_ref(i2);
        }

        for (size_t i = 0; i < indices2.size(); ++i) {
            uint64_t i1 = indices2[i];
            uint64_t i2 = ad_var_copy(i1);
            ad_var_dec_ref(i1);
            ad_mark_loop_boundary(i2);
            int unused = 0;
            indices2[i] = ad_var_schedule_force(i2, &unused);
            ad_var_dec_ref(i2);
        }

        write_cb(payload, indices2, false);
        indices1.release();
        indices1.swap(indices2);

        active_it = JitVar::borrow(cond_cb(payload));
        active_it.schedule_();
        active &= active_it;
        active.schedule_force_();
    }

    return it;
}

// Simple wavefront-style evaluated loop that progressively reduces the size
// of the loop state to ignore inactive entries
static size_t
ad_loop_evaluated_compress(JitBackend backend, const char *name, void *payload,
                           ad_loop_read read_cb, ad_loop_write write_cb,
                           ad_loop_cond cond_cb, ad_loop_body body_cb,
                           index64_vector indices,
                           JitVar active) {
    uint32_t size = (uint32_t) active.size(), it = 0;

    JitVar true_mask = JitVar::steal(jit_var_bool(backend, true)),
           zero = JitVar::steal(jit_var_u32(backend, 0)),
           idx = JitVar::steal(jit_var_counter(backend, size));

    dr::schedule(idx);

    index64_vector out_indices;
    dr::vector<bool> skip;

    skip.reserve(indices.size());
    out_indices.reserve(indices.size());

    // Filter out incompatible loop state variables
    for (uint64_t index: indices) {
        if (index) {
            VarInfo info = jit_set_backend((uint32_t) index);

            if (info.size == (size_t) size || info.size == 1) {
                out_indices.push_back_steal(
                    jit_var_undefined(backend, info.type, size));
                skip.push_back(false);
                continue;
            }
        }

        out_indices.push_back_borrow(index);
        skip.push_back(true);
    }

    /**
       This function implements two different strategies to progressively
       reduce the array contents.

       1. Reduce, then gather: evaluate all loop state, then launch a
          specialized reduction kernel that computes an index array. Next,
          gather the subset of remaining inputs using this index array and
          perform a single loop iteration.

          In addition to this, a separate kernel writes out the loop state of
          elements that finished in the current iteration.

          This variant launches two kernels per iteration. Compared to variant
          2 below, it prefers reads over writes and does not use atomics. For
          these reasons, it tends to run faster on the LLVM CPU backend.

       2. Reserve a spot, then scatter. This variant appends code at the
          end of each loop ieration that uses an atomic operation to reserve
          a spot in a set of loop state output arrays. It then scatters all
          updated state to the reverved position.

          This variant launches a single kernel per iteration, and it is
          fairly write and atomic-heavy. It tends to run faster on the CUDA
          backend
     */
    bool reduce_then_gather = backend == JitBackend::LLVM;

    while (true) {
        // Determine which entries aren't active, these must be written out
        JitVar not_active = JitVar::steal(jit_var_not(active.index()));

        uint32_t size_next = 0;

        if (reduce_then_gather) {
            for (uint64_t &index: indices) {
                int unused = 0;
                uint64_t index_new = ad_var_schedule_force(index, &unused);
                ad_var_dec_ref(index);
                index = index_new;
            }
            active.schedule_force_();

            // Evaluate the loop state
            jit_eval();

            // Reduce the array to the remaining active entries
            JitVar active_index =
                JitVar::steal(jit_var_compress(active.index()));
            size_next = (uint32_t) active_index.size();

            for (size_t i = 0; i < indices.size(); ++i) {
                if (skip[i])
                    continue;

                // Write entries that have become inactive to 'out_indices'
                uint64_t f_index = ad_var_scatter(
                    out_indices[i],
                    indices[i], idx.index(), not_active.index(),
                    ReduceOp::Identity,
                    ReduceMode::Permute);
                ad_var_dec_ref(out_indices[i]);
                out_indices[i] = f_index;
            }

            jit_eval();

            for (size_t i = 0; i < indices.size(); ++i) {
                // Gather remaining active entries. We always do this even when
                // the loop state was not compressed, which ensures identical code
                // generation in each iteration to benefit from kernel caching.
                uint32_t t_index = (uint32_t)
                    ad_var_gather(indices[i], active_index.index(),
                                  true_mask.index(), ReduceMode::Permute);
                ad_var_dec_ref(indices[i]);
                indices[i] = t_index;
            }

            idx = JitVar::steal((uint32_t) ad_var_gather(idx.index(), active_index.index(),
                                                         true_mask.index(), ReduceMode::Permute));
            dr::schedule(idx);
        } else {
            // Increase an atomic counter to determine the position in the output array
            uint32_t counter_tmp = jit_var_u32(backend, 0);
            JitVar slot = JitVar::steal(
                jit_var_scatter_inc(&counter_tmp, zero.index(), active.index()));
            JitVar counter = JitVar::steal(counter_tmp);

            for (size_t i = 0; i < indices.size(); ++i) {
                if (skip[i])
                    continue;

                // Write entries that have become inactive to 'out_indices'
                uint64_t f_index = ad_var_scatter(
                    out_indices[i], indices[i], idx.index(), not_active.index(),
                    ReduceOp::Identity, ReduceMode::Permute);

                ad_var_dec_ref(out_indices[i]);
                out_indices[i] = f_index;

                // Write remaining active entries into a new output buffer
                JitVar buffer = JitVar::steal(
                    jit_var_undefined(backend, jit_var_type((uint32_t) indices[i]), size));
                uint64_t t_index = ad_var_scatter(
                    buffer.index(), indices[i], slot.index(), active.index(),
                    ReduceOp::Identity, ReduceMode::Permute);
                ad_var_dec_ref(indices[i]);
                indices[i] = t_index;
            }

            JitVar buffer = JitVar::steal(jit_var_undefined(backend, VarType::UInt32, size));
            idx = JitVar::steal(jit_var_scatter(
                buffer.index(), idx.index(), slot.index(), active.index(),
                ReduceOp::Identity, ReduceMode::Permute));

            // Evaluate everything queued up to this point
            jit_eval();
            jit_var_read(counter.index(), 0, &size_next);

            if (size != size_next && size_next != 0) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (skip[i])
                        continue;
                    uint64_t new_index = ad_var_shrink(indices[i], size_next);
                    ad_var_dec_ref(indices[i]);
                    indices[i] = new_index;
                }
                idx = JitVar::steal((uint32_t) ad_var_shrink(idx.index(), size_next));
            }
        }

        active = not_active = JitVar();

        if (size_next == 0)
            break; // all done!

        if (size != size_next)
            jit_log(LogLevel::InfoSym,
                    "ad_loop_evaluated(\"%s\"): compressed loop state from %u "
                    "to %u entries.", name, size, size_next);

        size = size_next;
        write_cb(payload, indices, false);
        indices.release();

        jit_log(LogLevel::InfoSym,
                "ad_loop_evaluated(\"%s\"): executing loop iteration %zu.", name, ++it);

        // Execute the loop body
        {
            scoped_push_mask guard(backend, (uint32_t) true_mask.index());
            body_cb(payload);
        }

        active = JitVar::borrow(cond_cb(payload));
        read_cb(payload, indices);
    }

    if (it > 0)
        write_cb(payload, out_indices, false);

    return it;
}

static void ad_loop_evaluated(JitBackend backend, const char *name,
                              void *payload, ad_loop_read read_cb,
                              ad_loop_write write_cb,
                              ad_loop_cond cond_cb,
                              ad_loop_body body_cb,
                              bool compress) {
    index64_vector indices;

    jit_log(LogLevel::InfoSym,
            "ad_loop_evaluated(\"%s\"): evaluating initial loop state.", name);

    // Before the loop starts, make the loop state opaque to ensure proper kernel caching
    read_cb(payload, indices);
    for (uint64_t &index: indices) {
        int unused = 0;
        uint64_t index_new = ad_var_schedule_force(index, &unused);
        ad_var_dec_ref(index);
        index = index_new;
    }
    write_cb(payload, indices, false);

    // Evaluate the condition and merge it into 'active'
    uint32_t active_initial = cond_cb(payload);
    JitVar active = JitVar::steal(
        jit_var_mask_apply(active_initial, (uint32_t) jit_var_size(active_initial)));
    active.schedule_force_();

    size_t size = active.size();

    if (compress && size == 1) {
        jit_log(
            LogLevel::Warn,
            "ad_loop_evaluated(\"%s\"): loop state compression requires a "
            "non-scalar loop condition, switching to the default masked mode!");
        compress = false;
    }

    size_t it;
    if (compress)
        it = ad_loop_evaluated_compress(backend, name, payload, read_cb,
                                        write_cb, cond_cb, body_cb,
                                        std::move(indices), std::move(active));
    else
        it = ad_loop_evaluated_mask(backend, name, payload, read_cb, write_cb,
                                    cond_cb, body_cb, std::move(indices),
                                    std::move(active));

    jit_log(LogLevel::Debug,
            "ad_loop_evaluated(\"%s\"): loop finished after %zu iterations.", name, it);
}

/// CustomOp that hooks a recorded loop into the AD graph
struct LoopOp : public dr::detail::CustomOpBase {
public:
    LoopOp(JitBackend backend, const char *name, void *payload,
           ad_loop_read read_cb, ad_loop_write write_cb, ad_loop_cond cond_cb,
           ad_loop_body body_cb, ad_loop_delete delete_cb,
           const index64_vector &state,
           const dr::vector<uint32_t> &implicit_in,
           long long max_iterations)
        : m_backend(backend), m_name(name), m_payload(payload),
          m_read_cb(read_cb), m_write_cb(write_cb), m_cond_cb(cond_cb),
          m_body_cb(body_cb), m_delete_cb(delete_cb), m_diff_count(0),
          m_max_iterations(max_iterations), m_reset(false) {
        m_name_op = "Loop: " + m_name;

        m_inputs.reserve(state.size());

        for (size_t i = 0; i < state.size(); ++i) {
            uint32_t jit_index = (uint32_t) state[i];

            VarType vt = jit_var_type(jit_index);
            Input input{};
            input.index = jit_index;

            if (vt == VarType::Float16 || vt == VarType::Float32 ||
                vt == VarType::Float64) {
                uint32_t ad_index = (uint32_t) (state[i] >> 32);
                input.is_diff = true;
                input.has_grad_in = add_index(m_backend, ad_index, true);
                if (input.has_grad_in)
                    input.grad_in_index = m_input_indices.size() - 1;
                input.grad_in_offset = (uint32_t) m_diff_count++;
            }

            jit_var_inc_ref(jit_index);
            m_inputs.push_back(input);
        }

        m_implicit_in_offset = m_input_indices.size();
        for (uint32_t index: implicit_in)
            add_index(m_backend, index, true);

        m_state.reserve(state.size() + m_diff_count);
        m_state2.reserve(state.size());
    }

    ~LoopOp() {
        for (const Input &i : m_inputs) {
            jit_var_dec_ref(i.index);
            if (i.has_grad_out)
                ad_var_dec_ref(combine(m_output_indices[i.grad_out_offset]));
        }

        if (m_delete_cb)
            m_delete_cb(m_payload);
    }

    void disable_deleter() { m_delete_cb = nullptr; }

    void add_output(uint64_t index, size_t offset) {
        if (add_index(m_backend, index >> 32, false)) {
            Input &in = m_inputs[offset];
            in.has_grad_out = true;
            in.grad_out_offset = (uint32_t) m_output_indices.size() - 1;
            ad_var_inc_ref(combine(m_output_indices[in.grad_out_offset]));
        }
    }

    void read(dr::vector<uint64_t> &indices) {
        for (uint64_t index : m_state)
            indices.push_back(ad_var_inc_ref(index));
    }

    void write(const dr::vector<uint64_t> &indices, bool reset) {
        if (indices.size() != m_state.size())
            jit_fail("LoopOp::write(): internal error (received request to "
                     "write %zu indices, but state size is %zu)!",
                     indices.size(), m_state.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            uint64_t old_index = m_state[i];
            m_state[i] = ad_var_inc_ref(indices[i]);
            ad_var_dec_ref(old_index);
        }

        m_reset = reset;
    }

    uint64_t combine(uint32_t ad_index, uint32_t jit_index = 0) {
        return (((uint64_t) ad_index) << 32) + jit_index;
    }

    const char *name() const override { return m_name_op.c_str(); }

    // -------------------------------------------------------------------

    /* The forward() callbacks below implement the following logic:

         while cond(state):
             dr.enable_grad(state)
             dr.set_grad(state, grad_state)
             state = body(state)
             grad_state = dr.forward_from(state)
             dr.disable_grad(state)
     */

    uint32_t fwd_cond() {
        m_state2.release();
        for (size_t i = 0; i < m_inputs.size(); ++i)
            m_state2.push_back_borrow(m_state[i]);

        m_write_cb(m_payload, m_state2, m_reset);
        m_reset = false;
        m_state2.release();
        return m_cond_cb(m_payload);
    }

    void fwd_body() {
        // Create differentiable loop state variables
        m_state2.release();
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            uint64_t index;
            if (m_inputs[i].is_diff)
                index = ad_var_new((uint32_t) m_state[i]);
            else
                index = ad_var_inc_ref(m_state[i]);
            m_state2.push_back_steal(index);
        }

        // Run the loop body
        m_write_cb(m_payload, m_state2, true);

        {
            // Begin a recording session and abort it by not
            // calling .disarm(). This suppresses side effects.
            scoped_record record_guard(m_backend);

            m_body_cb(m_payload);
        }

        // AD forward propagation pass
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.is_diff)
                continue;
            ad_accum_grad(m_state2[i],
                          (uint32_t) m_state[m_inputs.size() + in.grad_in_offset]);
            ad_enqueue(dr::ADMode::Forward, m_state2[i]);
        }

        for (size_t i = m_implicit_in_offset; i < m_input_indices.size(); ++i)
            ad_enqueue(dr::ADMode::Forward, uint64_t(m_input_indices[i]) << 32);

        m_state2.release();
        ad_traverse(dr::ADMode::Forward, (uint32_t) dr::ADFlag::ClearNone);

        // Read the loop output + derivatives copy to loop state vars
        m_read_cb(m_payload, m_state2);
        m_state.release();
        for (size_t i = 0; i < m_inputs.size(); ++i)
            m_state.push_back_borrow((uint32_t) m_state2[i]);
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.is_diff)
                continue;
            m_state.push_back_steal(ad_grad(m_state2[i]));
        }

        m_state2.release();
    }

    void forward() override {
        std::string fwd_name = m_name + " [ad, fwd]";

        m_state.release();
        for (const Input &i : m_inputs)
            m_state.push_back_borrow(i.index);

        uint64_t zero = 0;
        size_t ctr = 0;
        for (const Input &in : m_inputs) {
            if (!in.is_diff)
                continue;

            uint32_t grad;
            if (in.has_grad_in)
                grad = ad_grad(combine(m_input_indices[ctr++]));
            else
                grad = jit_var_literal(m_backend, jit_var_type(in.index), &zero);

            m_state.push_back_steal(grad);
        }

        ad_loop(
            m_backend, 1, 0, 0, fwd_name.c_str(), this,
            [](void *p, dr::vector<uint64_t> &i) { ((LoopOp *) p)->read(i); },
            [](void *p, const dr::vector<uint64_t> &i, bool reset) { ((LoopOp *) p)->write(i, reset); },
            [](void *p) { return ((LoopOp *) p)->fwd_cond(); },
            [](void *p) { return ((LoopOp *) p)->fwd_body(); }, nullptr, false);

        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.has_grad_out)
                continue;

            ad_accum_grad(combine(m_output_indices[in.grad_out_offset]),
                          (uint32_t) m_state[m_inputs.size() + in.grad_in_offset]);
        }

        m_state.release();
    }

    // -------------------------------------------------------------------

    void backward() override {
        if (m_max_iterations == -1) {
            backward_simple();
        } else {
            jit_raise("CustomOp::backward(): the reverse-mode derivative of a "
                      "complex loop (with max_iterations != -1) is not yet "
                      "implemented!");
        }
    }

    // -------------------------------------------------------------------

    /* The backward_simple() callbacks below implement the following logic:

         state_o = <subset of 'state' for which derivative
                    tracking was enabled after, but not
                    before the loop>

         state_i = <subset of 'state' for which derivative
                    tracking was enabled before and after
                    the loop>

         grad_state_o = <zero-initialized gradients>

         while cond(state):
             dr.enable_grad(state_i)
             state = body(state)
             dr.set_grad(state_o, grad_state_o)
             grad_state_i += dr.backward_to(state_i)
             dr.disable_grad(state)
     */

    void bwd_body_simple() {
        // Create differentiable loop state variables
        m_state2.release();

        index32_vector tmp;
        size_t offset = m_inputs.size();
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];

            uint64_t index;
            if (in.has_grad_out && in.has_grad_in) {
                index = ad_var_new((uint32_t) m_state[i]);
                tmp.push_back_borrow((uint32_t) m_state[offset]);
            } else {
                index = ad_var_inc_ref(m_state[i]);
            }
            m_state2.push_back_steal(index);

            if (in.has_grad_out)
                offset++;
        }

        // Run the loop body
        m_write_cb(m_payload, m_state2, true);

        {
            // Begin a recording session and abort it by not
            // calling .disarm(). This suppresses side effects.
            scoped_record record_guard(m_backend);

            m_body_cb(m_payload);
        }
        m_state2.release();
        m_read_cb(m_payload, m_state2);

        // AD backward propagation pass
        offset = m_inputs.size();
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.has_grad_out)
                continue;

            ad_accum_grad(m_state2[i], (uint32_t) m_state[offset]);

            if (!in.has_grad_in)
                ad_enqueue(dr::ADMode::Backward, m_state2[i]);

            offset++;
        }

        ad_traverse(dr::ADMode::Backward, (uint32_t) dr::ADFlag::ClearNone);

        // Read the loop output + derivatives copy to loop state vars
        m_state.release();
        for (size_t i = 0; i < m_inputs.size(); ++i)
            m_state.push_back_borrow((uint32_t) m_state2[i]);

        offset = m_inputs.size();
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];

            if (!in.has_grad_out)
                continue;
            uint32_t grad = ad_grad(m_state2[i]);
            m_state.push_back_steal(grad);
            offset++;
        }

        m_state2.release();
    }

    void backward_simple() {
        std::string fwd_name = m_name + " [ad, bwd, simple]";

        m_state.release();
        for (const Input &i : m_inputs)
            m_state.push_back_borrow(i.index);

        for (const Input &in : m_inputs) {
            uint32_t grad;
            if (!in.has_grad_out)
                continue;

            if (in.has_grad_in) {
                uint64_t zero = 0;
                grad = jit_var_literal(m_backend, jit_var_type(in.index), &zero, jit_var_size(in.index));
            } else {
                grad = ad_grad(combine(m_output_indices[in.grad_out_offset]));
            }

            m_state.push_back_steal(grad);
        }

        ad_loop(
            m_backend, 1, 0, 0, fwd_name.c_str(), this,
            [](void *p, dr::vector<uint64_t> &i) { ((LoopOp *) p)->read(i); },
            [](void *p, const dr::vector<uint64_t> &i, bool reset) { ((LoopOp *) p)->write(i, reset); },
            [](void *p) { return ((LoopOp *) p)->fwd_cond(); },
            [](void *p) { return ((LoopOp *) p)->bwd_body_simple(); }, nullptr, false);


        size_t offset = m_inputs.size();
        for (const Input &in : m_inputs) {
            if (!in.has_grad_out)
                continue;

            if (in.has_grad_in) {
                ad_accum_grad(combine(m_input_indices[in.grad_in_index]),
                              (uint32_t) m_state[offset]);
            }

            offset++;
        }

        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.has_grad_out)
                continue;

            ad_accum_grad(combine(m_output_indices[in.grad_out_offset]),
                          (uint32_t) m_state[m_inputs.size() + in.grad_in_offset]);
        }


        m_state.release();
    }

private:
    struct Input {
        uint32_t index;

        /// Is this variable differentiable in principle?
        bool is_diff;

        /// Does the loop op. receive gradients for this variable?
        bool has_grad_in;

        /// Does the loop op. produce gradients for this variable?
        bool has_grad_out;

        uint32_t grad_in_index; // position in m_input_indices
        uint32_t grad_in_offset; // offset in m_state
        uint32_t grad_out_offset; // position in m_output_indices
    };
    dr::vector<Input> m_inputs;

    JitBackend m_backend;
    std::string m_name;
    std::string m_name_op;
    void *m_payload;
    ad_loop_read m_read_cb;
    ad_loop_write m_write_cb;
    ad_loop_cond m_cond_cb;
    ad_loop_body m_body_cb;
    ad_loop_delete m_delete_cb;
    /// Loop state of nested loop
    index64_vector m_state;
    /// Scratch array to call nested loop body/condition
    index64_vector m_state2;
    /// Total number of differentiable state variables
    size_t m_diff_count;
    // Offset of implicit indices in m_input_indices
    size_t m_implicit_in_offset;
    long long m_max_iterations;
    bool m_reset;
};

bool ad_loop(JitBackend backend, int symbolic, int compress,
             long long max_iterations, const char *name, void *payload,
             ad_loop_read read_cb, ad_loop_write write_cb, ad_loop_cond cond_cb,
             ad_loop_body body_cb, ad_loop_delete delete_cb, bool ad) {
    if (name == nullptr)
        name = "unnamed";

    if (strchr(name, '\n') || strchr(name, '\r'))
        jit_raise("'name' may not contain newline characters.");

    uint32_t flags = jit_flags();

    if (symbolic == -1) {
        if (flags & (uint32_t) JitFlag::SymbolicScope) {
            // We're inside some other symbolic operation, cannot use evaluated mode
            if (!jit_flag(JitFlag::SymbolicLoops))
                jit_log(LogLevel::Warn,
                        "ad_loop(\"%s\"): forcing conditional statement to "
                        "symbolic mode since the operation is nested within "
                        "another symbolic operation).", name);
            symbolic = 1;
        } else {
            symbolic = bool(flags & (uint32_t) JitFlag::SymbolicLoops);
        }
    }

    if (compress == -1)
        compress = (int) bool(flags & (uint32_t) JitFlag::CompressLoops);

    if (symbolic != 0 && symbolic != 1)
        jit_raise("'symbolic' must equal 0, 1, or -1.");

    if (compress != 0 && compress != 1)
        jit_raise("'compress' must equal 0, 1, or -1.");

    if (max_iterations < -1)
        jit_raise("'max_iterations' must be >= -1.");

    if (symbolic) {
        index64_vector indices_in;
        read_cb(payload, indices_in);
        dr::detail::ad_index32_vector implicit_in, implicit_out;

        bool needs_ad;
        {
            needs_ad = ad_loop_symbolic(backend, name, payload, read_cb,
                                        write_cb, cond_cb, body_cb, indices_in,
                                        implicit_in, implicit_out);
        }

        if (needs_ad && ad) {
            index64_vector indices_out;
            read_cb(payload, indices_out);

            nanobind::ref<LoopOp> op =
                new LoopOp(backend, name, payload, read_cb, write_cb,
                           cond_cb, body_cb, delete_cb, indices_in,
                           implicit_in, max_iterations);

            for (size_t i = 0; i < indices_out.size(); ++i) {
                VarType vt = jit_var_type((uint32_t) indices_out[i]);
                if (vt != VarType::Float16 && vt != VarType::Float32 &&
                    vt != VarType::Float64)
                    continue;

                if (max_iterations == 0 &&
                    (uint32_t) indices_in[i] == (uint32_t) indices_out[i]) {
                    // Keep unchanged variables out of the AD system
                    if (indices_in[i] != indices_out[i]) {
                        ad_var_inc_ref(indices_in[i]);
                        jit_var_dec_ref((uint32_t) indices_out[i]);
                        indices_out[i] = indices_in[i];
                    }
                } else {
                    uint64_t index = ad_var_new((uint32_t) indices_out[i]);
                    jit_var_dec_ref((uint32_t) indices_out[i]);
                    indices_out[i] = index;
                    op->add_output(index, i);
                }
            }

            if (ad_custom_op(op.get())) {
                write_cb(payload, indices_out, false);
                // LoopOp will eventually call delete_cb()
                return false;
            }

            // CustomOp was not needed, detach output again..
            op->disable_deleter();
            write_cb(payload, indices_out, false);
        }
    } else {
        if (jit_flag(JitFlag::SymbolicScope))
            jit_raise("Dr.Jit is currently recording symbolic computation and "
                      "cannot execute a loop in *evaluated mode*. You will likely "
                      "want to set the Jit flag dr.JitFlag.SymbolicLoops to True. "
                      "Alternatively, you could also annotate the loop condition "
                      "with dr.hint(.., symbolic=True) if it occurs inside a "
                      "@dr.syntax-annotated function. Please review the Dr.Jit "
                      "documentation of drjit.JitFlag.SymbolicLoops and "
                      "drjit.while_loop() for general information on symbolic and "
                      "evaluated loops, as well as their limitations.");

        scoped_isolation_boundary guard;
        ad_loop_evaluated(backend, name, payload, read_cb, write_cb,
                          cond_cb, body_cb, compress);
        guard.disarm();
    }

    return true; // Caller should directly call delete()
}
