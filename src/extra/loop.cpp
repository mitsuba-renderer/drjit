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
                             dr_index64_vector &backup) {
    dr_index64_vector indices1;
    dr::dr_vector<uint32_t> indices2;

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
        scoped_record record_guard(backend);

        // Rewrite the loop state variables
        JitVar loop = JitVar::steal(jit_var_loop_start(
            name, symbolic, indices2.size(), indices2.data()));

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

    return needs_ad;
}

static void ad_loop_evaluated(JitBackend backend, const char *name,
                              void *payload,
                              ad_loop_read read_cb, ad_loop_write write_cb,
                              ad_loop_cond cond_cb, ad_loop_body body_cb) {
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

/// CustomOp that hooks a recorded loop into the AD graph
struct LoopOp : public dr::detail::CustomOpBase {
public:
    LoopOp(JitBackend backend, const char *name, void *payload,
           ad_loop_read read_cb, ad_loop_write write_cb, ad_loop_cond cond_cb,
           ad_loop_body body_cb, ad_loop_delete delete_cb,
           const dr_index64_vector &state)
        : m_backend(backend), m_name(name), m_payload(payload),
          m_read_cb(read_cb), m_write_cb(write_cb), m_cond_cb(cond_cb),
          m_body_cb(body_cb), m_delete_cb(delete_cb), m_diff_count(0) {
        m_name_op = "Loop: " + m_name;

        m_inputs.reserve(state.size());
        m_rv.reserve(state.size());

        for (size_t i = 0; i < state.size(); ++i) {
            uint32_t jit_index = (uint32_t) state[i];

            VarType vt = jit_var_type(jit_index);
            Input input;
            input.index = jit_index;

            if (vt == VarType::Float16 || vt == VarType::Float32 ||
                vt == VarType::Float64) {
                uint32_t ad_index = (uint32_t) (state[i] >> 32);
                input.has_grad = true;
                input.has_grad_in = add_index(m_backend, ad_index, true);
                input.grad_offset = m_diff_count++;
            }

            jit_var_inc_ref(jit_index);
            m_inputs.push_back(input);
        }

        m_state.reserve(state.size() + m_diff_count);
        m_state2.reserve(state.size());
    }

    ~LoopOp() {
        for (const Input &i : m_inputs)
            jit_var_dec_ref(i.index);
        if (m_delete_cb)
            m_delete_cb(m_payload);
    }

    void disable_deleter() { m_delete_cb = nullptr; }

    void add_output(uint64_t index) {
        if (add_index(m_backend, index >> 32, false))
            m_rv.push_back_borrow((index >> 32) << 32);
    }

    void read(dr::dr_vector<uint64_t> &indices) {
        for (uint64_t index : m_state)
            indices.push_back(ad_var_inc_ref(index));
    }

    void write(const dr::dr_vector<uint64_t> &indices) {
        if (indices.size() != m_state.size())
            jit_fail("LoopOp::write(): internal error!");

        for (size_t i = 0; i < indices.size(); ++i) {
            uint64_t old_index = m_state[i];
            m_state[i] = ad_var_inc_ref(indices[i]);
            ad_var_dec_ref(old_index);
        }
    }

    // ---------------------------------------

    /* The forward() callbacks below implement the following logic:

         while cond(state):
             dr.enable_grad(state)
             dr.set_grad(state, grad_state)
             body(state)
             grad_state = dr.forward_from(state)
             dr.disable_grad(state)
     */

    uint32_t fwd_cond() {
        m_state2.release();
        for (size_t i = 0; i < m_inputs.size(); ++i)
            m_state2.push_back_borrow(m_state[i]);

        m_write_cb(m_payload, m_state2);
        m_state2.release();
        return m_cond_cb(m_payload);
    }

    void fwd_body() {
        // Create differentiable loop state variables
        m_state2.release();
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            uint64_t index;
            if (m_inputs[i].has_grad)
                index = ad_var_new((uint32_t) m_state[i]);
            else
                index = ad_var_inc_ref(m_state[i]);
            m_state2.push_back_steal(index);
        }

        // Run the loop body
        m_write_cb(m_payload, m_state2);
        m_body_cb(m_payload);

        // AD forward propagation pass
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.has_grad)
                continue;
            ad_accum_grad(m_state2[i],
                          (uint32_t) m_state[m_inputs.size() + in.grad_offset]);
            ad_enqueue(dr::ADMode::Forward, m_state2[i]);
        }
        m_state2.release();
        ad_traverse(dr::ADMode::Forward, (uint32_t) dr::ADFlag::ClearNone);

        // Read the loop output + derivatives copy to loop state vars
        m_read_cb(m_payload, m_state2);
        m_state.release();
        for (size_t i = 0; i < m_inputs.size(); ++i)
            m_state.push_back_borrow((uint32_t) m_state2[i]);
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            const Input &in = m_inputs[i];
            if (!in.has_grad)
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
        size_t ctr =0;
        for (const Input &in : m_inputs) {
            if (!in.has_grad)
                continue;

            uint32_t grad;
            if (in.has_grad_in)
                grad = ad_grad(combine(m_input_indices[ctr++]));
            else
                grad = jit_var_literal(m_backend, jit_var_type(in.index), &zero);

            m_state.push_back_steal(grad);
        }

        if (ctr != m_input_indices.size() ||
            m_state.size() - m_inputs.size() != m_output_indices.size())
            jit_fail("LoopOp::forward(): internal error!");

        ad_loop(
            m_backend, 1, fwd_name.c_str(), this,
            [](void *p, dr::dr_vector<uint64_t> &i) { ((LoopOp *) p)->read(i); },
            [](void *p, const dr::dr_vector<uint64_t> &i) { ((LoopOp *) p)->write(i); },
            [](void *p) { return ((LoopOp *) p)->fwd_cond(); },
            [](void *p) { return ((LoopOp *) p)->fwd_body(); }, nullptr, false);

        for (size_t i = 0; i < m_output_indices.size(); ++i)
            ad_accum_grad(combine(m_output_indices[i]),
                          (uint32_t) m_state[m_inputs.size() + i]);

        m_state.release();
    }

    uint64_t combine(uint32_t ad_index, uint32_t jit_index = 0) {
        return (((uint64_t) ad_index) << 32) + jit_index;
    }

    const char *name() const override { return m_name_op.c_str(); }

private:
    struct Input {
        uint32_t index;
        bool has_grad = false;
        bool has_grad_in = false;
        uint32_t grad_offset = 0;
    };
    dr::dr_vector<Input> m_inputs;

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
    dr_index64_vector m_state;
    /// Scratch array to call nested loop body/condition
    dr_index64_vector m_state2;
    dr_index64_vector m_rv;
    size_t m_diff_count;
};

bool ad_loop(JitBackend backend, int symbolic, const char *name, void *payload,
             ad_loop_read read_cb, ad_loop_write write_cb, ad_loop_cond cond_cb,
             ad_loop_body body_cb, ad_loop_delete delete_cb, bool ad) {
    try {
        if (name == nullptr)
            name = "unnamed";

        if (strchr(name, '\n') || strchr(name, '\r'))
            jit_raise("'name' may not contain newline characters.");

        if (symbolic == -1)
            symbolic = (int) jit_flag(JitFlag::SymbolicLoops);

        if (symbolic != 0 && symbolic != 1)
            jit_raise("'symbolic' must equal 0, 1, or -1.");


        if (symbolic) {
            dr_index64_vector indices_in;
            read_cb(payload, indices_in);

            bool needs_ad;
            {
                scoped_isolation_boundary guard;
                needs_ad =
                    ad_loop_symbolic(backend, name, payload, read_cb, write_cb,
                                     cond_cb, body_cb, indices_in);
                guard.defuse();
            }

            if (needs_ad && ad) {
                dr_index64_vector indices_out;
                read_cb(payload, indices_out);

                nanobind::ref<LoopOp> op =
                    new LoopOp(backend, name, payload, read_cb, write_cb,
                               cond_cb, body_cb, delete_cb, indices_in);

                for (size_t i = 0; i < indices_out.size(); ++i) {
                    VarType vt = jit_var_type(indices_out[i]);
                    if (vt != VarType::Float16 && vt != VarType::Float32 &&
                        vt != VarType::Float64)
                        continue;

                    uint64_t index = ad_var_new((uint32_t) indices_out[i]);
                    jit_var_dec_ref((uint32_t) indices_out[i]);
                    indices_out[i] = index;
                    op->add_output(index);
                }

                if (ad_custom_op(op.get())) {
                    write_cb(payload, indices_out);
                    // LoopOp will eventually call delete_cb()
                    return false;
                }

                // CustomOp was not needed, detach output again..
                op->disable_deleter();
            }
        } else {
            scoped_isolation_boundary guard;
            ad_loop_evaluated(backend, name, payload, read_cb, write_cb,
                              cond_cb, body_cb);
            guard.defuse();
        }

        return true; // Caller should directly call delete()
    } catch (...) {
        if (delete_cb)
            delete_cb(payload);
        throw;
    }
}
