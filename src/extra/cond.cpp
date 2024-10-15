/*
    extra/cond.cpp -- Logic to implement conditional statements through one
    common interface with support for symbolic and evaluated execution styles
    along with automatic differentiation.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"
#include "drjit-core/jit.h"
#include <drjit/custom.h>
#include <drjit-core/hash.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

namespace dr = drjit;

using JitVar = GenericArray<void>;

static void ad_cond_evaluated(JitBackend backend, const char *label,
                              void *payload, uint32_t cond_t, uint32_t cond_f,
                              const dr::vector<uint64_t> &args,
                              dr::vector<uint64_t> &rv, ad_cond_body body_cb) {
    jit_log(LogLevel::InfoSym,
            "ad_cond_evaluated(\"%s\"): executing conditional expression.",
            label);

    tsl::robin_map<uint32_t, size_t> arg_map;
    index64_vector args_t, args_f;
    size_t cond_size = jit_var_size((uint32_t) cond_t);

    // For differentiable inputs, create masked AD variables
    for (size_t i = 0; i < args.size(); ++i) {
        uint64_t index = args[i];
        uint32_t index_lo = (uint32_t) index;
        size_t size = jit_var_size((uint32_t) index);
        bool is_diff = index != index_lo;

        if (is_diff && (size == cond_size || size == 1 || cond_size == 1)) {
            // Force the creation of AD variables even when in an AD-suspended
            // scope. This is so that we can preserve the AD status of variables
            // that aren't changed by the loop
            scoped_force_grad_guard guard;

            uint64_t idx_t = ad_var_select(cond_t, index, index_lo),
                     idx_f = ad_var_select(cond_f, index, index_lo);
            uint32_t ad_idx_t = (uint32_t) (idx_t >> 32),
                     ad_idx_f = (uint32_t) (idx_f >> 32);

            if (ad_idx_t)
                arg_map[ad_idx_t] = index;
            if (ad_idx_f)
                arg_map[ad_idx_f] = index;

            args_t.push_back_steal(idx_t);
            args_f.push_back_steal(idx_f);
        } else {
            args_t.push_back_borrow(index);
            args_f.push_back_borrow(index);
        }
    }

    index64_vector rv_t, rv_f;
    rv_t.reserve(args.size());

    // Execute 'true_fn'
    {
        scoped_push_mask guard(backend, cond_t);
        body_cb(payload, true, args_t, rv_t);
    }

    rv_f.reserve(rv_t.size());
    rv.reserve(rv_t.size());

    // Execute 'false_fn'
    {
        scoped_push_mask guard(backend, cond_f);
        body_cb(payload, false, args_f, rv_f);
    }

    if (rv_t.size() != rv_f.size())
        jit_raise("ad_cond_evaluated(): inconsistent number of outputs!");

    // Combine the results
    for (size_t i = 0; i < rv_t.size(); ++i) {
        uint64_t idx_t = rv_t[i], idx_f = rv_f[i];

        uint32_t ad_idx_t = uint32_t(idx_t >> 32),
                 ad_idx_f = uint32_t(idx_f >> 32);

        // Unchanged differentiable outputs can be piped through directly
        // without being outputs of the CustomOp
        if (ad_idx_t && ad_idx_f) {
            auto prev_t = arg_map.find(ad_idx_t),
                 prev_f = arg_map.find(ad_idx_f);

            if (prev_t != arg_map.end() &&
                prev_f != arg_map.end() &&
                prev_t->second == prev_f->second) {
                uint64_t ref = ad_var_inc_ref(prev_t->second);
                rv.push_back(ref);
                continue;
            }
        }

        uint64_t i3;
        if (idx_t == idx_f || jit_var_is_dirty((uint32_t) idx_f))
            i3 = ad_var_inc_ref(idx_f);
        else
            i3 = ad_var_select(cond_t, idx_t, idx_f);

        rv.push_back(i3);
    }
}

/**
 * Record a symbolic conditional operation. The arguments have the following
 * roles
 *
 * - ``backend``: Computational backend used to compile the operation.
 * - ``name``: A descriptive string.
 * - ``payload``: An opaque payload forwarded to the callbacks.
 * - ``cond_t``: Should the ``true`` case be executed? (Mask ID)
 * - ``cond_f``: Should the ``false`` case be executed? (Mask ID)
 * - ``args``: Function argument indices.
 * - ``rv``: Result value indices (output argument)
 * - ``body_cb``: Callback for the function body.
 * - ``input_offsets``: Offset of differentiable parameters in argument list.
 * - ``output_offsets``: Offset of differentiable parameters in output list.
 * - ``implicit_in``: list of implicit differentiable input dependecies
 * - ``implicit_out``: list of implicit differentiable output dependecies
 */
static void ad_cond_symbolic(JitBackend backend, const char *label,
                             void *payload, uint32_t cond_t, uint32_t cond_f,
                             const dr::vector<uint64_t> &args,
                             dr::vector<uint64_t> &rv, ad_cond_body body_cb,
                             dr::vector<size_t> &input_offsets,
                             dr::vector<size_t> &output_offsets,
                             dr::vector<uint32_t> &implicit_in,
                             dr::vector<uint32_t> &implicit_out,
                             bool ad) {
    bool symbolic = jit_flag(JitFlag::SymbolicScope);

    /* Postponed operations captured by the isolation scope should only
     * be executed once we've exited the symbolic scope. We therefore
     * need to declare the AD isolation guard before the recording guard. */
    scoped_isolation_guard isolation_guard(1);
    scoped_record record_guard(backend);

    index64_vector args_t, args_f, rv_t, rv_f, cleanup;
    dr::vector<uint32_t> tmp;

    // For differentiable inputs, create new disconnected AD variables
    for (size_t i = 0; i < args.size(); ++i) {
        uint64_t index = args[i];
        uint32_t index_lo = (uint32_t) index;
        if (ad && (args[i] >> 32)) {
            // Force the creation of AD variables even when in an AD-suspended
            // scope. This is so that we can preserve the AD status of variables
            // that aren't changed by the loop
            scoped_force_grad_guard guard;

            uint64_t idx_t = ad_var_new(index_lo),
                     idx_f = ad_var_new(index_lo);

            uint32_t grad = ad_grad(index, true);
            if (grad) {
                ad_accum_grad(idx_t, grad);
                ad_accum_grad(idx_f, grad);
                jit_var_dec_ref(grad);
            }

            ad_var_map_put(index, idx_t);
            ad_var_map_put(index, idx_f);

            if ((idx_t >> 32) || (idx_f >> 32))
                input_offsets.push_back(i);

            args_t.push_back_steal(idx_t);
            args_f.push_back_steal(idx_f);
        } else {
            args_t.push_back_borrow(index_lo);
            args_f.push_back_borrow(index_lo);
        }
    }

    JitVar true_mask = JitVar::steal(jit_var_bool(backend, true));
    jit_new_scope(backend);

    JitVar handle =
        JitVar::steal(jit_var_cond_start(label, symbolic, cond_t, cond_f));

    // Execute 'true_fn'
    {
        uint32_t mask =
            backend == JitBackend::CUDA ? true_mask.index() : cond_t;
        scoped_push_mask guard(backend, mask);
        body_cb(payload, true, args_t, rv_t);
    }

    rv_f.reserve(rv_t.size());
    rv.reserve(rv_t.size());
    tmp.reserve(rv_t.size());

    // Collect return values
    for (size_t i = 0; i < rv_t.size(); ++i)
        tmp.push_back((uint32_t) rv_t[i]);

    // Collect gradients of differentiable inputs
    if (ad) {
        for (uint64_t index : args_t) {
            if (index >> 32) {
                uint32_t grad_idx = ad_grad(index);
                tmp.push_back(grad_idx);
                cleanup.push_back_steal(grad_idx);
            }
        }
    }

    JitVar handle_2 = JitVar::steal(jit_var_cond_append(
        handle.index(), tmp.data(), tmp.size()));

    // Execute 'false_fn'
    {
        uint32_t mask =
            backend == JitBackend::CUDA ? true_mask.index() : cond_f;
        scoped_push_mask guard(backend, mask);
        body_cb(payload, false, args_f, rv_f);
    }

    if (rv_t.size() != rv_f.size())
        jit_raise("ad_cond_symbolic(): inconsistent number of outputs!");

    // Collect return values
    tmp.clear();
    for (size_t i = 0; i < rv_f.size(); ++i)
        tmp.push_back((uint32_t) rv_f[i]);

    // Collect gradients of differentiable inputs
    if (ad) {
        for (uint64_t index : args_f) {
            if (index >> 32) {
                uint32_t grad_idx = ad_grad(index);
                tmp.push_back(grad_idx);
                cleanup.push_back_steal(grad_idx);
            }
        }
    }

    JitVar handle_3 = JitVar::steal(jit_var_cond_append(
        handle.index(), tmp.data(), tmp.size()));

    record_guard.disarm();

    jit_var_cond_end(handle.index(), tmp.data());

    for (size_t i = 0; i < rv_f.size(); ++i) {
        uint64_t idx_t = rv_t[i],
                 idx_f = rv_f[i];
        uint32_t idx_out = tmp[i];

        // Unchanged differentiable outputs can be piped through directly
        // without being outputs of the CustomOp
        if (ad && ((idx_f >> 32) || (idx_t >> 32))) {
            // Force the creation of AD variables even when in an AD-suspended
            scoped_force_grad_guard guard;
            idx_t = ad_var_map_get(idx_t);
            idx_f = ad_var_map_get(idx_f);

            if (idx_t == idx_f) {
                rv.push_back(ad_var_inc_ref(idx_t));
                jit_var_dec_ref(idx_out);
                continue;
            }

            output_offsets.push_back(i);
        }

        rv.push_back(idx_out);
    }

    // If ``true_fn`` or ``false_fn`` called backward() internally,
    // we need to backpropagate gradients from the inputs further
    if (ad) {
        size_t offset = rv_t.size();
        bool enqueued = false;
        for (size_t i = 0; i < args_t.size(); ++i) {
            if (args_f[i] >> 32) {
                uint32_t value = tmp[offset++];

                if (!jit_var_is_zero_literal(value)) {
                    ad_accum_grad(args[i], value);
                    ad_enqueue(dr::ADMode::Backward, args[i]);
                    enqueued = true;
                }

                jit_var_dec_ref(value);
            }
        }

        if (enqueued)
            ad_traverse(dr::ADMode::Backward,
                        (uint32_t) dr::ADFlag::ClearInterior);

        ad_copy_implicit_deps(implicit_in,  true);
        ad_copy_implicit_deps(implicit_out, false);
    }

    isolation_guard.disarm();
}

/// CustomOp that hooks a recorded conditional statement into the AD graph
struct CondOp : public dr::detail::CustomOpBase {
public:
    CondOp(JitBackend backend, const char *label, void *payload, uint32_t cond,
           ad_cond_body body_cb, ad_cond_delete delete_cb,
           const dr::vector<uint64_t> &args, dr::vector<uint64_t> &rv,
           const dr::vector<size_t> &input_offsets,
           const dr::vector<size_t> &output_offsets,
           const dr::vector<uint32_t> &implicit_in,
           const dr::vector<uint32_t> &implicit_out)
        : m_backend(backend), m_label(label), m_payload(payload), m_cond(cond),
          m_body_cb(body_cb), m_delete_cb(delete_cb) {
        m_label_op = "Cond: " + m_label;
        jit_var_inc_ref(m_cond);

        tsl::robin_set<uint32_t, UInt32Hasher> implicit;
        for (uint32_t index: implicit_in)
            implicit.insert(index);
        for (uint32_t index: implicit_out)
            implicit.insert(index);

        m_args.reserve(args.size());
        m_args_implicit.reserve(args.size());
        for (uint64_t index : args) {
            m_args.push_back_borrow(index);
            m_args_implicit.push_back(false);
        }

        // Keep track of differentiable inputs/outputs and their
        // position within the argument. Remove implicit inputs
        // and outputs (i.e. variables accessed via scatters/gathers)
        // from these lists.
        m_input_offsets.reserve(input_offsets.size());
        for (size_t offset : input_offsets) {
            uint32_t ad_index = uint32_t(m_args[offset] >> 32);
            ad_assert(ad_index != 0, "CondOp: internal error (1)");
            if (implicit.find(ad_index) != implicit.end()) {
                m_args_implicit[offset] = true;
                continue;
            }
            add_index(m_backend, ad_index, true);
            m_input_offsets.push_back(offset);
        }

        m_output_offsets.reserve(output_offsets.size());
        m_rv.reserve(rv.size());
        for (size_t offset : output_offsets) {
            uint64_t index = rv[offset];
            uint64_t index_new = ad_var_new((uint32_t) index);
            uint32_t ad_index_new = uint32_t(index_new >> 32);
            add_index(m_backend, ad_index_new, false);
            m_output_offsets.push_back(offset);
            m_rv.push_back_borrow(from_ad_index(ad_index_new));
            rv[offset] = index_new;
            ad_var_dec_ref(index);
        }

        if (!implicit_out.empty()) {
            implicit.clear();
            for (uint32_t index: implicit_out)
                implicit.insert(index);
            for (size_t i = 0; i < rv.size(); ++i) {
                uint32_t ad_index = to_ad_index(rv[i]);
                auto it = implicit.find(ad_index);
                if (it == implicit.end())
                    continue;
                // Recognize implicit outputs that are return values
                implicit.erase_fast(it);
                add_index(m_backend, ad_index, false);
                m_implicit_out.push_back_borrow(from_ad_index(ad_index));
                m_output_offsets.push_back(i);
            }
            // We don't support implicit outputs that aren't outputs
            // of the operation.
        }

        for (uint32_t index: implicit_in)
            add_index(m_backend, index, true);
    }

    ~CondOp() {
        if (m_delete_cb)
            m_delete_cb(m_payload);
        jit_var_dec_ref(m_cond);
    }


    uint32_t to_ad_index(uint64_t index) const {
        return uint32_t(index >> 32);
    }

    uint64_t from_ad_index(uint32_t ad_index) const {
        return ((uint64_t) ad_index) << 32;
    }

    void forward() override {
        dr::string label = m_label + " [ad, fwd]";

        index64_vector args, rv;
        args.reserve(m_args.size() + m_input_offsets.size());
        rv.reserve(m_output_offsets.size());

        for (uint64_t index : m_args)
            args.push_back_borrow((uint32_t) index);
        for (size_t i : m_input_offsets)
            args.push_back_steal(ad_grad(m_args[i]));

        ad_cond(
            m_backend, 1, label.c_str(), this, m_cond, args, rv,
            [](void *p, bool value, const dr::vector<uint64_t> &args,
               dr::vector<uint64_t> &rv) {
                ((CondOp *) p)->forward_cb(value, args, rv);
            },
            nullptr, false);

        ad_assert(rv.size() == m_output_offsets.size(),
                  "CondOp::forward(): size mismatch!");

        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            ad_accum_grad(from_ad_index(m_output_indices[i]), (uint32_t) rv[i]);
    }

    void backward() override {
        dr::string label = m_label + " [ad, bwd]";

        index64_vector args, rv;
        args.reserve(m_args.size() + m_output_offsets.size());
        rv.reserve(m_input_offsets.size());

        // Call the backward derivative with the inputs and the
        // derivative of the outputs. Skip implicit dependences.
        for (uint64_t index : m_args)
            args.push_back_borrow((uint32_t) index);
        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            args.push_back_steal(ad_grad(from_ad_index(m_output_indices[i])));

        ad_cond(
            m_backend, 1, label.c_str(), this, m_cond, args, rv,
            [](void *p, bool value, const dr::vector<uint64_t> &args,
               dr::vector<uint64_t> &rv) {
                ((CondOp *) p)->backward_cb(value, args, rv);
            },
            nullptr, false);

        ad_assert(rv.size() == m_input_offsets.size(),
                  "CondOp::backward(): size mismatch!");

        for (size_t i = 0; i < m_input_offsets.size(); ++i)
            ad_accum_grad(from_ad_index(m_input_indices[i]), (uint32_t) rv[i]);
    }

    // The symbolic conditional implementation will call this function twice
    // to generate the forward derivative of the 'if' and 'else' branch
    void forward_cb(bool value, const dr::vector<uint64_t> &args,
                    dr::vector<uint64_t> &rv) {
        index64_vector args2, rv2;

        args2.reserve(m_args.size());
        rv2.reserve(m_rv.size());

        for (size_t i = 0; i < m_args.size(); ++i)
            args2.push_back_borrow(m_args_implicit[i] ? m_args[i] : args[i]);

        for (size_t offset : m_input_offsets) {
            uint64_t &index = args2[offset],
                     index_new = ad_var_new((uint32_t) index);
            ad_var_dec_ref(index);
            index = index_new;
        }

        {
            // Begin a recording session and abort it by not
            // calling .disarm(). This suppresses side effects.
            scoped_record record_guard(m_backend);

            // Execute the body of the conditional operation
            m_body_cb(m_payload, value, args2, rv2);
        }

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t index = args2[m_input_offsets[i]];
            ad_accum_grad(index, (uint32_t) args[m_args.size() + i]);
            ad_enqueue(dr::ADMode::Forward, index);
        }

        // Enqueue implicit dependencies
        for (size_t i = m_input_offsets.size(); i < m_input_indices.size(); ++i)
            ad_enqueue(dr::ADMode::Forward, from_ad_index(m_input_indices[i]));

        ad_traverse(dr::ADMode::Forward, (uint32_t) dr::ADFlag::ClearNone);

        for (size_t offset : m_output_offsets)
            rv.push_back(ad_grad(rv2[offset]));
    }

    // The symbolic conditional implementation will call this function twice
    // to generate the backward derivative of the 'if' and 'else' branch
    void backward_cb(bool value, const dr::vector<uint64_t> &args,
                     dr::vector<uint64_t> &rv) {
        index64_vector args2, rv2;

        args2.reserve(m_args.size());
        rv2.reserve(m_rv.size());

        for (size_t i = 0; i < m_args.size(); ++i)
            args2.push_back_borrow(m_args_implicit[i] ? m_args[i] : args[i]);

        for (size_t offset : m_input_offsets) {
            uint64_t &index = args2[offset],
                     index_new = ad_var_new((uint32_t) index);
            ad_var_dec_ref(index);
            index = index_new;
        }

        {
            // Begin a recording session and abort it by not
            // calling .disarm(). This suppresses side effects.
            scoped_record record_guard(m_backend);

            // Execute the body of the conditional operation
            m_body_cb(m_payload, value, args2, rv2);
        }

        // Launch the AD system recursively to propagate derivatives
        for (size_t i = 0; i < m_output_offsets.size(); ++i) {
            // Enqueue regular output argument
            uint64_t index_new = ad_var_copy(rv2[m_output_offsets[i]]);
            ad_accum_grad(index_new, (uint32_t) args[m_args.size() + i]);
            ad_enqueue(dr::ADMode::Backward, index_new);
            ad_var_dec_ref(index_new);
        }

        ad_traverse(dr::ADMode::Backward, (uint32_t) dr::ADFlag::ClearNone);

        // Return gradients of non-implicit inputs
        for (size_t offset: m_input_offsets)
            rv.push_back(ad_grad(args2[offset]));
    }

    void disable(dr::vector<uint64_t> &rv) {
        m_delete_cb = nullptr;

        for (size_t offset : m_output_offsets) {
            uint64_t index = rv[offset];
            ad_assert(to_ad_index(index) != 0,
                      "CondOp::disable(\"%s\"): internal error!",
                      m_label.c_str());
            jit_var_inc_ref((uint32_t) index);
            ad_var_dec_ref(index);
            rv[offset] = (uint32_t) index;
        }
    }

    const char *name() const override { return m_label_op.c_str(); }

private:
    JitBackend m_backend;
    dr::string m_label;
    dr::string m_label_op;
    void *m_payload;
    uint32_t m_cond;
    ad_cond_body m_body_cb;
    ad_cond_delete m_delete_cb;
    index64_vector m_args;
    index64_vector m_rv;
    index64_vector m_implicit_out;
    dr::vector<bool> m_args_implicit;
    dr::vector<size_t> m_input_offsets;
    dr::vector<size_t> m_output_offsets;
};

bool ad_cond(JitBackend backend, int symbolic, const char *label, void *payload,
             uint32_t cond, const dr::vector<uint64_t> &args,
             dr::vector<uint64_t> &rv, ad_cond_body body_cb,
             ad_cond_delete delete_cb, bool ad) {
    if (label == nullptr)
        label = "unnamed";

    if (strchr(label, '\n') || strchr(label, '\r'))
        jit_raise("'label' may not contain newline characters.");

    if (symbolic == -1) {
        uint32_t flags = jit_flags();
        if (flags & (uint32_t) JitFlag::SymbolicScope) {
            // We're inside some other symbolic operation, cannot use evaluated mode
            if (!jit_flag(JitFlag::SymbolicCalls))
                jit_log(LogLevel::Warn,
                        "ad_cond(\"%s\"): forcing conditional statement to "
                        "symbolic mode since the operation is nested within "
                        "another symbolic operation).", label);
            symbolic = 1;
        } else {
            symbolic = bool(flags & (uint32_t) JitFlag::SymbolicConditionals);
        }
    }

    if (symbolic != 0 && symbolic != 1)
        jit_raise("'symbolic' must equal 0, 1, or -1.");

    if (jit_var_state(cond) == VarState::Literal) {
        jit_log(LogLevel::InfoSym,
                "ad_cond_evaluated(\"%s\"): removing conditional expression "
                "with uniform condition.", label);
        body_cb(payload, !jit_var_is_zero_literal(cond), args, rv);
        return true;
    }

    size_t size = jit_var_size(cond);

    JitVar true_mask = JitVar::steal(jit_var_mask_apply(cond, (uint32_t) size)),
           neg_mask = JitVar::steal(jit_var_not(cond)),
           false_mask = JitVar::steal(jit_var_mask_apply(neg_mask.index(), (uint32_t) size));

    if (symbolic) {
        dr::vector<size_t> input_offsets, output_offsets;
        dr::detail::ad_index32_vector implicit_in, implicit_out;

        {
            ad_cond_symbolic(backend, label, payload, true_mask.index(),
                             false_mask.index(), args, rv, body_cb, input_offsets,
                             output_offsets, implicit_in, implicit_out, ad);
        }

        if ((!input_offsets.empty() || !output_offsets.empty()) && !ad_grad_suspended()) {
            nanobind::ref<CondOp> op = new CondOp(
                backend, label, payload, cond, body_cb, delete_cb, args, rv,
                input_offsets, output_offsets, implicit_in, implicit_out);

            if (ad_custom_op(op.get())) {
                // CondOp will eventually call delete_cb()
                return false;
            }

            // CustomOp was not needed, detach output again..
            op->disable(rv);
        }
    } else {
        scoped_isolation_guard guard;
        ad_cond_evaluated(backend, label, payload, true_mask.index(),
                          false_mask.index(), args, rv, body_cb);
        guard.disarm();
    }

    return true; // Caller should directly call delete()
}
