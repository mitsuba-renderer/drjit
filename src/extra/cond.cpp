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
#include <drjit/custom.h>
#include <string>
#include <vector>

namespace dr = drjit;
using dr::dr_vector;

using JitVar = GenericArray<void>;

static void ad_cond_evaluated(JitBackend backend, const char *name,
                              void *payload, uint32_t cond_t, uint32_t cond_f,
                              const dr_vector<uint64_t> &args,
                              dr_vector<uint64_t> &rv, ad_cond_body body_cb) {
    jit_log(LogLevel::InfoSym,
            "ad_cond_evaluated(\"%s\"): executing conditional expression.",
            name);

    dr_index64_vector true_idx, false_idx;

    // Execute 'true_fn'
    {
        scoped_push_mask guard(backend, cond_t);
        body_cb(payload, true, args, true_idx);
    }

    false_idx.reserve(true_idx.size());
    rv.reserve(true_idx.size());

    // Execute 'false_fn'
    {
        scoped_push_mask guard(backend, cond_f);
        body_cb(payload, false, args, false_idx);
    }

    if (true_idx.size() != false_idx.size())
        jit_raise("ad_cond_evaluated(): inconsistent number of outputs!");

    // Combine the results
    for (size_t i = 0; i < true_idx.size(); ++i) {
        uint64_t i1 = true_idx[i], i2 = false_idx[i];

        rv.push_back(i1 == i2 ? ad_var_inc_ref(i1)
                              : ad_var_select(cond_t, i1, i2));
    }
}

static void ad_cond_symbolic(JitBackend backend, const char *name,
                             void *payload, uint32_t cond_t, uint32_t cond_f,
                             const dr_vector<uint64_t> &args_,
                             dr_vector<uint64_t> &rv, ad_cond_body body_cb,
                             std::vector<size_t> &input_offsets,
                             std::vector<size_t> &output_offsets, bool ad) {
    bool symbolic = jit_flag(JitFlag::SymbolicScope);

    scoped_record record_guard(backend);

    JitVar handle =
        JitVar::steal(jit_var_cond_start(name, symbolic, cond_t, cond_f));

    dr_index64_vector args, true_idx, false_idx;
    args.reserve(args_.size());

    // Detach from original computation in AD graph
    for (size_t i = 0; i < args_.size(); ++i) {
        uint64_t index = args_[i];

        if (ad && (index >> 32) != 0) {
            args.push_back_steal(ad_var_new((uint32_t) index));
            input_offsets.push_back(i);
        } else {
            args.push_back_borrow((uint32_t) index);
        }
    }

    // Execute 'true_fn'
    {
        scoped_push_mask guard(backend, cond_t);
        body_cb(payload, true, args, true_idx);
    }
    false_idx.reserve(true_idx.size());
    rv.reserve(true_idx.size());

    std::vector<uint32_t> tmp(true_idx.size());
    for (size_t i = 0; i < true_idx.size(); ++i)
        tmp[i] = (uint32_t) true_idx[i];

    JitVar handle_2 = JitVar::steal(jit_var_cond_append(
        handle.index(), tmp.data(), tmp.size()));

    // Execute 'false_fn'
    {
        scoped_push_mask guard(backend, cond_f);
        body_cb(payload, false, args, false_idx);
    }

    if (true_idx.size() != false_idx.size())
        jit_raise("ad_cond_symbolic(): inconsistent number of outputs!");

    for (size_t i = 0; i < false_idx.size(); ++i)
        tmp[i] = (uint32_t) false_idx[i];

    JitVar handle_3 = JitVar::steal(jit_var_cond_append(
        handle.index(), tmp.data(), tmp.size()));

    record_guard.disarm();

    jit_var_cond_end(handle.index(), tmp.data());

    for (size_t i = 0; i < tmp.size(); ++i) {
        uint32_t index = tmp[i];

        if (ad && (((true_idx[i] >> 32) != 0) || ((false_idx[i] >> 32) != 0)))
            output_offsets.push_back(i);

        rv.push_back(index);
    }
}

/// CustomOp that hooks a recorded conditional statement into the AD graph
struct CondOp : public dr::detail::CustomOpBase {
public:
    CondOp(JitBackend backend, const char *name, void *payload, uint32_t cond,
           ad_cond_body body_cb, ad_cond_delete delete_cb,
           const dr_vector<uint64_t> &args, dr_vector<uint64_t> &rv,
           std::vector<size_t> &&input_offsets, std::vector<size_t> &&output_offsets)
        : m_backend(backend), m_name(name), m_payload(payload), m_cond(cond),
          m_body_cb(body_cb), m_delete_cb(delete_cb),
          m_input_offsets(std::move(input_offsets)), m_output_offsets(std::move(output_offsets)) {
        m_name_op = "Cond: " + m_name;
        jit_var_inc_ref(m_cond);

        m_args.reserve(args.size());
        for (uint64_t index : args)
            m_args.push_back_borrow(index);

        m_rv.reserve(rv.size());
        for (size_t i : m_input_offsets)
            add_index(m_backend, m_args[i] >> 32, true);

        for (size_t i : m_output_offsets) {
            uint32_t index32 = (uint32_t) rv[i];
            uint64_t index64 = ad_var_new(index32);
            jit_var_dec_ref(index32);
            rv[i] = index64;
            add_index(m_backend, index64 >> 32, false);
            m_rv.push_back_borrow((index64 >> 32) << 32);
        }
    }

    ~CondOp() {
        if (m_delete_cb)
            m_delete_cb(m_payload);
        jit_var_dec_ref(m_cond);
    }


    uint64_t combine(uint32_t ad_index, uint32_t jit_index = 0) {
        return (((uint64_t) ad_index) << 32) + jit_index;
    }

    void forward() override {
        std::string name = m_name + " [ad, fwd]";

        dr_index64_vector args, rv;
        args.reserve(m_args.size() + m_input_offsets.size());
        rv.reserve(m_output_offsets.size());

        for (uint64_t index : m_args)
            args.push_back_borrow((uint32_t) index);
        for (size_t i : m_input_offsets)
            args.push_back_steal(ad_grad(m_args[i]));

        ad_cond(m_backend, 1, name.c_str(), this, m_cond, args, rv, forward_cb,
                nullptr, false);

        ad_assert(rv.size() == m_output_offsets.size(), "Size mismatch!");

        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            ad_accum_grad(combine(m_output_indices[i]), (uint32_t) rv[i]);
    }

    void backward() override {
        std::string name = m_name + " [ad, bwd]";

        dr_index64_vector args, rv;
        args.reserve(m_args.size() + m_output_offsets.size());
        rv.reserve(m_input_offsets.size());

        for (uint64_t index : m_args)
            args.push_back_borrow((uint32_t) index);
        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            args.push_back_steal(ad_grad(m_rv[i]));

        ad_cond(m_backend, 1, name.c_str(), this, m_cond, args, rv, backward_cb,
                nullptr, false);

        ad_assert(rv.size() == m_input_offsets.size(), "Size mismatch!");

        for (size_t i = 0; i < m_input_offsets.size(); ++i)
            ad_accum_grad(combine(m_input_indices[i]), (uint32_t) rv[i]);
    }

    static void forward_cb(void *p, bool value,
                            const dr_vector<uint64_t> &args,
                            dr_vector<uint64_t> &rv) {
        ((CondOp *) p)->forward_cb(value, args, rv);
    }

    static void backward_cb(void *p, bool value,
                            const dr_vector<uint64_t> &args,
                            dr_vector<uint64_t> &rv) {
        ((CondOp *) p)->backward_cb(value, args, rv);
    }

    void forward_cb(bool value, const dr_vector<uint64_t> &args,
                    dr_vector<uint64_t> &rv) {
        dr_index64_vector args2, rv2;

        args2.reserve(m_args.size());
        rv2.reserve(m_rv.size());

        for (size_t i = 0; i < m_args.size(); ++i)
            args2.push_back_borrow(args[i]);

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t &index     = args2[m_input_offsets[i]],
                      index_new = ad_var_new((uint32_t) index);
            ad_var_dec_ref(index);
            index = index_new;
        }

        m_body_cb(m_payload, value, args2, rv2);

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t index = args2[m_input_offsets[i]];
            ad_accum_grad(index, (uint32_t) args[m_args.size() + i]);
            ad_enqueue(dr::ADMode::Forward, index);
        }

        // Enqueue implicit dependencies
        for (size_t i = m_input_offsets.size(); i < m_input_indices.size(); ++i)
            ad_enqueue(dr::ADMode::Forward, combine(m_input_indices[i]));

        ad_traverse(dr::ADMode::Forward, (uint32_t) dr::ADFlag::ClearNone);

        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            rv.push_back(ad_grad(rv2[m_output_offsets[i]]));
    }

    void backward_cb(bool value, const dr_vector<uint64_t> &args,
                    dr_vector<uint64_t> &rv) {
        dr_index64_vector args2, rv2;

        args2.reserve(m_args.size());
        rv2.reserve(m_rv.size());

        for (size_t i = 0; i < m_args.size(); ++i)
            args2.push_back_borrow(args[i]);

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t &index     = args2[m_input_offsets[i]],
                      index_new = ad_var_new((uint32_t) index);
            ad_var_dec_ref(index);
            index = index_new;
        }

        m_body_cb(m_payload, value, args2, rv2);

        for (size_t i = 0; i < m_output_offsets.size(); ++i) {
            uint64_t index     = rv2[m_output_offsets[i]],
                     index_new = ad_var_copy(index);
            ad_accum_grad(index_new, (uint32_t) args[m_args.size() + i]);
            ad_enqueue(dr::ADMode::Backward, index_new);
            ad_var_dec_ref(index_new);
        }

        ad_traverse(dr::ADMode::Backward, (uint32_t) dr::ADFlag::ClearNone);

        for (size_t i = 0; i < m_input_offsets.size(); ++i)
            rv.push_back(ad_grad(args2[m_input_offsets[i]]));
    }

    void disable(dr_vector<uint64_t> &rv) {
        m_delete_cb = nullptr;

        for (size_t i = 0; i < rv.size(); ++i) {
            uint64_t index = rv[i];
            if ((index >> 32) != 0) {
                jit_var_inc_ref((uint32_t) index);
                ad_var_dec_ref(index);
                rv[i] = (uint32_t) index;
            }
        }
    }

    const char *name() const override { return m_name_op.c_str(); }

private:
    JitBackend m_backend;
    std::string m_name;
    std::string m_name_op;
    void *m_payload;
    uint32_t m_cond;
    ad_cond_body m_body_cb;
    ad_cond_delete m_delete_cb;
    dr_index64_vector m_args;
    dr_index64_vector m_rv;
    std::vector<size_t> m_input_offsets;
    std::vector<size_t> m_output_offsets;
};

bool ad_cond(JitBackend backend, int symbolic, const char *name, void *payload,
             uint32_t cond, const drjit::dr_vector<uint64_t> &args,
             drjit::dr_vector<uint64_t> &rv, ad_cond_body body_cb,
             ad_cond_delete delete_cb, bool ad) {
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
            body_cb(payload, !jit_var_is_zero_literal(cond), args, rv);
            return true;
        }

        size_t size = jit_var_size(cond);

        JitVar true_mask = JitVar::steal(jit_var_mask_apply(cond, size)),
               neg_mask = JitVar::steal(jit_var_not(cond)),
               false_mask = JitVar::steal(jit_var_mask_apply(neg_mask.index(), size));

        if (symbolic) {
            std::vector<size_t> input_offsets, output_offsets;
            dr_vector<uint32_t> implicit_in;
            {
                scoped_isolation_boundary guard;
                ad_cond_symbolic(backend, name, payload, true_mask.index(),
                                 false_mask.index(), args, rv, body_cb, input_offsets,
                                 output_offsets, ad);
                ad_copy_implicit_deps(implicit_in);
                guard.defuse();
            }

            if (!input_offsets.empty() || !output_offsets.empty()) {
                nanobind::ref<CondOp> op = new CondOp(
                    backend, name, payload, cond, body_cb, delete_cb, args, rv,
                    std::move(input_offsets), std::move(output_offsets));

                for (uint32_t index: implicit_in)
                    op->add_index(backend, index, true);

                if (ad_custom_op(op.get())) {
                    // CondOp will eventually call delete_cb()
                    return false;
                }

                // CustomOp was not needed, detach output again..
                op->disable(rv);
            }
        } else {
            scoped_isolation_boundary guard;
            ad_cond_evaluated(backend, name, payload, true_mask.index(),
                              false_mask.index(), args, rv, body_cb);
            guard.defuse();
        }

        return true; // Caller should directly call delete()
    } catch (...) {
        if (delete_cb)
            delete_cb(payload);
        throw;
    }
}
