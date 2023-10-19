/*
    extra/vcall.cpp -- Logic to dispatch virtual function calls, dr.switch(),
    and dr.dispatch() through one common interface with support for wavefront-
    and recorded execution styles along with automatic differentiation.

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <drjit/autodiff.h>
#include <drjit/custom.h>
#include <algorithm>
#include <string>
#include "common.h"

namespace dr = drjit;
using dr::dr_vector;

/// Index vector that decreases JIT refcounts when destructed
struct dr_index32_vector : dr_vector<uint32_t> {
    using Base = dr_vector<uint32_t>;
    using Base::Base;

    ~dr_index32_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            jit_var_dec_ref(operator[](i));
        Base::clear();
    }

    void push_back_steal(uint32_t index) { push_back(index); }
    void push_back_borrow(uint32_t index) {
        jit_var_inc_ref(index);
        push_back(index);
    }
};

/// Index vector that decreases JIT + AD refcounts when destructed
struct dr_index64_vector : dr_vector<uint64_t> {
    using Base = dr_vector<uint64_t>;
    using Base::Base;

    ~dr_index64_vector() { release(); }

    void release() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
        Base::clear();
    }

    void push_back_steal(uint64_t index) { push_back(index); }
    void push_back_borrow(uint64_t index) {
        push_back(ad_var_inc_ref(index));
    }
};

/// RAII helper to temporarily push a mask onto the Dr.Jit mask stack
struct scoped_set_mask {
    scoped_set_mask(JitBackend backend, uint32_t index) : backend(backend) {
        jit_var_mask_push(backend, index);
        jit_var_dec_ref(index);
    }

    ~scoped_set_mask() {
        jit_var_mask_pop(backend);
    }

    JitBackend backend;
};


/// RAII AD Isolation helper
struct scoped_isolation_boundary {
    scoped_isolation_boundary() {
        ad_scope_enter(dr::ADScope::Isolate, 0, nullptr);
    }

    ~scoped_isolation_boundary() {
        ad_scope_leave(success);
    }

    bool success = false;
};

/// RAII helper to temporarily record symbolic computation
struct scoped_record {
    scoped_record(JitBackend backend, const char *name) : backend(backend) {
        checkpoint = jit_record_begin(backend, name);
        scope = jit_new_scope(backend);
    }

    uint32_t checkpoint_and_rewind() {
        jit_set_scope(backend, scope);
        return jit_record_checkpoint(backend);
    }

    ~scoped_record() {
        jit_record_end(backend, checkpoint);
    }

    JitBackend backend;
    uint32_t checkpoint, scope;
};

using JitVar = GenericArray<void>;

// Forward declaration of a helper function full of checks (used by all strategies)
static void ad_vcall_check_rv(JitBackend backend, size_t size,
                              size_t callable_index,
                              dr_vector<uint64_t> &rv,
                              const dr_vector<uint64_t> &rv2);

// Strategy 1: record dynamic vcall by tracing all callables
static void ad_vcall_record(JitBackend backend, const char *domain,
                            const char *name, size_t size, uint32_t index,
                            uint32_t mask_, size_t callable_count,
                            const dr_vector<uint64_t> args,
                            dr_vector<uint64_t> &rv, dr_vector<bool> &rv_ad,
                            ad_vcall_callback callback, void *payload) {
    (void) domain;
    (void) size;

    JitVar mask;
    if (mask_)
        mask = JitVar::borrow(mask_);
    else
        mask = JitVar::steal(jit_var_bool(backend, true));

    dr_index64_vector args2;
    dr_vector<uint64_t> rv2;

    dr_index32_vector args3, rv3;

    args2.reserve(args.size());
    args2.reserve(args.size());

    dr_vector<uint32_t> checkpoints(callable_count + 1, 0),
                            inst_id(callable_count, 0);

    uint32_t se = 0; // operation representing side effects from the call
    {
        scoped_record rec(backend, name);

        // Wrap input arguments to clearly expose them as inputs of the vcall
        for (size_t i = 0; i < args.size(); ++i) {
            uint32_t wrapped = jit_var_wrap_vcall((uint32_t) args[i]);
            args3.push_back_steal(wrapped);

            if (args[i] >> 32)
                args2.push_back_steal(ad_var_new(wrapped));
            else
                args2.push_back_borrow(wrapped);
        }

        {
            scoped_set_mask mask_guard(backend, jit_var_vcall_mask(backend));
            for (size_t i = 0; i < callable_count; ++i) {
                checkpoints[i] = rec.checkpoint_and_rewind();
                inst_id[i] = i + 1;

                // Populate 'rv2' with function return values. This may raise
                // an exception, in which case everything should be properly
                // cleaned up in this function's scope
                rv2.clear();
                callback(payload, i, args2, rv2);

                for (uint64_t index: rv2)
                    ad_var_check_implicit(index);

                // Perform some sanity checks on the return values
                ad_vcall_check_rv(backend, size, i, rv, rv2);

                // Move return values to a separate array storing them for all callables
                if (i == 0) {
                    // Preallocate memory in the first iteration
                    rv3.reserve(rv2.size() * callable_count);
                    rv_ad.resize(rv2.size());
                    for (bool &b : rv_ad)
                        b = false;
                }

                for (size_t j = 0; j < rv2.size(); ++j) {
                    uint64_t index = rv2[j];
                    rv_ad[j] |= (index >> 32) != 0;
                    rv3.push_back_borrow((uint32_t) index);
                }
            }

            checkpoints[callable_count] = rec.checkpoint_and_rewind();
        }

        dr_vector<uint32_t> rv4;
        rv4.resize(rv.size());

        se = jit_var_vcall(name, index, mask.index(), callable_count,
                           inst_id.data(), (uint32_t) args3.size(),
                           args3.data(), (uint32_t) rv3.size(), rv3.data(),
                           checkpoints.data(), rv4.data());

        for (size_t i = 0; i < rv.size(); ++i) {
            ad_var_dec_ref(rv[i]);
            rv[i] = rv4[i];
        }
    }

    jit_var_mark_side_effect(se);
}

// Strategy 2: record dynamic vcall by grouping the arguments by callable (evaluates everything)
static void ad_vcall_reduce(JitBackend backend, const char *domain,
                            const char *name, size_t size, uint32_t index_,
                            uint32_t mask, size_t callable_count,
                            const dr_vector<uint64_t> args,
                            dr_vector<uint64_t> &rv,
                            ad_vcall_callback callback, void *payload) {
    (void) name; // unused

    JitVar index;
    if (mask)
        index = JitVar::steal(jit_var_and(index_, mask));
    else
        index = JitVar::borrow(index_);

    jit_var_schedule(index.index());
    for (uint64_t arg_i : args)
        jit_var_schedule((uint32_t) arg_i);

    uint32_t n_inst = callable_count;
    VCallBucket *buckets =
        jit_var_vcall_reduce(backend, domain, index.index(), &n_inst);

    dr_index64_vector args2(args.size(), 0);
    args2.clear();

    dr_vector<uint64_t> rv2;
    size_t last_size = 0;
    JitVar memop_mask = JitVar::steal(jit_var_bool(backend, true));

    for (size_t i = 0; i < n_inst ; ++i) {
        if (buckets[i].id == 0)
            continue;

        uint32_t callable_index = buckets[i].id - 1,
                 index2 = buckets[i].index;

        size_t wavefront_size = jit_var_size(index2);

        // Don't merge subsequent wavefronts into the same kernel,
        // which could happen if they have the same size
        if (last_size == wavefront_size)
            jit_eval();
        last_size = wavefront_size;

        // Fetch arguments
        scoped_set_mask mask_guard(
            backend, jit_var_mask_default(backend, wavefront_size));
        for (size_t j = 0; j < args.size(); ++j)
            args2.push_back_steal(
                ad_var_gather(args[j], index2, memop_mask.index(), true));

        // Populate 'rv2' with function return values. This may raise an
        // exception, in which case everything should be properly cleaned up in
        // this function's scope
        rv2.clear();
        callback(payload, callable_index, args2, rv2);

        // Perform some sanity checks on the return values
        ad_vcall_check_rv(backend, size, i, rv, rv2);

        // Merge 'rv2' into 'rv' (main function return values)
        for (size_t j = 0; j < rv2.size(); ++j) {
            uint64_t index =
                ad_var_scatter(rv[j], rv2[j], index2, memop_mask.index(),
                               ReduceOp::None, true);
            ad_var_dec_ref(rv[j]);
            rv[j] = index;
        }

        args2.release();
    }

    for (uint64_t index : rv)
        jit_var_schedule((uint32_t) index);
}

// Helper function full of checks (used by all strategies)
static void ad_vcall_check_rv(JitBackend backend, size_t size,
                              size_t callable_index,
                              dr_vector<uint64_t> &rv,
                              const dr_vector<uint64_t> &rv2) {
    // Examine return values
    if (rv.size() != rv2.size()) {
        if (!rv.empty())
            jit_raise(
                "ad_vcall(): callable %zu returned an unexpected "
                "number of return values (got %zu indices, expected %zu)",
                callable_index, rv2.size(), rv.size());

        // Allocate a zero-initialized output array in the first iteration
        rv.resize(rv2.size());

        uint64_t zero = 0;
        for (size_t i = 0; i < rv2.size(); ++i)
            rv[i] = jit_var_literal(backend, jit_var_type(rv2[i]), &zero, size);
    } else {
        // Some sanity checks
        for (size_t i = 0; i < rv.size(); ++i) {
            uint64_t i1 = rv[i], i2 = rv2[i];

            VarInfo v1 = jit_set_backend(i1),
                    v2 = jit_set_backend(i2);

            if (v2.backend != backend)
                jit_raise("ad_vcall(): callable %zu returned an array "
                          "with an inconsistent backend", callable_index);

            if (v1.type != v2.type)
                jit_raise("ad_vcall(): callable %zu returned an array "
                          "with an inconsistent type (%s vs %s)",
                          callable_index, jit_type_name(v1.type),
                          jit_type_name(v2.type));
        }
    }
}

/// CustomOp that hooks a recorded virtual function call into the AD graph
struct VCallOp : public dr::detail::CustomOpBase {
public:
    VCallOp(JitBackend backend, const char *name, const char *domain,
            uint32_t index, uint32_t mask, size_t callable_count,
            const dr_vector<uint64_t> &args, size_t rv_size, void *payload,
            ad_vcall_callback callback, ad_vcall_cleanup cleanup)
        : m_name(name), m_domain(domain), m_index(index), m_mask(mask),
          m_callable_count(callable_count), m_payload(payload),
          m_callback(callback), m_cleanup(cleanup) {
        m_backend = backend;

        jit_var_inc_ref(m_index);
        jit_var_inc_ref(m_mask);

        m_args.reserve(args.size());
        m_args2.reserve(args.size());
        m_rv.reserve(rv_size);
        m_rv2.reserve(rv_size);

        m_input_offsets.reserve(args.size());
        m_output_offsets.reserve(rv_size);
        m_temp.reserve(std::max(args.size(), rv_size));

        for (size_t i = 0; i < args.size(); ++i)
            m_args.push_back_borrow((uint32_t) args[i]);
    }

    ~VCallOp() {
        jit_var_dec_ref(m_index);
        jit_var_dec_ref(m_mask);
        if (m_cleanup)
            m_cleanup(m_payload);
    }

    uint64_t combine(uint32_t ad_index, uint32_t jit_index = 0) {
        return (((uint64_t) ad_index) << 32) + jit_index;
    }

    /// Implements f(arg..., grad(arg)...) -> grad(rv)...
    void forward() override {
        std::string name = m_name + " [ad, fwd]";

        dr_index64_vector args, rv;
        args.reserve(m_args.size() + m_input_offsets.size());
        rv.reserve(m_output_offsets.size());

        for (uint32_t index : m_args)
            args.push_back_borrow(index);
        for (size_t i = 0; i < m_input_offsets.size(); ++i)
            args.push_back_steal(ad_grad(combine(m_input_indices[i])));

        ad_vcall(m_backend, m_domain, name.c_str(), m_index, m_mask,
                 m_callable_count, args, rv, this, &forward_cb, nullptr,
                 false);

        ad_assert(rv.size() == m_output_offsets.size(), "Size mismatch!");

        m_args2.release();
        m_rv2.clear();
        m_temp.release();

        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            ad_accum_grad(combine(m_output_indices[i]), (uint32_t) rv[i]);
    }

    /// Implements f(arg..., grad(rv)...) -> grad(arg) ...
    void backward() override {
        std::string name = m_name + " [ad, bwd]";

        dr_index64_vector args, rv;
        args.reserve(m_args.size() + m_output_offsets.size());
        rv.reserve(m_input_offsets.size());

        for (uint32_t index : m_args)
            args.push_back_borrow(index);
        for (size_t i = 0; i < m_output_offsets.size(); ++i)
            args.push_back_steal(ad_grad(combine(m_output_indices[i])));

        ad_vcall(m_backend, m_domain, name.c_str(), m_index, m_mask,
                 m_callable_count, args, rv, this, &backward_cb, nullptr,
                 false);

        ad_assert(rv.size() == m_input_offsets.size(), "Size mismatch!");

        m_args2.release();
        m_rv2.clear();
        m_temp.release();

        for (size_t i = 0; i < m_input_offsets.size(); ++i)
            ad_accum_grad(combine(m_input_indices[i]), (uint32_t) rv[i]);
    }

    static void forward_cb(void *ptr, size_t index,
                           const dr_vector<uint64_t> &args,
                           dr_vector<uint64_t> &rv) {
        ((VCallOp *) ptr)->forward_cb(index, args, rv);
    }

    /// Forward AD callback (invoked by forward() once per callable)
    void forward_cb(size_t index, const dr_vector<uint64_t> &args,
                    dr_vector<uint64_t> &rv) {
        m_args2.release();
        for (size_t i = 0; i < m_args.size(); ++i)
            m_args2.push_back_borrow(args[i]);

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t &index     = m_args2[m_input_offsets[i]],
                      index_new = ad_var_new((uint32_t) index);
            ad_var_dec_ref(index);
            index = index_new;
        }

        m_rv2.clear();
        m_callback(m_payload, index, m_args2, m_rv2);
        ad_assert(m_rv2.size() == m_rv.size(), "Size mismatch!");

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t index = m_args2[m_input_offsets[i]];
            ad_accum_grad(index, (uint32_t) args[m_args.size() + i]);
            ad_enqueue(dr::ADMode::Forward, index);
        }

        for (size_t i = m_input_offsets.size(); i < m_input_indices.size(); ++i)
            ad_enqueue(dr::ADMode::Forward, ((uint64_t) m_input_indices[i]) << 32);

        // Enqueue implicit dependencies
        ad_traverse(dr::ADMode::Forward, (uint32_t) dr::ADFlag::ClearNone);

        m_temp.release();
        for (size_t i = 0; i < m_output_offsets.size(); ++i) {
            uint32_t index = ad_grad(m_rv2[m_output_offsets[i]]);
            m_temp.push_back_steal(index);
            rv.push_back(index);
        }
    }

    static void backward_cb(void *ptr, size_t index,
                           const dr_vector<uint64_t> &args,
                           dr_vector<uint64_t> &rv) {
        ((VCallOp *) ptr)->backward_cb(index, args, rv);
    }

    /// Backward AD callback (invoked by backward() once per callable)
    void backward_cb(size_t index, const dr_vector<uint64_t> &args,
                     dr_vector<uint64_t> &rv) {
        m_args2.release();
        for (size_t i = 0; i < m_args.size(); ++i)
            m_args2.push_back_borrow(args[i]);

        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint64_t &index     = m_args2[m_input_offsets[i]],
                      index_new = ad_var_new((uint32_t) index);
            ad_var_dec_ref(index);
            index = index_new;
        }

        m_rv2.clear();
        m_callback(m_payload, index, m_args2, m_rv2);
        ad_assert(m_rv2.size() == m_rv.size(), "Size mismatch!");

        for (size_t i = 0; i < m_output_offsets.size(); ++i) {
            uint64_t index     = m_rv2[m_output_offsets[i]],
                     index_new = ad_var_copy(index);
            ad_accum_grad(index_new, (uint32_t) args[m_args.size() + i]);
            ad_enqueue(dr::ADMode::Backward, index_new);
            ad_var_dec_ref(index_new);
        }

        ad_traverse(dr::ADMode::Backward, (uint32_t) dr::ADFlag::ClearNone);

        m_temp.release();
        for (size_t i = 0; i < m_input_offsets.size(); ++i) {
            uint32_t index = ad_grad(m_args2[m_input_offsets[i]]);
            m_temp.push_back_steal(index);
            rv.push_back(index);
        }
    }


    const char *name() const override { return m_name.c_str(); }

    void add_input(size_t i, uint64_t index) {
        if (add_index(m_backend, index >> 32, true))
            m_input_offsets.push_back((uint32_t) i);
    }

    void add_output(size_t i, uint64_t index) {
        if (add_index(m_backend, index >> 32, false)) {
            m_output_offsets.push_back((uint32_t) i);
            m_rv.push_back_borrow((index >> 32) << 32);
        }
    }

    void disable_cleanup() { m_cleanup = nullptr; }

private:
    std::string m_name;
    const char *m_domain;
    uint32_t m_index, m_mask;
    size_t m_callable_count;
    dr_index32_vector m_args;
    dr_index64_vector m_args2;
    dr_index64_vector m_rv;
    dr_vector<uint64_t> m_rv2;
    dr_index32_vector m_temp;
    dr_vector<uint32_t> m_input_offsets;
    dr_vector<uint32_t> m_output_offsets;
    void *m_payload;
    ad_vcall_callback m_callback;
    ad_vcall_cleanup m_cleanup;
};

// Generic checks, then forward either to ad_vcall_record or ad_vcall_reduce
bool ad_vcall(JitBackend backend, const char *domain, const char *name,
              uint32_t index, uint32_t mask, size_t callable_count,
              const dr_vector<uint64_t> &args, dr_vector<uint64_t> &rv,
              void *payload, ad_vcall_callback callback,
              ad_vcall_cleanup cleanup, bool ad) {

    if (callable_count == 0)
        jit_raise("ad_vcall(\"%s\"): callable list cannot be empty", name);
    if (index == 0)
        jit_raise("ad_vcall(\"%s\"): index list cannot be empty", name);

    size_t size = jit_var_size(index);
    if (mask) {
        size_t size_2 = jit_var_size(mask);

        if (size != size_2 && size != 1 && size_2 != 1)
            jit_raise(
                "ad_vcall(\"%s\"): mismatched argument sizes (%zu and %zu).",
                name, size, size_2);

        size = std::max(size, size_2);
    }

    bool needs_ad = false;
    for (uint64_t arg_i : args) {
        size_t size_2 = jit_var_size((uint32_t) arg_i);

        if (size != size_2 && size != 1 && size_2 != 1)
            jit_raise("ad_vcall(\"%s\"): mismatched argument sizes (%zu and %zu).",
                      name, size, size_2);

        size = std::max(size, size_2);
        needs_ad |= arg_i >> 32;
    }

    if (jit_flag(JitFlag::VCallRecord)) {
        dr_vector<bool> rv_ad;
        dr_vector<uint32_t> implicit_in;
        {
            scoped_isolation_boundary guard;
            ad_vcall_record(backend, domain, name, size, index, mask,
                            callable_count, args, rv, rv_ad, callback, payload);
            ad_copy_implicit_deps(implicit_in);
            guard.success = true;
        }

        for (bool b : rv_ad)
            needs_ad |= b;

        if (ad && needs_ad) {
            nanobind::ref<VCallOp> op =
                new VCallOp(backend, name, domain, index, mask, callable_count,
                            args, rv.size(), payload, callback, cleanup);

            for (size_t i = 0; i < args.size(); ++i)
                op->add_input(i, args[i]);

            for (uint32_t index: implicit_in)
                op->add_index(backend, index, true);

            for (size_t i = 0; i < rv.size(); ++i) {
                if (!rv_ad[i])
                    continue;

                uint64_t index = ad_var_new((uint32_t) rv[i]);

                jit_var_dec_ref((uint32_t) rv[i]);
                rv[i] = index;
                op->add_output(i, index);
            }

            if (ad_custom_op(op.get())) {
                // VCallOp will eventually call cleanup()
                return false;
            }

            // CustomOp was not needed, detach output again..
            op->disable_cleanup();
            for (size_t i = 0; i < rv.size(); ++i) {
                uint64_t index = rv[i],
                         index_d = (index << 32) >> 32;
                if (index == index_d)
                    continue;
                jit_var_inc_ref((uint32_t) index_d);
                ad_var_dec_ref(index);
                rv[i] = index_d;
            }
        }
    } else {
        ad_vcall_reduce(backend, domain, name, size, index, mask,
                        callable_count, args, rv, callback, payload);
    }

    // Caller should directly call cleanup()
    return true;
}

