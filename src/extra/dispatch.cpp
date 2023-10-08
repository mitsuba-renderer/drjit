#include <drjit/autodiff.h>
#include "common.h"
#include <algorithm>

namespace dr = drjit;

struct dr_index_vector : dr::dr_vector<uint64_t> {
    using Base = dr::dr_vector<uint64_t>;
    using Base::Base;

    ~dr_index_vector() { clear_and_decref(); }

    void clear_and_decref() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
        Base::clear();
    }
};

struct scoped_default_mask {
    scoped_default_mask(JitBackend backend, size_t size) : backend(backend) {
        uint32_t index = jit_var_mask_default(backend, size);
        jit_var_mask_push(backend, index);
        jit_var_dec_ref(index);
    }

    ~scoped_default_mask() {
        jit_var_mask_pop(backend);
    }

    JitBackend backend;
};

using JitVar = GenericArray<void>;

void ad_dispatch(JitBackend backend, const char *domain, uint32_t index_,
                 uint32_t mask_,
                 size_t callable_count, const drjit::dr_vector<uint64_t> args,
                 drjit::dr_vector<uint64_t> &rv, ad_dispatch_callback callback,
                 void *payload) {

    if (callable_count == 0)
        jit_raise("ad_dispatch(): callable list cannot be empty");
    if (index_ == 0)
        jit_raise("ad_dispatch(): index list cannot be empty");

    jit_var_schedule(index_);

    size_t size = jit_var_size(index_);
    if (mask_) {
        jit_var_schedule(mask_);
        size_t size_2 = jit_var_size(mask_);

        if (size != size_2 && size != 1 & size_2 != 1)
            jit_raise("ad_dispatch(): mismatched argument sizes (%zu and %zu).",
                      size, size_2);

        size = std::max(size, size_2);
    }

    for (uint64_t arg_i : args) {
        jit_var_schedule((uint32_t) arg_i);
        size_t size_2 = jit_var_size((uint32_t) arg_i);

        if (size != size_2 && size != 1 & size_2 != 1)
            jit_raise("ad_dispatch(): mismatched argument sizes (%zu and %zu).",
                      size, size_2);

        size = std::max(size, size_2);
    }

    // Account for the mask and increase the callable index by 1
    JitVar one   = JitVar::steal(jit_var_u32(backend, 1)),
           index = JitVar::steal(jit_var_add(index_, one.index()));
    if (mask_)
        index = JitVar::steal(jit_var_and(index.index(), mask_));

    JitVar mask = JitVar::steal(jit_var_bool(backend, true));

    uint32_t n_inst = callable_count;
    VCallBucket *buckets =
        jit_var_vcall_reduce(backend, domain, index.index(), &n_inst);

    dr_index_vector args2;
    dr::dr_vector<uint64_t> rv2;

    size_t last_size = 0;

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
        scoped_default_mask mask_guard(backend, wavefront_size);
        for (size_t j = 0; j < args.size(); ++j)
            args2.push_back(ad_var_gather(args[j], index2, mask.index(), true));

        // Populate 'rv2' with function return values. This may raise an
        // exception, in which case everything should be properly cleaned up in
        // this function's scope
        callback(payload, callable_index, args2, rv2);

        // Examine return values
        if (rv.size() != rv2.size()) {
            if (!rv.empty())
                jit_raise(
                    "ad_dispatch(): callable %u returned an unexpected "
                    "number of return values (got %zu indices, expected %zu)",
                    callable_index, rv2.size(), rv.size());

            // Allocate a zero-initialized output array in the first iteration
            uint64_t zero = 0;
            for (size_t i = 0; i < rv2.size(); ++i)
                rv.push_back(jit_var_literal(backend, jit_var_type(rv2[i]), &zero, size));
        } else {
            // Some sanity checks
            for (size_t i = 0; i < rv.size(); ++i) {
                uint64_t i1 = rv[i], i2 = rv2[i];
                VarInfo v1 = jit_set_backend(i1),
                        v2 = jit_set_backend(i2);
                if (v2.backend != backend)
                    jit_raise("ad_dispatch(): callable %u returned an array "
                              "with an inconsistent backend", callable_index);

                if (v1.type != v2.type)
                    jit_raise("ad_dispatch(): callable %u returned an array "
                              "with an inconsistent type (%s vs %s)", callable_index,
                              jit_type_name(v1.type),
                              jit_type_name(v2.type));
            }
        }

        // Merge into 'rv' (main function return values)
        for (size_t j = 0; j < rv2.size(); ++j) {
            uint64_t index = ad_var_scatter(rv[j], rv2[j], index2, mask.index(),
                                            ReduceOp::None, true);
            ad_var_dec_ref(rv[j]);
            rv[j] = index;
        }

        args2.clear_and_decref();
        rv2.clear();
    }

    for (uint64_t index : rv)
        jit_var_schedule((uint32_t) index);
}
