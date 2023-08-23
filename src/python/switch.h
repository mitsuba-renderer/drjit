#pragma once

#include <iostream>

#include <drjit/array_router.h>
#include <drjit/vcall.h>
#include <drjit-core/containers.h>
#include <drjit-core/state.h>

#include "common.h"

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

// ====================================================================
//                     Switch with recorded callables
// ====================================================================

template<typename UInt32>
py::object switch_record_impl(UInt32 indices, py::list funcs, py::args args) {
    static constexpr JitBackend Backend = backend_v<UInt32>;
    using Mask = dr::mask_t<UInt32>;
    using DiffType = leaf_array_t<dr::float_array_t<UInt32>>;

    isolate_grad<DiffType> guard;

    py::object detail = py::module_::import("drjit").attr("detail");
    auto apply_cpp = detail.attr("apply_cpp");

    uint32_t n_inst = 0;
    for (uint32_t i = 0; i < funcs.size(); ++i) {
        if (!funcs[i].is_none())
            n_inst++;
    }

    jit_new_scope(Backend);
    uint32_t scope = jit_scope(Backend);

    dr_index_vector indices_in, indices_out_all;
    dr_vector<uint32_t> state(n_inst + 1, 0);
    dr_vector<uint32_t> inst_id(n_inst, 0);

    // Wrap arguments with placeholders
    args = apply_cpp(args, py::cpp_function([&](uint32_t index) {
        uint32_t new_index = jit_var_wrap_vcall(index);
        indices_in.push_back(new_index);
        return new_index;
    }));

    detail::JitState<Backend> jit_state;
    jit_state.begin_recording();

    state[0] = jit_record_checkpoint(Backend);

    // Trace all Python functions
    py::object result;
    for (uint32_t i = 1, j = 1; i <= funcs.size(); ++i) {
        if (funcs[i-1].is_none())
            continue;

        jit_set_scope(Backend, scope);

        Mask vcall_mask = true;
        if constexpr (Backend == JitBackend::LLVM)
            vcall_mask = Mask::steal(jit_var_vcall_mask(Backend));
        jit_state.set_mask(vcall_mask.index());

        py::object result2;
        try {
            result2 = funcs[i-1](*args);
        } catch (const std::exception&) {
            // Special cleanup is necessary for interactive Python sessions
            py::object modules = py::module_::import("sys").attr("modules");
            if (modules.contains("ipykernel")) {
                for (uint32_t k = 0; k < indices_in.size(); k++)
                    jit_var_dec_ref(indices_in[k]);
            }
            throw;
        }

        // Check return type consistency
        if (result && result2 && !py::type::handle_of(result).is(py::type::handle_of(result2)))
            throw py::type_error("switch(): inconsistent return types!");
        result = result2;

        // Collect output indices
        apply_cpp(result, py::cpp_function([&](uint32_t index){
            indices_out_all.push_back(index);
        }));

        jit_state.clear_mask();

        state[j] = jit_record_checkpoint(Backend);
        inst_id[j - 1] = i;
        j++;
    }

    dr_vector<uint32_t> indices_out((uint32_t) indices_out_all.size() / n_inst, 0);

    Mask mask(true);
    uint32_t se = jit_var_vcall(
        "drjit::switch()", indices.index(), mask.index(), n_inst, inst_id.data(),
        (uint32_t) indices_in.size(), indices_in.data(),
        (uint32_t) indices_out_all.size(), indices_out_all.data(), state.data(),
        indices_out.data());

    jit_state.end_recording();
    jit_var_mark_side_effect(se);

    uint32_t offset = 0;
    return apply_cpp(result, py::cpp_function([&](uint32_t /*index*/) {
                         return indices_out[offset++];
                     }), false);
}

// ====================================================================
//                     Switch with reduced callables
// ====================================================================

template<typename UInt32>
py::object switch_reduce_impl(UInt32 indices, py::list funcs,
                              py::function gather_helper,
                              py::function zeros_helper,
                              py::function scatter_helper,
                              py::args args) {
    static constexpr JitBackend Backend = backend_v<UInt32>;
    using Mask = dr::mask_t<UInt32>;
    using DiffType = leaf_array_t<dr::float_array_t<UInt32>>;

    py::object detail = py::module_::import("drjit").attr("detail");
    auto apply_cpp = detail.attr("apply_cpp");

    // Schedule arguments, will be evaluated in jit_var_vcall_reduce
    apply_cpp(args, py::cpp_function([](uint32_t index) {
        jit_var_schedule(index);
    }));

    uint32_t n_inst = (uint32_t) funcs.size();
    VCallBucket *buckets = jit_var_vcall_reduce(Backend, nullptr,
                                                indices.index(), &n_inst);

    // Figure out the result type by tracing the first function and overwriting
    // all resulting values with zeros
    auto result = zeros_helper(funcs[0](*args));

    size_t last_size = 0;

    for (size_t i = 0; i < n_inst ; ++i) {
        if (buckets[i].id == 0 || funcs[buckets[i].id - 1].is_none())
            continue;

        UInt32 perm = UInt32::borrow(buckets[i].index);
        size_t wavefront_size = perm.size();

        MaskScope<Mask> scope(Mask::steal(
            jit_var_mask_default(Backend, (uint32_t) wavefront_size)));

        // Avoid merging multiple vcall launches if size repeats..
        if (wavefront_size != last_size) {
            last_size = wavefront_size;
        } else {
            apply_cpp(result, py::cpp_function([](uint32_t index) {
                jit_var_schedule(index);
            }));
            eval();
        }

        scatter_helper(result,
                       funcs[buckets[i].id - 1](*gather_helper(args, perm)),
                       perm);
    }

    // Schedule result
    apply_cpp(result, py::cpp_function([](uint32_t index) {
        jit_var_schedule(index);
    }));

    return result;
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
