#pragma once

#include <drjit/array.h>
#include <drjit-core/containers.h>


NAMESPACE_BEGIN(drjit)

std::vector<uint32_t> launch_frozen_kernel(JitBackend backend,
                                           size_t size,
                                           uint64_t kernel_hash_low,
                                           uint64_t kernel_hash_high,
                                           const std::string &kernel_ir,
                                           const std::vector<VarType> &return_types,
                                           const std::vector<uint32_t> &inputs,
                                           const std::vector<std::pair<bool, uint32_t>> &kernel_slot_to_flat_pos) {
    size_t n_outputs = return_types.size();
    std::vector<uint32_t> output_vars(n_outputs);
    // Create the output variables with the right types.
    for (size_t i = 0; i < n_outputs; ++i) {

      output_vars[i] = jit_var_nop(backend, return_types[i], size,
                                   /*placeholder*/ false,
                                   /* disable_lvn */ true);
    }

    std::vector<uint32_t> kernel_slot_to_var_index(inputs.size() + n_outputs);
    for (size_t i = 0; i < kernel_slot_to_flat_pos.size(); ++i) {
        auto [is_input, flat_pos] = kernel_slot_to_flat_pos[i];
        if (is_input)
            kernel_slot_to_var_index[i] = inputs[flat_pos];
        else
            kernel_slot_to_var_index[i] = output_vars[flat_pos];
    }

    uint32_t launch_var = jit_var_frozen_kernel(kernel_hash_low,
                                                kernel_hash_high,
                                                kernel_ir.c_str(),
                                                size,
                                                inputs.size(),
                                                inputs.data(),
                                                n_outputs,
                                                output_vars.data(),
                                                kernel_slot_to_var_index.data());

    jit_log(LogLevel::Debug,
            "launch_frozen_kernel(): backend = %u, size = %zu, "
            "kernel_hash_low = %zu, kernel_hash_high = %zu, "
            "%zu inputs and %zu outputs, frozen kernel variable r%u",
            (uint32_t) backend, size, kernel_hash_low, kernel_hash_high,
            inputs.size(), n_outputs, launch_var);

    return output_vars;
}

NAMESPACE_END(drjit)
