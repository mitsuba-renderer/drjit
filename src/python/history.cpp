/*
    history.cpp -- bindings for the kernel history

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "history.h"

void export_history(nb::module_ &m) {
    nb::object io = nb::module_::import_("io").attr("StringIO");
    m.def(
        "kernel_history",
        [io](std::vector<KernelType> types) {
            KernelHistoryEntry *data  = jit_kernel_history();
            KernelHistoryEntry *entry = data;
            nb::list history;
            while (entry && (uint32_t) entry->backend) {
                bool queried_type = types.size() == 0;
                for (KernelType t : types)
                    queried_type |= t == entry->type;

                if (queried_type) {
                    nb::dict dict;
                    dict["backend"] = entry->backend;
                    dict["type"]    = entry->type;
                    if (entry->type == KernelType::JIT) {
                        char kernel_hash[33];
                        snprintf(kernel_hash, sizeof(kernel_hash), "%016llx%016llx",
                                 (unsigned long long) entry->hash[1],
                                 (unsigned long long) entry->hash[0]);
                        dict["hash"] = kernel_hash;
                        dict["ir"]   = io(entry->ir);
                        dict["uses_optix"] = entry->uses_optix;
                        dict["cache_hit"]  = entry->cache_hit;
                        dict["cache_disk"] = entry->cache_disk;
                    }
                    dict["size"]         = entry->size;
                    dict["input_count"]  = entry->input_count;
                    dict["output_count"] = entry->output_count;
                    if (entry->type == KernelType::JIT) {
                        dict["operation_count"] = entry->operation_count;
                        dict["codegen_time"]   = entry->codegen_time;
                        dict["backend_time"]   = entry->backend_time;
                    }
                    dict["execution_time"] = entry->execution_time;

                    history.append(dict);
                }

                free(entry->ir);
                entry++;
            }
            free(data);
            return history;
        },
        "types"_a = nb::list(), doc_kernel_history);

    m.def("kernel_history_clear", &jit_kernel_history_clear,
          doc_kernel_history_clear);

    nb::enum_<KernelType>(m, "KernelType")
        .value("JIT", KernelType::JIT)
        .value("Reduce", KernelType::Reduce)
        .value("CallReduce", KernelType::CallReduce)
        .value("Other", KernelType::Other);
}
