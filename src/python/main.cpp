/*
    main.cpp -- Entry point of the Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "bind.h"
#include "base.h"
#include "shape.h"
#include "log.h"
#include "traits.h"
#include "scalar.h"
#include "llvm.h"
#include "reduce.h"
#include "eval.h"
#include "iter.h"
#include "init.h"
#include "memop.h"
#include "slice.h"
#include "dlpack.h"
#include "autodiff.h"
#include "inspect.h"

NB_MODULE(drjit_ext, m_) {
    (void) m_;
    nb::module_ m = nb::module_::import_("drjit");
    m.doc() = "A Just-In-Time-Compiler for Differentiable Rendering";

    export_log(m, nanobind_module_def_drjit_ext);

    uint32_t backends = 0;

#if defined(DRJIT_ENABLE_LLVM)
    backends |= (uint32_t) JitBackend::LLVM;

    nb::module_ llvm = m.def_submodule("llvm"),
                llvm_ad = llvm.def_submodule("ad");
#endif

#if defined(DRJIT_ENABLE_CUDA)
    backends |= (uint32_t) JitBackend::CUDA;

    nb::module_ cuda = m.def_submodule("cuda"),
                cuda_ad = cuda.def_submodule("ad");
#endif

    nb::enum_<JitBackend>(m, "JitBackend", doc_JitBackend)
        .value("None", JitBackend::None, doc_JitBackend_None)
        .value("CUDA", JitBackend::CUDA, doc_JitBackend_CUDA)
        .value("LLVM", JitBackend::LLVM, doc_JitBackend_LLVM);

    nb::enum_<VarType>(m, "VarType", doc_VarType)
        .value("Void", VarType::Void, doc_VarType_Void)
        .value("Bool", VarType::Bool, doc_VarType_Bool)
        .value("Int8", VarType::Int8, doc_VarType_Int8)
        .value("UInt8", VarType::UInt8, doc_VarType_UInt8)
        .value("Int16", VarType::Int16, doc_VarType_Int16)
        .value("UInt16", VarType::UInt16, doc_VarType_UInt16)
        .value("Int32", VarType::Int32, doc_VarType_Int32)
        .value("UInt32", VarType::UInt32, doc_VarType_UInt32)
        .value("Int64", VarType::Int64, doc_VarType_Int64)
        .value("UInt64", VarType::UInt64, doc_VarType_UInt64)
        .value("Pointer", VarType::Pointer, doc_VarType_Pointer)
        .value("Float16", VarType::Float16, doc_VarType_Float16)
        .value("Float32", VarType::Float32, doc_VarType_Float32)
        .value("Float64", VarType::Float64, doc_VarType_Float64);

    nb::enum_<ReduceOp>(m, "ReduceOp", doc_ReduceOp)
        .value("None", ReduceOp::None, doc_ReduceOp_None)
        .value("Add", ReduceOp::Add, doc_ReduceOp_Add)
        .value("Mul", ReduceOp::Mul, doc_ReduceOp_Mul)
        .value("Min", ReduceOp::Min, doc_ReduceOp_Min)
        .value("Max", ReduceOp::Max, doc_ReduceOp_Max)
        .value("And", ReduceOp::And, doc_ReduceOp_And)
        .value("Or", ReduceOp::Or, doc_ReduceOp_Or)
        .value("Count", ReduceOp::Count, doc_ReduceOp_Count);

    m.def("has_backend", &jit_has_backend, doc_has_backend);

    m.def("whos_str", &jit_var_whos);
    m.def("whos", []() { nb::print(jit_var_whos()); });
    m.attr("None") = nb::none();

    jit_init(backends);

    nb::module_ detail = m.attr("detail"),
                scalar = m.def_submodule("scalar");

    export_bind(detail);
    export_base(m);
    export_init(m);
    export_shape(m);
    export_traits(m);
    export_iter(detail);
    export_reduce(m);
    export_eval(m);
    export_memop(m);
    export_slice(m);
    export_dlpack(m);
    export_autodiff(m);
    export_inspect(m);

    export_scalar();

#if defined(DRJIT_ENABLE_LLVM)
    export_llvm();
    export_llvm_ad();
#endif
}
