/*
    main.cpp -- Entry point of the Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <nanobind/nanobind.h>
#define NB_INTRUSIVE_EXPORT NB_EXPORT
#include <nanobind/intrusive/counter.h>
#include "bind.h"
#include "base.h"
#include "shape.h"
#include "log.h"
#include "traits.h"
#include "scalar.h"
#include "llvm.h"
#include "cuda.h"
#include "reduce.h"
#include "eval.h"
#include "iter.h"
#include "init.h"
#include "memop.h"
#include "slice.h"
#include "dlpack.h"
#include "autodiff.h"
#include "inspect.h"
#include "dispatch.h"
#include "misc.h"

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
    nb::module_ detail = m.attr("detail"),
                scalar = m.def_submodule("scalar");

    m.attr("__version__") = DRJIT_VERSION;

    nb::enum_<JitBackend>(m, "JitBackend", doc_JitBackend)
        .value("None", JitBackend::None, doc_JitBackend_None)
        .value("CUDA", JitBackend::CUDA, doc_JitBackend_CUDA)
        .value("LLVM", JitBackend::LLVM, doc_JitBackend_LLVM);

    nb::enum_<JitFlag>(m, "JitFlag", doc_JitFlag)
        .value("ConstantPropagation", JitFlag::ConstantPropagation, doc_JitFlag_ConstantPropagation)
        .value("ValueNumbering", JitFlag::ValueNumbering, doc_JitFlag_ValueNumbering)
        .value("VCallRecord", JitFlag::VCallRecord, doc_JitFlag_VCallRecord)
        .value("IndexReuse", JitFlag::IndexReuse, doc_JitFlag_IndexReuse)
        .value("Default", JitFlag::Default, doc_JitFlag_Default);
        // .value("VCallDeduplicate", JitFlag::VCallDeduplicate, doc_JitFlag_VCallDeduplicate)
        // .value("VCallOptimize", JitFlag::VCallOptimize, doc_JitFlag_VCallOptimize);

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
        .value("Or", ReduceOp::Or, doc_ReduceOp_Or);

    nb::enum_<VarState>(m, "VarState", doc_VarState)
        .value("Invalid", VarState::Invalid, doc_VarState_Invalid)
        .value("Normal", VarState::Normal, doc_VarState_Normal)
        .value("Literal", VarState::Literal, doc_VarState_Literal)
        .value("Evaluated", VarState::Evaluated, doc_VarState_Evaluated)
        .value("Symbolic", VarState::Symbolic, doc_VarState_Symbolic)
        .value("Mixed", VarState::Mixed, doc_VarState_Mixed);

    m.def("has_backend", &jit_has_backend, doc_has_backend);

    m.def("whos_str", &jit_var_whos);
    m.def("whos", []() { nb::print(jit_var_whos()); });

    struct scoped_set_flag_py {
        JitFlag flag;
        bool value, backup = false;
        scoped_set_flag_py(JitFlag flag, bool value)
            : flag(flag), value(value) { }

        void __enter__() {
            backup = jit_flag(flag);
            jit_set_flag(flag, value);
        }

        void __exit__(nb::handle, nb::handle, nb::handle) {
            jit_set_flag(flag, backup);
        }
    };

    m.def("flag", &jit_flag, doc_flag);
    m.def("set_flag", &jit_set_flag, doc_set_flag);

    nb::class_<scoped_set_flag_py>(detail, "scoped_set_flag",
                                   doc_scoped_set_flag)
        .def(nb::init<JitFlag, bool>(), "flag"_a, "value"_a = true)
        .def("__enter__", &scoped_set_flag_py::__enter__)
        .def("__exit__", &scoped_set_flag_py::__exit__, nb::arg().none(),
             nb::arg().none(), nb::arg().none());

    m.attr("None") = nb::none();

    // Intrusive reference counting
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            if (!nb::is_alive())
                return;
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });

    nb::class_<nb::intrusive_base>(
        detail, "IntrusiveBase",
        nb::intrusive_ptr<nb::intrusive_base>(
            [](nb::intrusive_base *o, PyObject *po) noexcept {
                o->set_self_py(po);
            }));

    m.def("set_thread_count", &jit_llvm_set_thread_count);

    jit_init(backends);

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
    export_misc(m);
    export_dispatch(m);

    export_scalar(scalar);

#if defined(DRJIT_ENABLE_LLVM)
    export_llvm(llvm);
    export_llvm_ad(llvm_ad);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    export_cuda(cuda);
    export_cuda_ad(cuda_ad);
#endif
}
