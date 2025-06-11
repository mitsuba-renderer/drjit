/*
    main.cpp -- Entry point of the Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#define NB_INTRUSIVE_EXPORT NB_EXPORT

#include <nanobind/nanobind.h>
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
#include "switch.h"
#include "while_loop.h"
#include "if_stmt.h"
#include "detail.h"
#include "print.h"
#include "texture.h"
#include "history.h"
#include "profile.h"
#include "tracker.h"
#include "local.h"
#include "resample.h"
#include "coop_vec.h"
#include "reorder.h"

static int active_backend = -1;

static void set_flag_py(JitFlag flag, bool value) {
    if (flag == JitFlag::Debug) {
        if (value)
            enable_py_tracing();
        else
            disable_py_tracing();
    }
    jit_set_flag(flag, value);
}

NB_MODULE(_drjit_ext, m_) {
    (void) m_;
    nb::module_ m = nb::module_::import_("drjit");
    m.doc() = "A Just-In-Time-Compiler for Differentiable Rendering";

    export_log(m, nanobind_module_def__drjit_ext);

    uint32_t backends = 0;

#if defined(DRJIT_ENABLE_LLVM)
    backends |= (uint32_t) JitBackend::LLVM;

    nb::module_ llvm    = nb::module_::import_("drjit.llvm"),
                llvm_ad = nb::module_::import_("drjit.llvm.ad");
#endif

#if defined(DRJIT_ENABLE_CUDA)
    backends |= (uint32_t) JitBackend::CUDA;

    nb::module_ cuda    = nb::module_::import_("drjit.cuda"),
                cuda_ad = nb::module_::import_("drjit.cuda.ad");
#endif
    nb::module_ detail = m.attr("detail"),
                scalar = nb::module_::import_("drjit.scalar");

    m.attr("__version__") = DRJIT_VERSION;

    nb::enum_<JitBackend>(m, "JitBackend", doc_JitBackend)
        .value("Invalid", JitBackend::None, doc_JitBackend_Invalid)
        .value("CUDA", JitBackend::CUDA, doc_JitBackend_CUDA)
        .value("LLVM", JitBackend::LLVM, doc_JitBackend_LLVM);

    nb::enum_<JitFlag>(m, "JitFlag", doc_JitFlag, nb::is_arithmetic())
        .value("Debug", JitFlag::Debug, doc_JitFlag_Debug)
        .value("ReuseIndices", JitFlag::ReuseIndices, doc_JitFlag_ReuseIndices)
        .value("ConstantPropagation", JitFlag::ConstantPropagation, doc_JitFlag_ConstantPropagation)
        .value("ValueNumbering", JitFlag::ValueNumbering, doc_JitFlag_ValueNumbering)
        .value("FastMath", JitFlag::FastMath, doc_JitFlag_FastMath)
        .value("SymbolicLoops", JitFlag::SymbolicLoops, doc_JitFlag_SymbolicLoops)
        .value("OptimizeLoops", JitFlag::OptimizeLoops, doc_JitFlag_OptimizeLoops)
        .value("CompressLoops", JitFlag::CompressLoops, doc_JitFlag_CompressLoops)
        .value("SymbolicCalls", JitFlag::SymbolicCalls, doc_JitFlag_SymbolicCalls)
        .value("OptimizeCalls", JitFlag::OptimizeCalls, doc_JitFlag_OptimizeCalls)
        .value("MergeFunctions", JitFlag::MergeFunctions, doc_JitFlag_MergeFunctions)
        .value("PacketOps", JitFlag::PacketOps, doc_JitFlag_PacketOps)
        .value("ForceOptiX", JitFlag::ForceOptiX, doc_JitFlag_ForceOptiX)
        .value("PrintIR", JitFlag::PrintIR, doc_JitFlag_PrintIR)
        .value("KernelHistory", JitFlag::KernelHistory, doc_JitFlag_KernelHistory)
        .value("LaunchBlocking", JitFlag::LaunchBlocking, doc_JitFlag_LaunchBlocking)
        .value("ForbidSynchronization", JitFlag::ForbidSynchronization, doc_JitFlag_ForbidSynchronization)
        .value("ScatterReduceLocal", JitFlag::ScatterReduceLocal, doc_JitFlag_ScatterReduceLocal)
        .value("SymbolicConditionals", JitFlag::SymbolicConditionals, doc_JitFlag_SymbolicConditionals)
        .value("SymbolicScope", JitFlag::SymbolicScope, doc_JitFlag_SymbolicScope)
        .value("ShaderExecutionReordering", JitFlag::ShaderExecutionReordering, doc_JitFlag_ShaderExecutionReordering)
        .value("Default", JitFlag::Default, doc_JitFlag_Default)

        // Deprecated aliases
        .value("VCallRecord", JitFlag::VCallRecord, doc_JitFlag_VCallRecord)
        .value("VCallOptimize", JitFlag::VCallOptimize, doc_JitFlag_VCallOptimize)
        .value("VCallDeduplicate", JitFlag::VCallDeduplicate, doc_JitFlag_VCallDeduplicate)
        .value("LoopRecord", JitFlag::LoopRecord, doc_JitFlag_LoopRecord)
        .value("LoopOptimize", JitFlag::LoopOptimize, doc_JitFlag_LoopOptimize)
        .value("Recording", JitFlag::Recording, doc_JitFlag_Recording);

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
        .value("Identity", ReduceOp::Identity, doc_ReduceOp_Identity)
        .value("Add", ReduceOp::Add, doc_ReduceOp_Add)
        .value("Mul", ReduceOp::Mul, doc_ReduceOp_Mul)
        .value("Min", ReduceOp::Min, doc_ReduceOp_Min)
        .value("Max", ReduceOp::Max, doc_ReduceOp_Max)
        .value("And", ReduceOp::And, doc_ReduceOp_And)
        .value("Or", ReduceOp::Or, doc_ReduceOp_Or);

    nb::enum_<ReduceMode>(m, "ReduceMode", doc_ReduceMode)
        .value("Auto", ReduceMode::Auto, doc_ReduceMode_Auto)
        .value("Direct", ReduceMode::Direct, doc_ReduceMode_Direct)
        .value("Local", ReduceMode::Local, doc_ReduceMode_Local)
        .value("NoConflicts", ReduceMode::NoConflicts, doc_ReduceMode_NoConflicts)
        .value("Permute", ReduceMode::Permute, doc_ReduceMode_Permute)
        .value("Expand", ReduceMode::Expand, doc_ReduceMode_Expand);

    nb::enum_<VarState>(m, "VarState", doc_VarState)
        .value("Invalid", VarState::Invalid, doc_VarState_Invalid)
        .value("Literal", VarState::Literal, doc_VarState_Literal)
        .value("Undefined", VarState::Undefined, doc_VarState_Undefined)
        .value("Unevaluated", VarState::Unevaluated, doc_VarState_Unevaluated)
        .value("Evaluated", VarState::Evaluated, doc_VarState_Evaluated)
        .value("Dirty", VarState::Dirty, doc_VarState_Dirty)
        .value("Symbolic", VarState::Symbolic, doc_VarState_Symbolic)
        .value("Mixed", VarState::Mixed, doc_VarState_Mixed);

    nb::enum_<dr::FilterMode>(m, "FilterMode")
        .value("Nearest", dr::FilterMode::Nearest)
        .value("Linear", dr::FilterMode::Linear);

    nb::enum_<dr::WrapMode>(m, "WrapMode")
        .value("Repeat", dr::WrapMode::Repeat)
        .value("Clamp", dr::WrapMode::Clamp)
        .value("Mirror", dr::WrapMode::Mirror);

    m.def("has_backend", &jit_has_backend, doc_has_backend);

    m.def("sync_thread", &jit_sync_thread, doc_sync_thread)
     .def("flush_kernel_cache", &jit_flush_kernel_cache, doc_flush_kernel_cache)
     .def("flush_malloc_cache", &jit_flush_malloc_cache, doc_flush_malloc_cache)
     .def("malloc_clear_statistics", &jit_malloc_clear_statistics)
     .def("thread_count", &jit_llvm_thread_count, doc_thread_count)
     .def("set_thread_count", &jit_llvm_set_thread_count, doc_set_thread_count)
     .def("expand_threshold", &jit_llvm_expand_threshold, doc_expand_threshold)
     .def("set_expand_threshold", &jit_llvm_set_expand_threshold, doc_set_expand_threshold);

    m.def("flag", [](JitFlag f) { return jit_flag(f) != 0; }, doc_flag);
    m.def("set_flag", &set_flag_py, doc_set_flag);

    struct scoped_set_flag_py {
        JitFlag flag;
        bool value, backup = false;
        scoped_set_flag_py(JitFlag flag, bool value)
            : flag(flag), value(value) { }

        void __enter__() {
            backup = jit_flag(flag);
            set_flag_py(flag, value);
        }

        void __exit__(nb::handle, nb::handle, nb::handle) {
            set_flag_py(flag, backup);
        }
    };

    nb::class_<scoped_set_flag_py>(m, "scoped_set_flag",
                                   doc_scoped_set_flag)
        .def(nb::init<JitFlag, bool>(), "flag"_a, "value"_a = true)
        .def("__enter__", &scoped_set_flag_py::__enter__)
        .def("__exit__", &scoped_set_flag_py::__exit__, nb::arg().none(),
             nb::arg().none(), nb::arg().none());

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

    nb::class_<nb::intrusive_base> ib(
        detail, "IntrusiveBase",
        nb::intrusive_ptr<nb::intrusive_base>(
            [](nb::intrusive_base *o, PyObject *po) noexcept {
                o->set_self_py(po);
            }), doc_intrusive_base);

    jit_init_async(backends);

    python_cleanup_thread_static_initialization();
    nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
        dr::sync_thread(); // Finish any ongoing Dr.Jit computations.
        python_cleanup_thread_static_shutdown();
    }));

    export_bind(detail);
    export_coop_vec(m);
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
    export_detail(m);
    export_switch(m);
    export_while_loop(m);
    export_if_stmt(m);
    export_print(m);
    export_history(m);
    export_profile(m);
    export_tracker(detail);
    export_local(m);
    export_resample(m);
    export_reorder(m);

    export_scalar(scalar);

#if defined(DRJIT_ENABLE_LLVM)
    export_llvm(llvm);
    export_llvm_ad(llvm_ad);
#endif

#if defined(DRJIT_ENABLE_CUDA)
    export_cuda(cuda);
    export_cuda_ad(cuda_ad);
#endif

    /// Automatic backend selection
    auto set_backend = [](JitBackend backend) {
        const char *key = nullptr;
        if (active_backend == (int) backend)
            return;

        switch (backend) {
            case JitBackend::None: key = "scalar"; break;
            case JitBackend::CUDA: key = "cuda"; break;
            case JitBackend::LLVM: key = "llvm"; break;
            default: nb::raise("Unknown backend");
        }

        nb::object value = array_module.attr(key);
        nb::dict mods = nb::borrow<nb::dict>(nb::module_::import_("sys").attr("modules"));
        array_module.attr("auto") = value;

        mods["drjit.auto"] = value;
        if (backend == JitBackend::None && mods.contains("drjit.auto.ad"))
            nb::del(mods["drjit.auto.ad"]);
        else
            mods["drjit.auto.ad"] = value.attr("ad");

        active_backend = (int) backend;
    };

    m.def("set_backend",
          [=](const char *name) {
              JitBackend backend;
              if (strcmp(name, "cuda") == 0)
                  backend = JitBackend::CUDA;
              else if (strcmp(name, "llvm") == 0)
                  backend = JitBackend::LLVM;
              else if (strcmp(name, "scalar") == 0)
                  backend = JitBackend::None;
              else
                  nb::raise("set_backend(): argument must equal 'cuda', 'llvm', or 'scalar'!");
              set_backend(backend);
          },
          nb::sig("def set_backend(arg: Literal['cuda', 'llvm', 'scalar'], /)"), doc_set_backend);

    m.def("set_backend", set_backend);


    nb::module_ auto_ = m.def_submodule("auto"),
                auto_ad = auto_.def_submodule("ad");

    auto_.def("__getattr__", [=](nb::handle key) -> nb::object {
        if (jit_has_backend(JitBackend::CUDA))
            set_backend(JitBackend::CUDA);
        else if (jit_has_backend(JitBackend::LLVM))
            set_backend(JitBackend::LLVM);
        else
            set_backend(JitBackend::None);
        nb::object mod = nb::module_::import_("drjit.auto");
        return nb::steal(PyObject_GetAttr(mod.ptr(), key.ptr()));
    });

    auto_ad.def("__getattr__", [=](nb::handle key) -> nb::object {
        if (jit_has_backend(JitBackend::CUDA))
            set_backend(JitBackend::CUDA);
        else if (jit_has_backend(JitBackend::LLVM))
            set_backend(JitBackend::LLVM);
        else
            set_backend(JitBackend::None);
        nb::object mod = nb::module_::import_("drjit.auto.ad");
        return nb::steal(PyObject_GetAttr(mod.ptr(), key.ptr()));
    });
}
