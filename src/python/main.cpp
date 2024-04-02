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
#include <drjit/frozen.h>
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

    nb::enum_<JitFlag>(m, "JitFlag", doc_JitFlag)
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
        .value("ForceOptiX", JitFlag::ForceOptiX, doc_JitFlag_ForceOptiX)
        .value("PrintIR", JitFlag::PrintIR, doc_JitFlag_PrintIR)
        .value("KernelHistory", JitFlag::KernelHistory, doc_JitFlag_KernelHistory)
        .value("LaunchBlocking", JitFlag::LaunchBlocking, doc_JitFlag_LaunchBlocking)
        .value("ScatterReduceLocal", JitFlag::ScatterReduceLocal, doc_JitFlag_ScatterReduceLocal)
        .value("SymbolicConditionals", JitFlag::SymbolicConditionals, doc_JitFlag_SymbolicConditionals)
        .value("SymbolicScope", JitFlag::SymbolicScope, doc_JitFlag_SymbolicScope)
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
    jit_init(backends);

#if 0
    // TODO: this will need to be adapted to nanobind if kept
    m.def(
        "launch_frozen_kernel",
        [](JitBackend backend,
           uint64_t kernel_hash_low,
           uint64_t kernel_hash_high,
           const std::string &kernel_ir,
           const std::vector<VarType> &return_types,
           const std::vector<uint32_t> &inputs,
           const std::vector<std::pair<bool, uint32_t>> &kernel_slot_to_flat_pos,
           size_t size = 0) {

            if (backend == JitBackend::Invalid) {
                if (inputs.empty())
                    jit_raise("launch(): need at least one input to infer the backend");
                backend = jit_var_backend(inputs[0]);
            }
            if (backend == JitBackend::Invalid)
                jit_fail("launch_frozen_kernel(): must use a valid JIT backend");

            if (size == 0) {
                if (inputs.empty())
                    jit_raise("launch(): need at least one input to infer launch size");

                for (size_t i = 0; i < inputs.size(); ++i)
                    size = std::max(size, jit_var_size(inputs[i]));
            }

            std::vector<uint32_t> output_indices
                = dr::launch_frozen_kernel(backend,
                                           size,
                                           kernel_hash_low,
                                           kernel_hash_high,
                                           kernel_ir,
                                           return_types,
                                           inputs,
                                           kernel_slot_to_flat_pos);
            if (output_indices.size() != return_types.size()) {
                jit_fail("launch_frozen_kernel(): expected %zu outputs, found %zu",
                         return_types.size(), output_indices.size());
            }

            py::list result(output_indices.size());
            for (size_t i = 0; i < output_indices.size(); ++i) {
                size_t idx = output_indices[i];
                VarType type = return_types[i];

            // TODO: is there a better way to do this?
            // case VarType::Void: result[i] = T<void>::steal(idx);
#define SET_ARRAY(T)                                                             \
    switch (type) {                                                              \
        case VarType::Bool: result[i] = T<bool>::steal(idx); break;              \
        case VarType::Int8: result[i] = T<int8_t>::steal(idx); break;            \
        case VarType::UInt8: result[i] = T<uint8_t>::steal(idx); break;          \
        case VarType::Int16: result[i] = T<int16_t>::steal(idx); break;          \
        case VarType::UInt16: result[i] = T<uint16_t>::steal(idx); break;        \
        case VarType::Int32: result[i] = T<int32_t>::steal(idx); break;          \
        case VarType::UInt32: result[i] = T<uint32_t>::steal(idx); break;        \
        case VarType::Int64: result[i] = T<int64_t>::steal(idx); break;          \
        case VarType::UInt64: result[i] = T<uint32_t>::steal(idx); break;        \
        case VarType::Pointer: result[i] = T<void *>::steal(idx); break;         \
        case VarType::Float16: {                                                 \
            if constexpr (f16_enabled<T<float>>) {                               \
                result[i] = T<dr_half_t>::steal(idx); break;                     \
            } else {                                                             \
                jit_raise("Not support yet: float16 on LLVM backend.");          \
            }                                                                    \
        }                                                                        \
        case VarType::Float32: result[i] = T<float>::steal(idx); break;          \
        case VarType::Float64: result[i] = T<double>::steal(idx); break;         \
        default:                                                                 \
            jit_raise("launch_frozen_kernel(): unsupported result var type: %u", \
                        (uint32_t) type);                                        \
            break;                                                               \
    }

                if (backend == JitBackend::CUDA) {
                    SET_ARRAY(dr::CUDAArray);
                }
                else {
                    SET_ARRAY(dr::LLVMArray);
                }
#undef SET_ARRAY
            }

            return result;
        },
        "backend"_a,
        "kernel_hash_low"_a,
        "kernel_hash_high"_a,
        "kernel_ir"_a,
        "return_types"_a,
        "inputs"_a,
        "kernel_slot_to_flat_pos"_a,
        "size"_a = 0);
#endif



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
    export_detail(m);
    export_switch(m);
    export_while_loop(m);
    export_if_stmt(m);
    export_print(m);
    export_history(m);
    export_profile(m);
    export_tracker(detail);

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
