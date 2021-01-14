#include "bind.h"
#include <enoki/autodiff.h>
#include <enoki/idiv.h>
#include <enoki/loop.h>
#include <pybind11/stl.h>

extern void export_scalar(py::module_ &m);
extern void export_packet(py::module_ &m);

#if defined(ENOKI_ENABLE_JIT)
extern void export_cuda(py::module_ &m);
extern void export_llvm(py::module_ &m);
#if defined(ENOKI_ENABLE_AUTODIFF)
extern void export_cuda_ad(py::module_ &m);
extern void export_llvm_ad(py::module_ &m);
#endif
#endif

const uint32_t var_type_size[(int) VarType::Count] {
    (uint32_t) -1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 2, 4, 8
};

const bool var_type_is_float[(int) VarType::Count] {
    false, false, false, false, false, false, false, false,
    false, false, false, true, true, true
};

const bool var_type_is_unsigned[(int) VarType::Count] {
    false, false, false, true, false, true, false, true,
    false, true, true, false, false, false
};

const char* var_type_numpy[(int) VarType::Count] {
    "", "b1", "i1", "u1", "i2", "u2", "i4", "u4",
    "i8", "u8", "u8", "f2", "f4", "f8"
};

py::handle array_base, array_name, array_init, array_configure;

/// Placeholder base of all Enoki arrays in the Python domain
struct ArrayBase { };

PYBIND11_MODULE(enoki_ext, m_) {
#if defined(ENOKI_ENABLE_JIT)
    jit_set_log_level_stderr(LogLevel::Warn);
    jit_init_async((uint32_t) JitBackend::CUDA | (uint32_t) JitBackend::LLVM);
#endif

    (void) m_;
    py::module_ m = py::module_::import("enoki");

    // Look up some variables from the detail namespace
    py::module_ array_detail = (py::module_) m.attr("detail");
    array_name = array_detail.attr("array_name");
    array_init = array_detail.attr("array_init");
    array_configure = array_detail.attr("array_configure");

    m.attr("Dynamic") = ek::Dynamic;

    py::enum_<VarType>(m, "VarType", py::arithmetic())
        .value("Void", VarType::Void)
        .value("Bool", VarType::Bool)
        .value("Int8", VarType::Int8)
        .value("UInt8", VarType::UInt8)
        .value("Int16", VarType::Int16)
        .value("UInt16", VarType::UInt16)
        .value("Int32", VarType::Int32)
        .value("UInt32", VarType::UInt32)
        .value("Int64", VarType::Int64)
        .value("UInt64", VarType::UInt64)
        .value("Pointer", VarType::Pointer)
        .value("Float16", VarType::Float16)
        .value("Float32", VarType::Float32)
        .value("Float64", VarType::Float64)
        .def_property_readonly(
            "NumPy", [](VarType v) { return var_type_numpy[(int) v]; })
        .def_property_readonly(
            "Size", [](VarType v) { return var_type_size[(int) v]; });

    py::class_<ek::detail::reinterpret_flag>(array_detail, "reinterpret_flag")
        .def(py::init<>());

    array_base = py::class_<ArrayBase>(m, "ArrayBase");

    py::register_exception<enoki::Exception>(m, "Exception");
    array_detail.def("reinterpret_scalar", &reinterpret_scalar);
    array_detail.def("fmadd_scalar", [](double a, double b, double c) {
        return std::fma(a, b, c);
    });

    export_scalar(m);
    export_packet(m);

#if defined(ENOKI_ENABLE_JIT)
    export_cuda(m);
    export_llvm(m);
#if defined(ENOKI_ENABLE_AUTODIFF)
    export_cuda_ad(m);
    export_llvm_ad(m);
    m.def("ad_whos_str", &ek::ad_whos);
    m.def("ad_whos", []() { py::print(ek::ad_whos()); });
    m.def("ad_check_weights", [](bool value) { ek::ad_check_weights(value); });
#endif

    struct Scope {
        Scope(const std::string &name) : name(name) { }

        void enter() {
            #if defined(ENOKI_ENABLE_JIT)
                if (jit_has_backend(JitBackend::CUDA)) {
                    jit_prefix_push(JitBackend::CUDA, name.c_str());
                    pushed_cuda = true;
                }
                if (jit_has_backend(JitBackend::LLVM)) {
                    jit_prefix_push(JitBackend::LLVM, name.c_str());
                    pushed_llvm = true;
                }
            #endif
            #if defined(ENOKI_ENABLE_AUTODIFF)
                ek::ad_prefix_push(name.c_str());
            #endif
        }

        void exit(py::handle, py::handle, py::handle) {
            #if defined(ENOKI_ENABLE_JIT)
                if (pushed_cuda)
                    jit_prefix_pop(JitBackend::CUDA);
                if (pushed_llvm)
                    jit_prefix_pop(JitBackend::LLVM);
            #endif
            #if defined(ENOKI_ENABLE_AUTODIFF)
                ek::ad_prefix_pop();
            #endif
        }

        std::string name;
        #if defined(ENOKI_ENABLE_JIT)
            bool pushed_cuda = false;
            bool pushed_llvm = false;
        #endif
    };

    py::class_<Scope>(m, "Scope")
        .def(py::init<const std::string &>())
        .def("__enter__", &Scope::enter)
        .def("__exit__", &Scope::exit);

    py::enum_<LogLevel>(m, "LogLevel")
        .value("Disable", LogLevel::Disable)
        .value("Error", LogLevel::Error)
        .value("Warn", LogLevel::Warn)
        .value("Info", LogLevel::Info)
        .value("Debug", LogLevel::Debug)
        .value("Trace", LogLevel::Trace);
    py::implicitly_convertible<unsigned, LogLevel>();

    py::enum_<AllocType>(m, "AllocType")
        .value("Host", AllocType::Host)
        .value("HostAsync", AllocType::HostAsync)
        .value("HostPinned", AllocType::HostPinned)
        .value("Device", AllocType::Device)
        .value("Managed", AllocType::Managed)
        .value("ManagedReadMostly", AllocType::ManagedReadMostly);

    py::enum_<JitBackend>(m, "JitBackend")
        .value("CUDA", JitBackend::CUDA)
        .value("LLVM", JitBackend::LLVM);

    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("None", ReduceOp::None)
        .value("Add",  ReduceOp::Add)
        .value("Mul",  ReduceOp::Mul)
        .value("Min",  ReduceOp::Min)
        .value("Max",  ReduceOp::Max)
        .value("And",  ReduceOp::And)
        .value("Or",   ReduceOp::Or);

    py::enum_<JitFlag>(m, "JitFlag", py::arithmetic())
        .value("LoopRecord",          JitFlag::LoopRecord)
        .value("LoopOptimize",        JitFlag::LoopOptimize)
        .value("VCallRecord",         JitFlag::VCallRecord)
        .value("VCallOptimize",       JitFlag::VCallOptimize)
        .value("VCallBranch",         JitFlag::VCallBranch)
        .value("ForceOptiX",          JitFlag::ForceOptiX)
        .value("PostponeSideEffects", JitFlag::PostponeSideEffects);

    m.def("device_count", &jit_cuda_device_count);
    m.def("set_device", &jit_cuda_set_device, "device"_a);
    m.def("device", &jit_cuda_device);

    m.def("has_backend", &jit_has_backend);
    m.def("sync_thread", &jit_sync_thread);
    m.def("sync_device", &jit_sync_device);
    m.def("sync_all_devices", &jit_sync_all_devices);
    m.def("whos_str", &jit_var_whos);
    m.def("whos", []() { py::print(jit_var_whos()); });
    m.def("malloc_trim", &jit_malloc_trim);
    m.def("set_log_level", &jit_set_log_level_stderr);
    m.def("log_level", &jit_log_level_stderr);

    array_detail.def("graphviz", &jit_var_graphviz);
    array_detail.def("schedule", &jit_var_schedule);
    array_detail.def("eval", &jit_var_eval, py::call_guard<py::gil_scoped_release>());
    array_detail.def("eval", &jit_eval, py::call_guard<py::gil_scoped_release>());
    array_detail.def("to_dlpack", &to_dlpack, "owner"_a, "data"_a,
                     "type"_a, "device"_a, "shape"_a, "strides"_a);
    array_detail.def("from_dlpack", &from_dlpack);

    array_detail.def("device", &jit_cuda_device);
    array_detail.def("device", &jit_var_device);

    array_detail.def("printf_async", [](uint32_t mask_index, const char *fmt,
                                        std::vector<uint32_t> &indices) {
        jit_var_printf(JitBackend::CUDA, mask_index, fmt,
                       (uint32_t) indices.size(), indices.data());
    });

    m.def("set_flag", &jit_set_flag);
    m.def("flags", &jit_flags);

    /* Register a cleanup callback function that is invoked when
       the 'enoki::ArrayBase' Python type is garbage collected */
    py::cpp_function cleanup_callback(
        [](py::handle weakref) { py::gil_scoped_release gsr; jit_shutdown(false); }
    );

    (void) py::weakref(m.attr("ArrayBase"), cleanup_callback).release();
#else
    array_detail.def("schedule", [](uint32_t) { return false; });
    array_detail.def("eval", [](uint32_t) { return false; });
    array_detail.def("eval", []() { });
#endif

    array_detail.def("idiv", [](VarType t, py::object value) -> py::object {
        switch (t) {
            case VarType::Int32: {
                    enoki::divisor<int32_t> div(py::cast<int32_t>(value));
                    return py::make_tuple((int32_t) div.multiplier, div.shift);
                }
                break;

            case VarType::UInt32: {
                    enoki::divisor<uint32_t> div(py::cast<uint32_t>(value));
                    return py::make_tuple((uint32_t) div.multiplier, div.shift);
                }
                break;

            case VarType::Int64: {
                    enoki::divisor<int64_t> div(py::cast<int64_t>(value));
                    return py::make_tuple((int64_t) div.multiplier, div.shift);
                }
                break;

            case VarType::UInt64: {
                    enoki::divisor<uint64_t> div(py::cast<uint64_t>(value));
                    return py::make_tuple((uint64_t) div.multiplier, div.shift);
                }
                break;

            default:
                throw enoki::Exception("Unsupported integer type!");
        }
    });
}
