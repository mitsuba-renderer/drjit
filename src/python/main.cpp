#include "bind.h"
#include <enoki/autodiff.h>
#include <enoki/idiv.h>
#include <tsl/robin_map.h>

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
    (uint32_t) -1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8, 8
};

const char* var_type_numpy[(int) VarType::Count] {
    "", "?1", "b1", "B1", "i2", "u2", "i4", "u4",
    "i8", "u8", "f2", "f4", "f8", "u8"
};

py::handle array_base, array_name, array_init, array_configure;

/// Placeholder base of all Enoki arrays in the Python domain
struct ArrayBase { };

PYBIND11_MODULE(enoki_ext, m_) {
#if defined(ENOKI_ENABLE_JIT)
    jitc_set_log_level_stderr(LogLevel::Warn);
    jitc_init_async(1, 1);
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
        .value("Invalid", VarType::Invalid)
        .value("Int8", VarType::Int8)
        .value("UInt8", VarType::UInt8)
        .value("Int16", VarType::Int16)
        .value("UInt16", VarType::UInt16)
        .value("Int32", VarType::Int32)
        .value("UInt32", VarType::UInt32)
        .value("Int64", VarType::Int64)
        .value("UInt64", VarType::UInt64)
        .value("Float16", VarType::Float16)
        .value("Float32", VarType::Float32)
        .value("Float64", VarType::Float64)
        .value("Pointer", VarType::Pointer)
        .value("Bool", VarType::Bool)
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

    struct Scope {
        Scope(const std::string &name) : name(name) { }

        void enter() { ek::ad_prefix_push(name.c_str()); }
        void exit(py::handle, py::handle, py::handle) { ek::ad_prefix_pop(); }

        std::string name;
    };

    py::class_<Scope>(m, "Scope")
        .def(py::init<const std::string &>())
        .def("__enter__", &Scope::enter)
        .def("__exit__", &Scope::exit);
#endif

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

    m.def("device_count", &jitc_device_count);
    m.def("set_device", &jitc_set_device, "device"_a, "stream"_a = 0);
    m.def("stream", &jitc_stream);
    m.def("has_llvm", &jitc_has_llvm);
    m.def("has_cuda", &jitc_has_cuda);
    m.def("sync_stream", &jitc_sync_stream);
    m.def("sync_device", &jitc_sync_device);
    m.def("sync_all_devices", &jitc_sync_all_devices);
    m.def("whos_str", &jitc_var_whos);
    m.def("whos", []() { py::print(jitc_var_whos()); });
    m.def("malloc_trim", &jitc_malloc_trim);
    m.def("set_log_level", &jitc_set_log_level_stderr);
    m.def("log_level", &jitc_log_level_stderr);
    m.def("set_parallel_dispatch", &jitc_set_parallel_dispatch);
    m.def("parallel_dispatch", &jitc_parallel_dispatch);

    array_detail.def("graphviz", &jitc_var_graphviz);
    array_detail.def("schedule", &jitc_var_schedule);
    array_detail.def("eval", &jitc_var_eval, py::call_guard<py::gil_scoped_release>());
    array_detail.def("eval", &jitc_eval, py::call_guard<py::gil_scoped_release>());
    array_detail.def("to_dlpack", &to_dlpack, "owner"_a, "data"_a,
                     "type"_a, "device"_a, "shape"_a, "strides"_a);
    array_detail.def("from_dlpack", &from_dlpack);
    array_detail.def("device", &jitc_device);
    array_detail.def("device", &jitc_var_device);

    m.def("cse", &jitc_cse);
    m.def("set_cse", &jitc_set_cse);

    /* Register a cleanup callback funceion that is invoked when
       the 'enoki::ArrayBase' Python type is garbage collected */
    py::cpp_function cleanup_callback(
        [](py::handle weakref) { py::gil_scoped_release gsr; jitc_shutdown(false); }
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
                    return py::make_tuple(div.multiplier, div.shift);
                }
                break;

            case VarType::UInt32: {
                    enoki::divisor<uint32_t> div(py::cast<uint32_t>(value));
                    return py::make_tuple(div.multiplier, div.shift);
                }
                break;

            case VarType::Int64: {
                    enoki::divisor<int64_t> div(py::cast<int64_t>(value));
                    return py::make_tuple(div.multiplier, div.shift);
                }
                break;

            case VarType::UInt64: {
                    enoki::divisor<uint64_t> div(py::cast<uint64_t>(value));
                    return py::make_tuple(div.multiplier, div.shift);
                }
                break;

            default:
                throw enoki::Exception("Unsupported integer type!");
        }
    });
}
