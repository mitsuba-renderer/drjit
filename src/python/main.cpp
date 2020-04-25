#include "bind.h"

extern void export_scalar(py::module &m);
extern void export_packet(py::module &m);

#if defined(ENOKI_ENABLE_JIT)
extern void export_cuda(py::module &m);
extern void export_llvm(py::module &m);
#endif

const uint32_t var_type_size[(int) VarType::Count] {
    (uint32_t) -1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8, 1, 8
};

py::handle array_name, array_init;

PYBIND11_MODULE(enoki_ext, m_) {
#if defined(ENOKI_ENABLE_JIT)
    jitc_log_set_stderr(LogLevel::Warn);
    jitc_init_async(1, 1);
#endif

    (void) m_;
    py::module m = py::module::import("enoki");

    // Look up some variables from the detail namespace
    py::handle array_detail = m.attr("detail");
    array_name = array_detail.attr("array_name");
    array_init = array_detail.attr("array_init");

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
        .value("Bool", VarType::Bool)
        .def_property_readonly(
            "Size", [](VarType v) { return var_type_size[(int) v]; });

    py::class_<ek::ArrayBase>(m, "ArrayBase")
        .def_property("x",
            [](const py::object &self) -> py::object {
                return self[py::int_(0)];
            },
            [](const py::object &self, const py::object &v) {
                self[py::int_(0)] = v;
            })
        .def_property("y",
            [](const py::object &self) -> py::object {
                return self[py::int_(1)];
            },
            [](const py::object &self, const py::object &v) {
                self[py::int_(1)] = v;
            })
        .def_property("z",
            [](const py::object &self) -> py::object {
                return self[py::int_(2)];
            },
            [](const py::object &self, const py::object &v) {
                self[py::int_(2)] = v;
            })
        .def_property("w",
            [](const py::object &self) -> py::object {
                return self[py::int_(3)];
            },
            [](const py::object &self, const py::object &v) {
                self[py::int_(3)] = v;
            });

    py::register_exception<enoki::Exception>(m, "Exception");

    export_scalar(m);
    export_packet(m);

#if defined(ENOKI_ENABLE_JIT)
    export_cuda(m);
    export_llvm(m);

    py::enum_<LogLevel>(m, "LogLevel")
        .value("Disable", LogLevel::Disable)
        .value("Error", LogLevel::Error)
        .value("Warn", LogLevel::Warn)
        .value("Info", LogLevel::Info)
        .value("Debug", LogLevel::Debug)
        .value("Trace", LogLevel::Trace);

    py::enum_<AllocType>(m, "AllocType")
        .value("Host", AllocType::Host)
        .value("HostAsync", AllocType::HostAsync)
        .value("HostPinned", AllocType::HostPinned)
        .value("Device", AllocType::Device)
        .value("Managed", AllocType::Managed)
        .value("ManagedReadMostly", AllocType::ManagedReadMostly);

    m.def("device_count", &jitc_device_count);
    m.def("set_device", &jitc_set_device, "device"_a, "stream"_a = 0);
    m.def("has_llvm", &jitc_has_llvm);
    m.def("has_cuda", &jitc_has_cuda);
    m.def("sync_stream", &jitc_sync_stream);
    m.def("sync_device", &jitc_sync_device);
    m.def("eval", &jitc_eval);
    m.def("parallel_dispatch", &jitc_parallel_dispatch);
    m.def("set_parallel_dispatch", &jitc_set_parallel_dispatch);
    m.def("malloc_trim", &jitc_malloc_trim);
    m.def("log_set_stderr", &jitc_log_set_stderr);
    m.def("log_stderr", &jitc_log_stderr);

    /* Register a cleanup callback function that is invoked when
       the 'enoki::ArrayBase' Python type is garbage collected */
    py::cpp_function cleanup_callback(
        [](py::handle weakref) { jitc_shutdown(false); }
    );

    (void) py::weakref(m.attr("ArrayBase"), cleanup_callback).release();
#endif
}
