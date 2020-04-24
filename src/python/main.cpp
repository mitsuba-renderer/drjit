#include "common.h"
#include <enoki-jit/jit.h>

extern void export_route_basics(py::module &m);
extern void export_route_math(py::module &m);
extern void export_type_traits(py::module &m);
extern void export_constants(py::module &m);
extern void export_scalar(py::module &m);
extern void export_packet(py::module &m);

#if defined(ENOKI_ENABLE_JIT)
extern void export_cuda(py::module &m);
extern void export_llvm(py::module &m);
#endif

PYBIND11_MODULE(enoki, m) {
#if defined(ENOKI_ENABLE_JIT)
    jitc_init_async(1, 1);
#endif

    export_route_basics(m);
    export_route_math(m);
    export_type_traits(m);
    export_constants(m);
    export_scalar(m);
    export_packet(m);

#if defined(ENOKI_ENABLE_JIT)
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

    jitc_log_set_stderr(LogLevel::Trace);
    export_cuda(m);
    export_llvm(m);

    /* Register a cleanup callback function that is invoked when
       the 'enoki::ArrayBase' Python type is garbage collected */
    py::cpp_function cleanup_callback(
        [](py::handle weakref) { jitc_shutdown(false); }
    );

    (void) py::weakref(m.attr("ArrayBase"), cleanup_callback).release();
#endif
}
