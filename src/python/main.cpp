/*
    main.cpp -- main entry point of the Dr.Jit Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"

// Stores a reference to 'drjit.ArrayBase'
nb::handle array_base;

// Stores a reference to the 'drjit' namespace
nb::handle array_module;

/**
 * \brief Log callback that tries to print to the Python console if possible.
 *
 * It *never* tries to acquire the GIL to avoid deadlocks and instead falls
 * back to 'stderr' if needed.
 */
static void log_callback(LogLevel /* level */, const char *msg) {
    if (_Py_IsFinalizing()) {
        fprintf(stderr, "%s\n", msg);
    } else if (PyGILState_Check()) {
        nb::print(msg);
    } else {
        char *msg_copy = NB_STRDUP(msg);
        int rv = Py_AddPendingCall(
            [](void *p) -> int {
                nb::print((char *) p);
                free(p);
                return 0;
            }, msg_copy);
        if (rv)
            nb::detail::fail("log_callback(): Py_AddPendingCall(): failed!");
    }
}

NB_MODULE(drjit_ext, m_) {
#if defined(DRJIT_ENABLE_JIT)
    jit_set_log_level_stderr(LogLevel::Disable);
    jit_set_log_level_callback(LogLevel::Warn, log_callback);
    jit_init((uint32_t) JitBackend::CUDA |
                   (uint32_t) JitBackend::LLVM);
#endif

    (void) m_;
    nb::module_ m = nb::module_::import_("drjit"),
                scalar = m.def_submodule("scalar"),
                cuda = m.def_submodule("cuda"),
                cuda_ad = cuda.def_submodule("ad"),
                llvm = m.def_submodule("llvm"),
                llvm_ad = llvm.def_submodule("ad");

    bind_array_builtin(m);
    bind_array_math(m);
    bind_array_misc(m);
    bind_traits(m);
    bind_tensor(m);

    bind_scalar(scalar);
    bind_cuda(cuda);
    bind_cuda_ad(cuda_ad);
    bind_llvm(llvm);
    bind_llvm_ad(cuda_ad);

    // Type aliases
    scalar.attr("Int") = nb::handle(&PyLong_Type);
    scalar.attr("UInt") = nb::handle(&PyLong_Type);
    scalar.attr("Int64") = nb::handle(&PyLong_Type);
    scalar.attr("UInt64") = nb::handle(&PyLong_Type);
    scalar.attr("Float") = nb::handle(&PyFloat_Type);
    scalar.attr("Float64") = nb::handle(&PyFloat_Type);

    scalar.attr("newaxis") = nb::none();

    nb::enum_<LogLevel>(m, "LogLevel")
        .value("Disable", LogLevel::Disable)
        .value("Error", LogLevel::Error)
        .value("Warn", LogLevel::Warn)
        .value("Info", LogLevel::Info)
        .value("InfoSym", LogLevel::InfoSym)
        .value("Debug", LogLevel::Debug)
        .value("Trace", LogLevel::Trace);

    m.def("set_log_level", [](LogLevel level) {
        jit_set_log_level_callback(level, log_callback);
    });

    m.def("set_log_level", [](int level) {
        jit_set_log_level_callback((LogLevel) level, log_callback);
    });

    m.def("log_level", &jit_log_level_stderr);

    nb::enum_<JitBackend>(m, "JitBackend")
        .value("CUDA", JitBackend::CUDA)
        .value("LLVM", JitBackend::LLVM);

    m.def("has_backend", &jit_has_backend);
}
