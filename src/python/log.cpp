/*
    log.cpp -- Log callback that prints debug output through
    Python streams

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "log.h"

/**
 * \brief Log callback that tries to print to the Python console if possible.
 *
 * It *never* tries to acquire the GIL to avoid deadlocks and instead falls
 * back to 'stderr' if needed.
 */
static void log_callback(LogLevel /* level */, const char *msg) {
#if !defined(Py_LIMITED_API)
    if (_Py_IsFinalizing()) {
        fprintf(stderr, "%s\n", msg);
        return;
    }

    if (PyGILState_Check()) {
        nb::print(msg);
        return;
    }
#endif

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


void export_log(nb::module_ &m, PyModuleDef &pmd) {
    jit_set_log_level_stderr(LogLevel::Disable);
    jit_set_log_level_callback(LogLevel::Warn, log_callback);

    pmd.m_free = [](void *) {
        /// Shut down the JIT when the module is deallocated
        jit_set_log_level_stderr(LogLevel::Warn);
        jit_set_log_level_callback(LogLevel::Disable, nullptr);
        jit_shutdown(false);
    };

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
}
