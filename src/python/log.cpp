/*
    log.cpp -- Log callback that prints debug output through
    Python streams

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "log.h"
#include "init.h"

static bool running_in_jupyter_notebook = false;

/// Write a log message to the Python console. The caller must hold the GIL.
void log_write_locked(int level_, const char *msg) {
    LogLevel level = (LogLevel) level_;

    // Temporarily clear error status flags, if present
    nb::error_scope guard;

    // Write 'msg' (plus a newline) to a Python stream, optionally flushing it.
    auto emit = [msg](nb::handle file, bool flush) {
        nb::object write = file.attr("write");
        write(msg);
        write("\n");
        if (flush)
            file.attr("flush")();
    };

    // Writing through Python may raise (e.g. a closed or replaced stream);
    // fall back to raw stderr on failure.
    try {
        bool is_error = level == LogLevel::Error;
        bool err_out  = is_error || level == LogLevel::Warn;

        nb::handle file = PySys_GetObject(err_out ? "stderr" : "stdout");
        emit(file, is_error);

        if (is_error) {
            // A fatal error may bring down the process; also deliver to the
            // original stderr in case sys.stderr has been intercepted (e.g. in
            // a Jupyter notebook).
            nb::handle raw = PySys_GetObject("__stderr__");
            if (!file.is(raw))
                emit(raw, true);

            if (running_in_jupyter_notebook)
                nb::module_::import_("time").attr("sleep")(0.5);
        }
    } catch (...) {
        fputs(msg, stderr);
        fputc('\n', stderr);
        fflush(stderr);
    }
}

/**
 * \brief Log callback invoked by Dr.Jit-core to print messages.
 *
 * Dr.Jit-Core calls this with the central lock held. Although it would be nice
 * to do so, we cannot call into Python from here, not even when this thread
 * already holds the GIL, because stream I/O briefly releases the GIL, which can
 * lead to deadlocks. We therefore deliver log messages from the cleanup helper
 * thread, which holds neither lock.
 */
static void log_callback(LogLevel level, const char *msg) {
    if (!nb::is_alive() || level == LogLevel::Error) {
        // Route directly to the console if Python has already shut down, or
        // when a fatal error will in any case bring down the process.
        fputs(msg, stderr);
        fputc('\n', stderr);
        fflush(stderr);
    } else {
        enqueue_python_log((int) level, msg);
    }
}

/// Defined in init.cpp
extern int drjit_py_is_alive;

void export_log(nb::module_ &m, PyModuleDef &pmd) {
    nb::dict modules = nb::borrow<nb::dict>(PySys_GetObject("modules"));
    running_in_jupyter_notebook = modules.contains("ipykernel");

    jit_set_log_level_stderr(LogLevel::Disable);
    jit_set_log_level_callback(LogLevel::Warn, log_callback);

    pmd.m_free = [](void *) {
        // Switch from the Python logger to standard stderr output
        jit_set_log_level_stderr(LogLevel::Warn);
        jit_set_log_level_callback(LogLevel::Disable, nullptr);
    };

    // Shut down the Dr.Jit component when the Python interpreter
    // has been fully wound down. Doing it above (in pmd.m_free)
    // can lead to leak warnings.
    (void) Py_AtExit([] { drjit_py_is_alive = 0; jit_shutdown(false); });

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

    m.def("log_level", &jit_log_level_callback);
}
