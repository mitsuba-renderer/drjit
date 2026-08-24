/*
    funcenv.cpp -- Extract the environment that a wrapped callable reads for @dr.freeze

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "funcenv.h"
#include <nanobind/nanobind.h>
#include <nanobind/eval.h>
#include <exception>
#include <mutex>
#include <string>

/// Stand-in for a global variable not defined at call time
static nb::handle missing_global;

/// non-owning reference to ``types.CodeType``
static nb::handle code_type;

/**
 * This class scans the bytecode of a Python function to find references to
 * globals as identified by the ``LOAD_GLOBAL`` opcode. This is far more
 * selective than ``code.co_names``, which would produce many false positives.
 *
 * The Python bytecode encoding is not stable, hence there is a chance that this
 * code will not work on future Python versions. The ``configure()`` method
 * therefore first runs a test scan against a function with a known output.
 * In the case of failure, the implementation warns and then switches to a
 * ``dis``-based fallback implementation that is roughly 1000x slower(!)
 */
struct BytecodeScan {
    long load_global = -1;  /// Opcode ID of LOAD_GLOBAL
    long extended_arg = -1; /// Opcode ID of an opcode with a large payload

    /// Right shift turning an operand into an index into ``co_names``
    uint32_t shift = 0;

    /// Set when the calibration described above succeeded
    bool valid = false;

    /// Look up the opcodes and determine the operand encoding
    void configure() {
        try { valid = probe(); } catch (...) { valid = false; }

        if (!valid &&
            PyErr_WarnEx(
                PyExc_RuntimeWarning,
                "drjit.freeze(): could not interpret the bytecode of this "
                "Python version. Finding the global variables that a "
                "frozen function reads falls back to the 'dis' module, "
                "which has the same effect but takes about a thousand "
                "times longer, once per decorated function. Please report "
                "this, as it means that this Dr.Jit binary is older than "
                "the interpreter that loaded it.", 1) < 0)
            nb::raise_python_error();
    }

    /// Configure the bytecode decoder for the current Python version or fail
    bool probe() {
        // Resolve opcode IDs
        nb::object get_fn = nb::module_::import_("opcode").attr("opmap").attr("get");
        load_global  = nb::cast<long>(get_fn("LOAD_GLOBAL"));
        extended_arg = nb::cast<long>(get_fn("EXTENDED_ARG"));

        // As of Python 3.11, the LOAD_GLOBAL opcode stores its literal at bit offset 1.
        // Instead of hardcoding behavior based on the Python version, the following
        // tests the machinery on a specific function to both infer the shift and
        // test that this is still working correctly on a new/unseen Python version.
        constexpr size_t probe_size = 129;
        std::string src = "def probe():\n    return _g0";
        for (size_t i = 1; i < probe_size; ++i)
            src += ",_g" + std::to_string(i);
        src += "\n";

        nb::dict ns;
        nb::exec(nb::str(src.c_str()), ns);
        nb::object code = nb::getattr(ns["probe"], "__code__");

        for (uint32_t s : { 1u, 0u }) {
            shift = s;
            nb::list found;
            collect(code, found);
            if (found.size() == probe_size)
                return true;
        }

        return false;
    }

    /**
     * Given a function code object, ``collect()`` steps through the bytecode
     * one word at a time, collecting globals referenced by the ``LOAD_GLOBAL``
     * opcode, and chaining ``EXTENDED_ARG`` sequences that encode larger
     * literal parameters. The interpretation of the ``LOAD_GLOBAL`` literal
     * depends on the Python version and requires a calibrated shift determined
     * by ``probe()``.
     */
    void collect(nb::handle code, nb::list &out) const {
        nb::tuple names = nb::borrow<nb::tuple>(nb::getattr(code, "co_names"));
        nb::bytes bc = nb::borrow<nb::bytes>(nb::getattr(code, "co_code"));
        const uint8_t *p = (const uint8_t *) bc.c_str();
        drjit::vector<uint8_t> seen(names.size(), 0);

        // Bits contributed by a preceding ``EXTENDED_ARG`` chain
        uint32_t ext = 0;

        // Walk through the function's byte code one opcode at a time
        for (size_t i = 0, n = bc.size() / 2; i < n; ++i) {
            long op      = p[2 * i];
            uint32_t arg = p[2 * i + 1] | ext;

            // Accumulate longer literals signaled by an ``EXTENDED_ARG`` opcode
            if (op == extended_arg) {
                ext = arg << 8;
                continue;
            }

            // In Python 3.11, the lowest bit of the LOAD_GLOBAL opcode has a special
            // meaning and must be shifted out to get the referenced global name
            uint32_t index = arg >> shift;

            if (op == load_global && index < names.size() && !seen[index]) {
                seen[index] = 1;
                out.append(names[index]);
            }

            ext = 0;
        }
    }
};

static BytecodeScan bytecode_scan;
static std::once_flag bytecode_scan_configured;

/// Use the ``dis`` module to extract global reads. This is portable but also incredibly inefficient.
static void collect_via_dis(nb::handle code, nb::list &out) {
    nb::object instructions =
        nb::module_::import_("dis").attr("get_instructions")(code);
    nb::str load_global("LOAD_GLOBAL");

    for (nb::handle i : instructions)
        if (nb::getattr(i, "opname").equal(load_global))
            out.append(nb::getattr(i, "argval"));
}

/**
 * Append the names that ``code`` and the code objects nested in it read as
 * global variables to ``out``, with duplicates.
 *
 * The recursion matters because a comprehension (before Python 3.12), a
 * generator expression, a lambda or a nested function compiles to a code
 * object of its own, which the enclosing one merely refers to as a constant.
 * A global that only such a body reads would otherwise go unnoticed, and a
 * change of it would not invalidate a recording.
 */
static void collect_globals(nb::handle code, nb::list &out) {
    if (bytecode_scan.valid)
        bytecode_scan.collect(code, out);
    else
        collect_via_dis(code, out);

    for (nb::handle c : nb::borrow<nb::tuple>(nb::getattr(code, "co_consts")))
        if (c.type().is(code_type))
            collect_globals(c, out);
}

/// Names of the positional parameters of a code object
static nb::tuple positional_arg_names(nb::handle code) {
    nb::tuple names = nb::borrow<nb::tuple>(nb::getattr(code, "co_varnames"));
    size_t n_args   = nb::cast<size_t>(nb::getattr(code, "co_argcount"));

    nb::list result;
    for (size_t i = 0; i < n_args && i < names.size(); ++i)
        result.append(names[i]);

    return nb::tuple(result);
}

/// Names of the global variables that a function reads, deduplicated
static nb::tuple captured_global_names(nb::handle code, nb::dict globals) {
    std::call_once(bytecode_scan_configured, [] { bytecode_scan.configure(); });

    nb::list loaded;
    collect_globals(code, loaded);

    nb::dict builtins = nb::builtins();

    nb::set unique;
    nb::list result;
    for (nb::handle name : loaded) {
        // A name that only resolves to a builtin cannot change
        if (!globals.contains(name) && builtins.contains(name))
            continue;
        if (!unique.contains(name)) {
            unique.add(name);
            result.append(name);
        }
    }

    return nb::tuple(result);
}

FunctionEnvironment::FunctionEnvironment(nb::callable func) : func(func) {
    nb::object code = nb::getattr(func, "__code__");

    arg_names    = positional_arg_names(code);
    globals      = nb::borrow<nb::dict>(nb::getattr(func, "__globals__"));
    global_names = captured_global_names(code, globals);
    free_names   = nb::borrow<nb::tuple>(nb::getattr(code, "co_freevars"));
    closure      = nb::getattr(func, "__closure__");
}

nb::dict FunctionEnvironment::capture() const {
    nb::dict result;

    for (nb::handle name : global_names) {
        PyObject *value = PyDict_GetItem(globals.ptr(), name.ptr());
        result[name] = value ? nb::handle(value) : missing_global;
    }

    if (!closure.is_none()) {
        size_t i = 0;
        for (nb::handle cell : nb::borrow<nb::tuple>(closure))
            result[free_names[i++]] = nb::getattr(cell, DR_STR(cell_contents));
    }

    return result;
}

int FunctionEnvironment::tp_traverse(visitproc visit, void *arg) const {
    Py_VISIT(func.ptr());
    Py_VISIT(arg_names.ptr());
    Py_VISIT(global_names.ptr());
    Py_VISIT(globals.ptr());
    Py_VISIT(free_names.ptr());
    Py_VISIT(closure.ptr());
    return 0;
}

void FunctionEnvironment::tp_clear() {
    func.reset();
    arg_names.reset();
    global_names.reset();
    globals.reset();
    free_names.reset();
    closure.reset();
}

static int function_environment_tp_traverse(PyObject *self, visitproc visit,
                                          void *arg) noexcept {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self))
        return 0;
    return nb::inst_ptr<FunctionEnvironment>(self)->tp_traverse(visit, arg);
}

static int function_environment_tp_clear(PyObject *self) noexcept {
    if (nb::inst_ready(self))
        nb::inst_ptr<FunctionEnvironment>(self)->tp_clear();
    return 0;
}

static PyType_Slot slots[] = {
    { Py_tp_traverse, (void *) function_environment_tp_traverse },
    { Py_tp_clear, (void *) function_environment_tp_clear },
    { 0, nullptr }
};

void export_funcenv(nb::module_ &m) {
    missing_global = nb::builtins()["object"]().release();
    code_type = nb::object(nb::module_::import_("types").attr("CodeType"))
                    .release();

    nb::class_<FunctionEnvironment>(m, "FunctionEnvironment", nb::type_slots(slots))
        .def(nb::init<nb::callable>(), "func"_a)
        .def_ro("arg_names", &FunctionEnvironment::arg_names)
        .def_ro("global_names", &FunctionEnvironment::global_names)
        .def_ro("free_names", &FunctionEnvironment::free_names)
        .def("capture", &FunctionEnvironment::capture);
}
