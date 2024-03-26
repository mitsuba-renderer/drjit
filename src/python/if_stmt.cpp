/*
    if_stmt.cpp -- Python implementation of drjit.if_stmt() based on the
    abstract interface ad_cond() provided by the drjit-extra library

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "if_stmt.h"
#include "detail.h"
#include "base.h"
#include "tracker.h"
#include <nanobind/stl/optional.h>
#include <drjit/autodiff.h>

/// State object passed to callbacks that implement the Python interface around ad_cond().
struct IfState {
    nb::object args, rv;
    nb::callable true_fn, false_fn;
    dr::vector<StashRef> sr;
    dr::vector<dr::string> arg_labels;
    dr::vector<dr::string> rv_labels;
    VariableTracker tracker;

    IfState(nb::object &&args, nb::callable &&true_fn, nb::callable &&false_fn,
            dr::vector<dr::string> &&arg_labels,
            dr::vector<dr::string> &&rv_labels, bool strict)
        : args(std::move(args)), true_fn(std::move(true_fn)),
          false_fn(std::move(false_fn)),
          arg_labels(std::move(arg_labels)), rv_labels(std::move(rv_labels)),
          tracker(strict) { }
};

static void if_stmt_body_cb(void *p, bool cond_val,
                            const vector<uint64_t> &args_i,
                            vector<uint64_t> &rv_i) {
    IfState *is = (IfState *) p;
    nb::gil_scoped_acquire guard;

    // Reset the input state
    is->tracker.write(is->args, args_i, true, is->arg_labels, "args");

    // Run the 'true' or 'false' branch
    is->rv = tuple_call(cond_val ? is->true_fn
                                 : is->false_fn,
                        is->args);

    if (cond_val)
        stash_ref(is->args, is->sr);

    // Collect the output state
    is->tracker.read(is->rv, rv_i, is->rv_labels, "rv");
}

static void if_stmt_delete_cb(void *p) {
    if (!nb::is_alive())
        return;
    nb::gil_scoped_acquire guard;
    delete (IfState *) p;
}

nb::object if_stmt(nb::tuple args, nb::handle cond, nb::callable true_fn,
                   nb::callable false_fn, dr::vector<dr::string> &&arg_labels,
                   dr::vector<dr::string> &&rv_labels,
                   std::optional<dr::string> name,
                   std::optional<dr::string> mode,
                   bool strict) {
    try {
        (void) rv_labels;
        JitBackend backend = JitBackend::None;
        uint32_t cond_index = 0;

        bool is_scalar;
        if (mode.has_value())
            is_scalar = mode == "scalar";
        else
            is_scalar = cond.type().is(&PyBool_Type);

        if (!is_scalar) {
            nb::handle tp = cond.type();

            if (is_drjit_type(tp)) {
                const ArraySupplement &s = supp(tp);
                if ((VarType) s.type == VarType::Bool && s.ndim == 1 &&
                    (JitBackend) s.backend != JitBackend::None) {
                    backend = (JitBackend) s.backend;
                    cond_index = (uint32_t) s.index(inst_ptr(cond));
                    if (!cond_index)
                        nb::raise("'cond' cannot be empty.");
                }
            }

            if (!cond_index)
                nb::raise("'cond' must either be a Jit-compiled 1D Boolean "
                          "array or a scalar Python 'bool'");
        }

        if (is_scalar) {
            // If so, process it directly
            if (nb::cast<bool>(cond))
                return tuple_call(true_fn, args);
            else
                return tuple_call(false_fn, args);
        }

        // General case: call ad_cond() with a number of callbacks that
        // implement an interface to Python
        int symbolic = -1;
        if (!mode.has_value())
            symbolic = -1;
        else if (mode == "symbolic")
            symbolic = 1;
        else if (mode == "evaluated")
            symbolic = 0;
        else
            nb::raise("invalid 'mode' argument (must equal None, "
                      "\"scalar\", \"symbolic\", or \"evaluated\").");

        const char *name_cstr =
            name.has_value() ? name.value().c_str() : "unnamed";

        dr::unique_ptr<IfState> is(
            new IfState(std::move(args), std::move(true_fn),
                        std::move(false_fn), std::move(arg_labels),
                        std::move(rv_labels), strict));

        // Temporarily stash the reference counts of inputs. This influences the
        // behavior of copy-on-write (COW) operations like dr.scatter performed
        // within the symbolic region
        stash_ref(is->args, is->sr);

        dr::detail::index64_vector args_i, rv_i;
        is->tracker.read(is->args, args_i, is->arg_labels, "args");

        bool all_done =
            ad_cond(backend, symbolic, name_cstr, is.get(), cond_index, args_i,
                    rv_i, if_stmt_body_cb, if_stmt_delete_cb, true);

        is->tracker.write(is->rv, rv_i, false, is->rv_labels, "rv");
        is->rv.reset();
        is->sr.clear();

        is->tracker.restore(is->arg_labels, "args");
        is->tracker.restore(is->rv_labels, "rv");

        nb::object rv = is->tracker.rebuild(is->rv_labels, "rv");

        if (!all_done) {
            is->args = ::reset(is->args);
            is->tracker.clear();
            is.release();
        }

        return rv;
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.if_stmt(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(
            PyExc_RuntimeError,
            "dr.if_stmt(): %s. Please review the interface and assumptions "
            "of 'drjit.if_stmt()' as explained in the documentation "
            "(https://drjit.readthedocs.io/en/latest/reference.html#"
            "drjit.if_stmt).", e.what());
        throw nb::python_error();
    }
}

void export_if_stmt(nb::module_ &m) {
    m.def("if_stmt", &if_stmt, "args"_a, "cond"_a, "true_fn"_a, "false_fn"_a,
          "arg_labels"_a = nb::tuple(), "rv_labels"_a = nb::tuple(),
          "label"_a = nb::none(), "mode"_a = nb::none(), "strict"_a = true,
          doc_if_stmt,
          // Complicated signature to type-check if_stmt via TypeVarTuple
          nb::sig(
            "def if_stmt(args: tuple[*Ts], "
                        "cond: AnyArray | bool, "
                        "true_fn: typing.Callable[[*Ts], T], "
                        "false_fn: typing.Callable[[*Ts], T], "
                        "arg_labels: typing.Sequence[str] = (), "
                        "rv_labels: typing.Sequence[str] = (), "
                        "label: str | None = None, "
                        "mode: typing.Literal['scalar', 'symbolic', 'evaluated', None] = None, "
                        "strict: bool = True) "
            "-> T")
    );
}
