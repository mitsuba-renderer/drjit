/*
    while_loop.cpp -- Python implementation of drjit.while_loop() based on the
    abstract interface ad_loop() provided by the drjit-extra library

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "while_loop.h"
#include "base.h"
#include "detail.h"
#include "tracker.h"
#include <nanobind/stl/optional.h>

/**
 * \brief This data structure is responsible for capturing and updating the
 * state variables of a dr.while_loop() call and ensuring that they stay
 * consistent over time.
 * */
struct LoopState {
    /// State tuple
    nb::tuple state;
    /// Function that evaluates the loop condition
    nb::callable cond;
    /// Function that evolves the loop state
    nb::callable body;
    /// Holds a temporary reference to the loop condition
    nb::object active;
    size_t active_size;
    // Variable tracker, which monitors the evolution of 'state'
    VariableTracker tracker;

    LoopState(nb::tuple &&state, nb::callable &&cond, nb::callable &&body,
              dr::vector<dr::string> &&labels)
        : state(std::move(state)), cond(std::move(cond)), body(std::move(body)), active_size(1) {
        tracker.set_labels(VariableTracker::VariableGroup::Outputs, std::move(labels));
    }
};

/// Helper function to check that the type+size of the state variable returned
/// by 'body()' is sensible
static nb::tuple check_state(const char *name, nb::object &&o, const nb::tuple &old_state) {
    if (!o.type().is(&PyTuple_Type))
        nb::raise("the '%s' function must return a tuple", name);
    nb::tuple o_t = nb::borrow<nb::tuple>(o);
    if (nb::len(o_t) != nb::len(old_state))
        nb::raise("the '%s' function returned a tuple with an inconsistent size", name);
    return o_t;
}

/// Helper function to check that the return value of the loop conditional is sensible
static const ArraySupplement &check_cond(nb::handle h) {
    nb::handle tp = h.type();
    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if ((VarType) s.type == VarType::Bool && s.ndim == 1)
            return s;
    }

    nb::raise("the type of the loop condition ('%s') is not supported. The "
              "'cond' function must either return a Jit-compiled 1D Boolean "
              "array or a Python 'bool'.", nb::type_name(tp).c_str());
}

/// Callback functions that will be invoked by ad_loop()
static uint32_t while_loop_cond_cb(void *p) {
    nb::gil_scoped_acquire guard;
    LoopState *ls = (LoopState *) p;
    ls->active = tuple_call(ls->cond, ls->state);
    uint32_t active_index = (uint32_t) check_cond(ls->active).index(inst_ptr(ls->active));
    ls->active_size = jit_var_size(active_index);
    return active_index;
}

static void while_loop_body_cb(void *p) {
    nb::gil_scoped_acquire guard;
    LoopState *ls = (LoopState *) p;
    ls->state = check_state("body", tuple_call(ls->body, ls->state), ls->state);
};

static void while_loop_read_cb(void *p, dr::vector<uint64_t> &indices) {
    nb::gil_scoped_acquire guard;
    LoopState *ls = (LoopState *) p;
    ls->tracker.read(VariableTracker::VariableGroup::Outputs, ls->state, indices);
    ls->tracker.check_size(VariableTracker::VariableGroup::Outputs, ls->active_size);
}

static void while_loop_write_cb(void *p,
                                const dr::vector<uint64_t> &indices,
                                bool restart) {
    nb::gil_scoped_acquire guard;
    LoopState *ls = (LoopState *) p;
    if (restart) {
        ls->tracker.reset(VariableTracker::VariableGroup::Outputs);
        ls->active_size = 1;
    }
    ls->tracker.write(VariableTracker::VariableGroup::Outputs, ls->state, indices);
}

static void while_loop_delete_cb(void *p) {
    if (!nb::is_alive())
        return;
    nb::gil_scoped_acquire guard;
    delete (LoopState *) p;
}

nb::tuple while_loop(nb::tuple state, nb::callable cond, nb::callable body,
                     dr::vector<dr::string> &&labels,
                     std::optional<dr::string> name,
                     std::optional<dr::string> mode,
                     std::optional<bool> compress) {
    try {
        JitBackend backend = JitBackend::None;

        nb::object cond_val = tuple_call(cond, state);

        bool scalar_loop;
        if (mode.has_value())
            scalar_loop = mode == "scalar";
        else
            scalar_loop = cond_val.type().is(&PyBool_Type);

        if (scalar_loop) {
            // If so, process it directly
            while (nb::cast<bool>(cond_val)) {
                state = check_state("body", tuple_call(body, state), state);
                cond_val = tuple_call(cond, state);
            }

            return state;
        }

        // Temporarily stash the reference counts of inputs. This influences the
        // behavior of copy-on-write (COW) operations like dr.scatter performed
        // within the symbolic region
        dr::vector<StashRef> sr;
        stash_ref(state, sr);

        backend = (JitBackend) check_cond(cond_val).backend;
        cond_val.reset();

        // General case: call ad_loop() with a number of callbacks that
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

        LoopState *payload =
            new LoopState(std::move(state), std::move(cond), std::move(body),
                          std::move(labels));

        payload->tracker.set_check_size(!compress.has_value() || !compress.value());

        bool rv = ad_loop(backend, symbolic,
                          compress.has_value() ? (int) compress.value() : -1,
                          name_cstr, payload, while_loop_read_cb,
                          while_loop_write_cb, while_loop_cond_cb,
                          while_loop_body_cb, while_loop_delete_cb, true);

        payload->tracker.finalize();

        nb::tuple result = payload->state;

        if (rv) {
            delete payload;
        } else {
            payload->state = nb::borrow<nb::tuple>(reset(payload->state));
            payload->tracker.clear();
        }

        return result;
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.while_loop(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(
            PyExc_RuntimeError,
            "dr.while_loop(): %s. Please review the interface and assumptions "
            "of 'drjit.while_loop()' as explained in the documentation "
            "(https://drjit.readthedocs.io/en/latest/reference.html#"
            "drjit.while_loop).", e.what());
        throw nb::python_error();
    }
}

void export_while_loop(nb::module_ &m) {
    m.def("while_loop", &while_loop, "state"_a, "cond"_a, "body"_a,
          "labels"_a = nb::make_tuple(), "label"_a = nb::none(),
          "mode"_a = nb::none(), "compress"_a = nb::none(), doc_while_loop,
          // Complicated signature to type-check while_loop via TypeVarTuple
          nb::sig(
            "def while_loop(state: tuple[*Ts], "
                           "cond: typing.Callable[[*Ts], AnyArray | bool], "
                           "body: typing.Callable[[*Ts], tuple[*Ts]], "
                           "labels: typing.Sequence[str] = (), "
                           "label: str | None = None, "
                           "mode: typing.Literal['scalar', 'symbolic', 'evaluated'] | None = None, "
                           "compress: bool | None = None) "
            "-> tuple[*Ts]"
    ));
}
