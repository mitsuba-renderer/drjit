/*
    while_loop.cpp -- Python implementation of drjit.while_loop() based on the
    abstract interface ad_loop() provided by the drjit-extra library

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "while_loop.h"
#include "eval.h"
#include "base.h"
#include "reduce.h"
#include "detail.h"
#include "apply.h"
#include <nanobind/stl/optional.h>
#include <functional>

/**
 * \brief This data structure is responsible for capturing and updating the
 * state variables of a dr.while_loop() call and ensuring that they stay
 * consistent over time.
 * */
struct LoopState {
    /// State tuple
    nb::tuple state;
    /// Loop condition
    nb::callable cond;
    /// Function to evolve the loop state
    nb::callable body;
    /// Variable labels to provide nicer error messages
    dr::vector<dr::string> state_labels;

    /// Holds a temporary reference to the loop condition
    nb::object active;

    struct Entry {
        dr::string name;
        nb::handle type;
        uint64_t id;
        size_t size;
        Entry(const dr::string &name, nb::handle type, uint64_t id, size_t size)
            : name(name), type(nb::borrow(type)), id(id), size(size) { }
    };

    // Post-processed version of 'state'
    dr::vector<Entry> entries;
    /// Temporary stack to avoid infinite recursion
    dr::vector<PyObject*> stack;
    /// Temporary to assemble a per-variable name
    dr::string name;
    /// This variable is 'true' when traverse() is called for the first time
    bool first_time = true;
    /// Index into the 'entries' array when traverse() is called in later iterations
    size_t entry_pos = 0;
    /// Index into the 'indices' array when traverse() is called with <Write>
    size_t indices_pos = 0;
    /// Size of variables used by the loop
    size_t loop_size = 1;
    /// Have we already executed the loop body?
    bool body_executed = false;

    LoopState(nb::tuple &&state, nb::callable &&cond, nb::callable &&body,
              dr::vector<dr::string> &&state_labels)
        : state(std::move(state)), cond(std::move(cond)), body(std::move(body)),
          state_labels(std::move(state_labels)), first_time(true) { }

    /// Read or write the set of loop state variables
    template <bool Write, typename OutArray> void traverse(OutArray &indices) {
        size_t l1 = nb::len(state), l2 = state_labels.size();

        if (l2 && l1 != l2)
            nb::raise("the 'state' and 'state_labels' arguments have an inconsistent size.");

        if constexpr (Write)
            indices_pos = 0;

        entry_pos = 0;
        stack.clear();
        for (size_t i = 0; i < l1; ++i) {
            name = l2 ? state_labels[i] : ("arg" + dr::string(i));
            traverse<Write>(state[i], indices);
        }

        if constexpr (Write) {
            if (indices_pos != indices.size())
                nb::raise("traverse(): internal error, did not consume all indices.");
        }


        first_time = false;
    }

private:
    template <bool Write, typename OutArray> void traverse(nb::handle h, OutArray &indices) {
        if (std::find(stack.begin(), stack.end(), h.ptr()) != stack.end()) {
            PyErr_Format(
                PyExc_RecursionError,
                "Detected a cycle in field %s. This is not permitted.", name.c_str());
            nb::raise_python_error();
        }

        stack.push_back(h.ptr());
        nb::handle tp = h.type();

        size_t id;
        if (first_time) {
            id = entries.size();
            entries.emplace_back(name, tp, 0, 0);
        } else {
            id = entry_pos++;

            if (id >= entries.size())
                nb::raise("the number of loop state variables must stay "
                          "constant across iterations. However, Dr.Jit "
                          "detected a previously unobserved variable '%s' of "
                          "type '%s', which is not permitted. Please review "
                          "the interface and assumptions of dr.while_loop() as "
                          "explained in the Dr.Jit documentation.",
                          name.c_str(), nb::inst_name(h).c_str());

            Entry &e = entries[id];
            if (name != e.name)
                nb::raise(
                    "loop state variable '%s' of type '%s' created in a "
                    "previous iteration cannot be found anymore. "
                    "Instead, another variable '%s' of type '%s' was "
                    "found in its place, which is not permitted. Please "
                    "review the interface and assumptions of dr.while_loop() "
                    "as explained in the Dr.Jit documentation.", e.name.c_str(),
                    nb::type_name(e.type).c_str(), name.c_str(),
                    nb::type_name(tp).c_str());

            if (!tp.is(e.type))
                nb::raise_type_error(
                    "the body of this loop changed the type of loop state "
                    "variable '%s' from '%s' to '%s', which is not "
                    "permitted. Please review the interface and assumptions "
                    "of dr.while_loop() as explained in the Dr.Jit "
                    "documentation.",
                    name.c_str(), nb::type_name(e.type).c_str(),
                    nb::type_name(tp).c_str());
        }

        size_t name_size = name.size();
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);
            if (s.is_tensor) {
                name += ".array";
                traverse<Write>(nb::steal(s.tensor_array(h.ptr())), indices);
                name.resize(name_size);
            } else if (s.ndim > 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(h));

                for (Py_ssize_t i = 0; i < len; ++i) {
                    name += "[" + dr::string(i) + "]";
                    traverse<Write>(nb::steal(s.item(h.ptr(), i)), indices);
                    name.resize(name_size);
                }
            } else if (s.index) {
                uint64_t i1 = entries[id].id,
                         i2 = s.index(inst_ptr(h));

                size_t &s1 = entries[id].size,
                        s2 = jit_var_size((uint32_t) i2);

                if (!first_time && (s1 == 0 || s2 == 0))
                    nb::raise("loop state variable '%s' (which is of type "
                              "'%s') is uninitialized. Please review the "
                              "interface and assumptions of dr.while_loop() as "
                              "explained in the Dr.Jit documentation. %zu %zu",
                              name.c_str(), nb::inst_name(h).c_str(), s1, s2);

                if (!first_time && s1 != s2 && s1 != 1 && s2 != 1)
                    nb::raise("the body of this loop changed the size of loop "
                              "state variable '%s' (which is of type '%s') from "
                              "%zu to %zu. These sizes aren't compatible, and such "
                              "a change is therefore not permitted. Please review "
                              "the interface and assumptions of dr.while_loop() as "
                              "explained in the Dr.Jit documentation.",
                              name.c_str(), nb::inst_name(h).c_str(), s1, s2);

                if (body_executed && loop_size != s2 && i1 != i2 && !jit_var_is_dirty((uint32_t) i2)) {
                    if (loop_size != 1 && s2 != 1)
                        nb::raise("The body of this loop operates on arrays of "
                                  "size %zu. Loop state variable '%s' has an "
                                  "incompatible size %zu.",
                                  loop_size, name.c_str(), s2);
                    loop_size = std::max(loop_size, s2);
                }

                if constexpr (Write) {
                    if (indices_pos >= indices.size())
                        nb::raise("traverse(): internal error, ran out of indices.");

                    i2 = indices[indices_pos++];
                    s2 = jit_var_size((uint32_t) i2);

                    nb::handle ht = h.type();
                    nb::object tmp = nb::inst_alloc(ht);
                    supp(ht).init_index(i2, inst_ptr(tmp));
                    nb::inst_mark_ready(tmp);
                    nb::inst_replace_move(h, tmp);
                } else {
                    ad_var_inc_ref(i2);
                    indices.push_back(i2);
                }

                i1 = i2;
                s1 = s2;
            }
        } else if (tp.is(&PyList_Type)) {
            size_t ctr = 0;
            for (nb::handle v: nb::borrow<nb::list>(h)) {
                name += "[" + dr::string(ctr++) + "]";
                traverse<Write>(v, indices);
                name.resize(name_size);
            }
        } else if (tp.is(&PyTuple_Type)) {
            size_t ctr = 0;
            for (nb::handle v: nb::borrow<nb::tuple>(h)) {
                name += "[" + dr::string(ctr++) + "]";
                traverse<Write>(v, indices);
                name.resize(name_size);
            }
        } else if (tp.is(&PyDict_Type)) {
            for (nb::handle kv: nb::borrow<nb::dict>(h).items()) {
                nb::handle k = kv[0], v = kv[1];
                if (!nb::isinstance<nb::str>(k))
                    continue;
                if (stack.size() == 1)
                    name = nb::borrow<nb::str>(k).c_str();
                else
                    name += "['" + dr::string(nb::borrow<nb::str>(k).c_str()) + "']";
                traverse<Write>(v, indices);
                name.resize(name_size);
            }
        } else {
            nb::object dstruct = nb::getattr(tp, "DRJIT_STRUCT", nb::handle());
            nb::object traverse_cb = nb::getattr(
                tp, Write ? "_traverse_1_cb_rw" : "_traverse_1_cb_ro",
                nb::handle());

            if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
                for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                    name += "."; name += nb::str(k).c_str();
                    traverse<Write>(nb::getattr(h, k), indices);
                    name.resize(name_size);
                }
            } else if (traverse_cb.is_valid()) {
                nb::object cb_py = nb::cpp_function([&](uint64_t index) {
                    if constexpr (Write) {
                        if (indices_pos >= indices.size())
                            nb::raise("traverse(): internal error, "
                                      "ran out of indices.");
                        return indices[indices_pos++];
                    } else {
                        ad_var_inc_ref(index);
                        indices.push_back(index);
                    }
                });
                traverse_cb(h, cb_py);
            }
        }
        stack.pop_back();
    }
};

/// Helper function to check that the type+size of the state variable returned
/// by 'body()' is sensible
static nb::tuple check_state(const char *name, nb::object &&o, const nb::tuple &old_state) {
    if (!o.type().is(&PyTuple_Type))
        nb::raise("the '%s' function must return a tuple.", name);
    nb::tuple o_t = nb::borrow<nb::tuple>(o);
    if (nb::len(o_t) != nb::len(old_state))
        nb::raise("the '%s' function returned a tuple with an inconsistent size.", name);
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
    LoopState *lp = (LoopState *) p;
    lp->active = tuple_call(lp->cond, lp->state);
    uint32_t active_index = (uint32_t) check_cond(lp->active).index(inst_ptr(lp->active));
    lp->loop_size = jit_var_size(active_index);
    return active_index;
}

static void while_loop_body_cb(void *p) {
    nb::gil_scoped_acquire guard;
    LoopState *lp = (LoopState *) p;
    lp->state = check_state("body", tuple_call(lp->body, lp->state), lp->state);
    lp->body_executed = true;
};

static void while_loop_read_cb(void *p, dr::vector<uint64_t> &indices) {
    nb::gil_scoped_acquire guard;
    ((LoopState *) p)->traverse<false>(indices);
}

static void while_loop_write_cb(void *p,
                                const dr::vector<uint64_t> &indices,
                                bool restart) {
    nb::gil_scoped_acquire guard;
    LoopState *state = (LoopState *) p;
    if (restart)
        state->body_executed = false;
    state->traverse<true>(indices);
}

static void while_loop_delete_cb(void *p) {
    if (!nb::is_alive())
        return;
    nb::gil_scoped_acquire guard;
    delete (LoopState *) p;
}

nb::tuple while_loop(nb::tuple state, nb::callable cond, nb::callable body,
                     dr::vector<dr::string> &&state_labels,
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

        nb::object state_orig = state;

        // Temporarily stash the reference counts of inputs. This influences the
        // behavior of copy-on-write (COW) operations like dr.scatter performed
        // within the symbolic region
        dr::vector<StashRef> sr;
        stash_ref(state, sr);

        // Copy the loop inputs so that they cannot be mutated
        CopyMap copy_map;
        state = nb::borrow<nb::tuple>(copy(state, &copy_map));

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
                          std::move(state_labels));

        bool rv = ad_loop(backend, symbolic,
                          compress.has_value() ? (int) compress.value() : -1,
                          name_cstr, payload, while_loop_read_cb,
                          while_loop_write_cb, while_loop_cond_cb,
                          while_loop_body_cb, while_loop_delete_cb, true);

        // Undo the prior copy operation for unchanged parts of the PyTree
        nb::tuple result = nb::borrow<nb::tuple>(uncopy(payload->state, copy_map));

        if (rv) {
            delete payload;
        } else {
            payload->state = nb::borrow<nb::tuple>(reset(payload->state));
            payload->entries.clear();
            payload->first_time = true;
        }

        return result;
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.while_loop(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "dr.while_loop(): %s", e.what());
        throw nb::python_error();
    }
}

void export_while_loop(nb::module_ &m) {
    m.def("while_loop", &while_loop, "state"_a, "cond"_a, "body"_a,
          "state_labels"_a = nb::make_tuple(), "label"_a = nb::none(),
          "mode"_a = nb::none(), "compress"_a = nb::none(), doc_while_loop);
}
