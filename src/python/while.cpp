#include "while.h"
#include "eval.h"
#include "base.h"
#include "reduce.h"
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <functional>
#include <string>

struct PyVar {
    nb::object object;
    uint64_t index;

    PyVar(nb::handle h) : object(nb::borrow(h)), index(0) { }
    ~PyVar() { ad_var_dec_ref(index); }
    PyVar(PyVar &&v) : object(std::move(v.object)), index(v.index) {
        v.index = 0;
    }

    PyVar(const PyVar &) = delete;
    PyVar &operator=(const PyVar &) = delete;
    PyVar &operator=(PyVar &&v) {
        uint32_t index_old = index;
        index = v.index;
        v.index = 0;
        ad_var_dec_ref(index_old);
        object = std::move(v.object);
        return *this;
    }

    void set_index(uint64_t index_) {
        ad_var_inc_ref(index_);
        index = index_;
    }
};

using PyState = tsl::robin_map<std::string, PyVar, std::hash<std::string>>;
using Stack = std::vector<PyObject *>;

static void capture_state(std::string &name, Stack &stack, nb::handle h,
                          PyState &snapshot) {
    // Avoid infinite recursion
    if (std::find(stack.begin(), stack.end(), h.ptr()) != stack.end())
        return;

    stack.push_back(h.ptr());
    auto it = snapshot.emplace(name, nb::borrow(h)).first;

    size_t name_size = name.size();
    nb::handle tp = h.type();
    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.is_tensor) {
            name += ".array";
            capture_state(name, stack, nb::steal(s.tensor_array(h.ptr())), snapshot);
            name.resize(name_size);
        } else if (s.ndim > 1) {
            Py_ssize_t len = s.shape[0];
            if (len == DRJIT_DYNAMIC)
                len = s.len(inst_ptr(h));

            for (Py_ssize_t i = 0; i < len; ++i) {
                name += "[" + std::to_string(i) + "]";
                capture_state(name, stack, nb::steal(s.item(h.ptr(), i)), snapshot);
                name.resize(name_size);
            }
        } else  {
            uint32_t index = s.index(inst_ptr(h));
            if (!index)
                nb::raise("Loop state variable %s is uninitialized.", name.c_str());
            it.value().set_index(index);
        }
    } else if (tp.is(&PyList_Type)) {
        size_t ctr = 0;
        for (nb::handle v: nb::borrow<nb::list>(h)) {
            name += "[" + std::to_string(ctr++) + "]";
            capture_state(name, stack, v, snapshot);
            name.resize(name_size);
        }
    } else if (tp.is(&PyTuple_Type)) {
        size_t ctr = 0;
        for (nb::handle v: nb::borrow<nb::tuple>(h)) {
            name += "[" + std::to_string(ctr++) + "]";
            capture_state(name, stack, v, snapshot);
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
                name += "['" + std::string(nb::borrow<nb::str>(k).c_str()) + "']";
            capture_state(name, stack, v, snapshot);
            name.resize(name_size);
        }
    }
    stack.pop_back();
}

static PyState capture_state(nb::handle h) {
    std::string name;
    Stack stack;
    PyState snapshot;
    capture_state(name, stack, h, snapshot);
    return snapshot;
}

static void steal_and_replace(nb::handle h, uint64_t index) {
    nb::handle tp = h.type();
    nb::object tmp = nb::inst_alloc(tp);
    supp(tp).init_index(index, inst_ptr(tmp));
    nb::inst_mark_ready(tmp);
    ad_var_dec_ref(index);
    nb::inst_replace_move(h, tmp);
}

template <typename F>
static void rewrite_variables(PyState &s1, PyState &s2, const F &f) {
    for (PyState::iterator it1 = s1.begin(); it1 != s1.end(); ++it1) {
        PyState::iterator it2 = s2.find(it1->first);
        if (it2 == s2.end())
            nb::raise("Internal error: could not find loop state "
                      "variable '%s'.", it1->first.c_str());

        PyVar &v1 = it1.value(),
              &v2 = it2.value();

        if (!v1.object.type().is(v2.object.type()))
            nb::raise("The body of this loop changed the type of loop state "
                      "variable '%s' from '%s' to '%s', which is not "
                      "permitted. Please review the Dr.Jit "
                      "documentation on loops for details.",
                      it1->first.c_str(),
                      nb::inst_name(v1.object).c_str(),
                      nb::inst_name(v2.object).c_str());

        size_t size_1 = jit_var_size((uint32_t) v1.index),
               size_2 = jit_var_size((uint32_t) v2.index);

        if (size_1 != size_2 && size_1 != 1 && size_2 != 1)
            nb::raise(
                "The body of this loop changed the size of loop state "
                "variable '%s' (which is of type '%s') from %zu to %zu, "
                "which are not compatible. Please review the Dr.Jit "
                "documentation on loops for details.",
                it1->first.c_str(), nb::inst_name(v1.object).c_str(),
                size_1, size_2);

        if (v1.index)
            f(v1, v2);
    }
}

static uint32_t extract_index(nb::handle active) {
    nb::handle active_tp = active.type();
    if (is_drjit_type(active_tp)) {
        const ArraySupplement &s = supp(active_tp);
        if ((VarType) s.type == VarType::Bool && s.ndim == 1)
            return (uint32_t) s.index(inst_ptr(active));
    }

    nb::raise("The type of the loop condition ('%s') is not supported. "
              "You must either provide a 1D Dr.Jit boolean array or a "
              "Python 'bool' value.", nb::type_name(active_tp).c_str());
}

/// RAII helper to temporarily record symbolic computation
struct scoped_record {
    scoped_record(JitBackend backend) : backend(backend) {
        checkpoint = jit_record_begin(backend, nullptr);
    }

    void reset() {
        jit_record_end(backend, checkpoint);
        checkpoint = jit_record_begin(backend, nullptr);
    }

    ~scoped_record() {
        jit_record_end(backend, checkpoint);
    }

    JitBackend backend;
    uint32_t checkpoint;
};

void while_loop_symbolic(nb::object state, nb::handle cond, nb::handle step) {
    PyState s1 = capture_state(state);
    std::vector<uint32_t> indices;

    using JitVar = drjit::JitArray<JitBackend::None, void>;

    try {
        JitBackend backend = JitBackend::None;

        indices.reserve(s1.size());
        for (auto &[k, v] : s1) {
            if (v.index) {
                if (indices.empty()) {
                    backend = jit_set_backend(v.index).backend;
                }

                jit_var_inc_ref((uint32_t) v.index);
                indices.push_back((uint32_t) v.index);
            }
        }

        scoped_record record_guard(backend);

        JitVar loop =
            JitVar::steal(jit_var_loop_start(nullptr, indices.size(), indices.data()));

        // Rewrite the loop state variables
        size_t ctr = 0;
        for (auto &[k, v] : s1) {
            if (v.index)
                steal_and_replace(v.object, indices[ctr++]);
        }

        // Evaluate the loop condition
        JitVar loop_cond = JitVar::steal(
            jit_var_loop_cond(loop.index(), extract_index(cond(state))));

        PyState s2;
        do {
            // Evolve the loop state
            step(state);
            s2 = capture_state(state);

            // Ensure that modified loop state remains compatible and capture indices
            indices.clear();
            rewrite_variables(
                s1, s2,
                [&](const PyVar & /* unused */, const PyVar &v2) {
                    indices.push_back(v2.index);
                }
            );

            // Construct the loop object
            if (!jit_var_loop_end(loop.index(), loop_cond.index(), indices.data())) {
                record_guard.reset();

                // Re-run the loop recording process once more
                ctr = 0;
                for (auto &[k, v] : s2) {
                    if (v.index) {
                        jit_var_inc_ref((uint32_t) indices[ctr]);
                        steal_and_replace(v.object, indices[ctr]);
                        ctr++;
                    }
                }
                s2.clear();
                continue;
            }
            break;
        } while (true);

        // Rewrite the loop state variables
        ctr = 0;
        for (auto &[k, v] : s2) {
            if (v.index)
                steal_and_replace(v.object, indices[ctr++]);
        }

        for (auto &[k, v] : s1)
            jit_var_dec_ref((uint32_t) v.index);
    } catch (...) {
        // Restore all loop state variables to their original state
        for (auto &[k, v] : s1) {
            if (!v.index)
                continue;
            steal_and_replace(v.object, v.index);
        }
        throw;
    }
}

void while_loop_evaluated(nb::object state, nb::handle cond, nb::handle step) {
    if (jit_flag(JitFlag::Symbolic))
        nb::raise("Dr.Jit is currently recording symbolic computation and "
                  "cannot execute a loop in *evaluated mode*. You will likely "
                  "want to set ``drjit.JitFlag.SymbolicLoops`` to ``True``. "
                  "Please review the Dr.Jit documentation of both the flag "
                  "and ``drjit.while_loop()``.");

    nb::object active = nb::borrow(Py_True);
    while (true) {
        eval(state);
        nb::object value = cond(state);
        active &= value;

        uint32_t active_index = extract_index(active);

        nb::object reduced = any(active, std::nullopt);
        if (!nb::cast<bool>(reduced[0]))
            break;

        // Capture the state of all variables
        PyState s1 = capture_state(state);

        // Execute the loop body
        step(state);

        // Capture the state of all variables following execution of the loop body
        PyState s2 = capture_state(state);

        rewrite_variables(
            s1, s2,
            [active_index](PyVar &v1, PyVar &v2) {
                if (v1.index == v2.index)
                    return;

                nb::handle tp = v1.object.type();
                const ArraySupplement &s = supp(tp);
                nb::object tmp = nb::inst_alloc(tp);

                uint64_t new_index = ad_var_select(
                    active_index,
                    v2.index,
                    v1.index
                );

                s.init_index(new_index, inst_ptr(tmp));
                nb::inst_mark_ready(tmp);
                ad_var_dec_ref(new_index);
                nb::inst_replace_move(v2.object, tmp);
            }
        );
    }
}

void while_loop(nb::object state, nb::handle cond, nb::handle step) {
    try {
        // First, check if this is perhaps a scalar loop
        nb::object cond_val = cond(state);
        if (cond_val.type().is(&PyBool_Type)) {
            while (nb::cast<bool>(cond_val)) {
                step(state);
                cond_val = cond(state);
            }
        } else {
            cond_val.reset();
        }

        if (jit_flag(JitFlag::SymbolicLoops))
            while_loop_symbolic(state, cond, step);
        else
            while_loop_evaluated(state, cond, step);
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.while_loop(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "dr.while_loop(): %s", e.what());
        throw nb::python_error();
    }
}

void export_while(nb::module_ &m) {
    m.def("while_loop", &while_loop, "state"_a, "cond"_a, "step"_a);
}
