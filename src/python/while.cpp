#include "while.h"
#include "eval.h"
#include "base.h"
#include "reduce.h"
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <functional>
#include <string>

struct VarSnapshot {
    nb::object object;
    uint64_t index;
    VarSnapshot(nb::handle h) : object(nb::borrow(h)), index(0) { }
};

using Snapshot = tsl::robin_map<std::string, VarSnapshot, std::hash<std::string>>;
using Stack = std::vector<PyObject *>;

void capture_state(std::string &name, nb::handle h, Stack &stack, Snapshot &snapshot) {
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
            capture_state(name, nb::steal(s.tensor_array(h.ptr())), stack, snapshot);
            name.resize(name_size);
        } else if (s.ndim > 1) {
            Py_ssize_t len = s.shape[0];
            if (len == DRJIT_DYNAMIC)
                len = s.len(inst_ptr(h));

            for (Py_ssize_t i = 0; i < len; ++i) {
                name += "[" + std::to_string(i) + "]";
                capture_state(name, nb::steal(s.item(h.ptr(), i)), stack, snapshot);
                name.resize(name_size);
            }
        } else  {
            it.value().index = s.index(inst_ptr(h));
        }
    } else if (tp.is(&PyList_Type)) {
        size_t ctr = 0;
        for (nb::handle v: nb::borrow<nb::list>(h)) {
            name += "[" + std::to_string(ctr++) + "]";
            capture_state(name, v, stack, snapshot);
            name.resize(name_size);
        }
    } else if (tp.is(&PyTuple_Type)) {
        size_t ctr = 0;
        for (nb::handle v: nb::borrow<nb::tuple>(h)) {
            name += "[" + std::to_string(ctr++) + "]";
            capture_state(name, v, stack, snapshot);
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
            capture_state(name, v, stack, snapshot);
            name.resize(name_size);
        }
    }
    stack.pop_back();
}


nb::object while_loop(nb::object state, nb::handle cond, nb::handle step) {
    try {
        eval(state);

        nb::object active = nb::borrow(Py_True);

        Stack stack;
        Snapshot s1, s2;
        std::string state_str = "";

        while (true) {
            nb::object value = cond(state);
            active &= value;

            uint32_t active_index = 0;
            nb::handle tp = active.type();
            if (is_drjit_type(tp)) {
                const ArraySupplement &s = supp(tp);
                if ((VarType) s.type == VarType::Bool && s.ndim == 1 && (JitBackend) s.backend != JitBackend::None)
                    active_index = (uint32_t) s.index(inst_ptr(active));
            }

            nb::object reduced = any(active, std::nullopt);
            if (!nb::cast<bool>(reduced[0]))
                break;

            s1.clear();
            capture_state(state_str, state, stack, s1);

            step(state);

            s2.clear();
            capture_state(state_str, state, stack, s2);

            for (auto &kv : s1) {
                Snapshot::iterator it = s2.find(kv.first);
                if (it == s2.end())
                    nb::raise("Internal error: could not find loop state "
                              "variable '%s'.", kv.first.c_str());

                const VarSnapshot &v1 = kv.second;
                VarSnapshot &v2 = it.value();
                nb::handle tp = v1.object.type();

                if (!tp.is(v2.object.type()))
                    nb::raise("The body of this loop changed the type of loop state "
                              "variable '%s' from '%s' to '%s', which is not "
                              "permitted. Please review the Dr.Jit "
                              "documentation on loops for details.",
                              kv.first.c_str(),
                              nb::inst_name(v1.object).c_str(),
                              nb::inst_name(v2.object).c_str());

                if (v1.index == v2.index)
                    continue;

                size_t size_1 = jit_var_size((uint32_t) v1.index),
                       size_2 = jit_var_size((uint32_t) v2.index);

                if (size_1 != size_2 && size_1 != 1 && size_2 != 1)
                    nb::raise(
                        "The body of this loop changed the size of loop state "
                        "variable '%s' (which is of type '%s') from %zu to %zu, "
                        "which are not compatible. Please review the Dr.Jit "
                        "documentation on loops for details.",
                        kv.first.c_str(), nb::inst_name(v1.object).c_str(),
                        size_1, size_2);

                const ArraySupplement &s = supp(tp);
                nb::object tmp = nb::inst_alloc(tp);

                uint64_t new_index = ad_var_select(
                    active_index,
                    v2.index,
                    v1.index
                );
                s.init_index(new_index, inst_ptr(tmp));
                ad_var_dec_ref(new_index);
                nb::inst_mark_ready(tmp);
                nb::inst_replace_move(v2.object, tmp);
            }

            eval(state);
        }

        return state;
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
