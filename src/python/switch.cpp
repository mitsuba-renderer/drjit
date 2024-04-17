/*
    switch.cpp -- implementation of drjit.switch() and drjit.dispatch()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/autodiff.h>
#include "switch.h"
#include "base.h"
#include "detail.h"

/// Extract the mask parameter from a set of positional/keyword arguments
static nb::object extract_mask(nb::list &args, nb::kwargs &kwargs) {
    nb::str mask_key("active");
    nb::object mask;
    size_t argc = args.size();

    if (kwargs.contains(mask_key)) {
        mask = kwargs[mask_key];
        nb::del(kwargs[mask_key]);
        kwargs[mask_key] = mask.type()(true);
    } else if (argc > 0) {
        nb::object last = args[argc - 1];
        nb::handle last_tp = last.type();
        bool is_mask = false;

        if (is_drjit_type(last_tp)) {
            const ArraySupplement &s = supp(last_tp);
            if ((JitBackend) s.backend != JitBackend::None && s.ndim == 1 &&
                (VarType) s.type == VarType::Bool) {
                is_mask = true;
            }
        } else if (last_tp.is(&PyBool_Type)) {
            is_mask = true;
        }

        if (is_mask) {
            mask = last;
            nb::del(args[argc-1]);
            args.append(mask.type()(true));
        }
    }
    return mask;
}

static int extract_mode(nb::kwargs &kwargs) {
    nb::str mode_key("mode");
    if (!kwargs.contains(mode_key))
        return -1;

    nb::object mode = kwargs[mode_key];
    nb::del(kwargs[mode_key]);

    if (mode.is_none())
        return -1;

    if (!nb::isinstance<nb::str>(mode))
        nb::raise("'mode' must be None or of type 'str'!");

    const char *mode_str = nb::str(mode).c_str();
    if (strcmp(mode_str, "symbolic") == 0)
        return 1;
    else if (strcmp(mode_str, "evaluated") == 0)
        return 0;
    else
        nb::raise("'mode' must equal None, 'symbolic', or 'evaluated'");
}

static dr::string extract_label(nb::kwargs &kwargs, const char *def) {
    nb::str label_key("label");
    if (kwargs.contains(label_key)) {
        nb::object label = kwargs[label_key];
        nb::del(kwargs[label_key]);

        if (!label.is_none()) {
            if (!nb::isinstance<nb::str>(label))
                nb::raise("'label' must be None or of type 'str'!");
            else
                return ((nb::str) label).c_str();
        }
    }
    return def;
}

nb::object switch_impl(nb::handle index_, nb::sequence targets,
                       nb::args args_, nb::kwargs kwargs) {
    struct State {
        nb::tuple args_o;
        nb::object targets_o;
        nb::object rv_o;
    };

    try {
        nb::list args(args_);
        nb::object mask = extract_mask(args, kwargs);
        int symbolic = extract_mode(kwargs);
        dr::string label = extract_label(kwargs, "drjit.switch()");

        nb::handle index_tp = index_.type();
        if (index_tp.is(&PyLong_Type)) {
            if (mask.is_valid()) {
                bool mask_b = false;
                raise_if(!nb::try_cast(mask, mask_b),
                         "the provided 'mask' argument must be scalar if "
                         "'index' is scalar");
                if (!mask_b)
                    return nb::none();
            }

            return targets[index_](*args, **kwargs);
        }

        // Shift the func index (ad_call interprets 0 as 'disabled')
        nb::object index = index_ + nb::int_(1);
        index_tp = index.type();

        raise_if(!is_drjit_type(index_tp),
                 "the 'index' argument must be a Dr.Jit array");

        const ArraySupplement &s = supp(index_tp);
        raise_if((JitBackend) s.backend == JitBackend::None ||
                     (VarType) s.type != VarType::UInt32 || s.ndim != 1,
                 "the 'index' argument must be a Jit-compiled 1D 32-bit "
                 "unsigned integer array");

        ad_call_func func = [](void *ptr, void *self,
                               const dr::vector<uint64_t> &args_i,
                               dr::vector<uint64_t> &rv_i) {
            nb::gil_scoped_acquire guard;
            State &state = *(State *) ptr;
            state.args_o =
                nb::borrow<nb::tuple>(update_indices(state.args_o, args_i));

            uintptr_t index = (uintptr_t) self;
            nb::object result =
                state.targets_o[index](*state.args_o[0], **state.args_o[1]);

            if (state.rv_o.is_valid())
                check_compatibility(result, state.rv_o, false, "result");

            state.rv_o = std::move(result);
            ::collect_indices(state.rv_o, rv_i);
        };

        State *state =
            new State{ nb::make_tuple(args, kwargs), targets, nb::object() };

        ad_call_cleanup cleanup = [](void *ptr) {
            if (!nb::is_alive())
                return;
            nb::gil_scoped_acquire guard;
            delete (State *) ptr;
        };

        vector<uint64_t> args_i;
        dr::detail::index64_vector rv_i;
        ::collect_indices(state->args_o, args_i);

        if (mask.is_valid() && mask.type().is(&PyBool_Type)) {
            nb::type_object mask_tp = nb::borrow<nb::type_object>(s.mask);
            mask = mask_tp(mask);
        }

        bool done = ad_call(
            (JitBackend) s.backend, nullptr, symbolic, nb::len(targets),
            label.c_str(), false, (uint32_t) s.index(inst_ptr(index)),
            mask.is_valid() ? ((uint32_t) s.index(inst_ptr(mask))) : 0u, args_i,
            rv_i, state, func, cleanup, true);

        nb::object result = ::update_indices(state->rv_o, rv_i);

        if (done)
            cleanup(state);

        return result;
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "drjit.switch(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "drjit.switch(): %s", e.what());
        nb::raise_python_error();
    }
}

nb::object dispatch_impl(nb::handle_t<dr::ArrayBase> inst,
                         nb::callable target, nb::args args_,
                         nb::kwargs kwargs) {
    struct State {
        const std::type_info *type;
        nb::tuple args_o;
        nb::object target_o;
        nb::object rv_o;
        JitBackend backend;
        nb::str domain_name;

        ~State() {
            if (!nb::is_alive())
                return;
            nb::gil_scoped_acquire guard;
            args_o.reset();
            target_o.reset();
            rv_o.reset();
            domain_name.reset();
        }
    };

    const ArraySupplement &s = supp(inst.type());
    if (!s.is_class || s.ndim != 1)
        nb::raise("drjit.dispatch(): 'inst' parameter must be an instance array.");

    nb::object domain_name = nb::getattr(inst.type(), "Domain", nb::handle());
    if (!domain_name.is_valid() || !nb::isinstance<nb::str>(domain_name))
        nb::raise("drjit.dispatch(): The instance array type ('%s') lacks the "
                  "'Domain' name attribute.", nb::type_name(inst.type()).c_str());

    try {
        nb::list args(args_);
        nb::object mask = extract_mask(args, kwargs);
        int symbolic = extract_mode(kwargs);
        dr::string label = extract_label(kwargs, "drjit.dispatch()");

        ad_call_func target_cb = [](void *ptr, void *self,
                                    const dr::vector<uint64_t> &args_i,
                                    dr::vector<uint64_t> &rv_i) {
            nb::gil_scoped_acquire guard;
            State &state = *(State *) ptr;
            state.args_o =
                nb::borrow<nb::tuple>(update_indices(state.args_o, args_i));

            if (!self)
                self = jit_registry_peek(state.backend, state.domain_name.c_str());

            nb::object self_o = nb::steal(nb::detail::nb_type_put(
                state.type, self, nb::rv_policy::reference, nullptr));

            nb::object result =
                state.target_o(self_o, *state.args_o[0], **state.args_o[1]);

            if (state.rv_o.is_valid())
                check_compatibility(result, state.rv_o, false, "result");

            state.rv_o = std::move(result);
            ::collect_indices(state.rv_o, rv_i);
        };

        State *state = new State {
            &nb::type_info(s.value),
            nb::make_tuple(args, kwargs),
            target,
            nb::object(),
            (JitBackend) s.backend,
            nb::borrow<nb::str>(domain_name)
        };

        ad_call_cleanup cleanup = [](void *ptr) {
            if (!nb::is_alive())
                return;
            nb::gil_scoped_acquire guard;
            delete (State *) ptr;
        };

        vector<uint64_t> args_i;
        dr::detail::index64_vector rv_i;
        ::collect_indices(state->args_o, args_i);

        if (mask.is_valid() && mask.type().is(&PyBool_Type)) {
            nb::type_object mask_tp = nb::borrow<nb::type_object>(s.mask);
            mask = mask_tp(mask);
        }

        bool done = ad_call(
            (JitBackend) s.backend, state->domain_name.c_str(), symbolic, 0,
            label.c_str(), false, (uint32_t) s.index(inst_ptr(inst)),
            mask.is_valid() ? ((uint32_t) s.index(inst_ptr(mask))) : 0u, args_i,
            rv_i, state, target_cb, cleanup, true);

        nb::object result = ::update_indices(state->rv_o, rv_i);

        if (done)
            cleanup(state);

        return result;
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "drjit.dispatch(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "drjit.dispatch(): %s", e.what());
        nb::raise_python_error();
    }

    return nb::none();
}

void export_switch(nb::module_& m) {
    m.def("switch", &switch_impl, doc_switch, "index"_a,
          "targets"_a, "args"_a, "kwargs"_a)
     .def("dispatch", &dispatch_impl, doc_dispatch, "inst"_a,
          "target"_a, "args"_a, "kwargs"_a);
}
