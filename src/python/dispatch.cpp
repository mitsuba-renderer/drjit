/*
    dispatch.cpp -- implementation of drjit.switch() and drjit.dispatch()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "dispatch.h"
#include "base.h"
#include "misc.h"

struct dr_index_vector : dr::dr_vector<uint64_t> {
    using Base = dr::dr_vector<uint64_t>;
    using Base::Base;
    ~dr_index_vector() {
        for (size_t i = 0; i < size(); ++i)
            ad_var_dec_ref(operator[](i));
    }
};

nb::object switch_impl(nb::handle index_, nb::sequence callables,
                       nb::args args_, nb::kwargs kwargs) {
    try {
        // Extract the 'active' parameter, if provided.
        nb::str mask_key("active");
        nb::object mask;
        nb::list args(args_);
        size_t argc = args.size();

        if (kwargs.contains(mask_key)) {
            mask = kwargs[mask_key];
            nb::del(kwargs[mask_key]);
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
                argc--;
            }
        }

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

            return callables[index_](*args, **kwargs);
        }

        // Shift the callable index (ad_vcall interprets 0 as 'disabled')
        nb::object index = index_ + nb::int_(1);
        index_tp = index.type();

        raise_if(!is_drjit_type(index_tp),
                 "the 'index' argument must be a Dr.Jit array");

        const ArraySupplement &s = supp(index_tp);
        raise_if((JitBackend) s.backend == JitBackend::None ||
                     (VarType) s.type != VarType::UInt32 || s.ndim != 1,
                 "the 'index' argument must be a Jit-compiled 1D 32-bit "
                 "unsigned integer array");

        struct State {
            nb::tuple args_o;
            nb::object callables_o;
            nb::object rv_o;

            ~State() {
                if (!nb::is_alive())
                    return;
                nb::gil_scoped_acquire guard;
                args_o.reset();
                callables_o.reset();
                rv_o.reset();
            }
        };

        ad_vcall_callback callback = [](void *ptr, size_t index,
                                        const dr::dr_vector<uint64_t> &args_i,
                                        dr::dr_vector<uint64_t> &rv_i) {
            nb::gil_scoped_acquire guard;
            State &state = *(State *) ptr;
            state.args_o =
                nb::borrow<nb::tuple>(update_indices(state.args_o, args_i));

            nb::object result =
                state.callables_o[index](*state.args_o[0], **state.args_o[1]);

            if (state.rv_o.is_valid())
                check_compatibility(result, state.rv_o);

            state.rv_o = std::move(result);
            collect_indices(state.rv_o, rv_i);
        };

        ad_vcall_cleanup cleanup = [](void *ptr) { delete (State *) ptr; };

        State *state = new State{ nb::make_tuple(args, kwargs), callables,
                                  nb::object() };

        dr_index_vector rv_i;
        bool done = ad_vcall(
            (JitBackend) s.backend, nullptr, "dr.switch()",
            (uint32_t) s.index(inst_ptr(index)),
            mask.is_valid() ? ((uint32_t) s.index(inst_ptr(mask))) : 0u,
            nb::len(callables), collect_indices(state->args_o), rv_i, state,
            callback, cleanup, true);

        nb::object result = update_indices(state->rv_o, rv_i);

        if (done)
            cleanup(state);
        else
            state->rv_o.reset();

        return result;
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "drjit.switch(): encountered an exception (see above)!");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "drjit.switch(): %s!", e.what());
        nb::raise_python_error();
    }
}

void export_dispatch(nb::module_&m) {
    m.def("switch", &switch_impl, nb::raw_doc(doc_switch), "index"_a,
          "callables"_a, "args"_a, "kwargs"_a);
}
