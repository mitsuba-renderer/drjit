/*
    autodiff.cpp -- Bindings for autodiff utility functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "autodiff.h"
#include "apply.h"

struct SetGradEnabled : TraverseCallback {
    bool enable;
    SetGradEnabled(bool enable) : enable(enable) { }

    void operator()(nb::handle h) override {
        const ArraySupplement &s = supp(h.type());
        if (s.is_tensor) {
            operator()(nb::steal(s.tensor_array(h.ptr())));
        } else if (s.is_diff) {
            dr::ArrayBase *p = inst_ptr(h);
            uint64_t index = s.index(p);
            bool grad_enabled = ((uint32_t) index) != index;

            if (enable != grad_enabled) {
                uint64_t new_index;
                if (enable)
                    new_index = ad_var_new((uint32_t) index);
                else
                    new_index = (uint32_t) index;

                jit_var_inc_ref((uint32_t) new_index);
                nb::inst_destruct(h);
                s.init_index(new_index, p);
                nb::inst_mark_ready(h);
                ad_var_dec_ref(new_index);
            }
        }
    }
};

static void set_grad_enabled(nb::handle h, bool enable) {
    SetGradEnabled s { enable };
    traverse("drjit.set_grad_enabled", s, h);
}

static void enable_grad(nb::handle h) { set_grad_enabled(h, true); }
static void disable_grad(nb::handle h) { set_grad_enabled(h, false); }
static void enable_grad_2(nb::args args) { enable_grad((nb::handle) args); }
static void disable_grad_2(nb::args args) { disable_grad((nb::handle) args); }

void export_autodiff(nb::module_ &m) {
    m.def("set_grad_enabled", &set_grad_enabled, doc_set_grad_enabled)
     .def("enable_grad", &enable_grad, doc_enable_grad)
     .def("enable_grad", &enable_grad_2)
     .def("disable_grad", &disable_grad, doc_disable_grad)
     .def("disable_grad", &disable_grad_2);
}
