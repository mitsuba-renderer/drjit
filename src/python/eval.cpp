/*
    eval.cpp -- Bindings for drjit.eval() and drjit.schedule()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "eval.h"
#include "apply.h"

bool schedule(nb::handle h) {
    bool result_ = false;

    struct ScheduleCallback : TraverseCallback {
        bool &result;
        ScheduleCallback(bool &result) : result(result) { }

        void operator()(nb::handle h) const override {
            const ArraySupplement &s = supp(h.type());
            if (s.index)
                result |= jit_var_schedule(s.index(inst_ptr(h))) != 0;
        }
    };

    traverse("drjit.schedule", ScheduleCallback{ result_ }, h);
    return result_;
}

static bool schedule_2(nb::args args) { return schedule(args); }

static void eval(nb::handle h) {
    if (schedule(h))
        jit_eval();
}

static bool eval_2(nb::args args) {
    bool rv = schedule(args);
    if (rv || nb::len(args) == 0)
        jit_eval();
    return rv;
}

void export_eval(nb::module_ &m) {
    m.def("schedule", &schedule, doc_schedule)
     .def("schedule", &schedule_2)
     .def("eval", &eval, doc_eval)
     .def("eval", &eval_2);
}
