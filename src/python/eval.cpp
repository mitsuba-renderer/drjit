/*
    eval.cpp -- Bindings for drjit.eval() and drjit.schedule()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "eval.h"
#include "apply.h"
#include "local.h"
#include "coop_vec.h"

bool schedule(nb::handle h) {
    bool result_ = false;

    struct ScheduleCallback : TraverseCallback {
        bool &result;
        ScheduleCallback(bool &result) : result(result) { }

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.index)
                result |= (bool) jit_var_schedule((uint32_t) s.index(inst_ptr(h))) != 0;
        }

        void traverse_unknown(nb::handle h) override {
            if (h.type().is(local_type)) {
                Local & local = nb::cast<Local&>(h);
                for (uint32_t index : local.arrays())
                    result |= (bool) jit_var_schedule(index);
            }
            if (h.type().is(coop_vector_type))
                nb::raise("Cooperative vectors cannot be evaluated. They must be unpacked into regular variables.");
        }
    };

    ScheduleCallback sc{ result_ };
    traverse("drjit.schedule", sc, h);
    return result_;
}

static bool schedule_2(nb::args args) { return schedule(args); }

static void make_opaque(nb::handle h) {
    struct ScheduleForceCallback : TraverseCallback {
        bool result = false;

        void operator()(nb::handle h) override {
            nb::handle tp = h.type();
            const ArraySupplement &s = supp(tp);
            if (!s.index)
                return;

            ArrayBase *ptr = inst_ptr(h);

            int rv = 0;
            uint64_t index = s.index(ptr),
                     index_new = ad_var_schedule_force(index, &rv);

            if (rv)
                result = true;

            if (index != index_new)
                s.reset_index(index_new, ptr);

            ad_var_dec_ref(index_new);
        }

        void traverse_unknown(nb::handle h) override {
            if (h.type().is(local_type)) {
                Local & local = nb::cast<Local&>(h);
                for (uint32_t index : local.arrays())
                    result |= (bool) jit_var_schedule(index);
            }
            if (h.type().is(coop_vector_type))
                nb::raise("Cooperative vectors cannot be evaluated. They must be unpacked into regular variables.");
        }
    };

    ScheduleForceCallback sfc;
    traverse("drjit.make_opaque", sfc, h);
    if (sfc.result) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

static void make_opaque_2(nb::args args) { return make_opaque(args); }

/**
 * \brief Return an opaque copy of a PyTree
 *
 * This rebuilds the PyTree while giving each Dr.Jit array storage of its own,
 * so that the result is guaranteed to be distinct from every other variable in
 * the system. Differentiability is preserved.
 */
nb::object opaque(nb::handle h) {
    struct OpaqueOp : TransformCallback {
        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s = supp(h1.type());

            if (!s.index) {
                nb::inst_replace_copy(h2, h1);
                return;
            }

            uint64_t index = ad_var_copy_opaque(s.index(inst_ptr(h1)));
            s.init_index(index, inst_ptr(h2));
            ad_var_dec_ref(index);
        }

        nb::object transform_unknown(nb::handle h) const override {
            if (h.type().is(coop_vector_type))
                nb::raise("Cooperative vectors cannot be evaluated. They must "
                          "be unpacked into regular variables.");
            return nb::borrow(h);
        }
    };

    if (PyType_Check(h.ptr()))
        nb::raise("drjit.opaque(): expected an array, tensor, or PyTree, but "
                  "received a type object. Did you mean to write "
                  "drjit.opaque(dtype, value, shape)?");

    eval(h);

    OpaqueOp op;
    return transform("drjit.opaque", op, h);
}

bool eval(nb::handle h) {
    if (schedule(h)) {
        nb::gil_scoped_release guard;
        jit_eval();
        return true;
    }
    return false;
}

static bool eval_2(nb::args args) {
    bool rv = schedule(args);
    if (rv || nb::len(args) == 0) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
    return rv;
}

void export_eval(nb::module_ &m) {
    m.def("schedule", &schedule, doc_schedule)
     .def("schedule", &schedule_2)
     .def("eval", &eval, doc_eval)
     .def("eval", &eval_2)
     .def("make_opaque", &make_opaque, doc_make_opaque)
     .def("make_opaque", &make_opaque_2);
}
