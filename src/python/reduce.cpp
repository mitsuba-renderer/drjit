/*
    reduce.cpp -- Bindings for horizontal reduction operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "reduce.h"
#include "base.h"

nb::object all(nb::handle h) {
    nb::handle tp = h.type();
    if (tp.is(&PyBool_Type))
        return nb::borrow(h);

    if (is_drjit_array(h)) {
        const ArraySupplement &s = supp(tp);

        void *op = s.op[(int) ArrayOp::All];
        if (op == DRJIT_OP_NOT_IMPLEMENTED)
            throw nb::type_error(
                "drjit.all(): requires a Dr.Jit mask array or Python "
                "boolean sequence as input.");

        if (op != DRJIT_OP_DEFAULT) {
            nb::object result = nb::inst_alloc(tp);
            ((ArraySupplement::UnaryOp) op)(
                inst_ptr(h),
                inst_ptr(result));
            nb::inst_mark_ready(result);
            return result;
        }

        if (s.is_tensor && s.tensor_shape(inst_ptr(h)).size() <= 1)
            return all_nested(h);
    }

    nb::object result = nb::borrow(Py_True);

    size_t it = 0;
    for (nb::handle h2 : h) {
        if (it++ == 0)
            result = nb::borrow(h2);
        else
            result = result & h2;
    }

    return result;
}

nb::object any(nb::handle h) {
    nb::handle tp = h.type();
    if (tp.is(&PyBool_Type))
        return nb::borrow(h);

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        void *op = s.op[(int) ArrayOp::Any];
        if (op == DRJIT_OP_NOT_IMPLEMENTED)
            throw nb::type_error(
                "drjit.any(): requires a Dr.Jit mask array or Python "
                "boolean sequence as input.");

        if (op != DRJIT_OP_DEFAULT) {
            nb::object result = nb::inst_alloc(tp);
            ((ArraySupplement::UnaryOp) op)(
                inst_ptr(h),
                inst_ptr(result));
            nb::inst_mark_ready(result);
            return result;
        }

        if (s.is_tensor && s.tensor_shape(inst_ptr(h)).size() <= 1)
            return any_nested(h);
    }

    nb::object result = nb::borrow(Py_False);

    size_t it = 0;
    for (nb::handle h2 : h) {
        if (it++ == 0)
            result = borrow(h2);
        else
            result = result | h2;
    }


    return result;
}

nb::object all_nested(nb::handle h) {
    nb::handle tp_prev, tp_cur = h.type();

    if (is_drjit_type(tp_cur)) {
        const ArraySupplement &s = supp(tp_cur);
        if (s.is_tensor)
            return tp_cur(all(nb::steal(s.tensor_array(h.ptr()))), nb::tuple());
    }

    nb::object o = nb::borrow(h);
    do {
        tp_prev = tp_cur;
        o = all(o);
        tp_cur = o.type();
    } while (!tp_prev.is(tp_cur));

    return o;
}

nb::object any_nested(nb::handle h) {
    nb::handle tp_prev, tp_cur = h.type();

    if (is_drjit_type(tp_cur)) {
        const ArraySupplement &s = supp(tp_cur);
        if (s.is_tensor)
            return tp_cur(any(nb::steal(s.tensor_array(h.ptr()))), nb::tuple());
    }

    nb::object o = nb::borrow(h);
    do {
        tp_prev = tp_cur;
        o = any(o);
        tp_cur = o.type();
    } while (!tp_prev.is(tp_cur));

    return o;
}


void export_reduce(nb::module_ &m) {
    m.def("all", all, nb::raw_doc(doc_all));
    m.def("any", any, nb::raw_doc(doc_any));
    m.def("all_nested", all_nested, nb::raw_doc(doc_all_nested));
    m.def("any_nested", any_nested, nb::raw_doc(doc_any_nested));
}
