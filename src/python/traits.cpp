/*
    traits.cpp -- implementation of Dr.Jit type traits such as
    is_array_v, uint32_array_t, etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "traits.h"
#include "base.h"

static nb::handle scalar_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    while (is_drjit_type(tp))
        tp = supp(tp).value;
    return tp;
}

void export_traits(nb::module_ &m) {
    m.attr("Dynamic") = -1;

    m.def("is_array_v", [](nb::handle h) -> bool {
        return is_drjit_type(h.is_type() ? h : h.type());
    }, nb::raw_doc(doc_is_array_v));

    m.def("size_v", [](nb::handle h) -> Py_ssize_t {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            Py_ssize_t shape =
                supp(tp).shape[0];
            if (shape == DRJIT_DYNAMIC)
                shape = -1;
            return shape;
        } else {
            return 1;
        }
    }, nb::raw_doc(doc_size_v));

    m.def("is_jit_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            JitBackend backend =
                (JitBackend) supp(tp).backend;
            return backend != JitBackend::Invalid;
        }
        return false;
    }, nb::raw_doc(doc_is_jit_v));

    m.def("is_mask_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? (VarType) supp(tp).type == VarType::Bool
                                 : tp.is(&PyBool_Type);
    }, nb::raw_doc(doc_is_mask_v));

    m.def("is_dynamic_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);
            if (s.is_tensor)
                return true;
            for (int i = 0; i < s.ndim; ++i) {
                if (s.shape[i] == (uint8_t) dr::Dynamic)
                    return true;
            }
        }
        return false;
    }, nb::raw_doc(doc_is_dynamic_v));

    m.def("is_tensor_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).is_tensor : false;
    }, nb::raw_doc(doc_is_tensor_v));

    m.def("is_complex_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).is_complex : false;
    }, nb::raw_doc(doc_is_complex_v));

    m.def("is_quaternion_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).is_quaternion : false;
    }, nb::raw_doc(doc_is_quaternion_v));

    m.def("is_matrix_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).is_matrix : false;
    }, nb::raw_doc(doc_is_matrix_v));

    m.def("is_vector_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).is_vector : false;
    }, nb::raw_doc(doc_is_vector_v));

    m.def("depth_v", [](nb::handle h) -> size_t {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).ndim : 0;
    }, nb::raw_doc(doc_depth_v));

    m.def("value_t", [](nb::handle h) -> nb::type_object {
        nb::handle tp = h.is_type() ? h : h.type();
        return nb::borrow<nb::type_object>(
            is_drjit_type(tp) ? supp(tp).value : nb::none().type());
    }, nb::raw_doc(doc_value_t));

    m.def("mask_t", [](nb::handle h) -> nb::handle {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).mask : (PyObject *) &PyBool_Type;
    }, nb::raw_doc(doc_mask_t));

    m.def("array_t", [](nb::handle h) -> nb::handle {
        nb::handle tp = h.is_type() ? h : h.type();
        return is_drjit_type(tp) ? supp(tp).array : tp.ptr();
    }, nb::raw_doc(doc_array_t));

    m.def("scalar_t", scalar_t, nb::raw_doc(doc_scalar_t));

}
