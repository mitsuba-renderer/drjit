/*
    bind.cpp -- Central bind() function used to publish Dr.Jit type bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "bind.h"
#include "meta.h"
#include "base.h"
#include "init.h"

nb::object bind(const ArrayBinding &b) {
    const char *name = b.name;
    if (!name)
        name = meta_get_name(b);

    nb::detail::type_init_data d;

    d.flags = (uint32_t) nb::detail::type_init_flags::has_supplement |
              (uint32_t) nb::detail::type_init_flags::has_base_py |
              (uint32_t) nb::detail::type_init_flags::has_type_slots |
              (uint32_t) nb::detail::type_flags::is_final |
              (uint32_t) nb::detail::type_flags::is_destructible |
              (uint32_t) nb::detail::type_flags::is_copy_constructible |
              (uint32_t) nb::detail::type_flags::is_move_constructible;

    if (b.move) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_move;
        d.move = b.move;
    }

    if (b.copy) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_copy;
        d.copy = b.copy;
    }

    if (b.destruct) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_destruct;
        d.destruct = b.destruct;
    }

    d.align = b.talign;
    d.size = b.tsize_rel * b.talign;
    d.name = name;
    d.type = b.array_type;
    d.supplement = (uint32_t) sizeof(ArraySupplement);
    d.scope = b.scope.ptr();

    PyType_Slot slots [] = {
        { Py_tp_init, (void *) (b.is_tensor ? tp_init_tensor : tp_init_array) },
        { Py_sq_item, (void *) b.item },
        { Py_sq_ass_item, (void *) b.set_item },
        { 0, 0 }
    };

    d.type_slots = slots;
    d.type_slots_callback = nullptr;
    d.scope = meta_get_module(b).ptr();
    d.base_py = (PyTypeObject *) array_base.ptr();

    // Create the type and update its supplemental information
    nb::object tp = nb::steal(nb::detail::nb_type_new(&d));
    ArraySupplement &s = nb::type_supplement<ArraySupplement>(tp);
    s = b;

    // Register implicit cast predicate
    auto pred = [](PyTypeObject *tp_, PyObject *o,
                   nb::detail::cleanup_list *) -> bool {
        const ArraySupplement &s = supp(tp_);

        PyTypeObject *tp_o  = Py_TYPE(o),
                     *tp_t = (PyTypeObject *) s.value;

        do {
            if (tp_o == tp_t)
                return true;
            if (!is_drjit_type(tp_t))
                break;
            tp_t = (PyTypeObject *) supp(tp_t).value;
        } while (true);

        if (PyLong_CheckExact(o)) {
            VarType vt = (VarType) s.type;
            return vt == VarType::Float16 ||
                   vt == VarType::Float32 ||
                   vt == VarType::Float64;
        } else if (PySequence_Check(o)) {
            Py_ssize_t size = s.shape[0], len = PySequence_Length(o);
            if (len == -1)
                PyErr_Clear();
            return size == DRJIT_DYNAMIC || len == size;
        }
        return false;
    };

    nb::detail::implicitly_convertible(pred, b.array_type);

    VarType vt = (VarType) b.type;
    bool is_bool  = vt == VarType::Bool;
    bool is_float = vt == VarType::Float16 ||
                    vt == VarType::Float32 ||
                    vt == VarType::Float64;

    // Cache a reference to the underlying value type for use in the bindings
    nb::handle value_type_py;
    if (!b.value_type) {
        if (is_bool)
            value_type_py = &PyBool_Type;
        else if (is_float)
            value_type_py = &PyFloat_Type;
        else
            value_type_py = &PyLong_Type;
    } else {
        value_type_py = nb::detail::nb_type_lookup(b.value_type);
        if (!value_type_py.is_valid())
            nb::detail::fail(
                "nanobind.detail.bind(%s): element type '%s' not found!",
                d.type->name(), b.value_type->name());
    }
    s.value = value_type_py.ptr();

    // Cache a reference to the associated mask type for use in the bindings
    nb::handle mask_type_py;
    if (is_bool) {
        mask_type_py = tp;
    } else {
        ArrayMeta m2 = b;
        m2.type = (uint16_t) VarType::Bool;
        m2.is_vector = m2.is_complex = m2.is_quaternion =
            m2.is_matrix = false;
        mask_type_py = meta_get_type(m2);
    }
    s.mask = mask_type_py.ptr();

    // Cache a reference to the associated array type (for special types like matrices)
    nb::handle array_type_py;
    if (!s.is_tensor && !s.is_complex && !s.is_quaternion && !s.is_matrix) {
        array_type_py = tp;
    } else {
        ArrayMeta m2 = s;
        if (m2.is_tensor) {
            m2.shape[0] = DRJIT_DYNAMIC;
            m2.ndim = 1;
        }
        m2.is_vector = m2.is_complex = m2.is_quaternion = m2.is_matrix =
            m2.is_tensor = false;
        array_type_py = meta_get_type(m2);
    }

    s.array = array_type_py.ptr();

    return tp;
}


void export_bind(nb::module_ &detail) {
    nb::class_<ArrayBinding>(detail, "ArrayBinding");
    detail.def("bind", &bind);
}
