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
#include "slice.h"
#include "traits.h"

nb::object bind(const ArrayBinding &b) {
    // Compute the name of the type if not provided
    dr::string name = b.name ? b.name : meta_get_name(b);

    VarType vt = (VarType) b.type;
    bool is_bool  = vt == VarType::Bool;
    bool is_float = vt == VarType::Float16 ||
                    vt == VarType::Float32 ||
                    vt == VarType::Float64;

    nb::object self_type_o = nb::str(name.c_str());

    // Look up the scalar type underlying the array
    nb::object scalar_type_o;
    if (is_bool)
        scalar_type_o = nb::borrow(&PyBool_Type);
    else if (is_float)
        scalar_type_o = nb::borrow(&PyFloat_Type);
    else
        scalar_type_o = nb::borrow(&PyLong_Type);

    // Look up the value type underlying the array
    nb::object value_type_o;
    if (!b.value_type) {
        value_type_o = scalar_type_o;
    } else {
        value_type_o = nb::borrow(nb::detail::nb_type_lookup(b.value_type));
        if (!value_type_o.is_valid())
            nb::detail::raise(
                "nanobind.detail.bind(\"%s\"): element type \"%s\" not found.",
                name.c_str(), b.value_type->name());
        scalar_type_o = nb::borrow(scalar_t(value_type_o));
    }

    // Look up the mask type resulting from comparisons involving this type
    nb::object mask_type_o;
    if (is_bool) {
        mask_type_o = self_type_o; // reference self
    } else {
        ArrayMeta m2 = b;
        m2.type = (uint16_t) VarType::Bool;
        m2.is_vector = m2.is_complex = m2.is_quaternion =
            m2.is_matrix = false;
        mask_type_o = nb::borrow(meta_get_type(m2));
    }

    // Determine what other types 'b' are acceptable in an arithmetic operation
    // like 'a + b' or 'a | b' so that the result clearly has type 'a'
    nb::object compat_type_o = value_type_o, tmp_o = value_type_o;
    while (is_drjit_type(tmp_o)) {
        tmp_o = nb::borrow(supp(tmp_o).value);
        compat_type_o = compat_type_o | tmp_o;
    }

    // Determine the type of reduction operations that potentially strip the outermost dimension
    nb::object reduce_type_o;
    if (b.ndim == 1 && (JitBackend) b.backend != JitBackend::None)
        reduce_type_o = self_type_o; // reference self
    else
        reduce_type_o = value_type_o;

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
    d.name = name.c_str();
    d.type = b.array_type;
    d.supplement = (uint32_t) sizeof(ArraySupplement);

    PyType_Slot slots [] = {
        { Py_tp_init, (void *) (b.is_tensor ? tp_init_tensor : tp_init_array) },
        { Py_sq_item, (void *) (b.is_tensor ? sq_item_tensor : b.item) },
        { Py_sq_ass_item, (void *) (b.is_tensor ? sq_ass_item_tensor : b.set_item) },
        { 0, 0 }
    };

    d.type_slots = slots;
    d.type_slots_callback = nullptr;

    if (b.scope.is_valid())
        d.scope = b.scope.ptr();
    else
        d.scope = meta_get_module(b).ptr();

    // Parameterize generic base class
    nb::object base_o =
        array_base[nb::make_tuple(self_type_o, value_type_o, compat_type_o,
                                  mask_type_o, reduce_type_o)];

    d.base_py = (PyTypeObject *) base_o.ptr();

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

    // Cache a reference to the associated array type (for special types like matrices)
    nb::handle array_type_o;
    if (!s.is_tensor && !s.is_complex && !s.is_quaternion && !s.is_matrix) {
        array_type_o = tp;
    } else {
        ArrayMeta m2 = s;
        if (m2.is_tensor) {
            m2.shape[0] = DRJIT_DYNAMIC;
            m2.ndim = 1;
        }
        m2.is_vector = m2.is_complex = m2.is_quaternion = m2.is_matrix =
            m2.is_tensor = false;
        array_type_o = meta_get_type(m2);

        if (s.is_tensor) {
            m2.type = (uint32_t) VarType::UInt32;
            s.tensor_index = meta_get_type(m2).ptr();
        }
    }

    s.value = value_type_o.ptr();
    s.mask = is_bool ? tp.ptr() : mask_type_o.ptr();
    s.array = array_type_o.ptr();

    return tp;
}

void export_bind(nb::module_ &detail) {
    detail.def("bind", [](void *p) { return bind(*(ArrayBinding *) p); });
}
