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

// Return the name of a type alias storing types that are compatible with 'tp'
nb::object compat_type(nb::handle module_name, nb::handle tp) {
    if (!is_drjit_type(tp))
        return nb::borrow(tp);
    nb::object tp_module_name = tp.attr("__module__"),
               tp_name = tp.attr("__name__");

    nb::object result = nb::str("_") + tp_name + nb::str("Cp");
    if (nb::borrow(module_name).not_equal(tp_module_name))
        result = tp_module_name + nb::str(".") + result;
    return result;
}

// Create a union representing that are compatible with a given array
nb::object compat_types(nb::handle module_name, nb::handle self_name, nb::handle value_t, ArrayMeta m) {
    nb::list tp_list;
    tp_list.append(self_name);
    tp_list.append(is_drjit_type(value_t) ? compat_type(module_name, value_t)
                                          : nb::borrow(value_t));

    if ((JitBackend) m.backend != JitBackend::None && m.ndim > 0) {
        ArrayMeta m2 = m;
        m2.backend = (uint16_t) JitBackend::None;
        m2.ndim -= 1;
        tp_list.append(compat_type(module_name, meta_get_type(m2)));
    }

    if (m.is_diff) {
        ArrayMeta m2 = m;
        m2.is_diff = false;
        tp_list.append(compat_type(module_name, meta_get_type(m2)));
    }

    VarType vt;
    switch ((VarType) m.type) {
        case VarType::Float64: vt = VarType::Float32; break;
        case VarType::Float32: vt = VarType::Float16; break;
        case VarType::Float16: vt = VarType::UInt64; break;
        case VarType::UInt64:  vt = VarType::Int64; break;
        case VarType::Int64:   vt = VarType::UInt32; break;
        case VarType::UInt32:  vt = VarType::Int32; break;
        case VarType::Int32:   vt = VarType::Bool; break;
        default: vt = VarType::Void;
    }

    if (vt != VarType::Void) {
        ArrayMeta m2 = m;
        m2.type = (uint16_t) vt;
        nb::handle compat = meta_get_type(m2, false);
        if (compat.is_valid())
            tp_list.append(compat_type(module_name, compat));
    }

    return nb::module_::import_("typing").attr("Union")[nb::tuple(tp_list)];
}

nb::object bind(const ArrayBinding &b) {
    VarType vt = (VarType) b.type;

    // Compute the type name if not ready provided
    dr::string name = b.name ? b.name : meta_get_name(b);
    nb::object name_o = nb::str(name.c_str());

    // Determine where to install the type if not already provided
    nb::handle scope;
    if (b.scope.is_valid())
        scope = b.scope.ptr();
    else
        scope = meta_get_module(b).ptr();

    nb::object module_name = scope.attr("__name__");

    // Compute the generic parameters of ArrayBase. This is partly needed for
    // type checking, and partly to populate metadata ("array supplement") that
    // we embed into the created type object.

    // Look up the scalar type underlying the array
    nb::object scalar_t_o;
    if (vt == VarType::Bool)
        scalar_t_o = nb::borrow(&PyBool_Type);
    else if (is_float(b))
        scalar_t_o = nb::borrow(&PyFloat_Type);
    else
        scalar_t_o = nb::borrow(&PyLong_Type);

    // ``ValT``: the *value type* (i.e., the type of ``self[0]``)
    nb::object val_t_o;
    if (!b.value_type) {
        val_t_o = scalar_t_o;
    } else {
        val_t_o = nb::borrow(nb::detail::nb_type_lookup(b.value_type));
        if (!val_t_o.is_valid())
            nb::detail::raise(
                "nanobind.detail.bind(\"%s\"): element type \"%s\" not found.",
                name.c_str(), b.value_type->name());
        scalar_t_o = nb::borrow(scalar_t(val_t_o));
    }

    /// - ``SelfCpT`` and ``ValCpT``: types compatible with '' and 'ValT'
    nb::object self_cp_t_o = nb::str("_") + name_o + nb::str("Cp"),
               val_cp_t_o  = compat_type(module_name, val_t_o);
    scope.attr(self_cp_t_o) = compat_types(module_name, name_o, val_t_o, b);

    // - ``RedT``: type following reduction by 'dr.sum' or 'dr.all'
    nb::object red_t_o;
    if (b.ndim == 1 && (JitBackend) b.backend != JitBackend::None)
        red_t_o = name_o; // reference self
    else
        red_t_o = val_t_o;

    // - ``PlainT``: plain array type for special types like matrices
    nb::object plain_t_o;
    bool is_special = b.is_tensor || b.is_complex || b.is_quaternion || b.is_matrix;
    if (!is_special) {
        plain_t_o = name_o;
    } else {
        ArrayMeta m2 = b;
        if (m2.is_tensor) {
            m2.shape[0] = DRJIT_DYNAMIC;
            m2.ndim = 1;
        }
        m2.is_vector = m2.is_complex = m2.is_quaternion = m2.is_matrix =
            m2.is_tensor = false;
        plain_t_o = nb::borrow(meta_get_type(m2));
    }

    // - ``MaskT``: type produced by comparisons such as ``__eq__``";
    nb::object mask_t_o;
    if (vt == VarType::Bool) {
        mask_t_o = name_o; // reference self
    } else {
        ArrayMeta m2 = b;
        m2.type = (uint16_t) VarType::Bool;
        m2.is_vector = m2.is_complex = m2.is_quaternion =
            m2.is_matrix = false;
        mask_t_o = nb::borrow(meta_get_type(m2));
    }

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
    d.scope = scope.ptr();

    PyType_Slot slots [] = {
        { Py_tp_init, (void *) (b.is_tensor ? tp_init_tensor : tp_init_array) },
        { Py_sq_item, (void *) (b.is_tensor ? sq_item_tensor : b.item) },
        { Py_sq_ass_item, (void *) (b.is_tensor ? sq_ass_item_tensor : b.set_item) },
        { 0, 0 }
    };

    d.type_slots = slots;

    nb::object base_o = nb::borrow(array_base);

    #if PY_VERSION_HEX >= 0x03090000
        // Parameterize the generic base class if supported by Python
        if (b.is_tensor)
            base_o = base_o[nb::make_tuple(
                name_o, self_cp_t_o, name_o, self_cp_t_o, name_o, plain_t_o, mask_t_o)];
        else
            base_o = base_o[nb::make_tuple(
                name_o, self_cp_t_o, val_t_o, val_cp_t_o, red_t_o, plain_t_o, mask_t_o)];
    #endif

    d.base_py = (PyTypeObject *) base_o.ptr();

    // Type was already bound, let's create an alias
    nb::handle existing = nb::detail::nb_type_lookup(b.array_type);
    if (existing) {
        nb::handle(d.scope).attr(name.c_str()) = existing;
        return nb::borrow(existing);
    }

    // Create a new type and update its supplemental information
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
            return is_float(s);
        } else if (PySequence_Check(o)) {
            if (s.is_tensor)
                return true;

            if (is_drjit_type(tp_o) && supp(tp_o).ndim > s.ndim)
                return false;

            Py_ssize_t size = s.shape[0], len = PySequence_Length(o);
            if (len == -1)
                PyErr_Clear();

            return size == DRJIT_DYNAMIC || len == size;
        }
        return false;
    };

    nb::detail::implicitly_convertible(pred, b.array_type);

    s.value = val_t_o.ptr();
    s.array = is_special ? plain_t_o.ptr() : tp.ptr();
    s.mask = vt == VarType::Bool ? tp.ptr() : mask_t_o.ptr();

    if (s.is_tensor) {
        ArrayMeta m2 = s;
        m2.type = (uint32_t) VarType::UInt32;
        m2.shape[0] = DRJIT_DYNAMIC;
        m2.ndim = 1;
        m2.is_tensor = false;
        s.tensor_index = meta_get_type(m2).ptr();
    }

    return tp;
}

void export_bind(nb::module_ &detail) {
    detail.def("bind", [](void *p) { return bind(*(ArrayBinding *) p); });
}
