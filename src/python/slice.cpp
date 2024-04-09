/*
    slice.cpp -- implementation of drjit.slice_index() and
    map subscript operators

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "memop.h"
#include "base.h"
#include "init.h"
#include "shape.h"
#include "base.h"
#include "slice.h"
#include <vector>

/// Holds metadata about slicing component
struct Component {
    Py_ssize_t start, step, slice_size, size;
    nb::object object;

    Component(Py_ssize_t start, Py_ssize_t step, Py_ssize_t slice_size,
              Py_ssize_t size)
        : start(start), step(step), slice_size(slice_size), size(size) { }

    Component(nb::handle h, Py_ssize_t slice_size, Py_ssize_t size)
        : start(0), step(1), slice_size(slice_size), size(size),
          object(nb::borrow(h)) { }
};

std::pair<nb::tuple, nb::object>
slice_index(const nb::type_object_t<ArrayBase> &dtype,
            const nb::tuple &shape, const nb::tuple &indices) {
    const ArraySupplement &s = supp(dtype);

    if (s.ndim != 1 || s.shape[0] != DRJIT_DYNAMIC ||
        (VarType) s.type != VarType::UInt32)
        throw nb::type_error("drjit.slice_index(): dtype must be a dynamically "
                             "sized unsigned 32 bit Dr.Jit array.");

    size_t none_count = 0, ellipsis_count = 0;
    for (nb::handle h : indices) {
        ellipsis_count += h.type().is(&PyEllipsis_Type);
        none_count += h.is_none();
    }

    if (ellipsis_count > 1)
        nb::raise(
            "drjit.slice_tensor(): multiple ellipses (...) are not allowed.");

    size_t shape_offset = 0;
    size_t size_out = 1;
    nb::list shape_out;

    // Preallocate memory for computed slicing components
    size_t shape_len = nb::len(shape),
           indices_len = nb::len(indices);

    std::vector<Component> components;
    components.reserve(shape_len);

    for (nb::handle h : indices) {
        if (h.is_none()) {
            shape_out.append(1);
            continue;
        }

        if (shape_offset >= shape_len)
            nb::raise("drjit.slice_tensor(): too many indices.");

        Py_ssize_t size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
        nb::handle tp = h.type();

        if (tp.is(&PyLong_Type)) {
            Py_ssize_t v = nb::cast<Py_ssize_t>(h);
            if (v < 0)
                v += size;

            if (v < 0 || v >= size)
                nb::raise("drjit.slice_tensor(): index %zd is out of "
                          "bounds for axis %zu with size %zd.",
                          v, components.size(), size);

            components.emplace_back(v, 1, 1, size);
            continue;
        } else if (tp.is(&PySlice_Type)) {
            Py_ssize_t start, stop, step;
            size_t slice_length;
            nb::detail::slice_compute(h.ptr(), size, start, stop, step, slice_length);
            components.emplace_back(start, step, (Py_ssize_t) slice_length, size);
            shape_out.append(slice_length);
            size_out *= slice_length;
            continue;
        } else if (is_drjit_type(tp)) {
            ArrayMeta m = supp(tp);

            if (m.ndim == 1 && m.shape[0] == DRJIT_DYNAMIC) {
                VarType vt = (VarType) m.type;
                nb::object o = nb::borrow(h);

                size_t slice_size = nb::len(h);
                if (vt == VarType::Int8 || vt == VarType::Int16 ||
                    vt == VarType::Int32 || vt == VarType::Int64) {
                    o[o < nb::cast(0)] += nb::cast(size);
                }

                if (!o.type().is(dtype))
                    o = dtype(o);

                components.emplace_back(o, slice_size, size);
                shape_out.append(slice_size);
                size_out *= slice_size;
                continue;
            }
        } else if (tp.is(&PyEllipsis_Type)) {
            size_t true_indices = indices_len - none_count - ellipsis_count,
                   indices_to_add = shape_len - true_indices;
            --shape_offset;
            for (size_t i = 0; i <indices_to_add; ++i) {
                if (shape_offset >= shape_len)
                    nb::detail::fail("slice_index(): internal error.");
                size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
                components.emplace_back(0, 1, size, size);
                shape_out.append(size);
                size_out *= size;
            }
            continue;
        }

        nb::str tp_name = nb::type_name(tp);
        nb::raise(
            "drjit.slice_index(): unsupported type \"%s\" in slice expression.",
            tp_name.c_str());
    }

    // Implicit ellipsis at the end
    while (shape_offset != shape_len) {
        Py_ssize_t size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
        components.emplace_back(0, 1, size, size);
        shape_out.append(size);
        size_out *= size;
    }

    nb::object index = arange(dtype, 0, size_out, 1),
               index_out;

    nb::object active = nb::borrow(Py_True);
    if (size_out) {
        size_out = 1;
        index_out = dtype(0);

        for (auto it = components.rbegin(); it != components.rend(); ++it) {
            const Component &c = *it;
            nb::object index_next, index_rem;

            if (it + 1 != components.rend()) {
                index_next = index.floor_div(dtype(c.slice_size));
                index_rem = fma(index_next, dtype(uint32_t(-c.slice_size)), index);
            } else {
                index_rem = index;
            }

            nb::object index_val;
            if (!c.object.is_valid())
                index_val = fma(index_rem, dtype(uint32_t(c.step * size_out)),
                                dtype(uint32_t(c.start * size_out)));
            else
                index_val = gather(dtype, c.object, index_rem, active,
                                   ReduceMode::Auto) *
                            dtype(uint32_t(size_out));

            index_out += index_val;

            index = std::move(index_next);
            size_out *= c.size;
        }
    } else {
        index_out = dtype();
    }

    return { nb::tuple(shape_out), index_out };
}

PyObject *mp_subscript(PyObject *self, PyObject *key) noexcept {
    nb::handle self_tp = nb::handle(self).type(),
               key_tp = nb::handle(key).type();

    const ArraySupplement &s = supp(self_tp);

    try {
        bool key_is_array = is_drjit_type(key_tp);

        if (key_is_array && (VarType) supp(key_tp).type == VarType::Bool) {
            nb::object out = nb::inst_alloc(self_tp.ptr());
            nb::inst_copy(out, self);
            return out.release().ptr();
        }

        if (s.is_tensor) {
            nb::tuple key2;

            if (key_tp.is(&PyTuple_Type))
                key2 = nb::borrow<nb::tuple>(key);
            else
                key2 = nb::make_tuple(nb::handle(key));

            auto [out_shape, out_index] = slice_index(
                nb::borrow<nb::type_object_t<ArrayBase>>(s.tensor_index),
                nb::borrow<nb::tuple>(shape(self)), key2);

            nb::object source = nb::steal(s.tensor_array(self));

            nb::object out = gather(nb::borrow<nb::type_object>(s.array),
                                    source, out_index, nb::borrow(Py_True));

            return self_tp("array"_a = out, "shape"_a = out_shape)
                .release().ptr();
        }

        bool complex_case = false;
        if (key_tp.is(&PyLong_Type)) {
            Py_ssize_t index = PyLong_AsSsize_t(key);
            if (index < 0) {
                raise_if(index == -1 && PyErr_Occurred(),
                         "Invalid array index.");

                Py_ssize_t size = s.shape[0];
                if (size == DRJIT_DYNAMIC)
                    size = (Py_ssize_t) s.len(inst_ptr(self));

                index = size + index;
            }

            return s.item(self, index);
        } else if (key_tp.is(&PyTuple_Type)) {
            nb::object o = nb::borrow(self);
            Py_ssize_t size = NB_TUPLE_GET_SIZE(key);

            for (Py_ssize_t i = 0; i < size; ++i) {
                o = nb::steal(PyObject_GetItem(o.ptr(), NB_TUPLE_GET_ITEM(key, i)));
                raise_if(!o.is_valid(), "Item retrieval failed.");
            }

            return o.release().ptr();
        } else if (key == Py_None || key_tp.is(&PyEllipsis_Type) || key_tp.is(&PySlice_Type) || key_is_array) {
            complex_case = true;
        }

        if (complex_case) {
            nb::raise_type_error(
                "Complex slicing operations are currently only supported on tensors.");
        } else {
            nb::str key_name = nb::type_name(key_tp);
            nb::raise_type_error("Invalid key of type '%s' specified.",
                                 key_name.c_str());
        }
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(self_tp);
        e.restore();
        nb::chain_error(PyExc_TypeError, "%U.__getitem__(): internal error.",
                        tp_name.ptr());
        return nullptr;
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        nb::chain_error(PyExc_TypeError, "%U.__getitem__(): %s",
                        tp_name.ptr(), e.what());
        return nullptr;
    }
}

int mp_ass_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
    nb::handle self_tp = nb::handle(self).type(),
               key_tp = nb::handle(key).type();

    const ArraySupplement &s = supp(self_tp);

    try {
        bool key_is_array = is_drjit_type(key_tp);

        if (key_is_array && (VarType) supp(key_tp).type == VarType::Bool) {
            nb::object result = select(nb::borrow(key), nb::borrow(value), nb::borrow(self));
            nb::handle result_tp = result.type();
            if (!result_tp.is(self_tp))
                nb::raise("Incompatible mask assignment, type changed from "
                          "'%s' to '%s'",
                          type_name(self_tp).c_str(),
                          type_name(result_tp).c_str());
            nb::inst_replace_move(self, result);
            return 0;
        }

        if (s.is_tensor) {
            nb::tuple key2;

            if (key_tp.is(&PyTuple_Type))
                key2 = nb::borrow<nb::tuple>(key);
            else
                key2 = nb::make_tuple(nb::handle(key));

            auto [out_shape, out_index] = slice_index(
                nb::borrow<nb::type_object_t<ArrayBase>>(s.tensor_index),
                nb::borrow<nb::tuple>(shape(self)), key2);

            nb::object target = nb::steal(s.tensor_array(self));
            scatter(target, nb::borrow(value), out_index, nb::borrow(Py_True));

            return 0;
        }

        bool complex_case = false;
        if (key_tp.is(&PyLong_Type)) {
            Py_ssize_t index = PyLong_AsSsize_t(key);
            if (index < 0) {
                raise_if(index == -1 && PyErr_Occurred(),
                         "Invalid array index.");

                Py_ssize_t size = s.shape[0];
                if (size == DRJIT_DYNAMIC)
                    size = (Py_ssize_t) s.len(inst_ptr(self));

                index = size + index;
            }

            return s.set_item(self, index, value);
        } else if (key_tp.is(&PyTuple_Type)) {
            nb::object o = nb::borrow(self);
            Py_ssize_t size = NB_TUPLE_GET_SIZE(key);

            for (Py_ssize_t i = 0; i < size - 1; ++i) {
                o = nb::steal(PyObject_GetItem(o.ptr(), NB_TUPLE_GET_ITEM(key, i)));
                raise_if(!o.is_valid(), "Item retrival failed.");
            }

            if (size) {
                int rv = PyObject_SetItem(
                    o.ptr(), NB_TUPLE_GET_ITEM(key, size - 1), value);
                raise_if(rv != 0, "Item assigment failed.");
            }

            return 0;
        } else if (key == Py_None || key_tp.is(&PyEllipsis_Type) ||
                   key_tp.is(&PySlice_Type) || key_is_array) {
            complex_case = true;
        }

        if (complex_case) {
            nb::raise_type_error(
                "Complex slicing operations are currently only supported on tensors.");
        } else {
            nb::str key_name = nb::type_name(key_tp);
            nb::raise_type_error("Invalid key of type '%s' specified.",
                                 key_name.c_str());
        }
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(self_tp);
        e.restore();
        nb::chain_error(PyExc_TypeError, "%U.__setitem__(): internal error.",
                        tp_name.ptr());
        return -1;
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        PyErr_Format(PyExc_TypeError, "%U.__setitem__(): %s",
                     tp_name.ptr(), e.what());
        return -1;
    }
}

PyObject *sq_item_tensor(PyObject *self, Py_ssize_t index) noexcept {
    PyObject *key = PyLong_FromSsize_t(index);
    if (!key)
        return nullptr;
    return mp_subscript(self, key);
}

int sq_ass_item_tensor(PyObject *self, Py_ssize_t index, PyObject *value) noexcept {
    PyObject *key = PyLong_FromSsize_t(index);
    if (!key)
        return -1;
    return mp_ass_subscript(self, key, value);
}

void export_slice(nb::module_&m) {
    m.def("slice_index", &slice_index, doc_slice_index, "dtype"_a, "shape"_a,
          "indices"_a);
}
