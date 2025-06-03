/*
    slice.cpp -- implementation of drjit.slice_index() and
    map subscript operators

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "drjit/python.h"
#include "memop.h"
#include "base.h"
#include "init.h"
#include "shape.h"
#include "base.h"
#include "slice.h"
#include "meta.h"
#include <vector>

/// Holds metadata about slicing component
struct Component {
    Py_ssize_t start, step, slice_size, size;
    nb::object object;
    bool is_array = false;

    Component(Py_ssize_t start, Py_ssize_t step, Py_ssize_t slice_size,
              Py_ssize_t size)
        : start(start), step(step), slice_size(slice_size), size(size) { }

    Component(nb::handle h, Py_ssize_t slice_size, Py_ssize_t size,
              bool is_array = false)
        : start(0), step(1), slice_size(slice_size), size(size),
          object(nb::borrow(h)), is_array(is_array) {}
};

inline bool is_signed_int (VarType vt) {
    return vt == VarType::Int8 || vt == VarType::Int16 ||
           vt == VarType::Int32 || vt == VarType::Int64;
}

inline bool is_unsigned_int(VarType vt) {
    return vt == VarType::UInt8 || vt == VarType::UInt16 ||
           vt == VarType::UInt32 || vt == VarType::UInt64;
}

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

    size_t array_size = 0;
    int array_dim = 0;
    int array_dim_i = -1;

    for (uint32_t i = 0; i < indices.size(); ++i) {
        nb::handle h = indices[i];
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
            const ArraySupplement *s = &supp(tp);
            nb::object tmp;

            if (s->is_tensor) {
                const dr::vector<size_t> &shape = s->tensor_shape(inst_ptr(h));

                if (shape.size() != 1) {
                    nb::raise("drjit.slice_index(): encountered a %zu-D tensor "
                              "of type '%s' in slice expression. However, only "
                              "1D tensors are permitted.",
                              shape.size(), nb::inst_name(h).c_str());
                }

                tmp = nb::steal(s->tensor_array(h.ptr()));
                s = &supp(tmp.type());
                h = tmp;
            }

            if (s->ndim == 1 && s->shape[0] == DRJIT_DYNAMIC) {
                VarType vt = (VarType) s->type;
                nb::object o = nb::borrow(h);

                size_t slice_size = nb::len(h);
                if (is_signed_int(vt)) {
                    o = select(o.attr("__lt__")(nb::int_(0)),
                               o + nb::int_(size), o);
                }

                if (!o.type().is(dtype))
                    o = dtype(o);

                if (array_size <= 1)
                    array_size = slice_size;

                if (slice_size > 1 && array_size != slice_size)
                    jit_raise("Index size missmatch!");

                if (array_dim_i == ((int) i) - 1)
                    array_dim_i = i;
                else
                    array_dim = 0;

                if (array_dim_i < 0) {
                    array_dim_i = i;
                    array_dim = i;
                }

                components.emplace_back(o, slice_size, size, true);
                // shape_out.append(slice_size);
                // size_out *= slice_size;
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

    size_t slicing_size = size_out;
    if (array_size) {
        size_out *= array_size;
        shape_out.insert(array_dim, array_size);
    }

    nb::object index = arange(dtype, 0, size_out, 1),
               index_out;
    nb::object index_i = arange(dtype, 0, size_out, 1);

    nb::object active = nb::borrow(Py_True);
    if (size_out) {
        size_out = 1;
        index_out = dtype(0);

        for (auto it = components.rbegin(); it != components.rend(); ++it) {
            const Component &c = *it;
            nb::object index_next, index_rem;

            nb::object index_val;
            if (c.is_array) {
                jit_log(LogLevel::Warn, "is_array");
                index_next = index_i.floor_div(dtype(c.slice_size));
                index_rem =
                    fma(index_next, dtype(uint32_t(-c.slice_size)), index_i);
                jit_log(LogLevel::Warn, "index_rem=%s", nb::str(index_rem).c_str());

                index_val = gather(dtype, c.object, index_rem, active,
                                   ReduceMode::Auto) *
                            dtype(uint32_t(size_out));
                jit_log(LogLevel::Warn, "index_val=%s", nb::str(index_val).c_str());
            } else {
                if (it + 1 != components.rend()) {
                    index_next = index.floor_div(dtype(c.slice_size));
                    index_rem =
                        fma(index_next, dtype(uint32_t(-c.slice_size)), index);
                } else {
                    index_rem = index;
                }
                jit_log(LogLevel::Warn, "index_rem=%s", nb::str(index_rem).c_str());

                if (!c.object.is_valid())
                    index_val =
                        fma(index_rem, dtype(uint32_t(c.step * size_out)),
                            dtype(uint32_t(c.start * size_out)));
                else
                    index_val = gather(dtype, c.object, index_rem, active,
                                       ReduceMode::Auto) *
                                dtype(uint32_t(size_out));
                jit_log(LogLevel::Warn, "index_val=%s", nb::str(index_val).c_str());

                index_val = fma(index_rem, dtype(uint32_t(c.step * size_out)),
                                dtype(uint32_t(c.start * size_out)));
            }

            index = std::move(index_next);

            index_out += index_val;

            jit_log(LogLevel::Warn, "index_out=%s", nb::str(index_out).c_str());

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
        VarType key_type = VarType::Void;
        bool is_1d_ndim_array = s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC;
        bool is_slice = key_tp.is(&PySlice_Type);

        if (is_slice) {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(key, &start, &stop, &step))
                return nullptr;
            if (start == 0 && stop == PY_SSIZE_T_MAX && step == 1) {
                // x[:] style slice. Return the original object
                Py_INCREF(self);
                return self;
            }
        }

        if (is_drjit_type(key_tp))
            key_type = (VarType) supp(key_tp).type;

        if (key_type == VarType::Bool) {
            nb::object out = nb::inst_alloc(self_tp.ptr());
            nb::inst_copy(out, self);
            return out.release().ptr();
        } else if (is_1d_ndim_array &&
                   (is_signed_int(key_type) || is_unsigned_int(key_type))) {
            nb::object index = nb::borrow(key);

            if (is_signed_int(key_type))
                index = select(index.attr("__lt__")(nb::int_(0)),
                               index + nb::int_(nb::len(self)), index);

            return gather(nb::borrow<nb::type_object>(self_tp),
                          nb::borrow(self), index, nb::borrow(Py_True),
                          ReduceMode::Auto, nb::none())
                .release()
                .ptr();
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
        } else if (is_slice) {
            auto [start, end, step, slicelen] =
                nb::borrow<nb::slice>(key).compute(sq_length(self));

            if (is_1d_ndim_array) {
                ArrayMeta m = s;
                m.type = (uint16_t) VarType::UInt32;

                nb::type_object_t<ArrayBase> index_type =
                    nb::borrow<nb::type_object_t<ArrayBase>>(meta_get_type(m));

                nb::object index = arange(index_type, start, end, step);
                return gather(nb::borrow<nb::type_object>(self_tp), nb::borrow(self), index,
                              nb::borrow(Py_True), ReduceMode::Auto, nb::none())
                    .release()
                    .ptr();
            } else {
                ArrayMeta m2 = s;
                m2.shape[0] = (uint8_t) (slicelen <= 4 ? slicelen : DRJIT_DYNAMIC);
                nb::object result = meta_get_type(m2)();
                for (size_t i = 0; i < slicelen; ++i)
                    result[i] = nb::handle(self)[start + step * i];
                return result.release().ptr();
            }
        }

        if (key == Py_None || key_tp.is(&PyEllipsis_Type)) {
            nb::raise_type_error(
                "Complex slicing operations involving 'None' / '...' are "
                "currently only supported on tensors.");
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
        VarType key_type = VarType::Void;
        bool is_1d_ndim_array = s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC;
        bool is_slice = key_tp.is(&PySlice_Type);

        if (is_slice) {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(key, &start, &stop, &step))
                return -1;

            if (start == 0 && stop == PY_SSIZE_T_MAX && step == 1) {
                // x[:] = .. assignment
                nb::handle value_tp = nb::handle(value).type();
                if (value_tp.is(self_tp)) {
                    // Replace array contents

                    if (value == self)
                        return 0;
                    nb::inst_replace_copy(self, value);
                    return 0;
                } else if (value_tp.is(&PyLong_Type) || value_tp.is(&PyFloat_Type)) {
                    // Broadcast a scalar

                    dr::vector<size_t> shape;
                    if (!shape_impl(self, shape))
                        nb::raise("target is ragged!");
                    nb::object value2 = full("full", self_tp, value, shape);
                    return mp_ass_subscript(self, key, value2.ptr());
                } else {
                    nb::raise_type_error("Unsupported type in assignment: %s",
                                         nb::type_name(value_tp).c_str());
                }
            }

            // Other types of slice assignments are handled below (specific
            // cases for 1D, N-D arrays, and tensors)
        }

        if (is_drjit_type(key_tp))
            key_type = (VarType) supp(key_tp).type;

        if (key_type == VarType::Bool) {
            nb::object result = select(nb::borrow(key), nb::borrow(value), nb::borrow(self));
            nb::handle result_tp = result.type();
            if (!result_tp.is(self_tp))
                nb::raise("Incompatible mask assignment, type changed from "
                          "'%s' to '%s'",
                          type_name(self_tp).c_str(),
                          type_name(result_tp).c_str());
            nb::inst_replace_move(self, result);
            return 0;
        } else if (is_1d_ndim_array &&
                   (is_signed_int(key_type) || is_unsigned_int(key_type))) {
            nb::object index = nb::borrow(key);
            if (is_signed_int(key_type))
                index = select(index.attr("__lt__")(nb::int_(0)),
                               index + nb::int_(nb::len(self)), index);
            scatter(nb::borrow(self), nb::borrow(value), index,
                    nb::borrow(Py_True), ReduceMode::Auto);
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
            std::vector<dr::tuple<nb::object, nb::object, nb::object>> trail;

            for (Py_ssize_t i = 0; i < size - 1; ++i) {
                nb::object k = nb::borrow(NB_TUPLE_GET_ITEM(key, i));
                nb::object orig(o);
                o = nb::steal(PyObject_GetItem(o.ptr(), k.ptr()));
                if (!o.is_valid())
                    throw nb::python_error();
                if (!k.type().is(&PyLong_Type))
                    trail.emplace_back(orig, k, o);
            }

            if (size) {
                int rv = PyObject_SetItem(
                    o.ptr(), NB_TUPLE_GET_ITEM(key, size - 1), value);
                if (rv)
                    throw nb::python_error();
            }

            // Slice indexing creates new copies of arrays. For example, x[1:2]
            // creates a new array. If a slice is used as part of a nested
            // assignment, the previous logic made a change to the temporary
            // instead of the original object. So, at the end, we need to walk
            // backwards and propagate the changes.

            for (auto it = trail.rbegin(); it != trail.rend(); ++it) {
                auto [orig, k, o] = *it;
                if (PyObject_SetItem(orig.ptr(), k.ptr(), o.ptr()))
                    nb::raise_python_error();
            }

            return 0;
        } else if (is_slice) {
            auto [start, end, step, slicelen] =
                nb::borrow<nb::slice>(key).compute(sq_length(self));

            if (is_1d_ndim_array) {
                ArrayMeta m = s;
                m.type = (uint16_t) VarType::UInt32;

                nb::type_object_t<ArrayBase> index_type =
                    nb::borrow<nb::type_object_t<ArrayBase>>(meta_get_type(m));

                nb::object index = arange(index_type, start, end, step);
                scatter(nb::borrow(self), nb::borrow(value), nb::borrow(index),
                        nb::borrow(Py_True), ReduceMode::Auto);
                return 0;
            } else {
                bool length_matches =
                    nb::hasattr(value, "__len__") && nb::len(value) == slicelen;
                nb::handle self_o = self, value_o = value;
                for (size_t i = 0; i < slicelen; ++i)
                    self_o[start + step * i] =
                        length_matches ? value_o[i] : value_o;
                return 0;
            }
        }

        if (key == Py_None || key_tp.is(&PyEllipsis_Type)) {
            nb::raise_type_error(
                "Complex slicing operations involving 'None' / '...' are "
                "currently only supported on tensors.");
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
