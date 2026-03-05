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
#include <algorithm>
#include <vector>

/// Holds metadata about slicing component
struct Component {
    enum Type { None, Integer, Slice, Advanced, Ellipsis };

    Type type;
    Py_ssize_t start, step, slice_size, size;
    nb::object object;

    // Constructor for None indices
    Component(Type t)
        : type(t), start(0), step(1), slice_size(1), size(1) { }

    // Constructor for integer and slice indices
    Component(Type t, Py_ssize_t start, Py_ssize_t step, Py_ssize_t slice_size,
              Py_ssize_t size)
        : type(t), start(start), step(step), slice_size(slice_size), size(size) { }

    // Constructor for advanced indices (array indexing)
    Component(Type t, nb::handle h, Py_ssize_t slice_size, Py_ssize_t size)
        : type(t), start(0), step(1), slice_size(slice_size), size(size),
          object(nb::borrow(h)) { }
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
    components.reserve(indices_len);  // May include None indices

    // First pass: parse indices
    nb::list basic_shapes;       // Shapes from basic indexing (slices)
    size_t advanced_size = 0;    // Size of advanced index arrays (all must be same)

    for (nb::handle h : indices) {
        if (h.is_none()) {
            components.emplace_back(Component::None);
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

            components.emplace_back(Component::Integer, v, 1, 1, size);
            continue;
        } else if (tp.is(&PySlice_Type)) {
            Py_ssize_t start, stop, step;
            size_t slice_length;
            nb::detail::slice_compute(h.ptr(), size, start, stop, step, slice_length);
            components.emplace_back(Component::Slice, start, step, (Py_ssize_t) slice_length, size);
            basic_shapes.append(slice_length);
            continue;
        } else if (is_drjit_type(tp)) {
            const ArraySupplement *s2 = &supp(tp);
            nb::object tmp;

            if (s2->is_tensor) {
                const dr::vector<size_t> &tensor_shape = s2->tensor_shape(inst_ptr(h));

                if (tensor_shape.size() != 1) {
                    nb::raise("drjit.slice_index(): encountered a %zu-D tensor "
                              "of type '%s' in slice expression. However, only "
                              "1D tensors are permitted.",
                              tensor_shape.size(), nb::inst_name(h).c_str());
                }

                tmp = nb::steal(s2->tensor_array(h.ptr()));
                s2 = &supp(tmp.type());
                h = tmp;
            }

            if (s2->ndim == 1 && s2->shape[0] == DRJIT_DYNAMIC) {
                VarType vt = (VarType) s2->type;
                nb::object o = nb::borrow(h);

                size_t slice_size = nb::len(h);
                if (is_signed_int(vt)) {
                    o = select(o.attr("__lt__")(nb::int_(0)),
                               o + nb::int_(size), o);
                }

                if (!o.type().is(dtype))
                    o = dtype(o);

                components.emplace_back(Component::Advanced, o, slice_size, size);

                // Track the maximum size for broadcasting
                // PyTorch/NumPy broadcast all advanced indices to the same shape
                if (advanced_size == 0) {
                    advanced_size = slice_size;
                } else if (slice_size != 1 && advanced_size != 1 && advanced_size != slice_size) {
                    // Broadcasting rules: sizes must be 1 or equal
                    nb::raise("drjit.slice_index(): advanced index arrays with shapes %zu and %zu "
                              "cannot be broadcast together.", advanced_size, slice_size);
                } else if (slice_size > advanced_size) {
                    // Update to the larger size (broadcasting smaller arrays to match)
                    advanced_size = slice_size;
                }

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
                components.emplace_back(Component::Slice, 0, 1, size, size);
                basic_shapes.append(size);
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
        components.emplace_back(Component::Slice, 0, 1, size, size);
        basic_shapes.append(size);
    }

    // Build output shape following PyTorch/NumPy advanced indexing rules:
    // - None indices create new dimensions of size 1 at their positions
    // - Integer indices reduce dimensions (don't appear in output)
    // - Advanced indices: if consecutive, stay in place; if non-consecutive, move to front
    shape_out.clear();

    // Check if there are advanced indices and if they're consecutive
    int first_adv = -1, last_adv = -1;
    for (size_t i = 0; i < components.size(); ++i) {
        if (components[i].type == Component::Advanced) {
            if (first_adv == -1) first_adv = i;
            last_adv = i;
        }
    }

    bool has_advanced = (first_adv != -1);
    bool consecutive = true;
    if (has_advanced) {
        for (int i = first_adv; i <= last_adv; ++i) {
            if (components[i].type == Component::None) continue;  // None doesn't break consecutiveness
            if (components[i].type != Component::Advanced) {
                consecutive = false;
                break;
            }
        }
    }

    // Build output shape based on index arrangement
    if (has_advanced && consecutive) {
        // Advanced indices are consecutive: replace all with a single dimension
        bool advanced_added = false;
        for (const auto &comp : components) {
            if (comp.type == Component::None) {
                shape_out.append(1);
            } else if (comp.type == Component::Slice) {
                shape_out.append(comp.slice_size);
            } else if (comp.type == Component::Advanced) {
                if (!advanced_added) {
                    // All consecutive advanced indices produce a single dimension
                    shape_out.append(advanced_size);
                    advanced_added = true;
                }
                // Subsequent advanced indices don't add dimensions
            }
            // Integer indices don't contribute
        }
    } else if (has_advanced && !consecutive) {
        // Advanced indices are non-consecutive: move to front
        shape_out.append(advanced_size);
        for (const auto &comp : components) {
            if (comp.type == Component::None) {
                shape_out.append(1);
            } else if (comp.type == Component::Slice) {
                shape_out.append(comp.slice_size);
            }
            // Integer and Advanced (already added) don't contribute here
        }
    } else {
        // No advanced indexing: process each index type in order
        for (const auto &comp : components) {
            if (comp.type == Component::None) {
                shape_out.append(1);
            } else if (comp.type == Component::Slice) {
                shape_out.append(comp.slice_size);
            }
            // Integer indices don't contribute to shape
        }
    }

    // Calculate total size from the actual output shape
    size_out = 1;
    for (nb::handle h : shape_out)
        size_out *= nb::cast<size_t>(h);

    nb::object index = arange(dtype, 0, size_out, 1),
               index_out;

    nb::object active = nb::borrow(Py_True);
    if (size_out) {
        // Unified algorithm that handles both basic and advanced indexing
        index_out = dtype(0);

        // Calculate the stride multiplier for the input tensor dimensions
        // Skip None components as they don't correspond to input dimensions
        size_t input_stride = 1;
        std::vector<size_t> input_strides;
        for (auto it = components.rbegin(); it != components.rend(); ++it) {
            if (it->type == Component::None) {
                input_strides.push_back(0);  // Placeholder for None
            } else {
                input_strides.push_back(input_stride);
                input_stride *= it->size;
            }
        }
        std::reverse(input_strides.begin(), input_strides.end());

        // Decompose output index according to output shape
        nb::object remaining = index;
        std::vector<nb::object> output_dim_indices;

        // Decompose based on actual output shape (in reverse order)
        for (size_t i = nb::len(shape_out); i > 0; --i) {
            size_t dim_size = nb::cast<size_t>(shape_out[i - 1]);
            nb::object dim_idx;
            if (i > 1) {
                nb::object quotient = remaining.floor_div(dtype(dim_size));
                dim_idx = remaining - quotient * dtype(dim_size);
                remaining = quotient;
            } else {
                dim_idx = remaining;
            }
            output_dim_indices.insert(output_dim_indices.begin(), dim_idx);
        }

        // Check if there are advanced indices and if they're consecutive
        int first_adv = -1, last_adv = -1;
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].type == Component::Advanced) {
                if (first_adv == -1) first_adv = i;
                last_adv = i;
            }
        }

        bool has_advanced = (first_adv != -1);
        bool consecutive = true;
        if (has_advanced) {
            for (int i = first_adv; i <= last_adv; ++i) {
                if (components[i].type == Component::None) continue;
                if (components[i].type != Component::Advanced) {
                    consecutive = false;
                    break;
                }
            }
        }

        // Extract advanced_idx and basic indices from output_dim_indices
        nb::object advanced_idx = dtype(0);
        std::vector<nb::object> basic_dim_indices;
        size_t output_idx = 0;
        bool advanced_found = false;

        if (has_advanced && consecutive) {
            // Advanced indices are consecutive: they stay in their natural position
            for (const auto &comp : components) {
                if (comp.type == Component::None) {
                    output_idx++;
                } else if (comp.type == Component::Advanced) {
                    if (!advanced_found) {
                        advanced_idx = output_dim_indices[output_idx];
                        advanced_found = true;
                    }
                    output_idx++;
                } else if (comp.type == Component::Slice) {
                    basic_dim_indices.push_back(output_dim_indices[output_idx]);
                    output_idx++;
                }
            }
        } else if (has_advanced && !consecutive) {
            // Advanced indices are non-consecutive: they're moved to the front
            advanced_idx = output_dim_indices[0];
            output_idx = 1;
            for (const auto &comp : components) {
                if (comp.type == Component::None) {
                    if (output_idx < output_dim_indices.size()) {
                        output_idx++;
                    }
                } else if (comp.type == Component::Slice) {
                    if (output_idx < output_dim_indices.size()) {
                        basic_dim_indices.push_back(output_dim_indices[output_idx]);
                        output_idx++;
                    }
                }
            }
        } else {
            // No advanced indexing: just map output dimensions to input
            for (const auto &comp : components) {
                if (comp.type == Component::None) {
                    output_idx++;
                } else if (comp.type == Component::Slice) {
                    if (output_idx < output_dim_indices.size()) {
                        basic_dim_indices.push_back(output_dim_indices[output_idx]);
                        output_idx++;
                    }
                }
            }
        }

        // Map output indices back to input dimensions
        size_t basic_idx_counter = 0;
        for (size_t i = 0; i < components.size(); ++i) {
            const Component &c = components[i];

            // Skip None indices as they don't correspond to input dimensions
            if (c.type == Component::None)
                continue;

            nb::object dim_index;

            if (c.type == Component::Advanced) {
                // Advanced index: use the advanced_idx to gather from the index array
                // Handle broadcasting: if the index array has size 1, broadcast it
                if (c.slice_size == 1) {
                    dim_index = gather(dtype, c.object, dtype(0), active, ReduceMode::Auto);
                } else {
                    dim_index = gather(dtype, c.object, advanced_idx, active, ReduceMode::Auto);
                }
            } else if (c.type == Component::Integer) {
                // Integer index
                dim_index = dtype(c.start);
            } else if (c.type == Component::Slice) {
                // Basic slice: get the dimension index and apply slice transformation
                if (basic_idx_counter < basic_dim_indices.size()) {
                    dim_index = basic_dim_indices[basic_idx_counter];
                    dim_index = fma(dim_index, dtype(uint32_t(c.step)), dtype(uint32_t(c.start)));
                    basic_idx_counter++;
                } else {
                    dim_index = dtype(c.start);
                }
            }

            // Add contribution to output index
            index_out += dim_index * dtype(uint32_t(input_strides[i]));
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
                auto [orig, k, o2] = *it;
                if (PyObject_SetItem(orig.ptr(), k.ptr(), o2.ptr()))
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
