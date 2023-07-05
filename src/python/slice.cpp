#include "memop.h"
#include "base.h"
#include "init.h"
#include "shape.h"

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
slice_index(const nb::type_object_t<dr::ArrayBase> &dtype,
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
        nb::detail::raise(
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
            nb::detail::raise("drjit.slice_tensor(): too many indices.");

        Py_ssize_t size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
        nb::handle tp = h.type();

        if (tp.is(&PyLong_Type)) {
            Py_ssize_t v = nb::cast<Py_ssize_t>(h);
            if (v < 0)
                v += size;

            if (v < 0 || v >= size)
                nb::detail::raise("drjit.slice_tensor(): index %zd is out of "
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
                    nb::detail::fail("slice_index(): internal error!");
                size = nb::cast<Py_ssize_t>(shape[shape_offset++]);
                components.emplace_back(0, 1, size, size);
                shape_out.append(size);
                size_out *= size;
            }
            continue;
        }

        nb::str tp_name = nb::type_name(tp);
        nb::detail::raise(
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
                index_val = gather(dtype, c.object, index_rem, active) *
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

PyObject *mp_subscript(PyObject *self, PyObject *key) {
    nb::handle self_tp = nb::handle(self).type(),
               key_tp = nb::handle(key).type();

    const ArraySupplement &s = supp(self_tp);

    try {
        if (s.is_tensor) {
            nb::tuple key2;

            if (key_tp.is(&PyTuple_Type))
                key2 = nb::borrow<nb::tuple>(key);
            else
                key2 = nb::make_tuple(nb::handle(key));

            auto [out_shape, out_index] = slice_index(
                nb::borrow<nb::type_object_t<dr::ArrayBase>>(s.tensor_index),
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
                if (index == -1 && PyErr_Occurred())
                    nb::detail::raise("Invalid array index.");

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
                nb::handle key2 = NB_TUPLE_GET_ITEM(key, i);
                nb::object o2 = nb::steal(PyObject_GetItem(o.ptr(), key2.ptr()));
                if (!o2.is_valid())
                    nb::detail::raise_python_error();
                else
                    o = o2;
            }

            return o.release().ptr();
        } else if (is_drjit_type(key_tp)) {
            if ((VarType) supp(key_tp).type == VarType::Bool) {
                Py_INCREF(self);
                return self;
            } else {
                complex_case = true;
            }
        } else if (key == Py_None || key_tp.is(&PyEllipsis_Type) || key_tp.is(&PySlice_Type)) {
            complex_case = true;
        }

        if (complex_case) {
            nb::detail::raise_type_error(
                "Complex slicing operations are only supported on tensors.");
        } else {
            nb::str key_name = nb::type_name(key_tp);
            nb::detail::raise_type_error("Invalid key of type '%s' specified.",
                                         key_name.c_str());
        }
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(self_tp);
        e.restore();
        nb::chain_error(PyExc_TypeError, "%U.__getitem__(): internal error!",
                        tp_name.ptr());
        return nullptr;
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        PyErr_Format(PyExc_TypeError, "%U.__getitem__(): %s!",
                     tp_name.ptr(), e.what());
        return nullptr;
    }
}

void export_slice(nb::module_&m) {
    m.def("slice_index", &slice_index, doc_slice_index, "dtype"_a, "shape"_a,
          "indices"_a);
    m.attr("new_axis") = nb::none();
}
