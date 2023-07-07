/*
    memop.cpp -- Bindings for scatter/gather memory operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "memop.h"
#include "base.h"
#include "meta.h"
#include "init.h"
#include "shape.h"

nb::object gather(nb::type_object dtype, nb::object source,
                  nb::object index, nb::object active) {
    nb::handle source_tp = source.type();

    bool is_drjit_source_1d = is_drjit_type(source_tp);

    if (is_drjit_source_1d) {
        const ArraySupplement &s = supp(source_tp);
        is_drjit_source_1d = s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC;
    }

    // Recurse through pytrees
    if (!is_drjit_source_1d && source_tp.is(dtype)) {
        if (PySequence_Check(source.ptr())) {
            nb::list result;
            for (nb::handle value : source)
                result.append(gather(nb::borrow<nb::type_object>(value.type()),
                                     nb::borrow(value), index, active));

            if (!dtype.is(&PyList_Type))
                return dtype(result);
            else
                return result;
        } else if (source_tp.is(&PyDict_Type)) {
            nb::dict result;
            for (auto [k, v] : nb::borrow<nb::dict>(source))
                result[k] = gather(nb::borrow<nb::type_object>(v.type()),
                                   nb::borrow(v), index, active);

            return result;
        } else {
            nb::object dstruct = nb::getattr(dtype, "DRJIT_STRUCT", nb::handle());
            if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
                nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
                nb::dict out;

                for (auto [k, v] : dstruct_dict) {
                    if (!v.is_type())
                        throw nb::type_error("DRJIT_STRUCT invalid, expected types!");
                    nb::type_object sub_dtype = nb::borrow<nb::type_object>(v);
                    out[k] = gather(sub_dtype, nb::getattr(source, k), index, active);
                }

                return dtype(**out);
            }
        }
    }

    if (!is_drjit_type(dtype))
        throw nb::type_error("drjit.gather(): unsupported dtype!");

    if (!is_drjit_source_1d)
        throw nb::type_error(
            "drjit.gather(): 'source' argument must be a dynamic 1D array!");

    const ArraySupplement &source_supp = supp(source_tp);

    ArrayMeta source_meta = source_supp,
              active_meta = source_meta,
              index_meta  = source_meta;

    active_meta.type = (uint16_t) VarType::Bool;
    index_meta.type = (uint16_t) VarType::UInt32;

    nb::handle active_tp = meta_get_type(active_meta),
               index_tp = meta_get_type(index_meta);

    if (!index.type().is(index_tp)) {
        try {
            index = index_tp(index);
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_TypeError,
                "drjit.gather(): 'index' argument has an unsupported type, "
                "please provide an instance that is convertible to "
                "drjit.uint32_array_t(source).");
        }
    }

    if (!active.type().is(active_tp)) {
        try {
            active = active_tp(active);
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_TypeError,
                "drjit.gather(): 'active' argument has an unsupported type, "
                "please provide an instance that is convertible to "
                "drjit.mask_t(source).");
        }
    }

    const ArraySupplement &dtype_supp = supp(dtype);
    ArrayMeta dtype_meta = dtype_supp;

    if (dtype_meta == source_meta) {
        nb::object result = nb::inst_alloc(dtype);

        source_supp.gather(
            inst_ptr(source),
            inst_ptr(index),
            inst_ptr(active),
            inst_ptr(result)
        );

        nb::inst_mark_ready(result);
        return result;
    }

    ArrayMeta m = source_meta;
    m.is_vector = dtype_meta.is_vector;
    m.is_complex = dtype_meta.is_complex;
    m.is_matrix = dtype_meta.is_matrix;
    m.is_quaternion = dtype_meta.is_quaternion;
    m.ndim = dtype_meta.ndim;
    memcpy(m.shape, dtype_meta.shape, sizeof(ArrayMeta::shape));

    if (m == dtype_meta && m.ndim > 0 && m.shape[m.ndim - 1] == DRJIT_DYNAMIC &&
        m.shape[0] != DRJIT_DYNAMIC) {
        nb::object result = dtype();
        nb::type_object sub_tp = nb::borrow<nb::type_object>(dtype_supp.value);
        nb::int_ sub_size(m.shape[0]);

        for (size_t i = 0; i < m.shape[0]; ++i)
            result[i] =
                gather(sub_tp, source, index * sub_size + nb::int_(i), active);

        return result;
    }

    throw nb::type_error("drjit.gather(): unsupported dtype!");
}

void scatter(nb::object target, nb::object value, nb::object index,
             nb::object active) {
    nb::handle target_tp = target.type(),
                value_tp = value.type();

    bool is_drjit_target_1d = is_drjit_type(target_tp);

    if (is_drjit_target_1d) {
        const ArraySupplement &s = supp(target_tp);
        is_drjit_target_1d = s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC;
    }

    // Recurse through pytrees
    if (!is_drjit_target_1d && target_tp.is(value_tp)) {
        bool is_seq = PySequence_Check(value.ptr()),
             is_dict = value_tp.is(&PyDict_Type);

        size_t len = 0;
        if (is_seq || is_dict) {
            len = nb::len(value);

            if (len != nb::len(target))
                throw std::runtime_error("drjit.scatter(): 'target' and 'value' "
                                         "have incompatible lengths!");
        }

        if (is_seq) {
            for (size_t i = 0, l = len; i < l; ++i)
                scatter(target[i], value[i], index, active);
            return;
        }

        if (is_dict) {
            for (nb::handle k : nb::borrow<nb::dict>(value).keys())
                scatter(target[k], value[k], index, active);
            return;
        }

        nb::object dstruct = nb::getattr(target_tp, "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);

            for (auto [k, v] : dstruct_dict)
                scatter(nb::getattr(target, k), nb::getattr(value, k),
                        index, active);

            return;
        }
    }

    if (!is_drjit_target_1d)
        throw nb::type_error(
            "drjit.scatter(): 'target' argument must be a dynamic 1D array!");

    const ArraySupplement &target_supp = supp(target_tp);

    ArrayMeta target_meta = target_supp,
              active_meta = target_meta,
              index_meta = target_meta;

    active_meta.type = (uint16_t) VarType::Bool;
    index_meta.type = (uint16_t) VarType::UInt32;

    nb::handle active_tp = meta_get_type(active_meta),
               index_tp = meta_get_type(index_meta);

    if (!index.type().is(index_tp)) {
        try {
            index = index_tp(index);
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_TypeError,
                "drjit.scatter(): 'index' argument has an unsupported type, "
                "please provide an instance that is convertible to "
                "drjit.uint32_array_t(target).");
        }
    }

    if (!active.type().is(active_tp)) {
        try {
            active = active_tp(active);
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_TypeError,
                "drjit.scatter(): 'active' argument has an unsupported type, "
                "please provide an instance that is convertible to "
                "drjit.mask_t(target).");
        }
    }

    if (!is_drjit_type(value_tp)) {
        try {
            value = target.type()(value);
            value_tp = value.type();
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_TypeError,
                "drjit.scatter(): 'value' argument has an unsupported type! "
                "Please provide an instance that is convertible to "
                "type(target).");
        }
    }

    const ArraySupplement &value_supp = supp(value_tp);
    ArrayMeta value_meta = value_supp;

    if (value_meta == target_meta) {
        target_supp.scatter(
            inst_ptr(value),
            inst_ptr(index),
            inst_ptr(active),
            inst_ptr(target)
        );

        return;
    }

    ArrayMeta m = target_meta;
    m.is_vector = value_meta.is_vector;
    m.is_complex = value_meta.is_complex;
    m.is_matrix = value_meta.is_matrix;
    m.is_quaternion = value_meta.is_quaternion;
    m.ndim = value_meta.ndim;
    for (int i = 0; i < 4; ++i)
        m.shape[i] = value_meta.shape[i];

    if (m == value_meta && m.ndim > 0 && m.shape[m.ndim - 1] == DRJIT_DYNAMIC &&
        m.shape[0] != DRJIT_DYNAMIC) {
        nb::int_ sub_size(m.shape[0]);
        for (size_t i = 0; i < m.shape[0]; ++i)
            ::scatter(target, value[i],
                      index * sub_size + nb::int_(i), active);
        return;
    }

    throw nb::type_error("drjit.scatter(): 'value' type is unsupported.");
}

static void ravel_recursive(nb::handle result, nb::handle value,
                            nb::handle index_dtype, const size_t *shape,
                            const int64_t *strides, Py_ssize_t offset,
                            int depth, int stop_depth) {
    if (depth == stop_depth) {
        if (index_dtype.is_valid()) {
            nb::object index =
                arange(nb::borrow<nb::type_object_t<ArrayBase>>(index_dtype), offset,
                       offset + strides[depth] * shape[depth], strides[depth]);
            scatter(nb::borrow(result), nb::borrow(value), index, nb::cast(true));
        } else {
            result[offset] = value;
        }
    } else {
        for (size_t i = 0; i < shape[depth]; ++i) {
            ravel_recursive(result, value[i], index_dtype, shape, strides,
                            offset, depth + 1, stop_depth);
            offset += strides[depth];
        }
    }
}

nb::object ravel(nb::handle h, char order,
                 dr_vector<size_t> *shape_out,
                 dr_vector<int64_t> *strides_out,
                 const VarType *vt_in) {

    nb::handle tp = h.type();
    JitBackend backend = JitBackend::Invalid;
    VarType vt = VarType::Float32;
    bool is_dynamic = false;

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if (s.is_tensor) {
            if (order != 'C' && order != 'A')
                throw std::runtime_error("drjit.ravel(): tensors do not "
                                         "support F-style ordering for now.");

            const dr_vector<size_t> &shape = s.tensor_shape(inst_ptr(h));
            if (shape_out)
                *shape_out = shape;

            if (strides_out && !shape.empty()) {
                strides_out->resize(shape.size());

                int64_t stride = 1;
                for (size_t i = shape.size() - 1; ; --i) {
                    strides_out->operator[](i) = (int64_t) stride;
                    stride *= shape[i];
                    if (i == 0)
                        break;
                }
            }

            return nb::steal(s.tensor_array(h.ptr()));
        }

        if (s.ndim == 1 && s.shape[0] == DRJIT_DYNAMIC) {
            if (shape_out)
                shape_out->push_back(s.len(inst_ptr(h)));

            if (strides_out)
                strides_out->push_back(1);

            return nb::borrow(h);
        }

        backend = (JitBackend) s.backend;
        vt = (VarType) s.type;
        is_dynamic = s.shape[s.ndim - 1] == DRJIT_DYNAMIC;
    } else if (vt_in) {
        vt = (VarType) *vt_in;
    }

    dr_vector<size_t> shape;
    dr_vector<int64_t> strides;
    if (!shape_impl(h, shape))
        throw std::runtime_error("drjit.ravel(): ragged input not allowed.");
    strides.resize(shape.size());

    size_t stride = 1;
    if (order == 'C') {
        if (!shape.empty()) {
            for (size_t i = shape.size() - 1; ; --i) {
                strides[i] = (int64_t) stride;
                stride *= shape[i];
                if (i == 0)
                    break;
            }
        }
    } else if (order == 'F' || order == 'A') {
        for (size_t i = 0; i < shape.size() ; ++i) {
            strides[i] = (int64_t) stride;
            stride *= shape[i];
        }
    } else {
        throw std::runtime_error(
            "drjit.ravel(): order parameter must equal 'A', 'C', or 'F'.");
    }

    ArrayMeta m { };
    m.backend = (uint16_t) backend;
    m.type = (int16_t) vt;
    m.ndim = 1;
    m.shape[0] = DRJIT_DYNAMIC;

    size_t size = stride;

    // Create an empty array of the right shape
    nb::object result = full(meta_get_type(m), nb::handle(), 1, &size);

    nb::handle index_dtype;
    if (is_dynamic) {
        m.type = (uint16_t) VarType::UInt32;
        index_dtype = meta_get_type(m);
    }

    ravel_recursive(result, h, index_dtype, shape.data(), strides.data(), 0, 0,
                    (int) shape.size() - is_dynamic);

    if (shape_out)
        *shape_out = std::move(shape);

    if (strides_out)
        *strides_out = std::move(strides);

    return result;
}

static nb::object unravel_recursive(nb::handle dtype,
                                    nb::handle value,
                                    nb::handle index_dtype,
                                    const Py_ssize_t *shape,
                                    const Py_ssize_t *strides,
                                    Py_ssize_t offset, int depth,
                                    int stop_depth) {
    if (depth == stop_depth) {
        if (index_dtype.is_valid()) {
            nb::object index = arange(
                nb::borrow<nb::type_object_t<ArrayBase>>(index_dtype),
                offset, offset + strides[depth] * shape[depth], strides[depth]);
            return gather(nb::borrow<nb::type_object>(dtype), nb::borrow(value),
                          index, nb::cast(true));
        } else {
            return value[offset];
        }
    } else {
        const ArraySupplement &s = supp(dtype);

        nb::object result = dtype();
        for (Py_ssize_t i = 0; i < shape[depth]; ++i) {
            result[i] =
                unravel_recursive(s.value, value, index_dtype, shape, strides,
                                  offset, depth + 1, stop_depth);
            offset += strides[depth];
        }

        return result;
    }
}

nb::object unravel(const nb::type_object_t<ArrayBase> &dtype,
                   nb::handle_t<ArrayBase> array, char order) {
    const ArraySupplement &s = supp(dtype);
    if (s.is_tensor)
        throw nb::type_error(
            "drjit.unravel(): 'dtype' cannot be a tensor!");

    ArrayMeta m { };
    m.backend = s.backend;
    m.type = s.type;
    m.ndim = 1;
    m.shape[0] = DRJIT_DYNAMIC;

    nb::handle flat = meta_get_type(m);
    if (!flat.is(array.type())) {
        nb::str flat_name = nb::type_name(flat),
                actual_name = nb::inst_name(array);
        nb::detail::raise_type_error(
            "drjit.unravel(): expected array of type '%s', but got '%s'!",
            flat_name.c_str(), actual_name.c_str());
    }

    if (array.type().is(dtype))
        return nb::borrow(array);

    Py_ssize_t size = (Py_ssize_t) len(array);

    Py_ssize_t shape[4] { }, strides[4] { }, stride = 1;
    int ndim = s.ndim;
    for (int i = 0; i < ndim; ++i) {
        if (s.shape[i] == DRJIT_DYNAMIC) {
            if (i != s.ndim - 1)
                throw nb::type_error("drjit.unravel(): only the last dimension "
                                     "of 'dtype' may be dynamic!");
            shape[i] = size / stride;
        } else {
            shape[i] = s.shape[i];
        }
        stride *= shape[i];
    }

    if (size != stride)
        throw std::runtime_error("dtype.unravel(): input array size is not "
                                 "divisible by 'dtype' shape!");

    stride = 1;
    if (order == 'C') {
        for (int i = ndim - 1; ; --i) {
            strides[i] = stride;
            stride *= shape[i];
            if (i == 0)
                break;
        }
    } else if (order == 'F' || order == 'A') {
        for (int i = 0; i < ndim; ++i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    } else {
        throw std::runtime_error(
            "drjit.unravel(): order parameter must equal 'C' or 'F'.");
    }

    nb::handle index_dtype;
    if (s.shape[s.ndim - 1] == DRJIT_DYNAMIC) {
        m.type = (uint16_t) VarType::UInt32;
        index_dtype = meta_get_type(m);
    }

    return unravel_recursive(dtype, array, index_dtype, shape, strides, 0, 0,
                             (int) ndim - index_dtype.is_valid());
}

void export_memop(nb::module_ &m) {
    m.def("gather", &gather, "dtype"_a, "source"_a, "index"_a,
          "active"_a = true, nb::raw_doc(doc_gather))
     .def("scatter", &scatter, "target"_a, "value"_a, "index"_a,
          "active"_a = true, nb::raw_doc(doc_scatter))
     .def("ravel",
          [](nb::handle array, char order) {
              return ravel(array, order);
          }, "array"_a, "order"_a = 'A', nb::raw_doc(doc_ravel))
     .def("unravel",
          [](const nb::type_object_t<ArrayBase> &dtype,
             nb::handle_t<ArrayBase> array,
             char order) { return unravel(dtype, array, order); },
          "dtype"_a, "array"_a, "order"_a = 'A', nb::raw_doc(doc_unravel));
}
