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
            nb::inst_ptr<dr::ArrayBase>(source),
            nb::inst_ptr<dr::ArrayBase>(index),
            nb::inst_ptr<dr::ArrayBase>(active),
            nb::inst_ptr<dr::ArrayBase>(result)
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
        } catch (...) {
            throw nb::type_error(
                "drjit.scatter(): 'value' argument has an unsupported type! "
                "Please provide an instance that is convertible to "
                "type(target).");
        }
    }

    const ArraySupplement &value_supp = supp(value_tp);
    ArrayMeta value_meta = value_supp;

    if (value_meta == target_meta) {
        target_supp.scatter(
            nb::inst_ptr<dr::ArrayBase>(value),
            nb::inst_ptr<dr::ArrayBase>(index),
            nb::inst_ptr<dr::ArrayBase>(active),
            nb::inst_ptr<dr::ArrayBase>(target)
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

void export_memop(nb::module_ &m) {
    m.def("gather", &gather, "dtype"_a, "source"_a, "index"_a,
          "active"_a = true, nb::raw_doc(doc_gather));
    m.def("scatter", &scatter, "target"_a, "value"_a, "index"_a,
          "active"_a = true, nb::raw_doc(doc_scatter));
}
