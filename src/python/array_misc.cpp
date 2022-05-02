#include "python.h"
#include <nanobind/stl/vector.h>

nb::object full_alt(nb::type_object dtype, nb::handle value, size_t size);

nb::object full(nb::type_object dtype, nb::handle value,
                const std::vector<size_t> &shape) {
    if (is_drjit_type(dtype)) {
        const supp &s = nb::type_supplement<supp>(dtype);

        // if (s.meta.is_tensor) {
        //     size_t size = 1;
        //     for (size_t s : shape)
        //         size *= s;
        //     return dtype(full_alt(s.array_type, value, size), shape);
        // }
        //
        bool fail = false;
        if (s.meta.ndim == shape.size()) {
            for (uint16_t i = 0; i < s.meta.ndim; ++i) {
                if (s.meta.shape[i] != 0xFF && s.meta.shape[i] != shape[i])
                    fail = true;
            }
        } else {
            fail = true;
        }

        if (!fail) {
            const supp &s = nb::type_supplement<supp>(dtype);
            nb::object result = nb::inst_alloc(dtype);

            if (s.op_full && shape.size() == 1) {
                if ((VarType) s.meta.type == VarType::Bool && value.type().is(&PyLong_Type))
                    value = nb::cast<int>(value) ? Py_True : Py_False;

                s.op_full(value, shape[0], nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }

            nb::inst_zero(result);
            if (s.meta.shape[0] == 0xFF)
                s.init(nb::inst_ptr<void>(result), shape[0]);

            nb::type_object sub_type = nb::borrow<nb::type_object>(s.value);
            std::vector<size_t> sub_shape = shape;
            sub_shape.erase(sub_shape.begin());

            auto sq_ass_item =
                ((PyTypeObject *) dtype.ptr())->tp_as_sequence->sq_ass_item;

            for (size_t i = 0; i < shape[0]; ++i)
                sq_ass_item(result.ptr(), i, full(sub_type, value, sub_shape).ptr());

            return result;
        }
    } else if (dtype.is(&PyLong_Type) || dtype.is(&PyFloat_Type) || dtype.is(&PyBool_Type)) {
        if (shape.empty() || (shape.size() == 1 && shape[0] == 1))
            return dtype(value);
    } else {
        nb::object dstruct = nb::getattr(dtype, "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
            nb::object result = dtype();

            for (auto [k, v] : dstruct_dict) {
                nb::object entry;
                if (!v.is_type())
                    throw nb::type_error("DRJIT_STRUCT invalid, expected types!");

                nb::type_object sub_dtype = nb::borrow<nb::type_object>(v);

                if (is_drjit_type(v) && shape.size() == 1)
                    entry = full_alt(sub_dtype, value, shape[0]);
                else
                    entry = full(sub_dtype, value, shape);

                nb::setattr(result, k, entry);
            }

            return result;
        }

        throw nb::type_error("Unsupported dtype!");
    }

    nb::detail::raise(
        "The provided 'shape' and 'dtype' parameters are incompatible!");
}

nb::object full_alt(nb::type_object dtype, nb::handle value, size_t size) {
    std::vector<size_t> shape;

    if (is_drjit_type(dtype)) {
        const supp &s = nb::type_supplement<supp>(dtype);
        shape.reserve(s.meta.ndim);
        for (uint16_t i = 0; i < s.meta.ndim; ++i) {
            if (s.meta.shape[i] == 0xFF) {
                shape.push_back(size);
                break;
            } else {
                shape.push_back(s.meta.shape[i]);
            }
        }
    } else {
        shape.push_back(size);
    }

    return full(dtype, value, shape);
}

nb::object arange(nb::handle dtype, Py_ssize_t start, Py_ssize_t end, Py_ssize_t step) {
    if (is_drjit_type(dtype)) {
        const supp &s = nb::type_supplement<supp>(dtype);

        if (s.meta.ndim != 1 || s.meta.shape[0] != 0xFF)
            throw nb::type_error("drjit.arange(): unsupported 'dtype' -- must "
                                 "be a dynamically sized 1D array!");

        VarType vt = (VarType) s.meta.type;
        if (vt == VarType::Bool || vt == VarType::Pointer)
            throw nb::type_error("drjit.arange(): unsupported 'dtype' -- must "
                                 "be an integral type!");

        Py_ssize_t size = (end - start + step - (step > 0 ? 1 : -1)) / step;
        auto counter_meta = s.meta;
        counter_meta.type = (uint16_t) VarType::UInt32;
        nb::handle counter_tp = drjit::detail::array_get(counter_meta);
        const supp &counter_supp = nb::type_supplement<supp>(counter_tp);
        if (counter_supp.op_counter) {
            if (size == 0)
                return dtype();
            else if (size < 0)
                nb::detail::raise("drjit.arange(): size cannot be negative!");

            nb::object result = nb::inst_alloc(counter_tp);
            counter_supp.op_counter((size_t) size, nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);

            return array_module.attr("fma")(dtype(result), dtype(step), dtype(start));
        }
    }
    throw nb::type_error("drjit.arange(): unsupported dtype!");
}

nb::object linspace(nb::handle dtype, double start, double end, size_t size, bool endpoint) {
    if (is_drjit_type(dtype)) {
        const supp &s = nb::type_supplement<supp>(dtype);

        if (s.meta.ndim != 1 || s.meta.shape[0] != 0xFF)
            throw nb::type_error("drjit.linspace(): unsupported 'dtype' -- must "
                                 "be a dynamically sized 1D array!");

        VarType vt = (VarType) s.meta.type;
        if (vt != VarType::Float16 && vt != VarType::Float32 && vt != VarType::Float64)
            throw nb::type_error("drjit.linspace(): unsupported 'dtype' -- must "
                                 "be a floating point array!");

        auto counter_meta = s.meta;
        counter_meta.type = (uint16_t) VarType::UInt32;
        nb::handle counter_tp = drjit::detail::array_get(counter_meta);
        const supp &counter_supp = nb::type_supplement<supp>(counter_tp);

        if (counter_supp.op_counter) {
            if (size == 0)
                return dtype();
            else if (size < 0)
                nb::detail::raise("drjit.linspace(): size cannot be negative!");

            double step = (end - start) / (size - ((endpoint && size > 0) ? 1 : 0));

            nb::object result = nb::inst_alloc(counter_tp);
            counter_supp.op_counter((size_t) size, nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);

            return array_module.attr("fma")(dtype(result), dtype(step), dtype(start));
        }
    }

    throw nb::type_error("drjit.linspace(): unsupported dtype!");
}

nb::object gather_impl(nb::type_object dtype,
                       nb::handle_of<dr::ArrayBase> source,
                       nb::object index, nb::object active) {
    if (!is_drjit_type(dtype))
        throw nb::type_error(
            "drjit.gather(): 'dtype' argument must be a Dr.Jit array!");

    const supp &source_s = nb::type_supplement<supp>(source.type());

    if (source_s.meta.ndim != 1 || source_s.meta.shape[0] != 0xFF)
        throw nb::type_error(
            "drjit.gather(): 'source' argument must be a dynamic 1D array!");

    meta source_meta = source_s.meta,
         active_meta = source_meta,
         index_meta = source_meta;

    active_meta.type = (uint16_t) VarType::Bool;
    index_meta.type = (uint16_t) VarType::UInt32;

    nb::handle active_t = drjit::detail::array_get(active_meta),
               index_t = drjit::detail::array_get(index_meta);

    try {
        if (!index.type().is(index_t))
            index = index_t(index);
    } catch (...) {
        throw nb::type_error("drjit.gather(): 'index' argument has an "
                             "unsupported type, please provide an instance "
                             "that is convertible into drjit.mask_t(index).");
    }

    try {
        if (!active.type().is(active_t))
            active = active_t(active);
    } catch (...) {
        throw nb::type_error("drjit.gather(): 'active' argument has an "
                             "unsupported type, please provide an instance that "
                             "is convertible into drjit.mask_t(index).");
    }

    const supp &dtype_s = nb::type_supplement<supp>(dtype);
    meta dtype_meta = dtype_s.meta;

    if (dtype_meta == source_meta) {
        nb::object result = nb::inst_alloc(dtype);

        source_s.op_gather(
            nb::inst_ptr<void>(source),
            nb::inst_ptr<void>(index),
            nb::inst_ptr<void>(active),
            nb::inst_ptr<void>(result)
        );

        nb::inst_mark_ready(result);
        return result;
    }

    meta m = source_meta;
    m.is_vector = dtype_meta.is_vector;
    m.is_complex = dtype_meta.is_complex;
    m.is_matrix = dtype_meta.is_matrix;
    m.is_quaternion = dtype_meta.is_quaternion;
    m.ndim = dtype_meta.ndim;
    for (int i = 0; i < 4; ++i)
        m.shape[i] = dtype_meta.shape[i];

    if (m == dtype_meta && m.ndim > 0 && m.shape[m.ndim - 1] == 0xFF &&
        m.shape[0] != 0xFF) {
        nb::object result = dtype();
        for (size_t i = 0; i < m.shape[0]; ++i) {
            result[i] = gather_impl(nb::borrow<nb::type_object>(dtype_s.value),
                    source, index * nb::cast(m.shape[0]) + nb::cast(i),
                    active);
        }
        return result;
    }

    throw nb::type_error("drjit.gather(): 'dtype' unsupported!");
}

extern void bind_array_misc(nb::module_ m) {
    m.def("all", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_array(h)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce_mask op = s.op_all;
            if (!op)
                throw nb::type_error(
                    "drjit.all(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }
        }

        nb::object result = nb::borrow(Py_True);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = result & h2;
        }

        return result;
    });

    m.def("any", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_type(tp)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce_mask op = s.op_any;
            if (!op)
                throw nb::type_error(
                    "drjit.any(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }
        }

        nb::object result = nb::borrow(Py_False);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = result | h2;
        }

        return result;
    });

    m.def("full", full, "dtype"_a, "value"_a, "shape"_a, doc_full);
    m.def("full", full_alt, "dtype"_a, "value"_a, "shape"_a = 1);

    m.def(
        "zeros",
        [](nb::type_object dtype, const std::vector<size_t> &shape) {
            return full(dtype, nb::cast(0), shape);
        },
        "dtype"_a, "shape"_a, doc_zeros);

    m.def(
        "zeros",
        [](nb::type_object dtype, size_t size) {
            return full_alt(dtype, nb::cast(0), size);
        },
        "dtype"_a, "shape"_a = 1);

    m.def(
        "ones",
        [](nb::type_object dtype, const std::vector<size_t> &shape) {
            return full(dtype, nb::cast(1), shape);
        },
        "dtype"_a, "shape"_a, doc_ones);

    m.def(
        "ones",
        [](nb::type_object dtype, size_t size) {
            return full_alt(dtype, nb::cast(1), size);
        },
        "dtype"_a, "shape"_a = 1);

    m.def(
        "arange",
        [](nb::type_object dtype, Py_ssize_t size) {
            return arange(dtype, 0, size, 1);
        },
        "dtype"_a, "size"_a, doc_arange);

    m.def(
        "arange",
        [](nb::type_object dtype, Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step) {
            return arange(dtype, start, stop, step);
        },
        "dtype"_a, "start"_a, "stop"_a, "step"_a = 1);

    m.def(
        "linspace",
        [](nb::type_object dtype, double start, double stop, size_t num, bool endpoint) {
            return linspace(dtype, start, stop, num, endpoint);
        },
        "dtype"_a, "start"_a, "stop"_a, "num"_a, "endpoint"_a = true,
        doc_linspace);

    m.def("gather", &gather_impl, "dtype"_a, "source"_a, "index"_a,
          "active"_a = true, doc_gather);
}
