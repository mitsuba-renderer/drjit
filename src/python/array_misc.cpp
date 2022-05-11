#include "python.h"
#include <nanobind/stl/vector.h>

nb::object full_alt(nb::type_object dtype, nb::handle value, size_t size);
nb::object empty_alt(nb::type_object dtype, size_t size);

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

nb::object empty(nb::type_object dtype, const std::vector<size_t> &shape) {
    if (is_drjit_type(dtype)) {
        const supp &s = nb::type_supplement<supp>(dtype);

        // if (s.meta.is_tensor) {
        //     size_t size = 1;
        //     for (size_t s : shape)
        //         size *= s;
        //     return dtype(empty_alt(s.array_type, size), shape);
        // }

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

            if (s.op_empty && shape.size() == 1) {
                s.op_empty(shape[0], nb::inst_ptr<void>(result));
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
                sq_ass_item(result.ptr(), i, empty(sub_type, sub_shape).ptr());

            return result;
        }
    } else if (dtype.is(&PyLong_Type) || dtype.is(&PyFloat_Type) || dtype.is(&PyBool_Type)) {
        if (shape.empty() || (shape.size() == 1 && shape[0] == 1))
            return dtype(0);
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
                    entry = empty_alt(sub_dtype, shape[0]);
                else
                    entry = empty(sub_dtype, shape);

                nb::setattr(result, k, entry);
            }

            return result;
        }

        throw nb::type_error("Unsupported dtype!");
    }

    nb::detail::raise(
        "The provided 'shape' and 'dtype' parameters are incompatible!");
}


nb::object empty_alt(nb::type_object dtype, size_t size) {
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

    return empty(dtype, shape);
}

nb::object arange(nb::handle dtype, Py_ssize_t start, Py_ssize_t end,
                  Py_ssize_t step) {
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

            if (start == 0 && step == 1)
                return dtype(result);
            else
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

            double step = (end - start) / (size - ((endpoint && size > 0) ? 1 : 0));

            nb::object result = nb::inst_alloc(counter_tp);
            counter_supp.op_counter((size_t) size, nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);

            return array_module.attr("fma")(dtype(result), dtype(step), dtype(start));
        }
    }

    throw nb::type_error("drjit.linspace(): unsupported dtype!");
}

nb::object gather(nb::type_object dtype, nb::handle_of<dr::ArrayBase> source,
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

    if (!index.type().is(index_t)) {
        try {
            index = index_t(index);
        } catch (...) {
            throw nb::type_error(
                "drjit.gather(): 'index' argument has an unsupported type, "
                "please provide an instance that is convertible into "
                "drjit.uint32_array_t(type(source)).");
        }
    }

    if (!active.type().is(active_t)) {
        try {
            active = active_t(active);
        } catch (...) {
            throw nb::type_error(
                "drjit.gather(): 'active' argument has an unsupported type, "
                "please provide an instance that is convertible into "
                "drjit.mask_t(type(source)).");
        }
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
    for (int i = 0; i < m.ndim; ++i)
        m.shape[i] = dtype_meta.shape[i];

    if (m == dtype_meta && m.ndim > 0 && m.shape[m.ndim - 1] == 0xFF &&
        m.shape[0] != 0xFF) {
        nb::object result = dtype();
        for (size_t i = 0; i < m.shape[0]; ++i) {
            result[i] =
                gather(nb::borrow<nb::type_object>(dtype_s.value), source,
                       index * nb::cast(m.shape[0]) + nb::cast(i), active);
        }
        return result;
    }

    throw nb::type_error("drjit.gather(): 'dtype' unsupported!");
}

void scatter(nb::handle_of<dr::ArrayBase> target,
             nb::object value,
             nb::object index,
             nb::object active) {
    const supp &target_s = nb::type_supplement<supp>(target.type());

    if (target_s.meta.ndim != 1 || target_s.meta.shape[0] != 0xFF)
        throw nb::type_error(
            "drjit.scatter(): 'target' argument must be a dynamic 1D array!");

    meta target_meta = target_s.meta,
         active_meta = target_meta,
         index_meta = target_meta;

    active_meta.type = (uint16_t) VarType::Bool;
    index_meta.type = (uint16_t) VarType::UInt32;

    nb::handle active_t = drjit::detail::array_get(active_meta),
               index_t = drjit::detail::array_get(index_meta);

    if (!index.type().is(index_t)) {
        try {
            index = index_t(index);
        } catch (...) {
            throw nb::type_error(
                "drjit.scatter(): 'index' argument has an unsupported type, "
                "please provide an instance that is convertible into "
                "drjit.uint32_array_t(type(target)).");
        }
    }

    if (!active.type().is(active_t)) {
        try {
            active = active_t(active);
        } catch (...) {
            throw nb::type_error(
                "drjit.scatter(): 'active' argument has an unsupported type, "
                "please provide an instance that is convertible into "
                "drjit.mask_t(type(target)).");
        }
    }

    if (!is_drjit_type(value.type())) {
        try {
            value = target.type()(value);
        } catch (...) {
            throw nb::type_error(
                "drjit.scatter(): 'value' argument has an unsupported type! "
                "Please provide an instance that is convertible into "
                "type(target).");
        }
    }

    const supp &value_s = nb::type_supplement<supp>(value.type());
    meta value_meta = value_s.meta;

    if (value_meta == target_meta) {
        target_s.op_scatter(
            nb::inst_ptr<void>(value),
            nb::inst_ptr<void>(index),
            nb::inst_ptr<void>(active),
            nb::inst_ptr<void>(target)
        );
        return;
    }

    meta m = target_meta;
    m.is_vector = value_meta.is_vector;
    m.is_complex = value_meta.is_complex;
    m.is_matrix = value_meta.is_matrix;
    m.is_quaternion = value_meta.is_quaternion;
    m.ndim = value_meta.ndim;
    for (int i = 0; i < m.ndim; ++i)
        m.shape[i] = value_meta.shape[i];

    if (m == value_meta && m.ndim > 0 && m.shape[0] != 0xFF) {
        if (m.shape[m.ndim - 1] != 0xFF || m.ndim > 1) {
            for (size_t i = 0; i < m.shape[0]; ++i)
                ::scatter(target, value[i],
                          index * nb::cast(m.shape[0]) + nb::cast(i), active);
        } else {
            for (size_t i = 0; i < m.shape[0]; ++i)
                if (nb::cast<bool>(active[i]))
                    target[index * nb::cast(m.shape[0]) + nb::cast(i)] = value[i];
        }
        return;
    }

    throw nb::type_error("drjit.scatter(): 'value' type is unsupported.");
}

static void ravel_recursive(nb::handle result, nb::handle value,
                            nb::handle index_dtype, const Py_ssize_t *shape,
                            const Py_ssize_t *strides, Py_ssize_t offset,
                            int depth, int stop_depth) {
    if (depth == stop_depth) {
        if (index_dtype.is_valid()) {
            nb::object index =
                arange(index_dtype, offset,
                       offset + strides[depth] * shape[depth], strides[depth]);
            scatter(result, nb::borrow(value), index, nb::cast(true));
        } else {
            result[offset] = value;
        }
    } else {
        for (Py_ssize_t i = 0; i < shape[depth]; ++i) {
            ravel_recursive(result, value[i], index_dtype, shape, strides,
                            offset, depth + 1, stop_depth);
            offset += strides[depth];
        }
    }
}

nb::object ravel(nb::handle_of<dr::ArrayBase> h, char order,
                 std::vector<size_t> *shape_out,
                 std::vector<int64_t> *strides_out) {
    const supp &s = nb::type_supplement<supp>(h.type());

    if (s.meta.is_tensor) {
        if (order != 'C')
            throw std::runtime_error("drjit.ravel(): tensors only support "
                                     "C-style ordering for now.");

        return nb::steal(s.op_tensor_array(h.ptr()));
    }

    if (s.meta.ndim == 1 && s.meta.shape[0] == 0xFF)
        return borrow(h);

    nb::object shape_tuple = shape(h);
    if (shape_tuple.is_none())
        throw std::runtime_error("drjit.ravel(): ragged input not allowed.");

    Py_ssize_t shape[4] { }, strides[4] { }, stride = 1;

    size_t ndim = nb::len(shape_tuple);
    for (size_t i = 0; i < ndim; ++i)
        shape[i] = nb::cast<Py_ssize_t>(shape_tuple[i]);

    if (order == 'C') {
        for (size_t i = ndim - 1; ; --i) {
            strides[i] = stride;
            stride *= shape[i];
            if (i == 0)
                break;
        }
    } else if (order == 'F') {
        for (size_t i = 0; i < ndim; ++i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    } else {
        throw std::runtime_error(
            "drjit.ravel(): order parameter must equal 'C' or 'F'.");
    }

    meta m { };
    m.is_llvm = s.meta.is_llvm;
    m.is_cuda = s.meta.is_cuda;
    m.is_diff = s.meta.is_diff;
    m.type = s.meta.type;
    m.ndim = 1;
    m.shape[0] = 0xFF;

    nb::object result = empty_alt(
        nb::borrow<nb::type_object>(drjit::detail::array_get(m)), stride);

    nb::handle index_dtype;
    if (s.meta.shape[s.meta.ndim - 1] == 0xFF) {
        m.type = (uint16_t) VarType::UInt32;
        index_dtype = drjit::detail::array_get(m);
    }

    ravel_recursive(result, h, index_dtype, shape, strides, 0, 0,
                    (int) ndim - index_dtype.is_valid());

    if (shape_out && strides_out) {
        shape_out->resize(ndim);
        strides_out->resize(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            shape_out->operator[](i) = (size_t) shape[i];
            strides_out->operator[](i) = (int64_t) strides[i];
        }
    }

    return result;
}

bool schedule(nb::handle h) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (!s.meta.is_cuda && !s.meta.is_llvm)
            return false;

        if (s.meta.ndim == 1)
            return jit_var_schedule(s.op_index(nb::inst_ptr<void>(h)));
    }


    if (nb::isinstance<nb::sequence>(h)) {
        bool result = false;
        for (nb::handle h2 : h)
            result |= schedule(h2);
        return result;
    } else if (nb::isinstance<nb::mapping>(h)) {
        return schedule(nb::borrow<nb::mapping>(h).values());
    }

    nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
    if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
        nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
        bool result = false;
        for (auto [k, v] : dstruct_dict)
            result |= schedule(nb::getattr(h, k));
        return result;
    }

    return false;
}

bool eval(nb::handle h) {
    bool rv = schedule(h);
    if (rv)
        jit_eval();
    return rv;
}

static bool eval(nb::args args) {
    bool rv = schedule(args);
    if (rv || nb::len(args) == 0)
        jit_eval();
    return rv;
}

static bool schedule(nb::args args) {
    bool rv = false;
    for (nb::handle h : args)
        rv |= schedule(h);
    return rv;
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

    m.def("empty", empty, "dtype"_a, "shape"_a, doc_empty);
    m.def("empty", empty_alt, "dtype"_a, "shape"_a = 1);

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

    m.def("gather", &gather, "dtype"_a, "source"_a, "index"_a,
          "active"_a = true, doc_gather);

    m.def("scatter", &scatter, "target"_a, "value"_a, "index"_a,
          "active"_a = true, doc_scatter);

    m.def(
        "ravel",
        [](nb::handle_of<dr::ArrayBase> a, char order) {
            return ravel(a, order);
        },
        "arg"_a, "order"_a = 'C', doc_ravel);

    m.def("schedule", nb::overload_cast<nb::args>(schedule), doc_schedule);
    m.def("eval", nb::overload_cast<nb::args>(eval), doc_eval);
}
