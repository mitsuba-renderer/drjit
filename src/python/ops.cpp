#include "python.h"
#include "docstr.h"
#include <nanobind/stl/vector.h>
#include <iostream>


nb::object full_alt(nb::type_object dtype, nb::handle value, size_t size);
nb::object full(nb::type_object dtype, nb::handle value,
                const std::vector<size_t> &shape) {
    if (is_drjit_type(dtype)) {
        const supp &s = nb::type_supplement<supp>(dtype);
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

            if (s.ops.op_full && shape.size() == 1) {
                if ((VarType) s.meta.type == VarType::Bool && value.type().is(&PyLong_Type))
                    value = nb::cast<int>(value) ? Py_True : Py_False;

                s.ops.op_full(value, shape[0], nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }

            nb::inst_zero(result);
            if (s.meta.shape[0] == 0xFF)
                s.ops.init(nb::inst_ptr<void>(result), shape[0]);

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

    throw std::runtime_error(
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

extern void bind_ops(nb::module_ m) {
    m.def("all", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_array(h)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce_mask op = s.ops.op_all;
            if (!op)
                throw nb::type_error(
                    "drjit.all(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_ready(result);
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
            dr::detail::array_reduce_mask op = s.ops.op_any;
            if (!op)
                throw nb::type_error(
                    "drjit.any(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_ready(result);
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
}
