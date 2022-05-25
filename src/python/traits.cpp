/*
    traits.cpp -- implementation of Dr.Jit type traits such as
    is_array_v, uint32_array_t, etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"

static nb::handle scalar_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    while (is_drjit_type(tp))
        tp = nb::type_supplement<supp>(tp).value;
    return tp;
}

nb::handle reinterpret_array_t(nb::handle h, VarType vt) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        meta m = nb::type_supplement<supp>(tp).meta;
        m.type = (uint16_t) vt;
        tp = drjit::detail::array_get(m);
    } else {
        if (vt == VarType::Bool)
            tp = &PyBool_Type;
        else if (vt == VarType::Float16 ||
                 vt == VarType::Float32 ||
                 vt == VarType::Float64)
            tp = &PyFloat_Type;
        else
            tp = &PyLong_Type;
    }
    return tp;
}

static int itemsize_v(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    if (is_drjit_type(tp)) {
        switch ((VarType) nb::type_supplement<supp>(tp).meta.type) {
            case VarType::Bool:
            case VarType::Int8:
            case VarType::UInt8:
                return 1;
            case VarType::Int16:
            case VarType::UInt16:
            case VarType::Float16:
                return 2;
            case VarType::Int32:
            case VarType::UInt32:
            case VarType::Float32:
                return 4;
            case VarType::Int64:
            case VarType::UInt64:
            case VarType::Float64:
                return 8;
            default:
                break;
        }
    }
    throw nb::type_error("Unsupported input type!");
}

bool is_float_v(nb::handle h) {
    return (PyTypeObject *) scalar_t(h).ptr() == &PyFloat_Type;
}

nb::handle leaf_array_t(nb::handle h) {
    if (h.is_type()) {
        if (is_drjit_type(h)) {
            supp *s = (supp *) nb::detail::nb_type_supplement(h.ptr());
            while (s->meta.ndim > 1)
                s = (supp *) nb::detail::nb_type_supplement((PyObject *)s->value);
            return drjit::detail::array_get(s->meta);
        } else {
            nb::handle tp = nb::none();
            nb::object dstruct = nb::getattr(h, "DRJIT_STRUCT", nb::handle());
            if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
                for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                    tp = leaf_array_t(v);
                    if (is_drjit_type(tp)) {
                        const supp &s = nb::type_supplement<supp>(tp);
                        if (s.meta.is_diff && is_float_v(tp))
                            break;
                    }
                }
            }
            return tp;
        }
    }

    if (is_drjit_array(h)) {
        return leaf_array_t(h.type());
    } else if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        nb::handle tp = nb::none();
        for (auto h2 : h) {
            tp = leaf_array_t(h2);
            if (is_drjit_type(tp)) {
                const supp &s = nb::type_supplement<supp>(tp);
                if (s.meta.is_diff && is_float_v(tp))
                    break;
            }
        }
        return tp;
    } else if (nb::isinstance<nb::dict>(h)) {
        return leaf_array_t(nb::borrow<nb::dict>(h).values());
    } else {
        return leaf_array_t(h.type());
    }
}

nb::handle expr_t(nb::handle h0, nb::handle h1) {
    h0 = h0.is_type() ? h0 : h0.type();
    h1 = h1.is_type() ? h1 : h1.type();

    if (h0.ptr() == h1.ptr())
        return h0;

    meta m0;
    if (is_drjit_type(h0))
        m0 = nb::type_supplement<supp>(h0).meta;
    else
        m0 = meta_from_builtin(h0().ptr());

    meta m1;
    if (is_drjit_type(h1))
        m1 = nb::type_supplement<supp>(h1).meta;
    else
        m1 = meta_from_builtin(h1().ptr());

    meta m = meta_promote(m0, m1);

    if (!meta_check(m)) {
        PyErr_Format(PyExc_TypeError, "expr_t(): incompatible types  (%s, %s)!",
                     ((PyTypeObject *)h0.ptr())->tp_name,
                     ((PyTypeObject *)h1.ptr())->tp_name);
        return nb::handle();
    }

    return drjit::detail::array_get(m);
}

nb::handle expr_t(nb::args args) {
    nb::handle tp = args[0];
    for (auto h : args)
        tp = expr_t(tp, h);
    return tp;
}

nb::handle detached_t(nb::handle h) {
    h = h.is_type() ? h : h.type();

    if (is_drjit_type(h)) {
        const supp &s = nb::type_supplement<supp>(h);
        meta detached_meta = s.meta;
        detached_meta.is_diff = false;
        return drjit::detail::array_get(detached_meta);
    }

    return h;
}

extern void bind_traits(nb::module_ m) {
    m.attr("Dynamic") = (Py_ssize_t) -1;

    m.def("is_array_v", [](nb::handle h) -> bool {
        return is_drjit_type(h.is_type() ? h : h.type());
    }, nb::raw_doc(doc_is_array_v));

    m.def("size_v", [](nb::handle h) -> Py_ssize_t {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            Py_ssize_t shape = nb::type_supplement<supp>(tp).meta.shape[0];
            if (shape == DRJIT_DYNAMIC)
                shape = -1;
            return shape;
        } else {
            return 1;
        }
    }, nb::raw_doc(doc_size_v));

    m.def("depth_v", [](nb::handle h) -> size_t {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            const uint8_t *shape = nb::type_supplement<supp>(tp).meta.shape;
            int depth = 1;
            for (int i = 1; i < 4; ++i) {
                if (!shape[i])
                    break;
                depth++;
            }
            return depth;
        } else {
            return 0;
        }
    }, nb::raw_doc(doc_depth_v));

    m.def("itemsize_v", itemsize_v, nb::raw_doc(doc_itemsize_v));

    m.def("value_t", [](nb::handle h) -> nb::handle {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).value;
        else
            return tp;
    }, nb::raw_doc(doc_value_t));

    m.def("mask_t", [](nb::handle h) -> nb::handle {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).mask;
        else
            return &PyBool_Type;
    }, nb::raw_doc(doc_mask_t));

    m.def("array_t", [](nb::handle h) -> nb::handle {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).array;
        else
            return tp;
    }, nb::raw_doc(doc_array_t));

    m.def("scalar_t", scalar_t, nb::raw_doc(doc_scalar_t));

    m.def(
        "is_mask_v",
        [](nb::handle h) -> bool {
            return (PyTypeObject *) scalar_t(h).ptr() == &PyBool_Type;
        },
        nb::raw_doc(doc_is_mask_v));

    m.def("is_float_v", &is_float_v, nb::raw_doc(doc_is_float_v));

    m.def(
        "is_integral_v",
        [](nb::handle h) -> bool {
            return (PyTypeObject *) scalar_t(h).ptr() == &PyLong_Type;
        },
        nb::raw_doc(doc_is_integral_v));

    m.def(
        "is_arithmetic_v",
        [](nb::handle h) -> bool {
            PyTypeObject *s = (PyTypeObject *) scalar_t(h).ptr();
            return s == &PyLong_Type || s == &PyFloat_Type;
        },
        nb::raw_doc(doc_is_arithmetic_v));

    m.def(
        "is_signed_v",
        [](nb::handle h) -> bool {
            nb::handle tp = h.is_type() ? h : h.type();
            if (is_drjit_type(tp)) {
                VarType vt = (VarType) nb::type_supplement<supp>(tp).meta.type;
                return vt == VarType::Int8 || vt == VarType::Int16 ||
                       vt == VarType::Int32 || vt == VarType::Int64 ||
                       vt == VarType::Float16 || vt == VarType::Float32 ||
                       vt == VarType::Float64;
            } else {
                return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type);
            }
        },
        nb::raw_doc(doc_is_signed_v));

    m.def(
        "is_unsigned_v",
        [](nb::handle h) -> bool {
            nb::handle tp = h.is_type() ? h : h.type();
            if (is_drjit_type(tp)) {
                VarType vt = (VarType) nb::type_supplement<supp>(tp).meta.type;
                return vt == VarType::UInt8 || vt == VarType::UInt16 ||
                       vt == VarType::UInt32 || vt == VarType::UInt64 ||
                       vt == VarType::Bool;
            } else {
                return tp.is(&PyBool_Type);
            }
        },
        nb::raw_doc(doc_is_unsigned_v));

    m.def("is_jit_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            const auto &m = nb::type_supplement<supp>(tp).meta;
            return m.is_cuda || m.is_llvm;
        }
        return false;
    }, nb::raw_doc(doc_is_jit_v));

    m.def("is_llvm_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_llvm;
        return false;
    }, nb::raw_doc(doc_is_llvm_v));

    m.def("is_cuda_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_cuda;
        return false;
    }, nb::raw_doc(doc_is_cuda_v));

    m.def("is_diff_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_diff;
        return false;
    }, nb::raw_doc(doc_is_diff_v));

    m.def("is_complex_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_complex;
        return false;
    }, nb::raw_doc(doc_is_complex_v));

    m.def("is_quaternion_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_quaternion;
        return false;
    }, nb::raw_doc(doc_is_quaternion_v));

    m.def("is_matrix_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_matrix;
        return false;
    }, nb::raw_doc(doc_is_matrix_v));

    m.def("is_special_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            auto const &m = nb::type_supplement<supp>(tp).meta;
            return m.is_complex || m.is_quaternion || m.is_matrix;
        }
        return false;
    }, nb::raw_doc(doc_is_special_v));

    m.def("is_tensor_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp))
            return nb::type_supplement<supp>(tp).meta.is_tensor;
        return false;
    }, nb::raw_doc(doc_is_tensor_v));

    m.def("is_struct_v", [](nb::handle h) -> bool {
        nb::handle tp = h.is_type() ? h : h.type();
        return nb::hasattr(tp, "DRJIT_STRUCT");
    }, nb::raw_doc(doc_is_struct_v));

    m.def(
        "bool_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::Bool); },
        doc_uint32_array_t);

    m.def(
        "uint32_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::UInt32); },
        doc_uint32_array_t);

    m.def(
        "int32_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::Int32); },
        doc_int32_array_t);

    m.def(
        "uint64_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::UInt64); },
        doc_uint64_array_t);

    m.def(
        "int64_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::Int64); },
        doc_int64_array_t);

    m.def(
        "float32_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::Float32); },
        doc_float32_array_t);

    m.def(
        "float64_array_t",
        [](nb::handle h) { return reinterpret_array_t(h, VarType::Float64); },
        doc_float64_array_t);

    m.def(
        "uint_array_t",
        [](nb::handle h) {
            VarType vt;
            switch (itemsize_v(h)) {
                case 1: vt = VarType::UInt8; break;
                case 2: vt = VarType::UInt16; break;
                case 4: vt = VarType::UInt32; break;
                case 8: vt = VarType::UInt64; break;
                default: throw nb::type_error("Unsupported input type!");
            }
            return reinterpret_array_t(h, vt);
        },
        doc_uint_array_t);


    m.def(
        "int_array_t",
        [](nb::handle h) {
            VarType vt;
            switch (itemsize_v(h)) {
                case 1: vt = VarType::Int8; break;
                case 2: vt = VarType::Int16; break;
                case 4: vt = VarType::Int32; break;
                case 8: vt = VarType::Int64; break;
                default: throw nb::type_error("Unsupported input type!");
            }
            return reinterpret_array_t(h, vt);
        },
        doc_int_array_t);


    m.def(
        "float_array_t",
        [](nb::handle h) {
            VarType vt;
            switch (itemsize_v(h)) {
                case 2: vt = VarType::Float16; break;
                case 4: vt = VarType::Float32; break;
                case 8: vt = VarType::Float64; break;
                default: throw nb::type_error("Unsupported input type!");
            }
            return reinterpret_array_t(h, vt);
        },
        doc_float_array_t);

    m.def("detached_t", &detached_t, doc_detached_t);
    m.def("leaf_array_t", &leaf_array_t, doc_leaf_array_t);
    m.def("expr_t", nb::overload_cast<nb::args>(expr_t), doc_expr_t);
}
