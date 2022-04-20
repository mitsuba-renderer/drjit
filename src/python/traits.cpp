#include "python.h"
#include "docstr.h"

static nb::handle scalar_t(nb::handle h) {
    nb::handle tp = h.is_type() ? h : h.type();
    while (is_drjit_type(tp))
        tp = nb::type_supplement<supp>(tp).value;
    return tp;
}

extern void bind_traits(nb::module_ m) {
    m.attr("Dynamic") = (Py_ssize_t) -1;

    m.def("is_array_v", [](nb::handle h) -> bool {
        return is_drjit_type(h.is_type() ? h : h.type());
    }, nb::raw_doc(doc_is_array_v));

    m.def("array_size_v", [](nb::handle h) -> Py_ssize_t {
        nb::handle tp = h.is_type() ? h : h.type();
        if (is_drjit_type(tp)) {
            Py_ssize_t shape = nb::type_supplement<supp>(tp).meta.shape[0];
            if (shape == 0xFF)
                shape = -1;
            return shape;
        } else {
            return 1;
        }
    }, nb::raw_doc(doc_array_size_v));

    m.def("array_depth_v", [](nb::handle h) -> size_t {
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
    }, nb::raw_doc(doc_array_depth_v));

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

    m.def(
        "is_float_v",
        [](nb::handle h) -> bool {
            return (PyTypeObject *) scalar_t(h).ptr() == &PyFloat_Type;
        },
        nb::raw_doc(doc_is_float_v));

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
}
