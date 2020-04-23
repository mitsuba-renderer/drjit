#include "common.h"

void export_type_traits(py::module &m) {
    m.def("is_array_v", [](const py::handle h) {
        PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
        return strncmp(type->tp_name, "enoki.", 6) == 0;
    });

    m.def("is_mask_v", [](const py::handle h) {
        return var_type(h) == VarType::Bool;
    });

    m.def("is_arithmetic_v", [](const py::handle h) {
        return jitc_is_arithmetic(var_type(h));
    });

    m.def("is_floating_point_v", [](const py::handle h) {
        return jitc_is_floating_point(var_type(h));
    });

    m.def("is_integral_v", [](const py::handle h) {
        return jitc_is_integral(var_type(h));
    });

    m.def("is_cuda_array_v", [](const py::handle h) {
        PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
        return strncmp(type->tp_name, "enoki.cuda.", 11) == 0;
    });

    m.def("is_llvm_array_v", [](const py::handle h) {
        PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
        return strncmp(type->tp_name, "enoki.llvm.", 11) == 0;
    });

    m.def("is_jit_array_v", [](const py::handle h) {
        PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
        return strncmp(type->tp_name, "enoki.cuda.", 11) == 0 ||
               strncmp(type->tp_name, "enoki.llvm.", 11) == 0;
    });

    m.def("is_diff_array_v", [](const py::handle h) {
        PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
        return strstr(type->tp_name, ".autodiff.") != nullptr;
    });

    m.def("is_signed_v", [](const py::handle h) {
        return !jitc_is_unsigned(var_type(h));
    });

    m.def("is_unsigned_v", [](const py::handle h) {
        return jitc_is_unsigned(var_type(h));
    });

    m.def("array_size_v", [](py::handle h) -> py::object {
        if (var_is_enoki_type(h))
            return h.attr("Size");
        else
            return py::int_(1);
    });

    m.def("array_depth_v", [](py::handle h) -> py::object {
        if (var_is_enoki_type(h))
            return h.attr("Depth");
        else
            return py::int_(0);
    });

    m.attr("Dynamic") = ek::Dynamic;

    m.def("value_t", [](const py::object &o) -> py::object {
        if (var_is_enoki_type(o))
            return o.attr("Value");
        else
            return o;
    });

    m.def("scalar_t", [](const py::object &o) -> py::object {
        if (var_is_enoki_type(o))
            return o.attr("Scalar");
        else
            return o;
    });

    m.def("mask_t", [](py::handle h) -> py::object {
        if (!var_is_enoki_type(h))
            return py::reinterpret_borrow<py::object>((PyObject *) &PyBool_Type);
        py::object type = h.attr("ReplaceScalar")(VarType::Bool);
        return PyType_Check(h.ptr()) ? type : type(h);
    });

    m.def("int_array_t", [](py::handle h) -> py::object {
        if (!var_is_enoki_type(h))
            return py::reinterpret_borrow<py::object>((PyObject *) &PyLong_Type);
        size_t size = py::cast<size_t>(h.attr("Type").attr("Size"));
        VarType vt;
        switch (size) {
            case 1: vt = VarType::Int8; break;
            case 2: vt = VarType::Int16; break;
            case 4: vt = VarType::Int32; break;
            case 8: vt = VarType::Int64; break;
            default:
                throw std::runtime_error("int_array_t(): unsupported input type!");
        }
        py::object type = h.attr("ReplaceScalar")(vt);
        return PyType_Check(h.ptr()) ? type : type(h);
    });

    m.def("uint_array_t", [](py::handle h) -> py::object {
        if (!var_is_enoki_type(h))
            return py::reinterpret_borrow<py::object>((PyObject *) &PyLong_Type);
        size_t size = py::cast<size_t>(h.attr("Type").attr("Size"));
        VarType vt;
        switch (size) {
            case 1: vt = VarType::UInt8; break;
            case 2: vt = VarType::UInt16; break;
            case 4: vt = VarType::UInt32; break;
            case 8: vt = VarType::UInt64; break;
            default:
                throw std::runtime_error("uint_array_t(): unsupported input type!");
        }
        py::object type = h.attr("ReplaceScalar")(vt);
        return PyType_Check(h.ptr()) ? type : type(h);
    });

    m.def("float_array_t", [](py::handle h) -> py::object {
        if (!var_is_enoki_type(h))
            return py::reinterpret_borrow<py::object>((PyObject *) &PyFloat_Type);
        size_t size = py::cast<size_t>(h.attr("Type").attr("Size"));
        VarType vt;
        switch (size) {
            case 2: vt = VarType::Float16; break;
            case 4: vt = VarType::Float32; break;
            case 8: vt = VarType::Float64; break;
            default:
                throw std::runtime_error("float_array_t(): unsupported input type!");
        }
        py::object type = h.attr("ReplaceScalar")(vt);
        return PyType_Check(h.ptr()) ? type : type(h);
    });

    #define ENOKI_TYPE_CONVERSION(name, target, default)                           \
        m.def(name, [](py::handle h) -> py::object {                               \
            if (!var_is_enoki_type(h))                                             \
                return py::reinterpret_borrow<py::object>((PyObject *) &default);  \
            py::object type = h.attr("ReplaceScalar")(target);                     \
            return PyType_Check(h.ptr()) ? type : type(h);                         \
        })

    ENOKI_TYPE_CONVERSION("uint8_array_t",   VarType::UInt8,   PyLong_Type);
    ENOKI_TYPE_CONVERSION("uint16_array_t",  VarType::UInt16,  PyLong_Type);
    ENOKI_TYPE_CONVERSION("uint32_array_t",  VarType::UInt32,  PyLong_Type);
    ENOKI_TYPE_CONVERSION("uint64_array_t",  VarType::UInt64,  PyLong_Type);
    ENOKI_TYPE_CONVERSION("int8_array_t",    VarType::Int8,    PyLong_Type);
    ENOKI_TYPE_CONVERSION("int16_array_t",   VarType::Int16,   PyLong_Type);
    ENOKI_TYPE_CONVERSION("int32_array_t",   VarType::Int32,   PyLong_Type);
    ENOKI_TYPE_CONVERSION("int64_array_t",   VarType::Int64,   PyLong_Type);
    ENOKI_TYPE_CONVERSION("float16_array_t", VarType::Float16, PyFloat_Type);
    ENOKI_TYPE_CONVERSION("float32_array_t", VarType::Float32, PyFloat_Type);
    ENOKI_TYPE_CONVERSION("float64_array_t", VarType::Float64, PyFloat_Type);

    #undef ENOKI_TYPE_CONVERSION
}
