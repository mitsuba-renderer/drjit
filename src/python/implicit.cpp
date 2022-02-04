#include "common.h"

PyObject *implicit_conversion_handler(PyObject *obj, PyTypeObject *type_) {
    py::handle type((PyObject *) type_);

    const char *tp_name_src = obj->ob_type->tp_name;
    size_t Size = py::cast<size_t>(type.attr("Size"));

    bool pass = false;
    if (PyList_CheckExact(obj)) {
        pass = Size == dr::Dynamic || Size == (size_t) PyList_GET_SIZE(obj);
    } else if (PyTuple_CheckExact(obj)) {
        pass = Size == dr::Dynamic || Size == (size_t) PyTuple_GET_SIZE(obj);
    } else if (Size > 0 && PyNumber_Check(obj)) {
        pass = true;
    } else if (Size > 0 &&
                (strcmp(tp_name_src, "numpy.ndarray") == 0 ||
                 strcmp(tp_name_src, "Tensor") == 0)) {
        pass = true;
    } else if (strncmp(tp_name_src, "drjit.", 6) == 0 ||
               PyObject_HasAttrString(obj, "IsDrJit")) {
        if (py::cast<size_t>(py::handle(obj).attr("Size")) == Size) {
            pass = true;
        } else if (py::handle(obj).get_type().is(type.attr("Value"))) {
            pass = true;
        }
    }

    if (!pass)
        return nullptr;

    PyObject *args = PyTuple_New(1);
    Py_INCREF(obj);
    PyTuple_SET_ITEM(args, 0, obj);
    PyObject *result = PyObject_CallObject(type.ptr(), args);
    if (result == nullptr)
        PyErr_Clear();
    Py_DECREF(args);
    return result;
}

void register_implicit_conversions(const std::type_info &type) {
    auto tinfo = py::detail::get_type_info(type);
    tinfo->implicit_conversions.push_back(implicit_conversion_handler);
}
