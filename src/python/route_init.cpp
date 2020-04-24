#include "route.h"

/// Initialize an Enoki array from a variable-length argument list
void array_init(py::handle inst, const py::args &args, size_t size) {
    size_t argc = args.size();
    if (argc == 0)
        return;

    PyTypeObject *type = (PyTypeObject *) inst.get_type().ptr();

    py::object set_coeff = (py::object) inst.attr("set_coeff");
    bool dynamic = size == ek::Dynamic;

    bool success = false;
    try {
        py::object value_type = inst.attr("Value");
        if (argc == 1) {
            py::object o = args[0];
            PyTypeObject *type2 = (PyTypeObject *) o.get_type().ptr();

            if (strncmp(type2->tp_name, "enoki.", 6) == 0) {
                size_t other_size = py::len(o);
                if (dynamic) {
                    size = other_size;
                    inst.attr("init_")(size);
                }
                bool broadcast = other_size != size;
                for (size_t i = 0; i < size; ++i)
                    set_coeff(i, value_type(o[py::int_(broadcast ? 0 : i)]));
                success = true;
            } else if (py::isinstance<py::list>(o)) {
                py::list list = o;
                size_t other_size = list.size();
                if (dynamic) {
                    size = other_size;
                    inst.attr("init_")(size);
                }
                if (other_size == size) {
                    for (size_t i = 0; i < size; ++i)
                        set_coeff(i, list[i]);
                    success = true;
                }
            } else if (py::isinstance<py::tuple>(o)) {
                py::tuple tuple = o;
                size_t other_size = tuple.size();
                if (dynamic) {
                    size = other_size;
                    inst.attr("init_")(size);
                }
                if (other_size == size) {
                    for (size_t i = 0; i < size; ++i)
                        set_coeff(i, tuple[i]);
                    success = true;
                }
            } else {
                if (dynamic) {
                    size = 1;
                    inst.attr("init_")(size);
                }
                for (size_t i = 0; i < size; ++i)
                    set_coeff(i, o);
                success = true;
            }
        } else if (argc == size || dynamic) {
            if (dynamic) {
                size = argc;
                inst.attr("init_")(size);
            }
            for (size_t i = 0; i < size; ++i)
                set_coeff(i, value_type(args[i]));
            success = true;
        }

        coeff_evals += size;
    } catch (const py::error_already_set &) {
        // Discard exception
    }

    if (!success) {
        char tmp[256];
        if (dynamic) {
            snprintf(tmp, sizeof(tmp),
                     "%s constructor expects: arbitrarily many values of type '%s', a matching "
                     "list/tuple, or a NumPy/PyTorch array.",
                     std::string((py::str) inst.get_type().attr("__name__")).c_str(),
                     std::string((py::str) inst.get_type().attr("Value").attr("__name__")).c_str());
        } else {
            snprintf(tmp, sizeof(tmp),
                     "%s constructor expects: %s%zu values of type '%s', a matching "
                     "list/tuple, or a NumPy/PyTorch array.",
                     std::string((py::str) inst.get_type().attr("__name__")).c_str(),
                     size == 1 ? "" : "1 or ", size,
                     std::string((py::str) inst.get_type().attr("Value").attr("__name__")).c_str());
        }
        throw py::type_error(tmp);
    }
}

PyObject *implicit_conversion_handler(PyObject *obj, PyTypeObject *type_) {
    py::handle type((PyObject *) type_);

    const char *tp_name_src = obj->ob_type->tp_name;
    size_t Size = py::cast<size_t>(type.attr("Size"));

    bool pass = false;
    if (PyList_CheckExact(obj)) {
        pass = Size == ek::Dynamic || Size == PyList_GET_SIZE(obj);
    } else if (PyTuple_CheckExact(obj)) {
        pass = Size == ek::Dynamic || Size == PyTuple_GET_SIZE(obj);
    } else if (Size > 0 && PyNumber_Check(obj)) {
        pass = true;
    } else if (Size > 0 &&
                (strcmp(tp_name_src, "numpy.ndarray") == 0 ||
                 strcmp(tp_name_src, "Tensor") == 0)) {
        pass = true;
    } else if (var_is_enoki(obj)) {
        if (py::cast<size_t>(py::handle(obj).attr("Size")) == Size) {
            pass = true;
        } else if (py::handle(obj).is(type.attr("Type"))) {
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
