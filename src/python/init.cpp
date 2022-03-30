#include <drjit/python.h>
#include "../ext/nanobind/src/buffer.h"

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

namespace nb = nanobind;

bool array_resize(PyObject *self, const detail::array_supplement &supp,
                  Py_ssize_t len) {
    if (supp.meta.shape[0] == 0xFF) {
        try {
            supp.ops.init(nb::inst_ptr<void>(self), (size_t) len);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_TypeError, "%s.__init__(): %s",
                         Py_TYPE(self)->tp_name, e.what());
            return false;
        }

        return true;
    } else {
        if (supp.meta.shape[0] != len) {
            PyErr_Format(
                PyExc_TypeError,
                "%s.__init__(): input sequence has wrong size (expected %u, got %zd)!",
                Py_TYPE(self)->tp_name, (unsigned) supp.meta.shape[0], len);
            return false;
        }
        return true;
    }
}

int array_init(PyObject *self, PyObject *args, PyObject *kwds) {
    PyTypeObject *self_tp = Py_TYPE(self);
    const detail::array_supplement &supp =
        nb::type_supplement<detail::array_supplement>(self_tp);
    auto assign_item = self_tp->tp_as_sequence->sq_ass_item;

    if (kwds) {
        PyErr_Format(
            PyExc_TypeError,
            "%s.__init__(): constructor does not take keyword arguments!",
            self_tp->tp_name);
        return -1;
    }

    size_t argc = (size_t) PyTuple_GET_SIZE(args);
    if (argc == 0) {
        // Zero-initialize
        nb::detail::nb_inst_zero(self);
        return 0;
    } else if (supp.meta.shape[0] == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "%s.__init__(): too many arguments provided (expected 0, got %zu)!",
            Py_TYPE(self)->tp_name, argc);
        return -1;
    }

    if (argc == 1) {
        PyObject *arg = PyTuple_GET_ITEM(args, 0);
        PyTypeObject *arg_tp = Py_TYPE(arg);

        // Same type -> copy constructor
        if (arg_tp == self_tp) {
            nb::detail::nb_inst_copy(self, arg);
            return 0;
        }

        nb::detail::nb_inst_zero(self);

        // Fast path for tuples/list instances
        if (arg_tp == &PyTuple_Type) {
            Py_ssize_t len = PyTuple_GET_SIZE(arg);
            if (!array_resize(self, supp, len))
                return -1;

            for (Py_ssize_t i = 0; i < len; ++i) {
                if (assign_item(self, i, PyTuple_GET_ITEM(arg, i)))
                    return -1;
            }

            return 0;
        } else if (arg_tp == &PyList_Type) {
            Py_ssize_t len = PyList_GET_SIZE(arg);
            if (!array_resize(self, supp, len))
                return -1;

            for (Py_ssize_t i = 0; i < len; ++i) {
                if (assign_item(self, i, PyList_GET_ITEM(arg, i)))
                    return -1;
            }

            return 0;
        }

        if (arg_tp->tp_as_sequence && arg_tp != supp.value) {
            // General path for all sequence types
            auto arg_item = arg_tp->tp_as_sequence->sq_item;
            auto arg_length = arg_tp->tp_as_sequence->sq_length;

            if (arg_length && arg_item) {
                Py_ssize_t len = arg_length(arg);
                if (!array_resize(self, supp, len))
                    return -1;

                for (Py_ssize_t i = 0; i < len; ++i) {
                    PyObject *o = arg_item(arg, i);
                    if (!o)
                        return -1;
                    if (assign_item(self, i, o)) {
                        Py_DECREF(o);
                        return -1;
                    }
                    Py_DECREF(o);
                }
                return 0;
            }

        }

        if (arg_tp->tp_iter && arg_tp != supp.value) {
            PyObject *list = PySequence_List(arg);
            if (!list)
                return -1;
            PyObject *sub_args = PyTuple_New(1);
            PyTuple_SET_ITEM(sub_args, 0, list);
            int rv = array_init(self, sub_args, kwds);
            Py_DECREF(sub_args);
            return rv;
        }


        PyObject *args[2] = { nullptr, arg };
        PyObject *result =
            NB_VECTORCALL((PyObject *) supp.value, args + 1,
                          1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr);
        if (!result)
            return -1;

        Py_ssize_t len = supp.meta.shape[0];
        if (len == 0xFF) {
            len = 1;

            if (supp.ops.op_full) {
                supp.ops.op_full(result, len, nb::inst_ptr<void>(self));
                Py_DECREF(result);
                return 0;
            }

            if (!array_resize(self, supp, len)) {
                Py_DECREF(result);
                return -1;
            }
        }

        for (Py_ssize_t i = 0; i < len; ++i) {
            if (assign_item(self, i, result)) {
                Py_DECREF(result);
                return -1;
            }

        }

        Py_DECREF(result);

        return 0;
    } else {
        nb::detail::nb_inst_zero(self);

        Py_ssize_t len = PyTuple_GET_SIZE(args);
        if (!array_resize(self, supp, len))
            return -1;

        for (Py_ssize_t i = 0; i < len; ++i) {
            if (assign_item(self, i, PyTuple_GET_ITEM(args, i)))
                return -1;
        }

        return 0;
    }
}


NAMESPACE_END(detail)
NAMESPACE_END(drjit)
