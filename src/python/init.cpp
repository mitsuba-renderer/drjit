/*
    init.cpp -- implementation of drjit.ArrayBase.__init__(), which provides
    flexible and generic way fill a Dr.Jit array with contents

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"
#include "../ext/nanobind/src/buffer.h"

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

namespace nb = nanobind;

inline bool operator==(const meta &a, const meta &b) {
    return memcmp(&a, &b, sizeof(meta)) == 0;
}

bool array_resize(PyObject *self, const supp &s, Py_ssize_t len) {
    if (s.meta.shape[0] == 0xFF) {
        try {
            s.init(nb::inst_ptr<void>(self), (size_t) len);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_TypeError, "%s.__init__(): %s",
                         Py_TYPE(self)->tp_name, e.what());
            return false;
        }

        return true;
    } else {
        if (s.meta.shape[0] != len) {
            PyErr_Format(
                PyExc_TypeError,
                "%s.__init__(): input sequence has wrong size (expected %u, got %zd)!",
                Py_TYPE(self)->tp_name, (unsigned) s.meta.shape[0], len);
            return false;
        }
        return true;
    }
}

int array_init(PyObject *self, PyObject *args, PyObject *kwds) {
    PyTypeObject *self_tp = Py_TYPE(self);
    const supp &s = nb::type_supplement<supp>(self_tp);

    if (kwds) {
        PyErr_Format(
            PyExc_TypeError,
            "%s.__init__(): constructor does not take keyword arguments!",
            self_tp->tp_name);
        return -1;
    }

    auto assign_item = self_tp->tp_as_sequence->sq_ass_item;
    size_t argc = (size_t) PyTuple_GET_SIZE(args);

    if (argc == 0) {
        // Zero-initialize
        nb::detail::nb_inst_zero(self);
        return 0;
    } else if (s.meta.shape[0] == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "%s.__init__(): too many arguments provided (expected 0, got %zu)!",
            Py_TYPE(self)->tp_name, argc);
        return -1;
    }

    if (argc == 1) {
        PyObject *arg = PyTuple_GET_ITEM(args, 0);
        PyTypeObject *arg_tp = Py_TYPE(arg);

        // Copy/conversion from a compatible Dr.Jit array
        if (is_drjit_type(arg_tp)) {
            if (arg_tp == self_tp) {
                nb::detail::nb_inst_copy(self, arg);
                return 0;
            }

            meta arg_meta    = nb::type_supplement<supp>(arg_tp).meta,
                 arg_meta_cp = arg_meta;
            arg_meta_cp.type = s.meta.type;

            if (arg_meta_cp == s.meta && s.op_cast) {
                if (s.op_cast(nb::inst_ptr<void>(arg), (VarType) arg_meta.type,
                              nb::inst_ptr<void>(self)) == 0) {
                    nb::inst_mark_ready(self);
                    return 0;
                }
            }
        }

        nb::detail::nb_inst_zero(self);

        // Fast path for tuples/list instances
        if (arg_tp == &PyTuple_Type) {
            Py_ssize_t len = PyTuple_GET_SIZE(arg);
            if (!array_resize(self, s, len))
                return -1;

            for (Py_ssize_t i = 0; i < len; ++i) {
                if (assign_item(self, i, PyTuple_GET_ITEM(arg, i)))
                    return -1;
            }

            return 0;
        } else if (arg_tp == &PyList_Type) {
            Py_ssize_t len = PyList_GET_SIZE(arg);
            if (!array_resize(self, s, len))
                return -1;

            for (Py_ssize_t i = 0; i < len; ++i) {
                if (assign_item(self, i, PyList_GET_ITEM(arg, i)))
                    return -1;
            }

            return 0;
        }

        if (arg_tp->tp_as_sequence && arg_tp != s.value) {
            // General path for all sequence types
            auto arg_item = arg_tp->tp_as_sequence->sq_item;
            auto arg_length = arg_tp->tp_as_sequence->sq_length;

            if (arg_length && arg_item) {
                Py_ssize_t len = arg_length(arg);
                if (!array_resize(self, s, len))
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

        // Catch-all case for iterable types
        if (arg_tp->tp_iter && arg_tp != s.value) {
            PyObject *list = PySequence_List(arg);
            if (!list)
                return -1;
            PyObject *sub_args = PyTuple_New(1);
            PyTuple_SET_ITEM(sub_args, 0, list);
            int rv = array_init(self, sub_args, kwds);
            Py_DECREF(sub_args);
            return rv;
        }

        // No sequence/iterable type, broadcast to elements
        PyObject *result;
        if (arg_tp == s.value) {
            result = arg;
            Py_INCREF(result);
        } else {
            PyObject *args[2] = { nullptr, arg };
            result = NB_VECTORCALL((PyObject *) s.value, args + 1,
                                   1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr);
            if (!result)
                return -1;
        }

        Py_ssize_t len = s.meta.shape[0];
        if (len == 0xFF) {
            len = 1;

            if (s.op_full) {
                s.op_full(result, len, nb::inst_ptr<void>(self));
                Py_DECREF(result);
                return 0;
            }

            if (!array_resize(self, s, len)) {
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
        if (!array_resize(self, s, len))
            return -1;

        for (Py_ssize_t i = 0; i < len; ++i) {
            if (assign_item(self, i, PyTuple_GET_ITEM(args, i)))
                return -1;
        }

        return 0;
    }
}

int tensor_init(PyObject *self, PyObject *args, PyObject *kwds) {
    PyTypeObject *self_tp = Py_TYPE(self);
    const supp &s = nb::type_supplement<supp>(self_tp);

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
        s.op_tensor_shape(nb::inst_ptr<void>(self)).push_back(0);
        return 0;
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
        PyObject *value = s.op_tensor_array(self);
        if (array_init(value, args, kwds)) {
            Py_DECREF(value);
            return -1;
        }

        s.op_tensor_shape(nb::inst_ptr<void>(self)).push_back(len(value));
        Py_DECREF(value);
        return 0;
    }

    return -1;
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)
