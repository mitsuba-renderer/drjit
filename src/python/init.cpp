#include "meta.h"
#include "base.h"

#define raise_if(expr, msg)                                                    \
    do {                                                                       \
        if (NB_UNLIKELY(expr))                                                 \
            nb::detail::raise(msg);                                            \
    } while (false)

static void array_resize(PyObject *self, const ArraySupplement &s, Py_ssize_t len) {
    if (s.shape[0] == len)
        return;

    if (s.shape[0] == DRJIT_DYNAMIC)
        s.init(nb::inst_ptr<dr::ArrayBase>(self), (size_t) len);
    else
        nb::detail::raise(
            "Input has the wrong size (expected %u elements, got %zd).",
            (unsigned) s.shape[0], len);
}

int tp_init_array(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
    PyTypeObject *self_tp = Py_TYPE(self);
    const ArraySupplement &s = nb::type_supplement<ArraySupplement>(self_tp);
    Py_ssize_t argc = NB_TUPLE_GET_SIZE(args);
    ArraySupplement::SetItem set_item = s.set_item;

    try {
        raise_if(kwds, "Constructor does not take keyword arguments.");

        if (argc == 0) {
            // Default initialization, e.g., ``Array3f()``
            nb::detail::nb_inst_zero(self);
            return 0;
        } else if (argc > 1) {
            // Initialize from argument list, e.g., ``Array3f(1, 2, 3)``
            nb::detail::nb_inst_zero(self);

            array_resize(self, s, argc);
            for (Py_ssize_t i = 0; i < argc; ++i)
                raise_if(set_item(self, i, NB_TUPLE_GET_ITEM(args, i)),
                         "Item assignment failed.");

            return 0;
        } else {
            // Initialize from a single element, e.g., ``Array3f(other_array)``
            // or ``Array3f(1.0)``
            PyObject *arg = NB_TUPLE_GET_ITEM(args, 0);
            PyTypeObject *arg_tp = Py_TYPE(arg);
            bool try_sequence_import = arg_tp != (PyTypeObject *) s.value;

            // Initialization from another Dr.Jit array
            if (is_drjit_type(arg_tp)) {
                // Copy-constructor
                if (arg_tp == self_tp) {
                    nb::detail::nb_inst_copy(self, arg);
                    return 0;
                } else {
                    ArrayMeta m = nb::type_supplement<ArraySupplement>(arg_tp);

                    VarType vt = (VarType) m.type;
                    m.type = s.type;

                    if (m == s && s.cast) {
                        s.cast(nb::inst_ptr<dr::ArrayBase>(arg), vt,
                               nb::inst_ptr<dr::ArrayBase>(self));
                        nb::inst_mark_ready(self);
                    }

                    // Disallow inefficient element-by-element imports of JIT arrays
                    if (m.ndim == 1 && m.shape[0] == DRJIT_DYNAMIC)
                        try_sequence_import = false;
                }
            }

            nb::detail::nb_inst_zero(self);

            // Fast path for tuples/list instances
            if (arg_tp == &PyTuple_Type) {
                Py_ssize_t len = NB_TUPLE_GET_SIZE(arg);
                array_resize(self, s, len);

                for (Py_ssize_t i = 0; i < len; ++i)
                    raise_if(set_item(self, i, NB_TUPLE_GET_ITEM(arg, i)),
                             "Item assignment failed.");
                return 0;
            } else if (arg_tp == &PyList_Type) {
                Py_ssize_t len = NB_LIST_GET_SIZE(arg);
                array_resize(self, s, len);

                for (Py_ssize_t i = 0; i < len; ++i)
                    raise_if(set_item(self, i, NB_LIST_GET_ITEM(arg, i)),
                             "Item assignment failed.");
                return 0;
            }

            if (try_sequence_import) {
                ssizeargfunc arg_sq_item =
                    (ssizeargfunc) PyType_GetSlot(arg_tp, Py_sq_item);
                lenfunc arg_sq_length =
                    (lenfunc) PyType_GetSlot(arg_tp, Py_sq_length);

                // Special case for general sequence types
                if (arg_sq_length && arg_sq_item) {
                    Py_ssize_t len = arg_sq_length(arg);
                    array_resize(self, s, len);

                    for (Py_ssize_t i = 0; i < len; ++i) {
                        nb::object o = nb::steal(arg_sq_item(arg, i));
                        raise_if(!o.is_valid(), "Item retrieval failed.");
                        raise_if(set_item(self, i, o.ptr()),
                                 "Item assignment failed.");
                    }
                    return 0;
                }

                // Special case for general iterable types. Handled recursively
                getiterfunc arg_tp_iter =
                    (getiterfunc) PyType_GetSlot(arg_tp, Py_tp_iter);

                if (arg_tp_iter) {
                    nb::tuple args_2 =
                        nb::make_tuple(nb::steal(PySequence_List(arg)));
                    return tp_init_array(self, args_2.ptr(), kwds);
                }
            }

            // No sequence/iterable type, try broadcasting
            nb::object element;
            PyObject *value_type = s.value;

            if (s.is_matrix)
                value_type = nb::type_supplement<ArraySupplement>(value_type).value;

            if (arg_tp == (PyTypeObject *) s.value) {
                element = nb::borrow(arg);
            } else {
                PyObject *args[2] = { nullptr, arg };
                element = nb::steal(
                    NB_VECTORCALL(value_type, args + 1,
                                  1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));
                if (!element.is_valid()) {
                    nb::error_scope scope;
                    nb::str arg_tp_name = nb::type_name(arg_tp);
                    nb::detail::raise("Broadcast from type '%s' failed.",
                                      arg_tp_name.c_str());
                }
            }

            Py_ssize_t len = s.shape[0];
            raise_if(len == 0,
                     "Input has the wrong size (expected 0 elements, got 1).");

            if (len == DRJIT_DYNAMIC) {
                len = 1;
                array_resize(self, s, 1);
            }

            if (s.is_complex) {
                nb::float_ zero(0.0);
                raise_if(set_item(self, 0, element.ptr()) ||
                         set_item(self, 1, zero.ptr()),
                         "Item assignment failed.");
            } else if (s.is_quaternion) {
                nb::float_ zero(0.0);
                raise_if(set_item(self, 0, zero.ptr()) ||
                         set_item(self, 1, zero.ptr()) ||
                         set_item(self, 2, zero.ptr()) ||
                         set_item(self, 3, element.ptr()),
                         "Item assignment failed.");
            } else if (s.is_matrix) {
                nb::float_ zero(0.0);

                for (Py_ssize_t i = 0; i < len; ++i) {
                    nb::object col = nb::steal(s.item(self, i));
                    for (Py_ssize_t j = 0; j < len; ++j)
                        col[j] = (i == j) ? element : zero;
                }
            } else {
                for (Py_ssize_t i = 0; i < len; ++i)
                    raise_if(set_item(self, i, element.ptr()),
                             "Item assignment failed.");
            }

            return 0;
        }
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        nb::chain_error(PyExc_TypeError, "%U.__init__(): %s", tp_name.ptr(), e.what());
        return -1;
    }
}

int tp_init_tensor(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
    PyObject *array = nullptr, *shape = nullptr;
    const char *kwlist[3] = { "array", "shape", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO!", (char **) kwlist,
                                     &array, &PyTuple_Type, &shape))
        return -1;

    PyTypeObject *self_tp = Py_TYPE(self);

    if (!shape && !array) {
        nb::detail::nb_inst_zero(self);
        return 0;
    }

    if (!shape) {
        PyTypeObject *array_tp = Py_TYPE(array);

        // Same type -> copy constructor
        if (array_tp == self_tp) {
            nb::detail::nb_inst_copy(self, array);
            return 0;
        }

        /// XXX need dr.ravel(), and initialize shape here..
        // nb::detail::nb_inst_zero(self);
        // PyObject *value = s.op_tensor_array(self);
        // if (array_init(value, args, kwds)) {
        //     Py_DECREF(value);
        //     return -1;
        // }
        //
        // s.op_tensor_shape(nb::inst_ptr<void>(self)).push_back(len(value));
        // Py_DECREF(value);
        // return 0;
    }

    nb::str tp_name = nb::type_name(self_tp);
    PyErr_Format(PyExc_TypeError, "%U.__init__(): unsupported", tp_name.ptr());
    return -1;
}

