#include "meta.h"
#include "base.h"

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

int array_init(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
    PyTypeObject *self_tp = Py_TYPE(self);
    const ArraySupplement &s = nb::type_supplement<ArraySupplement>(self_tp);
    Py_ssize_t argc = NB_TUPLE_GET_SIZE(args);
    ArraySupplement::SetItem set_item = s.set_item;

    try {
        if (NB_UNLIKELY(kwds))
            nb::detail::raise("Constructor does not take keyword arguments.");

        if (argc == 0) {
            // Default initialization, e.g., ``Array3f()``
            nb::detail::nb_inst_zero(self);
            return 0;
        } else if (argc > 1) {
            // Initialize from argument list, e.g., ``Array3f(1, 2, 3)``
            nb::detail::nb_inst_zero(self);

            array_resize(self, s, argc);
            for (Py_ssize_t i = 0; i < argc; ++i) {
                if (NB_UNLIKELY(set_item(self, i, NB_TUPLE_GET_ITEM(args, i))))
                    nb::detail::raise("Item assignment failed.");
            }

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

                for (Py_ssize_t i = 0; i < len; ++i) {
                    if (NB_UNLIKELY(set_item(self, i, NB_TUPLE_GET_ITEM(arg, i))))
                        nb::detail::raise("Item assignment failed.");
                }
                return 0;
            } else if (arg_tp == &PyList_Type) {
                Py_ssize_t len = NB_LIST_GET_SIZE(arg);
                array_resize(self, s, len);

                for (Py_ssize_t i = 0; i < len; ++i) {
                    if (NB_UNLIKELY(set_item(self, i, NB_LIST_GET_ITEM(arg, i))))
                        nb::detail::raise("Item assignment failed.");
                }
                return 0;
            }

            if (try_sequence_import) {
                ssizeargfunc sq_item =
                    (ssizeargfunc) PyType_GetSlot(arg_tp, Py_sq_item);
                lenfunc sq_length =
                    (lenfunc) PyType_GetSlot(arg_tp, Py_sq_length);

                // Special case for general sequence types
                if (sq_length && sq_item) {
                    Py_ssize_t len = sq_length(arg);
                    array_resize(self, s, len);

                    for (Py_ssize_t i = 0; i < len; ++i) {
                        nb::object o = nb::steal(sq_item(arg, i));
                        if (!o.is_valid())
                            nb::detail::raise("Item retrieval failed.");
                        if (set_item(self, i, o.ptr()))
                            nb::detail::raise("Item assignment failed.");
                    }
                    return 0;
                }

                // Special case for general iterable types. Handled recursively
                getiterfunc tp_iter =
                    (getiterfunc) PyType_GetSlot(arg_tp, Py_tp_iter);

                if (tp_iter) {
                    nb::tuple args_2 =
                        nb::make_tuple(nb::steal(PySequence_List(arg)));
                    return array_init(self, args_2.ptr(), kwds);
                }
            }


            // No sequence/iterable type, try broadcasting
            nb::object element;
            if (arg_tp == (PyTypeObject *) s.value) {
                element = nb::borrow(arg);
            } else {
                PyObject *args[2] = { nullptr, arg };
                element = nb::steal(
                    NB_VECTORCALL(s.value, args + 1,
                                  1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));
                if (!element.is_valid()) {
                    nb::error_scope scope;
                    nb::str arg_tp_name = nb::type_name(arg_tp);
                    nb::detail::raise("Broadcast from type '%s' failed.",
                                      arg_tp_name.c_str());
                }
            }

            Py_ssize_t len = s.shape[0];
            if (len == 0)
                nb::detail::raise(
                    "Input has the wrong size (expected 0 elements, got 1).");

            if (len == DRJIT_DYNAMIC) {
                len = 1;
                array_resize(self, s, 1);
            }

            for (Py_ssize_t i = 0; i < len; ++i) {
                if (NB_UNLIKELY(set_item(self, i, element.ptr())))
                    nb::detail::raise("Item assignment failed.");
            }

            return 0;
        }
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(self_tp);
        nb::chain_error(PyExc_TypeError, "%U.__init__(): %s", tp_name.ptr(), e.what());
        return -1;
    }
}
