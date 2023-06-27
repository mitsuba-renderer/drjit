#include "meta.h"
#include "base.h"

/// Forward declaration
static bool array_init_seq(PyObject *self, const ArraySupplement &s, PyObject *seq);

/// Constructor for all dr.ArrayBase subclasses (except tensors)
int tp_init_array(PyObject *self, PyObject *args, PyObject *kwds) noexcept {
    PyTypeObject *self_tp = Py_TYPE(self);
    const ArraySupplement &s = supp(self_tp);
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
            raise_if(!array_init_seq(self, s, args),
                     "Could not initialize array from argument list.");
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
                    ArrayMeta m = supp(arg_tp);

                    VarType vt = (VarType) m.type;
                    m.type = s.type;

                    // Potentially do a cast
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

            // Try to construct from a sequence/iterable type
            if (try_sequence_import && array_init_seq(self, s, arg))
                return 0;

            // No sequence/iterable type, try broadcasting
            Py_ssize_t size = s.shape[0];
            raise_if(size == 0,
                     "Input has the wrong size (expected 0 elements, got 1).");

            nb::object element;
            PyObject *value_type = s.value;

            if (s.is_matrix)
                value_type = supp(value_type).value;

            if (arg_tp == (PyTypeObject *) s.value) {
                element = nb::borrow(arg);
            } else {
                PyObject *args[2] = { nullptr, arg };
                element = nb::steal(
                    NB_VECTORCALL(value_type, args + 1,
                                  1 | PY_VECTORCALL_ARGUMENTS_OFFSET, nullptr));
                if (NB_UNLIKELY(!element.is_valid())) {
                    nb::error_scope scope;
                    nb::str arg_tp_name = nb::type_name(arg_tp);
                    nb::detail::raise("Broadcast from type '%s' failed.",
                                      arg_tp_name.c_str());
                }
            }

            if (size == DRJIT_DYNAMIC) {
                if (s.init_const) {
                    s.init_const(1, element.ptr(), nb::inst_ptr<dr::ArrayBase>(self));
                    return 0;
                }

                size = 1;
                s.init(1, nb::inst_ptr<dr::ArrayBase>(self));
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

                for (Py_ssize_t i = 0; i < size; ++i) {
                    nb::object col = nb::steal(s.item(self, i));
                    for (Py_ssize_t j = 0; j < size; ++j)
                        col[j] = (i == j) ? element : zero;
                }
            } else {
                for (Py_ssize_t i = 0; i < size; ++i)
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

static bool array_init_seq(PyObject *self, const ArraySupplement &s, PyObject *seq) {
    ssizeargfunc sq_item = nullptr;
    lenfunc sq_length = nullptr;

    PyTypeObject *tp = Py_TYPE(seq);
#if defined(Py_LIMITED_API)
    sq_length = (lenfunc) PyType_GetSlot(tp, Py_sq_length);
    sq_item = (ssizeargfunc) PyType_GetSlot(tp, Py_sq_item);
#else
    PySequenceMethods *sm = tp->tp_as_sequence;
    if (sm) {
        sq_length = sm->sq_length;
        sq_item = sm->sq_item;
    }
#endif

    if (!sq_length || !sq_item) {
        // Special case for general iterable types. Handled recursively
        getiterfunc tp_iter;

#if defined(Py_LIMITED_API)
        tp_iter = (getiterfunc) PyType_GetSlot(tp, Py_tp_iter);
#else
        tp_iter = tp->tp_iter;
#endif

        if (tp_iter) {
            nb::object seq2 = nb::steal(PySequence_List(seq));
            raise_if(!seq2.is_valid(),
                     "Could not convert iterable into a sequence.");
            return array_init_seq(self, s, seq2.ptr());
        }

        return false;
    }

    Py_ssize_t size = sq_length(seq);
    raise_if(size < 0, "Unable to determine the size of the given sequence.");

    bool is_dynamic = s.shape[0] == DRJIT_DYNAMIC;
    raise_if(!is_dynamic && s.shape[0] != size,
             "Input has the wrong size (expected %u elements, got %zd).",
             (unsigned) s.shape[0], size);

    if (size == 1 && s.init_const) {
        nb::object o = nb::steal(sq_item(seq, 0));
        raise_if(!o.is_valid(), "Item retrival failed.");
        s.init_const((size_t) size, o.ptr(), nb::inst_ptr<dr::ArrayBase>(self));
        return true;
    }

    if (s.ndim == 1 && s.init_data) {
        size_t byte_size = jit_type_size((VarType) s.type) * (size_t) size;
        std::unique_ptr<uint8_t[]> storage(new uint8_t[byte_size]);
        bool fail = false;

        #define FROM_SEQ_IMPL(T)                                           \
        {                                                                  \
            nb::detail::make_caster<T> caster;                             \
            T *p = (T *) storage.get();                                    \
            for (Py_ssize_t i = 0; i < size; ++i) {                        \
                nb::object o = nb::steal(sq_item(seq, i));                 \
                if (NB_UNLIKELY(!o.is_valid() ||                           \
                    !caster.from_python(o,                                 \
                                        (uint8_t) nb::detail::cast_flags:: \
                                            convert, nullptr))) {          \
                    fail = true;                                           \
                    break;                                                 \
                }                                                          \
                p[i] = caster.value;                                       \
            }                                                              \
        }

        switch ((VarType) s.type) {
            case VarType::Bool:    FROM_SEQ_IMPL(bool);     break;
            case VarType::Float32: FROM_SEQ_IMPL(float);    break;
            case VarType::Float64: FROM_SEQ_IMPL(double);   break;
            case VarType::Int32:   FROM_SEQ_IMPL(int32_t);  break;
            case VarType::UInt32:  FROM_SEQ_IMPL(uint32_t); break;
            case VarType::Int64:   FROM_SEQ_IMPL(int64_t);  break;
            case VarType::UInt64:  FROM_SEQ_IMPL(uint64_t); break;
            default: fail = true;
        }

        raise_if(fail, "Could not construct from sequence (invalid type in input).");

        s.init_data((size_t) size, storage.get(),
                    nb::inst_ptr<dr::ArrayBase>(self));

        return true;
    }

    if (is_dynamic)
        s.init((size_t) size, nb::inst_ptr<dr::ArrayBase>(self));

    ArraySupplement::SetItem set_item = s.set_item;
    for (Py_ssize_t i = 0; i < size; ++i) {
        nb::object o = nb::steal(sq_item(seq, i));
        raise_if(!o.is_valid(),
                 "Item retrieval failed.");
        raise_if(set_item(self, i, o.ptr()),
                 "Item assignment failed.");
    }

    return true;
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

