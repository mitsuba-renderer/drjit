#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "python.h"

static nb::object nb_unop(const char *name, size_t offset,
                          PyObject *(*pyop)(PyObject *),
                          nb::handle h) noexcept {
    nb::handle tp = h.type();
    const supp &s = nb::type_supplement<supp>(tp);

    using UnOp = void (*) (const void *, void *);
    UnOp op;
    memcpy(&op, (uint8_t *) &s + offset, sizeof(UnOp));

    // The array potentially does not support the requested operation
    if (!op)
        return nb::borrow<nb::object>(Py_NotImplemented);

    nb::object result = nb::inst_alloc(tp);

    // If the array provides a direct callback for this operation, use it
    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(h.ptr()), nb::inst_ptr<void>(result.ptr()));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", s.name, name, e.what());
            result.clear();
        }
        return result;
    }

    // Otherwise, retry recursively
    nb::inst_zero(result);
    Py_ssize_t size = s.meta.shape[0];
    if (size == DRJIT_DYNAMIC) {
        size = s.sq_len(h.ptr());
        s.op_init(nb::inst_ptr<void>(result), size);
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
        nb::object v = nb::steal(s.sq_item(h.ptr(), i));

        if (!v.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = nb::steal(pyop(v.ptr()));
        if (!vr.is_valid() || s.sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }
    }

    return result;
}

static PyObject *nb_binop(const char *name, size_t ops_offset,
                          PyObject *(*pyop)(PyObject *, PyObject *),
                          PyObject *h0, PyObject *h1) noexcept {
    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0) == Py_TYPE(h1)) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0, h1 };
        if (!promote(name, o, 2))
            return nullptr;
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    /* Three possibilities must be handled:
        1. The requested operation is unsupported by the given array type
        2. The array type provides an implementation of the requested operation in s.
        3. The operation is supported, but no implementation is provided. Fall back
           to a generic operation that recurses
    */

    const supp &s = nb::type_supplement<supp>(o0.type());

    using BinOp = void (*) (const void *, const void *, void *);
    BinOp op;
    memcpy(&op, (uint8_t *) &s + ops_offset, sizeof(BinOp));

    // The array potentially does not support the requested operation
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(o0.type());

    // If the array provides a direct callback for this operation, use it
    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o0),
               nb::inst_ptr<void>(o1),
               nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", s.name, name,
                         e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    nb::inst_zero(result);

    Py_ssize_t s0 = s.meta.shape[0], s1 = s0, sr = s0;

    if (s0 == DRJIT_DYNAMIC) {
        s0 = s.sq_len(o0.ptr());
        s1 = s.sq_len(o1.ptr());

        sr = s0 > s1 ? s0 : s1;
        if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
            PyErr_Format(PyExc_IndexError,
                         "%s.%s(): binary operation involving arrays of "
                         "incompatible size: %zd and %zd!",
                         s.name, name, s0, s1);
            return nullptr;
        }

        s.op_init(nb::inst_ptr<void>(result), sr);
    }

    Py_ssize_t i0 = 0,
               i1 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1;

    for (Py_ssize_t i = 0; i < sr; ++i) {
        nb::object v0 = nb::steal(s.sq_item(o0.ptr(), i0)),
                   v1 = nb::steal(s.sq_item(o1.ptr(), i1));

        if (!v0.is_valid() || !v1.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = nb::steal(pyop(v0.ptr(), v1.ptr()));
        if (!vr.is_valid() || s.sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }

        i0 += k0; i1 += k1;
    }

    return result.release().ptr();
}

static PyObject *nb_inplace_binop(const char *name, size_t ops_offset,
                                  PyObject *(*pyop)(PyObject *,
                                                    PyObject *),
                                  PyObject *h0, PyObject *h1) noexcept {
    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0) == Py_TYPE(h1)) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0, h1 };
        if (!promote(name, o, 2))
            return nullptr;
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    const supp &s = nb::type_supplement<supp>(o0.type());

    using BinOp = void (*) (const void *, const void *, void *);
    BinOp op;
    memcpy(&op, (uint8_t *) &s + ops_offset, sizeof(BinOp));

    // The array potentially does not support the requested operation
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    // If the array provides a direct callback for this operation, use it
    if ((uintptr_t) op != 1) {
        nb::object result = nb::inst_alloc(o0.type());

        try {
            op(nb::inst_ptr<void>(o0),
               nb::inst_ptr<void>(o1),
               nb::inst_ptr<void>(result));

            nb::inst_mark_ready(result);
            nb::inst_destruct(o0);
            nb::inst_move(o0, result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", s.name,
                         name, e.what());
            o0.clear();
        }

        return o0.release().ptr();
    }

    Py_ssize_t s0 = s.meta.shape[0], s1 = s0, sr = s0;

    if (s0 == DRJIT_DYNAMIC) {
        s0 = s.sq_len(o0.ptr());
        s1 = s.sq_len(o1.ptr());

        sr = s0 > s1 ? s0 : s1;
        if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
            PyErr_Format(PyExc_IndexError,
                         "%s.%s(): binary operation involving arrays of "
                         "incompatible size: %zd and %zd!",
                         s.name, name, s0, s1);
            return nullptr;
        }
    }

    if (s0 != sr) {
        nb::object result = nb::inst_alloc(o0.type());
        nb::inst_zero(result);

        if (s.meta.shape[0] == DRJIT_DYNAMIC)
            s.op_init(nb::inst_ptr<void>(result), sr);

        nb::object v = nb::steal(s.sq_item(o0.ptr(), 0));
        if (!v.is_valid())
            return nullptr;

        for (Py_ssize_t i = 0; i < sr; ++i) {
            if (s.sq_ass_item(result.ptr(), i, v.ptr()))
                return nullptr;
        }

        o0 = result;
    }

    Py_ssize_t i1 = 0, k1 = s1 == 1 ? 0 : 1;
    for (Py_ssize_t i = 0; i < sr; ++i) {
        nb::object v0 = nb::steal(s.sq_item(o0.ptr(), i)),
                   v1 = nb::steal(s.sq_item(o1.ptr(), i1));

        if (!v0.is_valid() || !v1.is_valid()) {
            o0.clear();
            break;
        }

        nb::object vr = nb::steal(pyop(v0.ptr(), v1.ptr()));
        if (!vr.is_valid() || s.sq_ass_item(o0.ptr(), i, vr.ptr())) {
            o0.clear();
            break;
        }

        i1 += k1;
    }

    return o0.release().ptr();
}

static PyObject *nb_select(PyObject *h0, PyObject *h1, PyObject *h2) noexcept {
    PyObject *o[3] = { h0, h1, h2 };
    if (!promote("select", o, 3, true))
        return nullptr;

    nb::object o0 = nb::steal(o[0]),
               o1 = nb::steal(o[1]),
               o2 = nb::steal(o[2]);

    const supp &s = nb::type_supplement<supp>(o1.ptr());
    const supp &sm = nb::type_supplement<supp>(o0.type());

    using TernOp = void (*)(const void *, const void *, const void *, void *);
    TernOp op = s.op_select;

    // The array potentially does not support the requested operation
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(o1.type());

    // If the array provides a direct callback for this operation, use it
    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o0),
               nb::inst_ptr<void>(o1),
               nb::inst_ptr<void>(o2),
               nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "drjit.select(): %s!", e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    nb::object py_op = array_module.attr("select");

    Py_ssize_t s0 = sm.sq_len(o0.ptr()),
               s1 = s.sq_len(o1.ptr()),
               s2 = s.sq_len(o2.ptr()),
               st = s0 > s1 ? s0 : s1,
               sr = s2 > st ? s2 : st;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1) ||
        (s2 != sr && s2 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "drjit.select(): operation involving arrays of "
                     "incompatible size: %zd, %zd, and %zd!", s0, s1, s2);
        return nullptr;
    }

    nb::inst_zero(result);

    if (s.meta.shape[0] == DRJIT_DYNAMIC)
        s.op_init(nb::inst_ptr<void>(result), sr);

    Py_ssize_t i0 = 0,
               i1 = 0,
               i2 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1,
               k2 = s2 == 1 ? 0 : 1;

    try {
        for (Py_ssize_t i = 0; i < sr; ++i) {
            nb::object v0 = nb::steal(sm.sq_item(o0.ptr(), i0)),
                       v1 = nb::steal(s.sq_item(o1.ptr(), i1)),
                       v2 = nb::steal(s.sq_item(o2.ptr(), i2));

            if (!v0.is_valid() || !v1.is_valid() || !v2.is_valid()) {
                result.clear();
                break;
            }

            nb::object vr = py_op(v0, v1, v2);
            if (!vr.is_valid() || s.sq_ass_item(result.ptr(), i, vr.ptr())) {
                result.clear();
                break;
            }

            i0 += k0; i1 += k1; i2 += k2;
        }
    } catch (const std::exception &e) {
        PyErr_Format(PyExc_RuntimeError, "drjit.select(): %s!", e.what());
        result.clear();
    }

    return result.release().ptr();
}

static PyObject *tp_richcompare(PyObject *h0, PyObject *h1, int op) noexcept {
    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0) == Py_TYPE(h1)) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0, h1 };
        if (!promote("__richcmp__", o, 2))
            return nullptr;
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    /* Three possibilities must be handled:
        1. The requested operation is unsupported by the given array type
        2. The array type provides an implementation of the requested operation in s.
        3. The operation is supported, but no implementation is provided. Fall back
           to a generic operation that recurses
    */

    const supp &s = nb::type_supplement<supp>(o0.type());
    const supp &sm = nb::type_supplement<supp>(s.mask);

    // The array potentially does not support the requested operation
    if (!s.op_richcmp)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(s.mask);

    if ((uintptr_t) s.op_richcmp != 1) {
        try {
            s.op_richcmp(nb::inst_ptr<void>(o0),
                             nb::inst_ptr<void>(o1),
                             op,
                             nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.__richcmp__(): %s!", s.name,
                         e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    Py_ssize_t s0 = s.sq_len(o0.ptr()),
               s1 = s.sq_len(o1.ptr()),
               sr = s0 > s1 ? s0 : s1;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "%s.__richcmp__(): binary operation involving arrays of "
                     "incompatible size: %zd and %zd!",
                     s.name, s0, s1);
        return nullptr;
    }

    nb::inst_zero(result);

    if (s.meta.shape[0] == DRJIT_DYNAMIC)
        s.op_init(nb::inst_ptr<void>(result), sr);

    Py_ssize_t i0 = 0,
               i1 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1;

    for (Py_ssize_t i = 0; i < sr; ++i) {
        nb::object v0 = nb::steal(s.sq_item(o0.ptr(), i0)),
                   v1 = nb::steal(s.sq_item(o1.ptr(), i1));

        if (!v0.is_valid() || !v1.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = nb::steal(PyObject_RichCompare(v0.ptr(), v1.ptr(), op));
        if (!vr.is_valid() || sm.sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }

        i0 += k0; i1 += k1;
    }

    return result.release().ptr();
}

#define DRJIT_BINOP(name, pyop, label, ilabel)                                 \
    static PyObject *nb_##name(PyObject *h0, PyObject *h1) noexcept {          \
        return nb_binop(label, offsetof(supp, op_##name), pyop, h0, h1);       \
    }                                                                          \
    static PyObject *nb_inplace_##name(PyObject *h0, PyObject *h1) noexcept {  \
        return nb_inplace_binop(ilabel, offsetof(supp, op_##name), pyop, h0,   \
                                h1);                                           \
    }

DRJIT_BINOP(add, PyNumber_Add, "__add__", "__iadd__")
DRJIT_BINOP(subtract, PyNumber_Subtract, "__sub__", "__isub__")
DRJIT_BINOP(multiply, PyNumber_Multiply, "__mul__", "__imul__")
DRJIT_BINOP(remainder, PyNumber_Remainder, "__mod__", "__imod__")
DRJIT_BINOP(floor_divide, PyNumber_FloorDivide, "__floordiv__", "__ifloordiv__")
DRJIT_BINOP(true_divide, PyNumber_TrueDivide, "__truediv__", "__itruediv__")
DRJIT_BINOP(and, PyNumber_And, "__and__", "__iand__")
DRJIT_BINOP(or, PyNumber_Or, "__or__", "__ior__")
DRJIT_BINOP(xor, PyNumber_Xor, "__xor__", "__ixor__")
DRJIT_BINOP(lshift, PyNumber_Lshift, "__lshift__", "__ilshift__")
DRJIT_BINOP(rshift, PyNumber_Rshift, "__rshift__", "__irshift__")


static int nb_bool(PyObject *o) noexcept {
    const supp &s = nb::type_supplement<supp>(nb::handle(o).type());

    if (s.meta.type != (uint16_t) VarType::Bool || s.meta.shape[1] > 0) {
        PyErr_Format(PyExc_TypeError,
                     "%s.__bool__(): implicit conversion to 'bool' is only "
                     "supported for scalar mask arrays!", s.name);
        return -1;
    }

    Py_ssize_t length = s.meta.shape[0];
    if (length == DRJIT_DYNAMIC)
        length = s.sq_len(o);

    if (length != 1) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.__bool__(): implicit conversion to 'bool' requires a "
                     "scalar mask array (array size was %zd).", s.name, length);
        return -1;
    }

    PyObject *result = s.sq_item(o, 0);
    if (!result)
        return -1;
    Py_DECREF(result);

    if (result == Py_True) {
        return 1;
    } else if (result == Py_False) {
        return 0;
    } else {
        PyErr_Format(PyExc_RuntimeError, "%s.__bool__(): internal error!");
        return -1;
    }
}

static PyObject *nb_positive(PyObject *o) noexcept {
    Py_INCREF(o);
    return o;
}

static PyObject *nb_negative(PyObject *o) noexcept {
    return nb_unop("__neg__", offsetof(supp, op_negative),
                   &PyNumber_Negative, o).release().ptr();
}

static PyObject *nb_absolute(PyObject *o) noexcept {
    return nb_unop("__abs__", offsetof(supp, op_absolute),
                   &PyNumber_Absolute, o).release().ptr();
}

static PyObject *nb_invert(PyObject *o) noexcept {
    return nb_unop("__invert__", offsetof(supp, op_invert),
                   &PyNumber_Invert, o).release().ptr();
}

extern PyObject *tp_repr(PyObject *self);

struct dr_iter {
    PyObject_HEAD
    PyObject *o;
    Py_ssize_t i, size;
};

static PyObject *tp_iter_next(PyObject *o) {
    dr_iter *it = (dr_iter *) o;
    if (it->i >= it->size)
        return nullptr;
    PyObject *result = Py_TYPE(it->o)->tp_as_sequence->sq_item(it->o, it->i);
    it->i++;
    return result;
}

static void tp_iter_dealloc(PyObject *self) {
    dr_iter *it = (dr_iter *) self;
    PyTypeObject *tp = Py_TYPE(self);
    Py_DECREF(it->o);
    tp->tp_free(self);
}

PyTypeObject dr_iter_type = {
    .tp_name = "dr_iter",
    .tp_basicsize = sizeof(dr_iter),
    .tp_dealloc = tp_iter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Dr.Jit iterator",
    .tp_iternext = tp_iter_next
};

static PyObject *tp_iter(PyObject *o) {
    const supp &s = nb::type_supplement<supp>(Py_TYPE(o));

    dr_iter *iter = PyObject_NEW(dr_iter, &dr_iter_type);
    iter->o = o;
    Py_INCREF(o);
    iter->i = 0;
    iter->size = s.sq_len(o);
    return (PyObject *) iter;
}

template <int Index> nb::object ab_getter(nb::handle_t<dr::ArrayBase> h) {
    PyTypeObject *tp = (PyTypeObject *) h.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    if (s.meta.is_tensor || s.meta.shape[0] == DRJIT_DYNAMIC || Index >= s.meta.shape[0]) {
        char tmp[128];
        snprintf(tmp, sizeof(tmp), "%s: does not have a '%c' component!",
                 s.name, "xyzw"[Index]);
        throw nb::type_error(tmp);
    }

    return nb::steal(tp->tp_as_sequence->sq_item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index> void ab_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
    PyTypeObject *tp = (PyTypeObject *) h.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    if (s.meta.is_tensor || s.meta.shape[0] == DRJIT_DYNAMIC || Index >= s.meta.shape[0]) {
        char tmp[128];
        snprintf(tmp, sizeof(tmp), "%s: does not have a '%c' component!",
                 s.name, "xyzw"[Index]);
        throw nb::type_error(tmp);
    }

    if (tp->tp_as_sequence->sq_ass_item(h.ptr(), (Py_ssize_t) Index,
                                        value.ptr()))
        nb::detail::raise_python_error();
}

static PyObject *mp_subscript(PyObject *self, PyObject *key) {
    PyTypeObject *self_tp = Py_TYPE(self),
                 *key_tp = Py_TYPE(key);

    const supp &s = nb::type_supplement<supp>(self_tp);

    if (s.meta.is_tensor) {
        nb::tuple key2;

        if (key_tp == &PyTuple_Type)
            key2 = nb::borrow<nb::tuple>(key);
        else
            key2 = nb::make_tuple(key);

        nb::object result;
        try {
            nb::tuple tensor_shape = nb::borrow<nb::tuple>(shape(self));
            nb::object source = nb::steal(s.op_tensor_array(self));

            auto [shape, index] = slice_index(nb::borrow<nb::type_object>(self_tp),
                                              tensor_shape, key2);

            result = nb::handle(self_tp)(
                gather(nb::borrow<nb::type_object>(source.type()), source,
                       index, nb::borrow(Py_True)));
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.__getitem__(): %s!",
                         s.name, e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    bool cast_to_tensor = false;
    if (key_tp == &PyLong_Type) {
        Py_ssize_t size = PyLong_AsSsize_t(key);
        if (size < 0) {
            if (size == -1 && PyErr_Occurred())
                return nullptr;
            size = s.sq_len(self) + size;
        }
        return self_tp->tp_as_sequence->sq_item(self, size);
    } else if (key_tp == &PyTuple_Type) {
        PyObject *o = self;
        Py_INCREF(o);
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(key); ++i) {
            PyObject *key2 = PyTuple_GET_ITEM(key, i),
                     *o2 = PyObject_GetItem(o, key2);
            if (!o2) {
                Py_DECREF(o);
                return nullptr;
            }
            Py_DECREF(o);
            o = o2;
        }
        return o;
    } else if (is_drjit_type(key_tp)) {
        const supp &sk = nb::type_supplement<supp>(key_tp);
        if ((VarType) sk.meta.type == VarType::Bool) {
            Py_INCREF(self);
            return self;
        }
        cast_to_tensor = true;
    } else if (key == Py_None || key_tp == &PyEllipsis_Type || key_tp == &PySlice_Type) {
        cast_to_tensor = true;
    }

    if (cast_to_tensor)
        PyErr_Format(PyExc_TypeError,
                     "%s.__getitem__(): complex slicing operations are only "
                     "supported on tensors.", s.name);
    else
        PyErr_Format(PyExc_TypeError,
                     "%s.__getitem__(): invalid key of type '%s' specified!",
                     s.name, key_tp->tp_name);
    return nullptr;
}

static int mp_ass_subscript(PyObject *self, PyObject *key, PyObject *value) {
    PyTypeObject *self_tp = Py_TYPE(self),
                 *key_tp = Py_TYPE(key);

    const supp &s = nb::type_supplement<supp>(self_tp);

    if (s.meta.is_tensor) {
    }

    bool cast_to_tensor = false;
    if (key_tp == &PyLong_Type) {
        Py_ssize_t size = PyLong_AsSsize_t(key);
        if (size < 0) {
            if (size == -1 && PyErr_Occurred())
                return -1;
            size = s.sq_len(self) + size;
        }
        return self_tp->tp_as_sequence->sq_ass_item(self, size, value);
    } else if (is_drjit_type(key_tp)) {
        const supp &sk = nb::type_supplement<supp>(key_tp);
        if ((VarType) sk.meta.type == VarType::Bool) {
            PyObject *result = nb_select(key, value, self);
            if (!result)
                return -1;
            nb::inst_destruct(self);
            nb::inst_move(self, result);
            Py_DECREF(result);
            return 0;
        }
        cast_to_tensor = true;
    } else if (key == Py_None || key_tp == &PyEllipsis_Type || key_tp == &PySlice_Type) {
        cast_to_tensor = true;
    }

    if (cast_to_tensor)
        PyErr_Format(PyExc_TypeError,
                     "%s.__setitem__(): complex slicing operations are only "
                     "supported on tensors.", s.name);
    else
        PyErr_Format(PyExc_TypeError,
                     "%s.__setitem__(): invalid key of type '%s' specified!",
                     s.name, key_tp->tp_name);
    return -1;
}

nb::dlpack::dtype dlpack_dtype(VarType vt) {
    switch (vt) {
        case VarType::Bool:
        case VarType::UInt8:   return nb::dtype<uint8_t>(); break;
        case VarType::Int8:    return nb::dtype<int8_t>(); break;
        case VarType::UInt16:  return nb::dtype<uint16_t>(); break;
        case VarType::Int16:   return nb::dtype<int16_t>(); break;
        case VarType::UInt32:  return nb::dtype<uint32_t>(); break;
        case VarType::Int32:   return nb::dtype<int32_t>(); break;
        case VarType::UInt64:  return nb::dtype<uint64_t>(); break;
        case VarType::Int64:   return nb::dtype<int64_t>(); break;
        case VarType::Float32: return nb::dtype<float>(); break;
        case VarType::Float64: return nb::dtype<double>(); break;
        default:
            throw nb::type_error(
                "dtype is not understood by the DLPack protocol");
    }
}


template <bool ForceCPU, typename... Ts>
nb::tensor<Ts...> dlpack(nb::handle_t<dr::ArrayBase> h) {
    const supp &s = nb::type_supplement<supp>(h.type());
    bool is_dynamic = false;

    if (s.meta.is_tensor) {
        is_dynamic = true;
    } else {
        for (int i = 0; i < s.meta.ndim; ++i)
            is_dynamic |= s.meta.shape[i] == DRJIT_DYNAMIC;
    }

    nb::dlpack::dtype dtype = dlpack_dtype((VarType) s.meta.type);

    std::vector<size_t> shape;
    std::vector<int64_t> strides;
    int32_t device_id = 0, device_type = nb::device::cpu::value;

    void *ptr;
    if (is_dynamic) {
        nb::object raveled = ravel(h, 'C', &shape, &strides);

        const supp &s2 = nb::type_supplement<supp>(raveled.type());

        if (s2.op_index) {
            uint32_t index = s2.op_index(nb::inst_ptr<void>(raveled));

            bool is_cuda = s.meta.is_cuda;
            if constexpr (ForceCPU) {
                if (is_cuda) {
                    nb::object tmp = raveled.type()();
                    index = jit_var_migrate(index, AllocType::Host);
                    s2.op_set_index(nb::inst_ptr<void>(tmp), index);
                    raveled = std::move(tmp);
                    is_cuda = false;
                }
            }

            jit_var_eval(index);
            ptr = jit_var_ptr(index);
            if (is_cuda) {
                device_type = nb::device::cuda::value;
                device_id = jit_var_device(index);
            } else {
                jit_sync_thread();
            }
        } else {
            ptr = s2.op_ptr(h.ptr());
        }

        return {
            ptr,
            shape.size(),
            shape.data(),
            raveled,
            strides.data(),
            dtype,
            device_type,
            device_id
        };
    } else {
        int64_t stride = 1;
        for (int i = s.meta.ndim - 1; ; --i) {
            shape.push_back(s.meta.shape[i]);
            strides.push_back(stride);
            stride *= s.meta.shape[i];
            if (i == 0)
                break;
        }

        return {
            nb::inst_ptr<void>(h),
            s.meta.ndim,
            shape.data(),
            nb::borrow(h),
            strides.data(),
            dtype,
            device_type,
            device_id
        };
    }
}

void bind_array_builtin(nb::module_ m) {
    if (PyType_Ready(&dr_iter_type))
        nb::detail::fail("Issue initializing iterator type");

    auto callback = [](PyType_Slot **s) noexcept {
        *(*s)++ = { Py_tp_iter, (void *) tp_iter };
        *(*s)++ = { Py_mp_subscript, (void *) mp_subscript };
        *(*s)++ = { Py_mp_ass_subscript, (void *) mp_ass_subscript };
        *(*s)++ = { Py_nb_add, (void *) nb_add };
        *(*s)++ = { Py_nb_inplace_add, (void *) nb_inplace_add };
        *(*s)++ = { Py_nb_subtract, (void *) nb_subtract };
        *(*s)++ = { Py_nb_inplace_subtract, (void *) nb_inplace_subtract };
        *(*s)++ = { Py_nb_multiply, (void *) nb_multiply };
        *(*s)++ = { Py_nb_inplace_multiply, (void *) nb_inplace_multiply };
        *(*s)++ = { Py_nb_remainder, (void *) nb_remainder };
        *(*s)++ = { Py_nb_inplace_remainder, (void *) nb_inplace_remainder };
        *(*s)++ = { Py_nb_floor_divide, (void *) nb_floor_divide };
        *(*s)++ = { Py_nb_inplace_floor_divide, (void *) nb_inplace_floor_divide };
        *(*s)++ = { Py_nb_true_divide, (void *) nb_true_divide };
        *(*s)++ = { Py_nb_inplace_true_divide, (void *) nb_inplace_true_divide };
        *(*s)++ = { Py_nb_and, (void *) nb_and };
        *(*s)++ = { Py_nb_inplace_and, (void *) nb_inplace_and };
        *(*s)++ = { Py_nb_xor, (void *) nb_xor };
        *(*s)++ = { Py_nb_inplace_xor, (void *) nb_inplace_xor };
        *(*s)++ = { Py_nb_or, (void *) nb_or };
        *(*s)++ = { Py_nb_inplace_or, (void *) nb_inplace_or };
        *(*s)++ = { Py_nb_lshift, (void *) nb_lshift };
        *(*s)++ = { Py_nb_inplace_lshift, (void *) nb_inplace_lshift };
        *(*s)++ = { Py_nb_rshift, (void *) nb_rshift };
        *(*s)++ = { Py_nb_inplace_rshift, (void *) nb_inplace_rshift };
        *(*s)++ = { Py_nb_bool, (void *) nb_bool };
        *(*s)++ = { Py_nb_positive, (void *) nb_positive };
        *(*s)++ = { Py_nb_negative, (void *) nb_negative };
        *(*s)++ = { Py_nb_absolute, (void *) nb_absolute };
        *(*s)++ = { Py_nb_invert, (void *) nb_invert };
        *(*s)++ = { Py_tp_repr, (void *) tp_repr };
        *(*s)++ = { Py_tp_richcompare, (void *) tp_richcompare };
    };

    nb::class_<dr::ArrayBase> ab(m, "ArrayBase", nb::type_callback(callback));

    m.def("shape", &shape, nb::raw_doc(doc_shape));

    ab.def_property_readonly("shape", shape, nb::raw_doc(doc_ArrayBase_shape));
    ab.def_property_readonly(
        "array",
        [](nb::handle_t<dr::ArrayBase> h) -> nb::object {
            const supp &s = nb::type_supplement<supp>(h.type());
            if (s.meta.is_tensor)
                return nb::steal(s.op_tensor_array(h.ptr()));
            else
                return nb::borrow(h);
        },
        nb::raw_doc(doc_ArrayBase_array));

    ab.def_property("x", ab_getter<0>, ab_setter<0>, nb::raw_doc(doc_ArrayBase_x));
    ab.def_property("y", ab_getter<1>, ab_setter<1>, nb::raw_doc(doc_ArrayBase_y));
    ab.def_property("z", ab_getter<2>, ab_setter<2>, nb::raw_doc(doc_ArrayBase_z));
    ab.def_property("w", ab_getter<3>, ab_setter<3>, nb::raw_doc(doc_ArrayBase_w));

    ab.def_property_readonly(
        "index",
        [](nb::handle_t<dr::ArrayBase> h) -> uint32_t {
            const supp &s = nb::type_supplement<supp>(h.type());
            if (!s.op_index)
                return 0;
            return s.op_index(nb::inst_ptr<void>(h));
        },
        nb::raw_doc(doc_ArrayBase_index));

    ab.def_property_readonly(
        "index_ad",
        [](nb::handle_t<dr::ArrayBase> h) -> uint32_t {
            PyTypeObject *tp = (PyTypeObject *) h.type().ptr();
            auto &s = nb::type_supplement<supp>(tp);
            if (!s.op_index_ad)
                return 0;
            return s.op_index_ad(nb::inst_ptr<void>(h));
        },
        nb::raw_doc(doc_ArrayBase_index_ad));

    ab.def("__dlpack_device__",
        [](nb::handle_t<dr::ArrayBase> h) -> std::pair<int, int> {
            const supp &s = nb::type_supplement<supp>(h.type());
            if (s.meta.is_cuda) {
                uint32_t device;
                if (s.op_index)
                    device = jit_var_device(s.op_index(nb::inst_ptr<void>(h)));
                else
                    device = jit_cuda_device();

                return { nb::device::cuda::value, device };
            } else {
                return { nb::device::cpu::value, 0 };
            }
        }, doc_dlpack_device);

    ab.def("__dlpack__", &dlpack<false>, doc_dlpack);
    ab.def("__array__", &dlpack<true, nb::numpy>, doc_array);

    m.def(
        "select",
        [](bool condition, nb::handle x, nb::handle y) -> nb::object {
            return borrow(condition ? x : y);
        },
        nb::raw_doc(doc_select), "condition"_a, "x"_a, "y"_a);

    m.def(
        "select",
        [](nb::handle condition, nb::handle x, nb::handle y) {
            if (!is_drjit_array(condition) && !is_drjit_array(x) &&
                !is_drjit_array(y))
                throw nb::next_overload();
            return nb::steal(nb_select(condition.ptr(), x.ptr(), y.ptr()));
        },
        "condition"_a, "x"_a, "y"_a);

    array_base = ab;
    array_module = m;
}
