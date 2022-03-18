#include "python.h"

nb::handle array_base;
nb::handle array_module;

static bool meta_check(meta m) {
    return m.is_valid &&
           (m.type > (uint8_t) VarType::Void &&
            m.type < (uint8_t) VarType::Count) &&
           (m.is_vector + m.is_complex + m.is_quaternion + m.is_matrix +
                m.is_tensor <= 1) &&
           (m.is_cuda + m.is_llvm <= 1);
}

#if 0
static void meta_print(meta m) {
    printf("meta[\n"
           "  is_vector=%u,\n"
           "  is_complex=%u,\n"
           "  is_quaternion=%u,\n"
           "  is_matrix=%u,\n"
           "  is_tensor=%u,\n"
           "  is_diff=%u,\n"
           "  is_llvm=%u,\n"
           "  is_cuda=%u,\n"
           "  is_valid=%u,\n"
           "  type=%u,\n"
           "  shape=(%u, %u, %u, %u)\n"
           "]\n",
           m.is_vector, m.is_complex, m.is_quaternion, m.is_matrix, m.is_tensor,
           m.is_diff, m.is_llvm, m.is_cuda, m.is_valid, m.type, m.shape[0],
           m.shape[1], m.shape[2], m.shape[3]);
}
#endif

/// Compute the metadata type of an operation combinining 'a' and 'b'
static meta meta_promote(meta a, meta b) {
    meta r;
    r.is_vector = a.is_vector | b.is_vector;
    r.is_complex = a.is_complex | b.is_complex;
    r.is_quaternion = a.is_quaternion | b.is_quaternion;
    r.is_matrix = a.is_matrix | b.is_matrix;
    r.is_tensor = a.is_tensor | b.is_tensor;
    r.is_diff = a.is_diff | b.is_diff;
    r.is_llvm = a.is_llvm | b.is_llvm;
    r.is_cuda = a.is_cuda | b.is_cuda;
    r.is_valid = a.is_valid & b.is_valid;
    r.type = a.type > b.type ? a.type : b.type;
    r.unused = 0;
    r.tsize_rel = r.talign = 0;

    int ndim_a = -1, ndim_b = -1;
    for (int i = 0; i < 4; ++i) {
        r.shape[i] = 0;
        if (a.shape[i])
            ndim_a = i;
        if (b.shape[i])
            ndim_b = i;
    }

    int ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    for (int i = ndim; i >= 0; --i) {
        int value_a = 1, value_b = 1;

        if (ndim_a >= 0)
            value_a = a.shape[ndim_a--];
        if (ndim_b >= 0)
            value_b = b.shape[ndim_b--];

        if (value_a == value_b)
            r.shape[i] = value_a;
        else if (value_a != value_b && (value_a == 1 || value_b == 1))
            r.shape[i] = value_a > value_b ? value_a : value_b;
        else
            r.is_valid = 0;
    }

    return r;
}

/**
  Given a list of Dr.Jit arrays and scalars, determine the flavor and shape of
  the result array and broadcast/convert everything into this form.
 */
static bool promote(const char *op, PyObject **o, size_t n) {
    meta m;
    memset(&m, 0, sizeof(meta));
    m.is_valid = 1;

    for (size_t i = 0; i < n; ++i) {
        PyTypeObject *tp = Py_TYPE(o[i]);

        if (PyType_IsSubtype(tp, (PyTypeObject *) array_base.ptr())) {
            m = meta_promote(m, nb::type_supplement<supp>(tp).meta);
            continue;
        }

        uint8_t type;
        if (tp == &PyLong_Type) {
            long long result = PyLong_AsLongLong(o[i]);

            if (result == -1 && PyErr_Occurred()) {
                PyErr_Format(PyExc_RuntimeError,
                             "%s.%s(): integer overflow during type promotion!",
                             Py_TYPE(o[0])->tp_name, op);
                return false;
            }

            type = (uint8_t) ((result >= INT32_MIN && result <= INT32_MAX)
                                  ? VarType::Int32
                                  : VarType::Int64);

        } else if (tp == &PyBool_Type) {
            type = (uint8_t) VarType::Bool;
        } else {
            PyErr_Format(PyExc_TypeError,
                         "%s.%s(): encountered an unsupported argument of type "
                         "'%s' (must be a Dr.Jit array or a Python scalar)!",
                         Py_TYPE(o[0])->tp_name, op, tp->tp_name);
            return false;
        }

        if (m.type < type)
            m.type = type;
    }

    if (!meta_check(m)) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.%s(): incompatible arguments!", Py_TYPE(o[0])->tp_name, op);
        return false;
    }

    nb::handle h = drjit::detail::array_get(m);
    PyObject *args[2];

    for (size_t i = 0; i < n; ++i) {
        if (Py_TYPE(o[i]) == (PyTypeObject *) h.ptr()) {
            Py_INCREF(o[i]);
        } else {
            args[0] = nullptr;
            args[1] = o[i];
            PyObject *res = NB_VECTORCALL(
                h.ptr(), args + 1, PY_VECTORCALL_ARGUMENTS_OFFSET | 1, nullptr);

            if (!res) {
                PyErr_Clear();
                PyErr_Format(PyExc_RuntimeError,
                             "%s(): type promotion from '%s' to '%s' failed!", op,
                             Py_TYPE(o[i])->tp_name, ((PyTypeObject *) h.ptr())->tp_name);
                for (size_t j = 0; j < i; ++j)
                    Py_CLEAR(o[j]);
                return false;
            }

            o[i] = res;
        }
    }

    return true;
}

static PyObject *nb_math_unop(const char *name, size_t ops_offset,
                              PyObject *o) noexcept {
    PyTypeObject *tp = (PyTypeObject *) Py_TYPE(o);
    const supp &s = nb::type_supplement<supp>(tp);

    using UnOp = void (*) (const void *, void *);
    UnOp op;
    memcpy(&op, (uint8_t *) &s.ops + ops_offset, sizeof(UnOp));
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(tp);

    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o),
               nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", tp->tp_name, name,
                         e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    nb::object py_op = array_module.attr(name);

    auto sq_length = tp->tp_as_sequence->sq_length;
    auto sq_item = tp->tp_as_sequence->sq_item;
    auto sq_ass_item = tp->tp_as_sequence->sq_ass_item;

    if (!sq_length || !sq_item || !sq_ass_item) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.%s(): cannot perform operation (missing "
                     "number/sequence protocol operations)!",
                     tp->tp_name, name);
        return nullptr;
    }

    Py_ssize_t size = sq_length(o);

    nb::inst_zero(result);

    if (s.meta.shape[0] == 0xFF)
        s.ops.init(nb::inst_ptr<void>(result), size);

    for (Py_ssize_t i = 0; i < size; ++i) {
        nb::object v = nb::steal(sq_item(o, i));

        if (!v.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = py_op(v);
        if (!vr.is_valid() || sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }
    }

    return result.release().ptr();
}

static PyObject *nb_unop(const char *name, size_t ops_offset,
                         size_t nb_offset, PyObject *o) noexcept {
    PyTypeObject *tp = (PyTypeObject *) Py_TYPE(o);
    const supp &s = nb::type_supplement<supp>(tp);

    using UnOp = void (*) (const void *, void *);

    UnOp op;
    memcpy(&op, (uint8_t *) &s.ops + ops_offset, sizeof(UnOp));
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(tp);

    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o),
               nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", tp->tp_name, name,
                         e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    unaryfunc nb_op = nullptr;
    PyObject *tp_value = PyDict_GetItemString(tp->tp_dict, "Value"); // borrowed
    if (tp_value && PyType_Check(tp_value))
        memcpy(&nb_op,
               (uint8_t *) ((PyTypeObject *) tp_value)->tp_as_number + nb_offset,
               sizeof(unaryfunc));

    auto sq_length = tp->tp_as_sequence->sq_length;
    auto sq_item = tp->tp_as_sequence->sq_item;
    auto sq_ass_item = tp->tp_as_sequence->sq_ass_item;

    if (!nb_op || !sq_length || !sq_item || !sq_ass_item) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.%s(): cannot perform operation (missing "
                     "number/sequence protocol operations)!",
                     tp->tp_name, name);
        return nullptr;
    }

    Py_ssize_t size = sq_length(o);

    nb::inst_zero(result);

    if (s.meta.shape[0] == 0xFF)
        s.ops.init(nb::inst_ptr<void>(result), size);

    for (Py_ssize_t i = 0; i < size; ++i) {
        nb::object v = nb::steal(sq_item(o, i));

        if (!v.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = nb::steal(nb_op(v.ptr()));
        if (!vr.is_valid() || sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }
    }

    return result.release().ptr();
}

static PyObject *nb_binop(const char *name, size_t ops_offset, size_t nb_offset,
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
        2. The array type provides an implementation of the requested operation in s.ops.
        3. The operation is supported, but no implementation is provided. Fall back
           to a generic operation that recurses
    */

    PyTypeObject *tp = (PyTypeObject *) o0.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    using BinOp = void (*) (const void *, const void *, void *);

    BinOp op;
    memcpy(&op, (uint8_t *) &s.ops + ops_offset, sizeof(BinOp));
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(tp);

    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o0),
               nb::inst_ptr<void>(o1),
               nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", tp->tp_name, name,
                         e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    binaryfunc nb_op = nullptr;
    PyObject *tp_value = PyDict_GetItemString(tp->tp_dict, "Value"); // borrowed
    if (tp_value && PyType_Check(tp_value))
        memcpy(&nb_op,
               (uint8_t *) ((PyTypeObject *) tp_value)->tp_as_number + nb_offset,
               sizeof(binaryfunc));

    auto sq_length = tp->tp_as_sequence->sq_length;
    auto sq_item = tp->tp_as_sequence->sq_item;
    auto sq_ass_item = tp->tp_as_sequence->sq_ass_item;

    if (!nb_op || !sq_length || !sq_item || !sq_ass_item) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.%s(): cannot perform operation (missing "
                     "number/sequence protocol operations)!",
                     tp->tp_name, name);
        return nullptr;
    }

    Py_ssize_t s0 = sq_length(o0.ptr()),
               s1 = sq_length(o1.ptr()),
               sr = s0 > s1 ? s0 : s1;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "%s.%s(): binary operation involving arrays of "
                     "incompatible size: %zd and %zd!",
                     tp->tp_name, name, s0, s1);
        return nullptr;
    }

    nb::inst_zero(result);

    if (s.meta.shape[0] == 0xFF)
        s.ops.init(nb::inst_ptr<void>(result), sr);

    Py_ssize_t i0 = 0,
               i1 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1;

    for (Py_ssize_t i = 0; i < sr; ++i) {
        nb::object v0 = nb::steal(sq_item(o0.ptr(), i0)),
                   v1 = nb::steal(sq_item(o1.ptr(), i1));

        if (!v0.is_valid() || !v1.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = nb::steal(nb_op(v0.ptr(), v1.ptr()));
        if (!vr.is_valid() || sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }

        i0 += k0; i1 += k1;
    }

    return result.release().ptr();
}

static PyObject *nb_inplace_binop(const char *name, size_t ops_offset,
                                  size_t nb_offset, PyObject *h0,
                                  PyObject *h1) noexcept {
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

    PyTypeObject *tp = (PyTypeObject *) o0.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    using BinOp = void (*) (const void *, const void *, void *);

    BinOp op;
    memcpy(&op, (uint8_t *) &s.ops + ops_offset, sizeof(BinOp));
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    if ((uintptr_t) op != 1) {
        nb::object result = nb::inst_alloc(tp);

        try {
            op(nb::inst_ptr<void>(o0),
               nb::inst_ptr<void>(o1),
               nb::inst_ptr<void>(result));

            nb::inst_mark_ready(result);
            nb::inst_destruct(o0);
            nb::inst_move(o0, result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.%s(): %s!", tp->tp_name, name,
                         e.what());
            o0.clear();
        }

        return o0.release().ptr();
    }

    binaryfunc nb_op = nullptr;
    PyObject *tp_value = PyDict_GetItemString(tp->tp_dict, "Value"); // borrowed
    if (tp_value && PyType_Check(tp_value))
        memcpy(&nb_op,
               (uint8_t *) ((PyTypeObject *) tp_value)->tp_as_number + nb_offset,
               sizeof(binaryfunc));

    auto sq_length = tp->tp_as_sequence->sq_length;
    auto sq_item = tp->tp_as_sequence->sq_item;
    auto sq_ass_item = tp->tp_as_sequence->sq_ass_item;

    if (!nb_op || !sq_length || !sq_item || !sq_ass_item) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.%s(): cannot perform operation (missing "
                     "number/sequence protocol operations)!",
                     tp->tp_name, name);
        return nullptr;
    }

    Py_ssize_t s0 = sq_length(o0.ptr()),
               s1 = sq_length(o1.ptr()),
               sr = s0 > s1 ? s0 : s1;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "%s.%s(): binary operation involving arrays of "
                     "incompatible size: %zd and %zd!",
                     tp->tp_name, name, s0, s1);
        return nullptr;
    }

    if (s0 != sr) {
        nb::object result = nb::inst_alloc(tp);
        nb::inst_zero(result);

        if (s.meta.shape[0] == 0xFF)
            s.ops.init(nb::inst_ptr<void>(result), sr);

        nb::object v = nb::steal(sq_item(o0.ptr(), 0));
        if (!v.is_valid())
            return nullptr;

        for (Py_ssize_t i = 0; i < sr; ++i) {
            if (sq_ass_item(result.ptr(), i, v.ptr()))
                return nullptr;
        }

        o0 = result;
    }

    Py_ssize_t i1 = 0, k1 = s1 == 1 ? 0 : 1;
    for (Py_ssize_t i = 0; i < sr; ++i) {
        nb::object v0 = nb::steal(sq_item(o0.ptr(), i)),
                   v1 = nb::steal(sq_item(o1.ptr(), i1));

        if (!v0.is_valid() || !v1.is_valid()) {
            o0.clear();
            break;
        }

        nb::object vr = nb::steal(nb_op(v0.ptr(), v1.ptr()));
        if (!vr.is_valid() || sq_ass_item(o0.ptr(), i, vr.ptr())) {
            o0.clear();
            break;
        }

        i1 += k1;
    }

    return o0.release().ptr();
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
        2. The array type provides an implementation of the requested operation in s.ops.
        3. The operation is supported, but no implementation is provided. Fall back
           to a generic operation that recurses
    */

    PyTypeObject *tp = (PyTypeObject *) o0.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    if (!s.ops.op_richcmp)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(s.mask);

    if ((uintptr_t) s.ops.op_richcmp != 1) {
        try {
            s.ops.op_richcmp(nb::inst_ptr<void>(o0),
                             nb::inst_ptr<void>(o1),
                             op,
                             nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "%s.__richcmp__(): %s!",
                         tp->tp_name, e.what());
            result.clear();
        }

        return result.release().ptr();
    }

    auto nb_op = s.value->tp_richcompare;
    auto sq_length = tp->tp_as_sequence->sq_length;
    auto sq_item = tp->tp_as_sequence->sq_item;
    auto sq_ass_item = s.mask->tp_as_sequence->sq_ass_item;

    if (!nb_op || !sq_length || !sq_item || !sq_ass_item) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.__richcmp__(): cannot perform operation (missing "
                     "number/sequence protocol operations)!",
                     tp->tp_name);
        return nullptr;
    }

    Py_ssize_t s0 = sq_length(o0.ptr()),
               s1 = sq_length(o1.ptr()),
               sr = s0 > s1 ? s0 : s1;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "%s.__richcmp__(): binary operation involving arrays of "
                     "incompatible size: %zd and %zd!",
                     tp->tp_name, s0, s1);
        return nullptr;
    }

    nb::inst_zero(result);

    if (s.meta.shape[0] == 0xFF)
        s.ops.init(nb::inst_ptr<void>(result), sr);

    Py_ssize_t i0 = 0,
               i1 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1;

    for (Py_ssize_t i = 0; i < sr; ++i) {
        nb::object v0 = nb::steal(sq_item(o0.ptr(), i0)),
                   v1 = nb::steal(sq_item(o1.ptr(), i1));

        if (!v0.is_valid() || !v1.is_valid()) {
            result.clear();
            break;
        }

        nb::object vr = nb::steal(nb_op(v0.ptr(), v1.ptr(), op));
        if (!vr.is_valid() || sq_ass_item(result.ptr(), i, vr.ptr())) {
            result.clear();
            break;
        }

        i0 += k0; i1 += k1;
    }

    return result.release().ptr();
}

#define DRJIT_BINOP(name, label, ilabel)                                       \
    static PyObject *nb_##name(PyObject *h0, PyObject *h1) noexcept {          \
        return nb_binop(label, offsetof(ops, op_##name),                       \
                        offsetof(PyNumberMethods, nb_##name), h0, h1);         \
    }                                                                          \
    static PyObject *nb_inplace_##name(PyObject *h0, PyObject *h1) noexcept {  \
        return nb_inplace_binop(ilabel, offsetof(ops, op_##name),              \
                                offsetof(PyNumberMethods, nb_##name), h0, h1); \
    }

DRJIT_BINOP(add, "__add__", "__iadd__")
DRJIT_BINOP(subtract, "__sub__", "__isub__")
DRJIT_BINOP(multiply, "__mul__", "__imul__")
DRJIT_BINOP(remainder, "__mod__", "__imod__")
DRJIT_BINOP(floor_divide, "__floordiv__", "__ifloordiv__")
DRJIT_BINOP(true_divide, "__truediv__", "__itruediv__")
DRJIT_BINOP(and, "__and__", "__iand__")
DRJIT_BINOP(or, "__or__", "__ior__")
DRJIT_BINOP(xor, "__xor__", "__ixor__")
DRJIT_BINOP(lshift, "__lshift__", "__ilshift__")
DRJIT_BINOP(rshift, "__rshift__", "__irshift__")


static int nb_bool(PyObject *o) noexcept {
    PyTypeObject *tp = Py_TYPE(o);
    const supp &s = nb::type_supplement<supp>(tp);

    if (s.meta.type != (uint16_t) VarType::Bool || s.meta.shape[1] > 0) {
        PyErr_Format(PyExc_TypeError,
                     "%s.__bool__(): implicit conversion to 'bool' is only "
                     "supported for scalar mask arrays!",
                     tp->tp_name);
        return -1;
    }

    Py_ssize_t length = s.meta.shape[0];
    if (length == 0xFF)
        length = (Py_ssize_t) s.ops.len(nb::inst_ptr<void>(o));

    if (length != 1) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s.__bool__(): implicit conversion to 'bool' requires a "
                     "scalar mask array (array size was %zd).", tp->tp_name, length);
        return -1;
    }

    PyObject *result = tp->tp_as_sequence->sq_item(o, 0);
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
    return nb_unop("__neg__", offsetof(ops, op_negative),
                   offsetof(PyNumberMethods, nb_negative), o);
}

static PyObject *nb_absolute(PyObject *o) noexcept {
    return nb_unop("__abs__", offsetof(ops, op_absolute),
                   offsetof(PyNumberMethods, nb_absolute), o);
}

static PyObject *nb_invert(PyObject *o) noexcept {
    return nb_unop("__invert__", offsetof(ops, op_invert),
                   offsetof(PyNumberMethods, nb_invert), o);
}

extern PyObject *tp_repr(PyObject *self);

#define DR_MATH_UNOP(name)                                                     \
    m.def(#name, [](double h) { return dr::name(h); });                        \
    m.def(#name, [](nb::handle_of<dr::ArrayBase> h) {                          \
        return nb::steal(                                                      \
            nb_math_unop(#name, offsetof(ops, op_##name), h.ptr()));           \
    });

void bind_arraybase(nb::module_ m) {
    auto callback = [](PyTypeObject *tp) noexcept {
        tp->tp_as_sequence->sq_length = len;
        tp->tp_as_number->nb_add = nb_add;
        tp->tp_as_number->nb_inplace_add = nb_inplace_add;
        tp->tp_as_number->nb_subtract = nb_subtract;
        tp->tp_as_number->nb_inplace_subtract = nb_inplace_subtract;
        tp->tp_as_number->nb_multiply = nb_multiply;
        tp->tp_as_number->nb_inplace_multiply = nb_inplace_multiply;
        tp->tp_as_number->nb_remainder = nb_remainder;
        tp->tp_as_number->nb_inplace_remainder = nb_inplace_remainder;
        tp->tp_as_number->nb_floor_divide = nb_floor_divide;
        tp->tp_as_number->nb_inplace_floor_divide = nb_inplace_floor_divide;
        tp->tp_as_number->nb_true_divide = nb_true_divide;
        tp->tp_as_number->nb_inplace_true_divide = nb_inplace_true_divide;
        tp->tp_as_number->nb_and = nb_and;
        tp->tp_as_number->nb_inplace_and = nb_inplace_and;
        tp->tp_as_number->nb_xor = nb_xor;
        tp->tp_as_number->nb_inplace_xor = nb_inplace_xor;
        tp->tp_as_number->nb_or = nb_or;
        tp->tp_as_number->nb_inplace_or = nb_inplace_or;
        tp->tp_as_number->nb_lshift = nb_lshift;
        tp->tp_as_number->nb_inplace_lshift = nb_inplace_lshift;
        tp->tp_as_number->nb_rshift = nb_rshift;
        tp->tp_as_number->nb_inplace_rshift = nb_inplace_rshift;
        tp->tp_as_number->nb_bool = nb_bool;
        tp->tp_as_number->nb_positive = nb_positive;
        tp->tp_as_number->nb_negative = nb_negative;
        tp->tp_as_number->nb_absolute = nb_absolute;
        tp->tp_as_number->nb_invert = nb_invert;
        tp->tp_repr = tp_repr;
        tp->tp_richcompare = tp_richcompare;
    };

    nb::class_<dr::ArrayBase> ab(m, "ArrayBase", nb::type_callback(callback));
    array_base = ab;
    array_module = m;

    DR_MATH_UNOP(sin);
    DR_MATH_UNOP(cos);
    DR_MATH_UNOP(tan);
    DR_MATH_UNOP(asin);
    DR_MATH_UNOP(acos);
    DR_MATH_UNOP(atan);
    DR_MATH_UNOP(sinh);
    DR_MATH_UNOP(cosh);
    DR_MATH_UNOP(tanh);
    DR_MATH_UNOP(asinh);
    DR_MATH_UNOP(acosh);
    DR_MATH_UNOP(atanh);
    DR_MATH_UNOP(sqrt);
    DR_MATH_UNOP(cbrt);
}
