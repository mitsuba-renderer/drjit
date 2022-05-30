#include "python.h"

namespace drjit {
    template <typename T> T fma(const T &a, const T &b, const T &c) {
        return fmadd(a, b, c);
    }
};


static nb::object nb_math_unop(const char *name, size_t ops_offset,
                               nb::handle h) noexcept {
    nb::handle tp = h.type();
    const supp &s = nb::type_supplement<supp>(tp);

    // if (s.is_tensor) {
    //     void *ptr = nb::inst_ptr(h);
    //
    //     return tp(
    //         nb_math_unop(name, ops_offset, s.op_tensor_array(ptr)),
    //         s.op_tensor_shape(ptr)
    //     );
    // }

    using UnOp = void (*) (const void *, void *);
    UnOp op;
    memcpy(&op, (uint8_t *) &s + ops_offset, sizeof(UnOp));
    if (!op) {
        PyErr_Format(PyExc_TypeError, "drjit.%s(): unsupported type %s.", name,
                     ((PyTypeObject *) tp.ptr())->tp_name);
        return nb::object();
    }

    nb::object result = nb::inst_alloc(tp);

    try {
        if ((uintptr_t) op != 1) {
            op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
            return result;
        }

        nb::object py_op = array_module.attr(name);

        auto [sq_length, sq_item, sq_ass_item] = get_sq(tp, name, py_op.ptr());
        if (!sq_length)
            return nb::object();

        Py_ssize_t size = sq_length(h.ptr());

        nb::inst_zero(result);

        if (s.meta.shape[0] == DRJIT_DYNAMIC)
            s.init(nb::inst_ptr<void>(result), size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            nb::object v = nb::steal(sq_item(h.ptr(), i));

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
    } catch (const std::exception &e) {
        PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name, e.what());
        result.clear();
    }

    return result;
}

static PyObject *nb_math_unop_2(const char *name, size_t ops_offset,
                                PyObject *o) noexcept {
    PyTypeObject *tp = (PyTypeObject *) Py_TYPE(o);
    const supp &s = nb::type_supplement<supp>(tp);

    using UnOp2 = void (*) (const void *, void *, void *);
    UnOp2 op;
    memcpy(&op, (uint8_t *) &s + ops_offset, sizeof(UnOp2));
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result_1 = nb::inst_alloc(tp),
               result_2 = nb::inst_alloc(tp),
               result = nb::make_tuple(result_1, result_2);

    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o),
               nb::inst_ptr<void>(result_1),
               nb::inst_ptr<void>(result_2));
            nb::inst_mark_ready(result_1);
            nb::inst_mark_ready(result_2);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name,
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
                     "drjit.%s(): cannot perform operation!", name);
        return nullptr;
    }

    Py_ssize_t size = sq_length(o);

    nb::inst_zero(result_1);
    nb::inst_zero(result_2);

    if (s.meta.shape[0] == DRJIT_DYNAMIC)
        s.init(nb::inst_ptr<void>(result), size);

    try {
        for (Py_ssize_t i = 0; i < size; ++i) {
            nb::object v = nb::steal(sq_item(o, i));

            if (!v.is_valid()) {
                result.clear();
                break;
            }

            nb::object vr = py_op(v);
            if (!vr.is_valid() || sq_ass_item(result_1.ptr(), i, vr[0].ptr()) ||
                sq_ass_item(result_2.ptr(), i, vr[1].ptr())) {
                result.clear();
                break;
            }
        }
    } catch (const std::exception &e) {
        PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name, e.what());
        result.clear();
    }


    return result.release().ptr();
}

static PyObject *nb_math_binop(const char *name, size_t ops_offset,
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

    PyTypeObject *tp = (PyTypeObject *) o0.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    using BinOp = void (*) (const void *, const void *, void *);

    BinOp op;
    memcpy(&op, (uint8_t *) &s + ops_offset, sizeof(BinOp));
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
            PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name, e.what());
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
                     "drjit.%s(): cannot perform operation!",
                     tp->tp_name, name);
        return nullptr;
    }

    Py_ssize_t s0 = sq_length(o0.ptr()),
               s1 = sq_length(o1.ptr()),
               sr = s0 > s1 ? s0 : s1;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "drjit.%s(): binary operation involving arrays of "
                     "incompatible size: %zd and %zd!",
                     name, s0, s1);
        return nullptr;
    }

    nb::inst_zero(result);

    if (s.meta.shape[0] == DRJIT_DYNAMIC)
        s.init(nb::inst_ptr<void>(result), sr);

    Py_ssize_t i0 = 0,
               i1 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1;

    try {
        for (Py_ssize_t i = 0; i < sr; ++i) {
            nb::object v0 = nb::steal(sq_item(o0.ptr(), i0)),
                       v1 = nb::steal(sq_item(o1.ptr(), i1));

            if (!v0.is_valid() || !v1.is_valid()) {
                result.clear();
                break;
            }

            nb::object vr = py_op(v0, v1);
            if (!vr.is_valid() || sq_ass_item(result.ptr(), i, vr.ptr())) {
                result.clear();
                break;
            }

            i0 += k0; i1 += k1;
        }
    } catch (const std::exception &e) {
        PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name, e.what());
        result.clear();
    }

    return result.release().ptr();
}

static PyObject *nb_math_ternop(const char *name, size_t ops_offset,
                                PyObject *h0, PyObject *h1, PyObject *h2) noexcept {
    nb::object o0, o1, o2;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0) == Py_TYPE(h1) && Py_TYPE(h1) == Py_TYPE(h2)) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
        o2 = nb::borrow(h2);
    } else {
        PyObject *o[3] = { h0, h1, h2 };
        if (!promote(name, o, 3))
            return nullptr;
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
        o2 = nb::steal(o[2]);
    }

    PyTypeObject *tp = (PyTypeObject *) o0.type().ptr();
    const supp &s = nb::type_supplement<supp>(tp);

    using TernOp = void (*) (const void *, const void *, const void *, void *);

    TernOp op;
    memcpy(&op, (uint8_t *) &s + ops_offset, sizeof(TernOp));
    if (!op)
        return nb::handle(Py_NotImplemented).inc_ref().ptr();

    nb::object result = nb::inst_alloc(tp);

    if ((uintptr_t) op != 1) {
        try {
            op(nb::inst_ptr<void>(o0),
               nb::inst_ptr<void>(o1),
               nb::inst_ptr<void>(o2),
               nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
        } catch (const std::exception &e) {
            PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name,
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
                     "drjit.%s(): cannot perform operation!", name);
        return nullptr;
    }

    Py_ssize_t s0 = sq_length(o0.ptr()),
               s1 = sq_length(o1.ptr()),
               s2 = sq_length(o2.ptr()),
               st = s0 > s1 ? s0 : s1,
               sr = s2 > st ? s2 : st;

    if ((s0 != sr && s0 != 1) || (s1 != sr && s1 != 1) ||
        (s2 != sr && s2 != 1)) {
        PyErr_Format(PyExc_IndexError,
                     "drjit.%s(): ternary operation involving arrays of "
                     "incompatible size: %zd, %zd, and %zd!",
                     name, s0, s1, s2);
        return nullptr;
    }

    nb::inst_zero(result);

    if (s.meta.shape[0] == DRJIT_DYNAMIC)
        s.init(nb::inst_ptr<void>(result), sr);

    Py_ssize_t i0 = 0,
               i1 = 0,
               i2 = 0,
               k0 = s0 == 1 ? 0 : 1,
               k1 = s1 == 1 ? 0 : 1,
               k2 = s2 == 1 ? 0 : 1;

    try {
        for (Py_ssize_t i = 0; i < sr; ++i) {
            nb::object v0 = nb::steal(sq_item(o0.ptr(), i0)),
                       v1 = nb::steal(sq_item(o1.ptr(), i1)),
                       v2 = nb::steal(sq_item(o2.ptr(), i2));

            if (!v0.is_valid() || !v1.is_valid() || !v2.is_valid()) {
                result.clear();
                break;
            }

            nb::object vr = py_op(v0, v1, v2);
            if (!vr.is_valid() || sq_ass_item(result.ptr(), i, vr.ptr())) {
                result.clear();
                break;
            }

            i0 += k0; i1 += k1; i2 += k2;
        }
    } catch (const std::exception &e) {
        PyErr_Format(PyExc_RuntimeError, "drjit.%s(): %s!", name, e.what());
        result.clear();
    }

    return result.release().ptr();
}

#define DR_MATH_UNOP(name)                                                     \
    m.def(                                                                     \
        #name, [](double d) { return dr::name(d); }, nb::raw_doc(doc_##name)); \
    m.def(#name, [](nb::handle_t<dr::ArrayBase> h) {                           \
        return nb_math_unop(#name, offsetof(supp, op_##name), h);              \
    });

#define DR_MATH_UNOP_2(name)                                                   \
    m.def(                                                                     \
        #name, [](double d) { return dr::name(d); }, nb::raw_doc(doc_##name)); \
    m.def(#name, [](nb::handle_t<dr::ArrayBase> h) {                           \
        return nb::steal(                                                      \
            nb_math_unop_2(#name, offsetof(supp, op_##name), h.ptr()));        \
    });

#define DR_MATH_BINOP(name)                                                    \
    m.def(                                                                     \
        #name, [](double d1, double d2) { return dr::name(d1, d2); },          \
        nb::raw_doc(doc_##name));                                              \
    m.def(#name, [](nb::handle h1, nb::handle h2) {                            \
        if (!is_drjit_array(h1) && !is_drjit_array(h2))                        \
            throw nb::next_overload();                                         \
        return nb::steal(nb_math_binop(#name, offsetof(supp, op_##name),       \
                                       h1.ptr(), h2.ptr()));                   \
    });

#define DR_MATH_TERNOP(name)                                                   \
    m.def(                                                                     \
        #name,                                                                 \
        [](double d1, double d2, double d3) { return dr::name(d1, d2, d3); },  \
        nb::raw_doc(doc_##name));                                              \
    m.def(#name, [](nb::handle h1, nb::handle h2, nb::handle h3) {             \
        if (!is_drjit_array(h1) && !is_drjit_array(h2) && !is_drjit_array(h3)) \
            throw nb::next_overload();                                         \
        return nb::steal(nb_math_ternop(#name, offsetof(supp, op_##name),      \
                                        h1.ptr(), h2.ptr(), h3.ptr()));        \
    });

static nb::object minimum(nb::handle h0, nb::handle h1) {
    if (!is_drjit_array(h0) && !is_drjit_array(h1)) {
        try {
            double a = nb::cast<double>(h0);
            double b = nb::cast<double>(h1);
            return a > b ? nb::borrow(h1) : nb::borrow(h0);
        } catch (...) {}

        try {
            nb::list a = nb::cast<nb::list>(h0);
            nb::list b = nb::cast<nb::list>(h1);

            nb::list result;
            for (size_t i = 0, l = nb::len(a); i < l; i++)
                result.append(minimum(h0[i], h1[i]));
            return std::move(result);
        } catch (...) {}

        PyErr_Format(PyExc_TypeError, "minimum(): invalid arguments!");
        return nb::object();
    }

    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0.ptr()) == Py_TYPE(h1.ptr())) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0.ptr(), h1.ptr() };
        if (!promote("minimum", o, 2))
            return nb::object();
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    return nb::steal(
        nb_math_binop("minimum", offsetof(supp, op_minimum), h0.ptr(), h1.ptr()));
}

static nb::object maximum(nb::handle h0, nb::handle h1) {
    if (!is_drjit_array(h0) && !is_drjit_array(h1)) {
        try {
            double a = nb::cast<double>(h0);
            double b = nb::cast<double>(h1);
            return a > b ? nb::borrow(h0) : nb::borrow(h1);
        } catch (...) {}

        try {
            nb::list a = nb::cast<nb::list>(h0);
            nb::list b = nb::cast<nb::list>(h1);

            nb::list result;
            for (size_t i = 0, l = nb::len(a); i < l; i++)
                result.append(maximum(h0[i], h1[i]));
            return std::move(result);
        } catch (...) {}

        PyErr_Format(PyExc_TypeError, "maximum(): invalid arguments!");
        return nb::object();
    }

    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0.ptr()) == Py_TYPE(h1.ptr())) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0.ptr(), h1.ptr() };
        if (!promote("maximum", o, 2))
            return nb::object();
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    return nb::steal(
        nb_math_binop("maximum", offsetof(supp, op_maximum), h0.ptr(), h1.ptr()));
}

static nb::object sum(nb::handle h) {
    nb::handle tp = h.type();
    if (tp.is(&PyFloat_Type) || tp.is(&PyLong_Type) || tp.is(&PyBool_Type))
        return borrow(h);

    if (is_drjit_type(tp)) {
        const supp &s = nb::type_supplement<supp>(tp);
        dr::detail::array_reduce op = s.op_sum;
        if (!op)
            throw nb::type_error(
                "drjit.sum(): requires a Dr.Jit array or Python "
                "sequence as input.");

        if ((uintptr_t) op != 1) {
            nb::object result = nb::inst_alloc(tp);
            op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
            return result;
        }
    }

    nb::object result = nb::borrow(PyFloat_FromDouble(0.0));
    for (nb::handle h2 : h)
        result = result + h2;
    return result;
}

static nb::object prod(nb::handle h) {
    nb::handle tp = h.type();
    if (tp.is(&PyFloat_Type) || tp.is(&PyLong_Type) || tp.is(&PyBool_Type))
        return borrow(h);

    if (is_drjit_type(tp)) {
        const supp &s = nb::type_supplement<supp>(tp);
        dr::detail::array_reduce op = s.op_prod;
        if (!op)
            throw nb::type_error(
                "drjit.prod(): requires a Dr.Jit array or Python "
                "sequence as input.");

        if ((uintptr_t) op != 1) {
            nb::object result = nb::inst_alloc(tp);
            op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
            nb::inst_mark_ready(result);
            return result;
        }
    }

    nb::object result = nb::borrow(PyFloat_FromDouble(1.0));
    for (nb::handle h2 : h)
        result = result * h2;
    return result;
}

static nb::object dot(nb::handle h0, nb::handle h1) {
    if (!is_drjit_array(h0) && !is_drjit_array(h1)) {
        try {
            nb::list a = nb::cast<nb::list>(h0);
            nb::list b = nb::cast<nb::list>(h1);

            nb::object result;
            for (size_t i = 0, l = nb::len(a); i < l; i++) {
                if (i == 0)
                    result = h0[i] * h1[i];
                else
                    result = result + h0[i] * h1[i];
            }
            return result;
        } catch (...) {}

        PyErr_Format(PyExc_TypeError, "dot(): invalid arguments!");
        return nb::object();
    }

    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0.ptr()) == Py_TYPE(h1.ptr())) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0.ptr(), h1.ptr() };
        if (!promote("maximum", o, 2))
            return nb::object();
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    const supp &s = nb::type_supplement<supp>(o0.type());
    if (s.meta.ndim == 1) {
        return sum(h0 * h1);
    } else {
        PySequenceMethods *sm = ((PyTypeObject *) o0.type().ptr())->tp_as_sequence;
        nb::object result = o0[0] * o1[0];
        if (is_float_v(o0.type())) {
            for (Py_ssize_t i = 1; i < sm->sq_length(o0.ptr()); ++i)
                result = drjit::fmadd(o0[i], o1[i], result);
        } else {
            for (Py_ssize_t i = 1; i < sm->sq_length(o0.ptr()); ++i)
                result = result + o0[i] * o1[i];
        }
        return result;
    }
}


void bind_array_math(nb::module_ m) {
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
    DR_MATH_UNOP(exp);
    DR_MATH_UNOP(exp2);
    DR_MATH_UNOP(log);
    DR_MATH_UNOP(log2);
    DR_MATH_UNOP(sqrt);
    DR_MATH_UNOP(cbrt);
    DR_MATH_UNOP(floor);
    DR_MATH_UNOP(ceil);
    DR_MATH_UNOP(round);
    DR_MATH_UNOP(trunc);
    DR_MATH_UNOP(rcp);
    DR_MATH_UNOP(rsqrt);
    DR_MATH_BINOP(atan2);
    DR_MATH_BINOP(ldexp);
    DR_MATH_TERNOP(fma);
    DR_MATH_UNOP_2(sincos);
    DR_MATH_UNOP_2(sincosh);
    DR_MATH_UNOP_2(frexp);

    m.def("abs", [](double d) { return dr::abs(d); }, nb::raw_doc(doc_abs));
    m.def("abs", [](Py_ssize_t i) { return dr::abs(i); });
    m.def("abs", [](nb::handle_t<dr::ArrayBase> h) {
        return nb_math_unop("abs", offsetof(supp, op_absolute), h);
    });

    m.def("minimum", [](double d1, double d2) { return dr::min(d1, d2); }, nb::raw_doc(doc_minimum));
    m.def("minimum", [](Py_ssize_t d1, Py_ssize_t d2) { return dr::min(d1, d2); });
    m.def("minimum", [](nb::handle h1, nb::handle h2) { return minimum(h1, h2); });

    m.def("maximum", [](double d1, double d2) { return dr::max(d1, d2); }, nb::raw_doc(doc_maximum));
    m.def("maximum", [](Py_ssize_t d1, Py_ssize_t d2) { return dr::max(d1, d2); });
    m.def("maximum", [](nb::handle h1, nb::handle h2) { return maximum(h1, h2); });

    m.def("all", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_array(h)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce op = s.op_all;
            if (!op)
                throw nb::type_error(
                    "drjit.all(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }
        }

        nb::object result = nb::borrow(Py_True);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = result & h2;
        }

        return result;
    }, nb::raw_doc(doc_all));

    m.def("any", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_type(tp)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce op = s.op_any;
            if (!op)
                throw nb::type_error(
                    "drjit.any(): requires a Dr.Jit mask array or Python "
                    "boolean sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }
        }

        nb::object result = nb::borrow(Py_False);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = result | h2;
        }

        return result;
    }, nb::raw_doc(doc_any));

    m.def("min", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyFloat_Type) || tp.is(&PyLong_Type) || tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_type(tp)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce op = s.op_min;
            if (!op)
                throw nb::type_error(
                    "drjit.min(): requires a Dr.Jit array or Python "
                    "sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }
        }

        nb::object result;

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = minimum(result, h2);
        }
        return result;
    }, nb::raw_doc(doc_min));

    m.def("max", [](nb::handle h) -> nb::object {
        nb::handle tp = h.type();
        if (tp.is(&PyFloat_Type) || tp.is(&PyLong_Type) || tp.is(&PyBool_Type))
            return borrow(h);

        if (is_drjit_type(tp)) {
            const supp &s = nb::type_supplement<supp>(tp);
            dr::detail::array_reduce op = s.op_max;
            if (!op)
                throw nb::type_error(
                    "drjit.max(): requires a Dr.Jit array or Python "
                    "sequence as input.");

            if ((uintptr_t) op != 1) {
                nb::object result = nb::inst_alloc(tp);
                op(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            }
        }

        nb::object result;

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = maximum(result, h2);
        }

        return result;
    }, nb::raw_doc(doc_max));

    m.def("sum", &sum, nb::raw_doc(doc_sum));
    m.def("prod", &prod, nb::raw_doc(doc_prod));
    m.def("dot", &dot, nb::raw_doc(doc_dot));
    m.def("norm", [](nb::handle h) -> nb::object {
        return nb_math_unop("sqrt", offsetof(supp, op_sqrt), dot(h, h));
    }, nb::raw_doc(doc_norm));
}
