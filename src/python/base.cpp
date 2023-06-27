#include <nanobind/stl/string.h>
#include "base.h"
#include "meta.h"
#include "repr.h"
#include "iter.h"
#include "shape.h"
#include "apply.h"

nb::handle array_base;

static PyObject *nb_negative(PyObject *h0) noexcept {
    return apply<Normal>(ArrayOp::Neg, Py_nb_negative,
                         std::make_index_sequence<1>(), h0);
}

static PyObject *nb_absolute(PyObject *h0) noexcept {
    return apply<Normal>(ArrayOp::Abs, Py_nb_absolute, std::make_index_sequence<1>(), h0);
}

static PyObject *nb_invert(PyObject *h0) noexcept {
    return apply<Normal>(ArrayOp::Invert, Py_nb_invert,
                         std::make_index_sequence<1>(), h0);
}

static PyObject *nb_add(PyObject *h0, PyObject *h1) noexcept {
    return apply<Normal>(ArrayOp::Add, Py_nb_add,
                         std::make_index_sequence<2>(), h0, h1);
}

static PyObject *nb_subtract(PyObject *h0, PyObject *h1) noexcept {
    return apply<Normal>(ArrayOp::Sub, Py_nb_subtract,
                         std::make_index_sequence<2>(), h0, h1);
}

static PyObject *nb_multiply(PyObject *h0, PyObject *h1) noexcept {
    return apply<Normal>(ArrayOp::Mul, Py_nb_multiply,
                         std::make_index_sequence<2>(), h0, h1);
}

static PyObject *nb_and(PyObject *h0, PyObject *h1) noexcept {
    return apply<Normal>(ArrayOp::And, Py_nb_and, std::make_index_sequence<2>(),
                         h0, h1);
}

static PyObject *nb_or(PyObject *h0, PyObject *h1) noexcept {
    return apply<Normal>(ArrayOp::Or, Py_nb_or, std::make_index_sequence<2>(),
                         h0, h1);
}

static PyObject *nb_xor(PyObject *h0, PyObject *h1) noexcept {
    return apply<Normal>(ArrayOp::Xor, Py_nb_xor, std::make_index_sequence<2>(),
                         h0, h1);
}


static PyObject *tp_richcompare(PyObject *h0, PyObject *h1, int slot) noexcept {
    return apply<RichCompare>(ArrayOp::Richcmp, slot,
                              std::make_index_sequence<2>(), h0, h1);
}

template <int Index> nb::object xyzw_getter(nb::handle_t<dr::ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY((!s.is_vector && !s.is_quaternion) || s.ndim == 0 ||
                    s.shape[0] == DRJIT_DYNAMIC || Index >= s.shape[0])) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%c' component.", name.c_str(),
                          "xyzw"[Index]);
    }

    return nb::steal(s.item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index> void xyzw_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY((!s.is_vector && !s.is_quaternion) || s.ndim == 0 ||
                    s.shape[0] == DRJIT_DYNAMIC || Index >= s.shape[0])) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%c' component.", name.c_str(),
                          "xyzw"[Index]);
    }

    if (s.set_item(h.ptr(), (Py_ssize_t) Index, value.ptr()))
        nb::detail::raise_python_error();
}

template <int Index> nb::object complex_getter(nb::handle_t<dr::ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%s' component.", name.c_str(),
                          Index == 0 ? "real" : "imaginary");
    }

    return nb::steal(s.item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index> void complex_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%s' component.", name.c_str(),
                          Index == 0 ? "real" : "imaginary");
    }

    if (s.set_item(h.ptr(), (Py_ssize_t) Index, value.ptr()))
        nb::detail::raise_python_error();
}

static int nb_bool(PyObject *o) noexcept {
    PyTypeObject *tp = Py_TYPE(o);
    const ArraySupplement &s = supp(tp);

    if (NB_UNLIKELY(s.type != (uint16_t) VarType::Bool)) {
        nb::str name = nb::type_name(tp);
        PyErr_Format(PyExc_TypeError,
                     "%U.__bool__(): implicit conversion to 'bool' is only "
                     "supported for scalar mask arrays.", name.ptr());
        return -1;
    }

    if (NB_UNLIKELY(s.ndim != 1)) {
        nb::str name = nb::type_name(tp);
        PyErr_Format(
            PyExc_RuntimeError,
            "%U.__bool__(): implicit conversion to 'bool' requires an "
            "array with at most 1 dimension (this one has %i dimensions).",
            name.ptr(), (int) s.ndim);
        return -1;
    }

    Py_ssize_t length = s.shape[0];
    if (length == DRJIT_DYNAMIC)
        length = (Py_ssize_t) s.len(nb::inst_ptr<dr::ArrayBase>(o));

    if (NB_UNLIKELY(length != 1)) {
        nb::str name = nb::type_name(tp);
        PyErr_Format(
            PyExc_RuntimeError,
            "%U.__bool__(): implicit conversion to 'bool' requires an "
            "array with at most 1 element (this one has %zd elements).",
            name.ptr(), length);
        return -1;
    }

    PyObject *result = s.item(o, 0);
    if (!result)
        return -1;
    Py_DECREF(result);

    if (result == Py_True) {
        return 1;
    } else if (result == Py_False) {
        return 0;
    } else {
        nb::str name = nb::type_name(tp);
        PyErr_Format(PyExc_RuntimeError, "%U.__bool__(): internal error!");
        return -1;
    }
}

static PyObject *nb_positive(PyObject *o) noexcept {
    Py_INCREF(o);
    return o;
}

#define DR_ARRAY_SLOT(name) { Py_##name, (void *) name }

static PyType_Slot array_base_slots[] = {
    // Unary operations
    DR_ARRAY_SLOT(nb_absolute),
    DR_ARRAY_SLOT(nb_negative),
    DR_ARRAY_SLOT(nb_invert),
    DR_ARRAY_SLOT(nb_positive),

    /// Binary arithmetic operations
    DR_ARRAY_SLOT(nb_add),
    DR_ARRAY_SLOT(nb_subtract),
    DR_ARRAY_SLOT(nb_multiply),

    /// Binary bit/mask operations
    DR_ARRAY_SLOT(nb_and),
    DR_ARRAY_SLOT(nb_or),
    DR_ARRAY_SLOT(nb_xor),

    /// Miscellaneous
    DR_ARRAY_SLOT(tp_iter),
    DR_ARRAY_SLOT(tp_repr),
    DR_ARRAY_SLOT(tp_richcompare),
    DR_ARRAY_SLOT(sq_length),
    DR_ARRAY_SLOT(nb_bool),

    { 0, nullptr }
};


void export_base(nb::module_ &m) {
    nb::class_<dr::ArrayBase> ab(m, "ArrayBase",
                                 nb::type_slots(array_base_slots),
                                 nb::supplement<ArraySupplement>());

    ab.def_prop_ro_static("__meta__", [](nb::handle h) {
        return meta_str(nb::type_supplement<ArraySupplement>(h));
    });

    ab.def_prop_ro("shape", &shape, nb::raw_doc(doc_ArrayBase_shape));
    ab.def_prop_rw("x", xyzw_getter<0>, xyzw_setter<0>, nb::raw_doc(doc_ArrayBase_x));
    ab.def_prop_rw("y", xyzw_getter<1>, xyzw_setter<1>, nb::raw_doc(doc_ArrayBase_y));
    ab.def_prop_rw("z", xyzw_getter<2>, xyzw_setter<2>, nb::raw_doc(doc_ArrayBase_z));
    ab.def_prop_rw("w", xyzw_getter<3>, xyzw_setter<3>, nb::raw_doc(doc_ArrayBase_w));
    ab.def_prop_rw("real", complex_getter<0>, complex_setter<0>, nb::raw_doc(doc_ArrayBase_real));
    ab.def_prop_rw("imag", complex_getter<1>, complex_setter<1>, nb::raw_doc(doc_ArrayBase_imag));

    array_base = ab;
}
