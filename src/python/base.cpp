/*
    base.cpp -- Bindings of the dr::ArrayBase type underlying
    all Dr.Jit arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "base.h"
#include "apply.h"
#include "iter.h"
#include "meta.h"
#include "repr.h"
#include "shape.h"
#include <nanobind/stl/string.h>

#define DR_NB_UNOP(name, op)                                                   \
    static PyObject *nb_##name(PyObject *h0) noexcept {                        \
        return apply<Normal>(op, Py_nb_##name, std::make_index_sequence<1>(),  \
                             h0);                                              \
    }

#define DR_NB_BINOP(name, op)                                             \
    static PyObject *nb_##name(PyObject *h0, PyObject *h1) noexcept {          \
        return apply<Normal>(op, Py_nb_##name, std::make_index_sequence<2>(),  \
                             h0, h1);                                          \
    }                                                                          \
    static PyObject *nb_inplace_##name(PyObject *h0, PyObject *h1) noexcept {  \
        return apply<InPlace>(op, Py_nb_##name, std::make_index_sequence<2>(), \
                              h0, h1);                                         \
    }

#define DR_MATH_BINOP(name, op)                                                \
    m.def(#name, [](nb::handle h0, nb::handle h1) {                            \
        if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1)))           \
            throw nb::next_overload();                                         \
        return nb::steal(apply<Normal>(                                        \
            op, #name, std::make_index_sequence<2>(), h0.ptr(), h1.ptr()));    \
    }, doc_##name);                                                            \
    m.def(                                                                     \
        #name, [](double v0, double v1) { return dr::name(v0, v1); },          \
        nb::raw_doc(doc_##name));

#define DR_MATH_TERNOP(name, op)                                               \
    m.def(#name, [](nb::handle h0, nb::handle h1, nb::handle h2) {             \
        if (!is_drjit_array(h0) && !is_drjit_array(h1) && !is_drjit_array(h2)) \
            throw nb::next_overload();                                         \
        return nb::steal(apply<Normal>(op, #name,                              \
                                       std::make_index_sequence<3>(),          \
                                       h0.ptr(), h1.ptr(), h2.ptr()));         \
    }, doc_##name);                                                            \
    m.def(                                                                     \
        #name,                                                                 \
        [](double v0, double v1, double v2) { return dr::name(v0, v1, v2); },  \
        nb::raw_doc(doc_##name));

nb::handle array_module;
nb::handle array_submodules[5];
nb::handle array_base;

DR_NB_UNOP(negative, ArrayOp::Neg)
DR_NB_UNOP(absolute, ArrayOp::Abs)
DR_NB_UNOP(invert, ArrayOp::Invert)
DR_NB_BINOP(add, ArrayOp::Add)
DR_NB_BINOP(subtract, ArrayOp::Sub)
DR_NB_BINOP(multiply, ArrayOp::Mul)
DR_NB_BINOP(true_divide, ArrayOp::TrueDiv)
DR_NB_BINOP(floor_divide, ArrayOp::FloorDiv)
DR_NB_BINOP(lshift, ArrayOp::LShift)
DR_NB_BINOP(rshift, ArrayOp::RShift)
DR_NB_BINOP(remainder, ArrayOp::Mod)
DR_NB_BINOP(and, ArrayOp::And)
DR_NB_BINOP(or, ArrayOp::Or)
DR_NB_BINOP(xor, ArrayOp::Xor)

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

template <int Index>
void xyzw_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
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

template <int Index>
void complex_setter(nb::handle_t<dr::ArrayBase> h, nb::handle value) {
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
                     "supported for scalar mask arrays.",
                     name.ptr());
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

#define DR_ARRAY_SLOT(name)                                                    \
    { Py_##name, (void *) name }

static PyType_Slot array_base_slots[] = {
    // Unary operations
    DR_ARRAY_SLOT(nb_absolute),
    DR_ARRAY_SLOT(nb_negative),
    DR_ARRAY_SLOT(nb_invert),
    DR_ARRAY_SLOT(nb_positive),

    /// Binary arithmetic operations
    DR_ARRAY_SLOT(nb_add),
    DR_ARRAY_SLOT(nb_inplace_add),
    DR_ARRAY_SLOT(nb_subtract),
    DR_ARRAY_SLOT(nb_inplace_subtract),
    DR_ARRAY_SLOT(nb_multiply),
    DR_ARRAY_SLOT(nb_inplace_multiply),
    DR_ARRAY_SLOT(nb_true_divide),
    DR_ARRAY_SLOT(nb_inplace_true_divide),
    DR_ARRAY_SLOT(nb_floor_divide),
    DR_ARRAY_SLOT(nb_inplace_floor_divide),
    DR_ARRAY_SLOT(nb_lshift),
    DR_ARRAY_SLOT(nb_inplace_lshift),
    DR_ARRAY_SLOT(nb_rshift),
    DR_ARRAY_SLOT(nb_inplace_rshift),
    DR_ARRAY_SLOT(nb_remainder),
    DR_ARRAY_SLOT(nb_inplace_remainder),

    /// Binary bit/mask operations
    DR_ARRAY_SLOT(nb_and),
    DR_ARRAY_SLOT(nb_inplace_and),
    DR_ARRAY_SLOT(nb_or),
    DR_ARRAY_SLOT(nb_inplace_or),
    DR_ARRAY_SLOT(nb_xor),
    DR_ARRAY_SLOT(nb_inplace_xor),

    /// Miscellaneous
    DR_ARRAY_SLOT(tp_iter),
    DR_ARRAY_SLOT(tp_repr),
    DR_ARRAY_SLOT(tp_richcompare),
    DR_ARRAY_SLOT(sq_length),
    DR_ARRAY_SLOT(nb_bool),

    { 0, nullptr }
};

namespace drjit {
    template <typename T> T fma(const T &a, const T &b, const T &c) {
        return fmadd(a, b, c);
    }
};

void export_base(nb::module_ &m) {
    nb::class_<dr::ArrayBase> ab(m, "ArrayBase",
                                 nb::type_slots(array_base_slots),
                                 nb::supplement<ArraySupplement>());

    ab.def_prop_ro_static("__meta__", [](nb::handle h) {
        return meta_str(nb::type_supplement<ArraySupplement>(h));
    });

    ab.def_prop_ro("shape", &shape, nb::raw_doc(doc_ArrayBase_shape));
    ab.def_prop_rw("x", xyzw_getter<0>, xyzw_setter<0>,
                   nb::raw_doc(doc_ArrayBase_x));
    ab.def_prop_rw("y", xyzw_getter<1>, xyzw_setter<1>,
                   nb::raw_doc(doc_ArrayBase_y));
    ab.def_prop_rw("z", xyzw_getter<2>, xyzw_setter<2>,
                   nb::raw_doc(doc_ArrayBase_z));
    ab.def_prop_rw("w", xyzw_getter<3>, xyzw_setter<3>,
                   nb::raw_doc(doc_ArrayBase_w));
    ab.def_prop_rw("real", complex_getter<0>, complex_setter<0>,
                   nb::raw_doc(doc_ArrayBase_real));
    ab.def_prop_rw("imag", complex_getter<1>, complex_setter<1>,
                   nb::raw_doc(doc_ArrayBase_imag));

    m.def("minimum",
          [](Py_ssize_t a, Py_ssize_t b) { return dr::minimum(a, b); }, doc_minimum);
    DR_MATH_BINOP(minimum, ArrayOp::Minimum);

    m.def("maximum",
          [](Py_ssize_t a, Py_ssize_t b) { return dr::maximum(a, b); }, doc_maximum);
    DR_MATH_BINOP(maximum, ArrayOp::Maximum);

    DR_MATH_TERNOP(fma, ArrayOp::Fma);

    m.def("select",
          [](nb::handle h0, nb::handle h1, nb::handle h2) {
              if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1) &&
                              !is_drjit_array(h2)))
                  throw nb::next_overload();
              return nb::steal(apply<Select>(ArrayOp::Select, "select",
                                             std::make_index_sequence<3>(),
                                             h0.ptr(), h1.ptr(), h2.ptr()));
          }, nb::raw_doc(doc_select));

    m.def("select",
          [](bool mask, nb::handle a, nb::handle b) {
              return nb::borrow(mask ? a : b);
          });

    array_base = ab;
    array_module = m;
    array_submodules[0] = m.attr("scalar");

#if defined(DRJIT_ENABLE_CUDA)
    array_submodules[1] = m.attr("cuda");
    array_submodules[2] = array_submodules[1].attr("ad");
#endif

#if defined(DRJIT_ENABLE_LLVM)
    array_submodules[3] = m.attr("llvm");
    array_submodules[4] = array_submodules[3].attr("ad");
#endif
}
