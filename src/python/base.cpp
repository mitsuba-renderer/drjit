/*
    base.cpp -- Bindings of the ArrayBase type underlying
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
#include "slice.h"
#include <nanobind/stl/string.h>

#define DR_NB_UNOP(name, op)                                                   \
    static PyObject *nb_##name(PyObject *h0) noexcept {                        \
        return apply<Normal>(op, Py_nb_##name, std::make_index_sequence<1>(),  \
                             h0);                                              \
    }

#define DR_NB_BINOP(name, op)                                                  \
    static PyObject *nb_##name(PyObject *h0, PyObject *h1) noexcept {          \
        return apply<Normal>(op, Py_nb_##name, std::make_index_sequence<2>(),  \
                             h0, h1);                                          \
    }                                                                          \
    static PyObject *nb_inplace_##name(PyObject *h0, PyObject *h1) noexcept {  \
        return apply<InPlace>(op, Py_nb_##name, std::make_index_sequence<2>(), \
                              h0, h1);                                         \
    }

#define DR_MATH_UNOP(name, op)                                                 \
    m.def(#name, [](nb::handle_t<ArrayBase> h0) {                              \
        return nb::steal(apply<Normal>(                                        \
            op, #name, std::make_index_sequence<1>(), h0.ptr()));              \
    }, nb::raw_doc(doc_##name));                                               \
    m.def(#name, [](double v0) { return dr::name(v0); });

#define DR_MATH_UNOP_PAIR(name, op)                                            \
    m.def(#name, [](nb::handle_t<ArrayBase> h0) {                              \
        return apply_ret_pair(op, #name, h0);                                  \
    }, nb::raw_doc(doc_##name));                                               \
    m.def(#name, [](double v0) { return dr::name(v0); });

#define DR_MATH_BINOP(name, op)                                                \
    m.def(#name, [](nb::handle h0, nb::handle h1) {                            \
        if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1)))           \
            throw nb::next_overload();                                         \
        return nb::steal(apply<Normal>(                                        \
            op, #name, std::make_index_sequence<2>(), h0.ptr(), h1.ptr()));    \
    }, nb::raw_doc(doc_##name));                                               \
    m.def(#name, [](double v0, double v1) { return dr::name(v0, v1); });

#define DR_MATH_TERNOP(name, op)                                               \
    m.def(#name,                                                               \
          [](nb::handle h0, nb::handle h1, nb::handle h2) {                    \
              if (!is_drjit_array(h0) && !is_drjit_array(h1) &&                \
                  !is_drjit_array(h2))                                         \
                  throw nb::next_overload();                                   \
              return nb::steal(apply<Normal>(op, #name,                        \
                                             std::make_index_sequence<3>(),    \
                                             h0.ptr(), h1.ptr(), h2.ptr()));   \
          }, nb::raw_doc(doc_##name));                                         \
    m.def(#name, [](double v0, double v1, double v2) {                         \
        return dr::name(v0, v1, v2);                                           \
    });

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

static PyObject *nb_power(PyObject *h0_, PyObject *h1_) noexcept {
    nb::handle h0 = h0_, h1 = h1_;

    try {
        Py_ssize_t i1 = 0;
        double d1 = 0.0;
        if (nb::try_cast(h1, i1) || (nb::try_cast(h1, d1) && (i1 = (Py_ssize_t) d1, (double) i1 == d1))) {
            if (i1 == PY_SSIZE_T_MIN)
                nb::detail::raise("Negative exponent is too large!");

            Py_ssize_t u1 = (i1 < 0) ? -i1 : i1;

            nb::object result = array_module.attr("ones")(h0.type(), nb::len(h0)),
                       x = nb::borrow(h0);

            while (u1) {
                if (u1 & 1)
                    result *= x;
                x *= x;
                u1 >>= 1;
            }

            if (i1 < 0)
                result = array_module.attr("rcp")(result);

            return result.release().ptr();
        } else {
            nb::object log2 = array_module.attr("log2"),
                       exp2 = array_module.attr("exp2");

            return exp2(log2(h0) * h1).release().ptr();
        }
    } catch (nb::python_error &e) {
        nb::str tp0_name = nb::inst_name(h0), tp1_name = nb::inst_name(h1);
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.power(<%U>, <%U>): failed (see above)!",
                        tp0_name.ptr(), tp1_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp0_name = nb::inst_name(h0), tp1_name = nb::inst_name(h1);
        nb::chain_error(PyExc_RuntimeError, "drjit.power(<%U>, <%U>): %s",
                        tp0_name.ptr(), tp1_name.ptr(), e.what());
    }

    return nullptr;
}

static PyObject *nb_inplace_power(PyObject *h0, PyObject *h1) noexcept {
    PyObject *r = nb_power(h0, h1);
    if (!r)
        return nullptr;

    if (Py_TYPE(r) == Py_TYPE(h0) && h0 != r) {
        nb::inst_replace_move(h0, r);
        Py_INCREF(h0);
        Py_DECREF(r);
        return h0;
    } else {
        return r;
    }
}

static PyObject *tp_richcompare(PyObject *h0, PyObject *h1, int slot) noexcept {
    return apply<RichCompare>(ArrayOp::Richcmp, slot,
                              std::make_index_sequence<2>(), h0, h1);
}


template <int Index> nb::object xyzw_getter(nb::handle_t<ArrayBase> h) {
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
void xyzw_setter(nb::handle_t<ArrayBase> h, nb::handle value) {
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

template <int Index> nb::object complex_getter(nb::handle_t<ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        nb::str name = nb::inst_name(h);
        nb::detail::raise("%s does not have a '%s' component.", name.c_str(),
                          Index == 0 ? "real" : "imaginary");
    }

    return nb::steal(s.item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index>
void complex_setter(nb::handle_t<ArrayBase> h, nb::handle value) {
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

    if (s.is_tensor) {
        nb::object array = nb::steal(s.tensor_array(o));
        return nb_bool(array.ptr());
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
        length = (Py_ssize_t) s.len(inst_ptr(o));

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
    DR_ARRAY_SLOT(nb_power),
    DR_ARRAY_SLOT(nb_inplace_power),

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
    DR_ARRAY_SLOT(mp_length),
    DR_ARRAY_SLOT(nb_bool),
    DR_ARRAY_SLOT(mp_subscript),
    DR_ARRAY_SLOT(mp_ass_subscript),

    { 0, nullptr }
};

namespace drjit {
    template <typename T> T fma(const T &a, const T &b, const T &c) {
        return fmadd(a, b, c);
    }
};

nb::object fma(nb::handle h0, nb::handle h1, nb::handle h2) {
    if (!is_drjit_array(h0) && !is_drjit_array(h1) && !is_drjit_array(h2))
        throw nb::next_overload();

    PyObject *o =
        apply<Normal>(ArrayOp::Fma, "fma", std::make_index_sequence<3>(),
                      h0.ptr(), h1.ptr(), h2.ptr());

    if (!o)
        nb::detail::raise_python_error();

    return nb::steal(o);
}

nb::object select(nb::handle h0, nb::handle h1, nb::handle h2) {
    if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1) &&
                    !is_drjit_array(h2)))
        throw nb::next_overload();

    PyObject *o =
        apply<Select>(ArrayOp::Select, "select", std::make_index_sequence<3>(),
                      h0.ptr(), h1.ptr(), h2.ptr());

    if (!o)
        nb::detail::raise_python_error();

    return nb::steal(o);
}

void export_base(nb::module_ &m) {
    nb::class_<ArrayBase> ab(m, "ArrayBase",
                                 nb::type_slots(array_base_slots),
                                 nb::supplement<ArraySupplement>());

    ab.def_prop_ro_static("__meta__", [](nb::type_object_t<ArrayBase> tp) {
        return meta_str(supp(tp));
    });

    ab.def_prop_ro("ndim", &ndim, nb::raw_doc(doc_ArrayBase_ndim));
    ab.def_prop_ro("shape", &shape, nb::raw_doc(doc_ArrayBase_shape));
    ab.def_prop_ro(
        "array",
        [](nb::handle_t<ArrayBase> h) -> nb::object {
            const ArraySupplement &s = supp(h.type());
            if (s.is_tensor)
                return nb::steal(s.tensor_array(h.ptr()));
            else if (s.is_matrix || s.is_quaternion || s.is_complex)
                return nb::handle(s.array)(h);
            else
                return nb::borrow(h);
        },
        nb::raw_doc(doc_ArrayBase_array));

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

    ab.def_prop_ro(
        "index",
        [](nb::handle_t<dr::ArrayBase> h) -> uint32_t {
            const ArraySupplement &s = supp(h.type());
            if (s.is_tensor || !s.index)
                return 0;
            return (uint32_t) s.index(inst_ptr(h));
        },
        nb::raw_doc(doc_ArrayBase_index));

    ab.def_prop_ro(
        "index_ad",
        [](nb::handle_t<dr::ArrayBase> h) -> uint32_t {
            const ArraySupplement &s = supp(h.type());
            if (s.is_tensor || !s.index)
                return 0;
            return s.index(inst_ptr(h)) >> 32;
        },
        nb::raw_doc(doc_ArrayBase_index_ad));

    m.def("abs", [](Py_ssize_t a) { return dr::abs(a); });
    DR_MATH_UNOP(abs, ArrayOp::Abs);
    DR_MATH_UNOP(sqrt, ArrayOp::Sqrt);
    DR_MATH_UNOP(rcp, ArrayOp::Rcp);
    DR_MATH_UNOP(rsqrt, ArrayOp::Rsqrt);
    DR_MATH_UNOP(cbrt, ArrayOp::Cbrt);

    DR_MATH_UNOP(round, ArrayOp::Round);
    DR_MATH_UNOP(trunc, ArrayOp::Trunc);
    DR_MATH_UNOP(ceil, ArrayOp::Ceil);
    DR_MATH_UNOP(floor, ArrayOp::Floor);

    DR_MATH_UNOP(exp, ArrayOp::Exp);
    DR_MATH_UNOP(exp2, ArrayOp::Exp2);
    DR_MATH_UNOP(log, ArrayOp::Log);
    DR_MATH_UNOP(log2, ArrayOp::Log2);

    DR_MATH_UNOP(sin, ArrayOp::Sin);
    DR_MATH_UNOP(cos, ArrayOp::Cos);
    DR_MATH_UNOP(tan, ArrayOp::Tan);
    DR_MATH_UNOP(asin, ArrayOp::Asin);
    DR_MATH_UNOP(acos, ArrayOp::Acos);
    DR_MATH_UNOP(atan, ArrayOp::Atan);

    DR_MATH_UNOP(sinh, ArrayOp::Sinh);
    DR_MATH_UNOP(cosh, ArrayOp::Cosh);
    DR_MATH_UNOP(tanh, ArrayOp::Tanh);
    DR_MATH_UNOP(asinh, ArrayOp::Asinh);
    DR_MATH_UNOP(acosh, ArrayOp::Acosh);
    DR_MATH_UNOP(atanh, ArrayOp::Atanh);

    DR_MATH_UNOP(erf, ArrayOp::Erf);

    DR_MATH_UNOP_PAIR(sincos, ArrayOp::Sincos);
    DR_MATH_UNOP_PAIR(sincosh, ArrayOp::Sincosh);

    m.def("sqr", [](nb::handle h) { return h*h; }, doc_sqr);

    m.def("minimum",
          [](Py_ssize_t a, Py_ssize_t b) { return dr::minimum(a, b); });
    DR_MATH_BINOP(minimum, ArrayOp::Minimum);

    m.def("maximum",
          [](Py_ssize_t a, Py_ssize_t b) { return dr::maximum(a, b); });
    DR_MATH_BINOP(maximum, ArrayOp::Maximum);

    DR_MATH_BINOP(atan2, ArrayOp::Atan2);

    m.def("fma", [](Py_ssize_t a, Py_ssize_t b, Py_ssize_t c) {
        return dr::fma(a, b, c);
    });

    DR_MATH_TERNOP(fma, ArrayOp::Fma);

    m.def("select",
          nb::overload_cast<nb::handle, nb::handle, nb::handle>(&select),
          nb::raw_doc(doc_select));

    m.def("select",
          [](bool mask, nb::handle a, nb::handle b) {
              return nb::borrow(mask ? a : b);
          });

    m.def(
        "power", [](Py_ssize_t arg0, Py_ssize_t arg1) { return std::pow(arg0, arg1); },
        doc_pow);

    m.def(
        "power", [](double arg0, double arg1) { return std::pow(arg0, arg1); });

    m.def("power",
          [](nb::handle h0, nb::handle h1) {
              if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1)))
                  throw nb::next_overload();
              return nb::steal(nb_power(h0.ptr(), h1.ptr()));
          }
    );

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
