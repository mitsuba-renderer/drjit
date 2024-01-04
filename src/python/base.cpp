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
#include "print.h"
#include "shape.h"
#include "slice.h"
#include "inspect.h"
#include "traits.h"
#include "init.h"
#include "autodiff.h"
#include "reduce.h"
#include <cmath>
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

#define DR_MATH_UNOP_UINT32(name, op)                                          \
    m.def(#name, [](nb::handle_t<ArrayBase> h0) {                              \
        return nb::steal(apply<Normal>(                                        \
            op, #name, std::make_index_sequence<1>(), h0.ptr()));              \
    }, nb::raw_doc(doc_##name));                                               \
    m.def(#name, [](uint32_t v0) { return dr::name(v0); });

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

#define Py_nb_absolute_base Py_nb_absolute
#define Py_nb_multiply_base Py_nb_multiply
#define Py_nb_true_divide_base Py_nb_true_divide

DR_NB_UNOP(negative, ArrayOp::Neg)
DR_NB_UNOP(absolute_base, ArrayOp::Abs)
DR_NB_UNOP(invert, ArrayOp::Invert)
DR_NB_BINOP(add, ArrayOp::Add)
DR_NB_BINOP(subtract, ArrayOp::Sub)
DR_NB_BINOP(multiply_base, ArrayOp::Mul)
DR_NB_BINOP(true_divide_base, ArrayOp::TrueDiv)
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
                nb::raise("Negative exponent is too large!");

            Py_ssize_t u1 = (i1 < 0) ? -i1 : i1;

            size_t size = width(h0);
            nb::object result = array_module.attr("ones")(h0.type(), size),
                       x = nb::borrow(h0);

            while (u1) {
                if (u1 & 1)
                    result *= x;
                x = x * x;
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
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.power(<%U>, <%U>): failed (see above)!",
                        nb::inst_name(h0).ptr(), nb::inst_name(h1).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "drjit.power(<%U>, <%U>): %s",
                        nb::inst_name(h0).ptr(), nb::inst_name(h1).ptr(), e.what());
    }

    return nullptr;
}

static PyObject *nb_inplace_power(PyObject *h0, PyObject *h1) noexcept {
    PyObject *r = nb_power(h0, h1);
    if (!r || r == Py_NotImplemented)
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

static nb::object matmul(nb::handle h0, nb::handle h1);

static PyObject *nb_multiply(PyObject *h0, PyObject *h1) noexcept {
    if (!is_matrix_v(h0) && !is_matrix_v(h1)) {
        return nb_multiply_base(h0, h1);
    } else {
        try {
            return matmul(h0, h1).release().ptr();
        } catch (...) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }
}

static PyObject *nb_inplace_multiply(PyObject *h0, PyObject *h1) noexcept {
    if (!is_matrix_v(h0) && !is_matrix_v(h1)) {
        return nb_inplace_multiply_base(h0, h1);
    } else {
        PyObject *r;

        try {
            r = matmul(h0, h1).release().ptr();
        } catch (...) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }

        if (Py_TYPE(r) == Py_TYPE(h0) && h0 != r) {
            nb::inst_replace_move(h0, r);
            Py_INCREF(h0);
            Py_DECREF(r);
            return h0;
        } else {
            return r;
        }
    }
}

static PyObject *nb_absolute(PyObject *h0) noexcept {
    nb::object o = nb::steal(nb_absolute_base(h0));
    if (o.is_valid()) {
        const ArraySupplement &s = supp(o.type());

        if (s.is_complex)
            o = o[0];
        else if (s.is_quaternion)
            o = o[3];
    }
    return o.release().ptr();
}


static nb::object rcp(nb::handle_t<ArrayBase> h0);

static bool needs_special_division(nb::handle h0, nb::handle h1) {
    return is_special_v(h0) || is_special_v(h1) ||
           (is_drjit_array(h0) && is_drjit_array(h1) &&
            supp(h0.type()).ndim > supp(h1.type()).ndim);
}

static PyObject *nb_true_divide(PyObject *h0, PyObject *h1) noexcept {
    if (!needs_special_division(h0, h1)) {
        return nb_true_divide_base(h0, h1);
    } else {
        try {
            return (nb::handle(h0) * array_module.attr("rcp")(nb::handle(h1))).release().ptr();
        } catch (...) {
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }
}

static PyObject *nb_inplace_true_divide(PyObject *h0, PyObject *h1) noexcept {
    if (!needs_special_division(h0, h1)) {
        return nb_inplace_true_divide_base(h0, h1);
    } else {
        PyObject *r = nb_true_divide(h0, h1);
        if (!r || r == Py_NotImplemented)
            return r;

        if (Py_TYPE(r) == Py_TYPE(h0) && h0 != r) {
            nb::inst_replace_move(h0, r);
            Py_INCREF(h0);
            Py_DECREF(r);
            return h0;
        } else {
            return r;
        }
    }
}

static PyObject *nb_matrix_multiply(PyObject *h0, PyObject *h1) noexcept {
    try {
        return matmul(h0, h1).release().ptr();
    } catch (...) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}

static PyObject *nb_inplace_matrix_multiply(PyObject *h0, PyObject *h1) noexcept {
    try {
        PyObject *r = matmul(h0, h1).release().ptr();

        if (Py_TYPE(r) == Py_TYPE(h0) && h0 != r) {
            nb::inst_replace_move(h0, r);
            Py_INCREF(h0);
            Py_DECREF(r);
            return h0;
        } else {
            return r;
        }
    } catch (...) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
}

static PyObject *tp_richcompare(PyObject *h0, PyObject *h1, int slot) noexcept {
    return apply<RichCompare>(ArrayOp::Richcmp, slot,
                              std::make_index_sequence<2>(), h0, h1);
}

static PyObject *tp_hash(PyObject *h) noexcept {
    return PyLong_FromSize_t((size_t) (((uintptr_t) h) / sizeof(void *)));
}

template <int Index> nb::object xyzw_getter(nb::handle_t<ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY((!s.is_vector && !s.is_quaternion) || s.ndim == 0 ||
                    s.shape[0] == DRJIT_DYNAMIC || Index >= s.shape[0])) {
        nb::str name = nb::inst_name(h);
        nb::raise("%s does not have a '%c' component.", name.c_str(),
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
        nb::raise("%s does not have a '%c' component.", name.c_str(),
                  "xyzw"[Index]);
    }

    if (s.set_item(h.ptr(), (Py_ssize_t) Index, value.ptr()))
        nb::raise_python_error();
}

template <int Index> nb::object complex_getter(nb::handle_t<ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        if constexpr (Index == 0) {
            return nb::borrow(h);
        } else {
            return array_module.attr("zeros")(h.type(), array_module.attr("shape")(h));
        }
    }

    return nb::steal(s.item(h.ptr(), (Py_ssize_t) Index));
}

template <int Index>
void complex_setter(nb::handle_t<ArrayBase> h, nb::handle value) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_complex)) {
        nb::raise_type_error(
            "type '%s' is not complex-valued, cannot assign to '%s'.",
            nb::inst_name(h).c_str(), Index == 0 ? "real" : "imag");
    }

    if (s.set_item(h.ptr(), (Py_ssize_t) Index, value.ptr()))
        nb::raise_python_error();
}

static nb::object transpose_getter(nb::handle_t<ArrayBase> h) {
    const ArraySupplement &s = supp(h.type());

    if (NB_UNLIKELY(!s.is_matrix))
        nb::raise_type_error("'%s' is not a matrix type.", nb::inst_name(h).c_str());

    nb::object result = nb::inst_alloc_zero(h.type());
    for (size_t i = 0; i < s.shape[0]; ++i)
        for (size_t j = 0; j < s.shape[1]; ++j)
            result[i][j] = h[j][i];
    return result;
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
    DR_ARRAY_SLOT(nb_matrix_multiply),
    DR_ARRAY_SLOT(nb_inplace_matrix_multiply),

    /// Binary bit/mask operations
    DR_ARRAY_SLOT(nb_and),
    DR_ARRAY_SLOT(nb_inplace_and),
    DR_ARRAY_SLOT(nb_or),
    DR_ARRAY_SLOT(nb_inplace_or),
    DR_ARRAY_SLOT(nb_xor),
    DR_ARRAY_SLOT(nb_inplace_xor),

    /// Miscellaneous
    DR_ARRAY_SLOT(tp_iter),
    DR_ARRAY_SLOT(tp_hash),
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
        nb::raise_python_error();

    return nb::steal(o);
}

static nb::object rcp(nb::handle_t<ArrayBase> h0) {
    return nb::steal(apply<Normal>(
        ArrayOp::Rcp, "rcp", std::make_index_sequence<1>(), h0.ptr()));
}

nb::object matmul(nb::handle h0, nb::handle h1) {
    const nb::handle tp0 = h0.type(),
                     tp1 = h1.type();
    bool d0 = is_drjit_type(tp0), d1 = is_drjit_type(tp1);

    bool type_error = false;
    try {
        auto outer_static = [](const ArrayMeta &m) {
            return m.ndim > 0 && m.shape[0] != DRJIT_DYNAMIC;
        };

        if (d0 && d1) {
            const ArraySupplement &s0 = supp(tp0), &s1 = supp(tp1);

            if (s0.is_tensor || s1.is_tensor)
                nb::raise(
                    "this operation only handles fixed-sizes arrays. A different "
                    "approach is needed for multiplications involving potentially "
                    "large dynamic arrays/tensors. Other other tools like PyTorch, "
                    "JAX, or Tensorflow will be preferable in such situations "
                    "(e.g., to train neural networks");

            if (s0.is_complex || s1.is_complex || s0.is_quaternion || s1.is_quaternion)
                nb::raise("complex/quaternion-valued inputs not supported.");

            // Resolve overloaded binding instead of calling fma() defined here
            nb::object fma = array_module.attr("fma");

            size_t n = s0.shape[0];

            auto is_dyn_1d = [](const ArrayMeta &m) {
                return m.ndim == 1 && m.shape[0] == DRJIT_DYNAMIC;
            };

            auto is_vector = [n](const ArrayMeta &m) {
                return (m.ndim == 1 || (m.ndim == 2 && m.shape[1] == DRJIT_DYNAMIC)) &&
                       m.shape[0] == n;
            };

            auto is_matrix = [n](const ArrayMeta &m) {
                return (m.ndim == 2 || (m.ndim == 3 && m.shape[2] == DRJIT_DYNAMIC)) &&
                       m.shape[0] == n && m.shape[1] == n;
            };

            // Matrix-matrix product
            if (is_matrix(s0) && is_matrix(s1)) {
                nb::object result = expr_t(tp0, tp1)();

                for (size_t i = 0; i < n; ++i) {
                    nb::object h0i = h0[i], ri = result[i];
                    for (size_t j = 0; j < n; ++j) {
                        nb::object v = h0i[0] * h1[0][j];
                        for (size_t k = 1; k < n; ++k)
                            v = fma(h0i[k], h1[k][j], v);
                        ri[j] = v;
                    }
                }
                return result;
            }

            // Matrix-vector product
            if (is_matrix(s0) && is_vector(s1)) {
                nb::object result = expr_t(value_t(tp0), tp1)();
                for (size_t i = 0; i < n; ++i) {
                    nb::object h0i = h0[i],
                               v   = h0i[0] * h1[0];
                    for (size_t k = 1; k < n; ++k)
                        v = fma(h0i[k], h1[k], v);
                    result[i] = v;
                }
                return result;
            }

            // Vector-matrix product
            if (is_vector(s0) && is_matrix(s1)) {
                nb::object result = expr_t(tp0, value_t(tp1))();
                for (size_t i = 0; i < n; ++i) {
                    nb::object v = h0[0] * h1[0][i];
                    for (size_t k = 1; k < n; ++k)
                        v = fma(h0[k], h1[k][i], v);
                    result[i] = v;
                }
                return result;
            }

            // Inner product, case 1
            if (is_vector(s0) && is_vector(s1)) {
                nb::object result = h0[0] * h1[0];
                for (size_t k = 1; k < n; ++k)
                    result = fma(h0[k], h1[k], result);
                return result;
            }

            // Inner product, case 2
            if (is_dyn_1d(s0) && is_dyn_1d(s1) && nb::len(h0) == nb::len(h1))
                return sum(h0 * h1, 0);

            // Scalar product
            if (is_dyn_1d(s0) && outer_static(s1)) {
                nb::object result = expr_t(tp0, tp1)();
                size_t n2 = nb::len(h1);
                for (size_t i = 0; i < n2; ++i)
                    result[i] = h0 * h1[i];
                return result;
            } else if (is_dyn_1d(s1) && outer_static(s0)) {
                nb::object result = expr_t(tp0, tp1)();
                size_t n2 = nb::len(h0);
                for (size_t i = 0; i < n2; ++i)
                    result[i] = h0[i] * h1;
                return result;
            }
        }

        if (d0 && (tp1.is(&PyLong_Type) || tp1.is(&PyFloat_Type))) {
            const ArraySupplement &s0 = supp(tp0);

            if (outer_static(s0)) {
                nb::object result = expr_t(tp0, tp1)();
                for (size_t i = 0; i < s0.shape[0]; ++i)
                    result[i] = h0[i] * h1;
                return result;
            } else {
                return nb::steal(nb_multiply_base(h0.ptr(), h1.ptr()));
            }
        } else if (d1 && (tp0.is(&PyLong_Type) || tp0.is(&PyFloat_Type))) {
            const ArraySupplement &s1 = supp(tp1);

            if (outer_static(s1)) {
                nb::object result = expr_t(tp0, tp1)();
                for (size_t i = 0; i < s1.shape[0]; ++i)
                    result[i] = h0 * h1[i];
                return result;
            } else {
                return nb::steal(nb_multiply_base(h0.ptr(), h1.ptr()));
            }
        }

        type_error = true;
        nb::raise("unsupported input types.");
    } catch (nb::python_error &e) {
        nb::raise_from(e, PyExc_RuntimeError,
                       "drjit.matmul(<%U>, <%U>): failed (see above)!",
                       nb::inst_name(h0).ptr(), nb::inst_name(h1).ptr());
    } catch (const std::exception &e) {
        nb::chain_error(type_error ? PyExc_TypeError : PyExc_RuntimeError,
                        "drjit.matmul(<%U>, <%U>): %s", nb::inst_name(h0).ptr(),
                        nb::inst_name(h1).ptr(), e.what());
        nb::raise_python_error();
    }
}

nb::object select(nb::handle h0, nb::handle h1, nb::handle h2) {
    if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1) &&
                    !is_drjit_array(h2)))
        throw nb::next_overload();

    PyObject *o =
        apply<Select>(ArrayOp::Select, "select", std::make_index_sequence<3>(),
                      h0.ptr(), h1.ptr(), h2.ptr());

    if (!o)
        nb::raise_python_error();

    return nb::steal(o);
}

nb::object reinterpret_array(nb::type_object_t<dr::ArrayBase> t, nb::handle_t<dr::ArrayBase> h) {
    struct Reinterpret : TransformCallback {
        VarType source_type;
        VarType target_type;

        Reinterpret(VarType source_type, VarType target_type)
            : source_type(source_type), target_type(target_type) { }

        nb::handle transform_type(nb::handle tp) const override {
            ArrayMeta m = supp(tp);
            m.type = (uint8_t) target_type;
            return meta_get_type(m);
        }

        virtual void operator()(nb::handle h1, nb::handle h2) override {
            supp(h2.type()).cast(
                inst_ptr(h1), source_type, 1, inst_ptr(h2)
            );
        }
    };

    ArrayMeta mt = supp(t), ms = supp(h.type());
    if (mt == ms)
        return nb::borrow(h);

    VarType source_type = (VarType) ms.type,
            target_type = (VarType) mt.type;

    ms.type = mt.type;
    if (ms != mt)
        nb::raise("drjit.reinterpret_array(): input and target type are incompatible.");

    Reinterpret r(source_type, target_type);
    return transform("drjit.reinterpret_array", r, h);
}

static VarState get_state(nb::handle h_) {
    struct GetState : TraverseCallback {
        VarState state = VarState::Invalid;
        size_t count = 0;

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (!s.index)
                return;

            VarState vs = jit_var_state((uint32_t) s.index(inst_ptr(h)));
            if (count++ == 0)
                state = vs;
            if (state != vs)
                state = VarState::Mixed;
        }
    };

    GetState gs;
    traverse("drjit.ArrayBase.state", gs, h_);
    return gs.state;
}

void export_base(nb::module_ &m) {
    nb::class_<ArrayBase> ab(m, "ArrayBase",
                                 nb::type_slots(array_base_slots),
                                 nb::supplement<ArraySupplement>());

    ab.def_prop_ro_static("__meta__", [](nb::type_object_t<ArrayBase> tp) {
        return meta_str(supp(tp));
    });

    ab.def_prop_ro("state", &get_state, nb::raw_doc(doc_ArrayBase_state));
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

    ab.def_prop_ro("T", transpose_getter, nb::raw_doc(doc_ArrayBase_T));

    ab.def_prop_rw(
        "grad", [](nb::handle_t<ArrayBase> h) { return ::grad(h); },
        [](nb::handle_t<ArrayBase> h, nb::handle h2) { ::set_grad(h, h2); },
        nb::raw_doc(doc_ArrayBase_grad));

    ab.def_prop_rw("label",
        [](nb::handle_t<dr::ArrayBase> h) -> nb::object {
            const ArraySupplement &s = supp(h.type());
            if ((JitBackend) s.backend != JitBackend::None) {
                const char *str = jit_var_label((uint32_t) s.index(inst_ptr(h)));
                if (str)
                    return nb::str(str);
            }
            return nb::none();
        },
        &set_label,
        nb::raw_doc(doc_ArrayBase_label)
    );

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

    m.def("abs", [](Py_ssize_t v) { return dr::abs(v); })
     .def("abs", [](double v) { return dr::abs(v); })
     .def("abs", [](nb::handle_t<ArrayBase> h0) {
                     return nb::steal(nb_absolute(h0.ptr()));
                 }, nb::raw_doc(doc_abs)
     );

    DR_MATH_UNOP(sqrt, ArrayOp::Sqrt);

    m.def("rcp", [](double v0) { return dr::rcp(v0); })
     .def("rcp", ::rcp, nb::raw_doc(doc_rcp));

    DR_MATH_UNOP(rsqrt, ArrayOp::Rsqrt);
    DR_MATH_UNOP(cbrt, ArrayOp::Cbrt);

    DR_MATH_UNOP(round, ArrayOp::Round);
    DR_MATH_UNOP(trunc, ArrayOp::Trunc);
    DR_MATH_UNOP(ceil, ArrayOp::Ceil);
    DR_MATH_UNOP(floor, ArrayOp::Floor);

    DR_MATH_UNOP_UINT32(popcnt, ArrayOp::Popcnt);
    DR_MATH_UNOP_UINT32(lzcnt, ArrayOp::Lzcnt);
    DR_MATH_UNOP_UINT32(tzcnt, ArrayOp::Tzcnt);

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

    m.def("square", [](nb::handle h) { return h*h; }, doc_square);
    m.def("matmul", &matmul, doc_matmul);

    m.def("minimum",
          [](Py_ssize_t a, Py_ssize_t b) { return dr::minimum(a, b); });
    DR_MATH_BINOP(minimum, ArrayOp::Minimum);

    m.def("maximum",
          [](Py_ssize_t a, Py_ssize_t b) { return dr::maximum(a, b); });
    DR_MATH_BINOP(maximum, ArrayOp::Maximum);

    DR_MATH_BINOP(atan2, ArrayOp::Atan2);

    m.def("fma",
          [](Py_ssize_t a, Py_ssize_t b, Py_ssize_t c) {
              return dr::fma(a, b, c);
          }, nb::raw_doc(doc_fma))
     .def("fma",
          [](double a, double b, double c) {
              return dr::fma(a, b, c);
          })
     .def("fma", (nb::object (*)(nb::handle, nb::handle, nb::handle)) fma);

    m.def("select",
          nb::overload_cast<nb::handle, nb::handle, nb::handle>(&select),
          nb::raw_doc(doc_select));

    m.def("select",
          [](bool mask, nb::handle a, nb::handle b) {
              return nb::borrow(mask ? a : b);
          });

    m.def("power",
          [](Py_ssize_t arg0, Py_ssize_t arg1) { return std::pow(arg0, arg1); },
          doc_pow);

    m.def("power", [](double arg0, double arg1) { return std::pow(arg0, arg1); });

    m.def("power",
          [](nb::handle h0, nb::handle h1) {
              if (NB_UNLIKELY(!is_drjit_array(h0) && !is_drjit_array(h1)))
                  throw nb::next_overload();
              return nb::steal(nb_power(h0.ptr(), h1.ptr()));
          }
    );

    m.def("reinterpret_array", &reinterpret_array, doc_reinterpret_array);

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
