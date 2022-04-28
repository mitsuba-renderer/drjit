/*
    promote.cpp -- promote a sequence of Dr.Jit arrays to a compatible type

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"

/// Check if the given metadata record is valid
static bool meta_check(meta m) {
    return m.is_valid &&
           (m.type > (uint8_t) VarType::Void &&
            m.type < (uint8_t) VarType::Count) &&
           (m.is_vector + m.is_complex + m.is_quaternion + m.is_matrix +
                m.is_tensor <= 1) &&
           (m.is_cuda + m.is_llvm <= 1);
}

/// Compute the metadata type of an operation combinining 'a' and 'b'
static meta meta_promote(meta a, meta b) {
    int ndim_a = a.ndim, ndim_b = b.ndim;

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
    r.tsize_rel = r.talign = 0;
    r.ndim = ndim_a > ndim_b ? ndim_a : ndim_b;

    memset(r.shape, 0, sizeof(r.shape));

    for (int i = r.ndim; i >= 0; --i) {
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

    if (r.is_tensor) {
        r.ndim = 0;
        memset(r.shape, 0, sizeof(r.shape));
    }

    return r;
}

/**
 * \brief Given a list of Dr.Jit arrays and scalars, determine the flavor and
 * shape of the result array and broadcast/convert everything into this form.
 *
 * \param op
 *    Name of the operation for error messages
 *
 * \param o
 *    Array of input operands of size 'n'
 *
 * \param n
 *    Number of operands
 *
 * \param select
 *    Should be 'true' if this is a drjit.select() operation, in which case the
 *    first operand will be promoted to a mask array
 */
bool promote(const char *op, PyObject **o, size_t n, bool select) {
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
        } else if (tp == &PyFloat_Type) {
            type = (uint8_t) VarType::Float32;
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

    auto m_type = m.type;
    nb::handle h;

    if (!select)
        h = drjit::detail::array_get(m);

    PyObject *args[2];

    for (size_t i = 0; i < n; ++i) {
        if (select) {
            m.type = (i == 0) ? (uint16_t) VarType::Bool : m_type;
            h = drjit::detail::array_get(m);
        }

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
