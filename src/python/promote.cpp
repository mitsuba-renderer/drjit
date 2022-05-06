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
    r.ndim = a.ndim > b.ndim ? a.ndim : b.ndim ;

    memset(r.shape, 0, sizeof(r.shape));

    if (r.is_tensor || !r.is_valid) {
        r.ndim = 0;
    } else {
		int ndim_a = 0, ndim_b = 0;
        for (int i = 0; i < r.ndim; ++i) {
            int value_a = 1, value_b = 1;

            if (ndim_a < a.ndim)
                value_a = a.shape[ndim_a++];
            if (ndim_b < b.ndim)
                value_b = b.shape[ndim_b++];

            if (value_a == value_b)
                r.shape[i] = value_a;
            else if (value_a == 1 || value_b == 1)
                r.shape[i] = value_a > value_b ? value_a : value_b;
            else
                r.is_valid = 0;
        }
    }

    return r;
}

static meta meta_from_builtin(PyObject *o) {
    meta m { };
    m.is_valid = true;

    if (PyNumber_Check(o)) {
        if (PyBool_Check(o)) {
            m.type = (uint8_t) VarType::Bool;
        } else if (PyLong_Check(o)) {
            long long result = PyLong_AsLongLong(o);

            if (result == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                m.is_valid = false;
            } else {
                bool is_i32 = (result >= (long long) INT32_MIN &&
                               result <= (long long) INT32_MAX);
                m.type = (uint16_t) (is_i32 ? VarType::Int32 : VarType::Int64);
            }
        } else if (PyFloat_Check(o)) {
            m.type = (uint8_t) VarType::Float32;
        } else {
            m.is_valid = false;
        }
    } else if (PyTuple_Check(o) || PyList_Check(o)) {
        Py_ssize_t len = PySequence_Size(o);

        if (len < 0) {
            PyErr_Clear();
            m.is_valid = false;
        } else {
            for (Py_ssize_t i = 0; i < len; ++i) {
                PyObject *o2 = PySequence_GetItem(o, i);
                if (!o2) {
                    PyErr_Clear();
                    m.is_valid = false;
                    break;
                }

                meta m2 = meta_from_builtin(o2);
                Py_DECREF(o2);

                if (m2.ndim >= 3) {
                    m.is_valid = false;
                    break;
                }

                for (int j = 0; j < m2.ndim; ++j)
                    m2.shape[j + 1] = m2.shape[j];
                m2.shape[0] = len > 4 ? 0xFF : (uint8_t) len;
                m2.ndim++;

                m = meta_promote(m, m2);
            }
        }
    } else {
        m.is_valid = false;
    }

    return m;
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
    PyTypeObject * base = (PyTypeObject *) array_base.ptr();
    meta m;
    memset(&m, 0, sizeof(meta));
    m.is_valid = 1;

    nb::handle h;
    for (size_t i = 0; i < n; ++i) {
        PyTypeObject *tp = Py_TYPE(o[i]);
        bool is_drjit_array = PyType_IsSubtype(tp, base);

        meta m2;
        if (is_drjit_array)
            m2 = nb::type_supplement<supp>(tp).meta;
        else
            m2 = meta_from_builtin(o[i]);

        if (!m2.is_valid) {
            PyErr_Format(PyExc_TypeError,
                         "%s.%s(): encountered an unsupported argument of type "
                         "'%s' (must be a Dr.Jit array or a type that can be "
                         "converted into one)",
                         Py_TYPE(o[0])->tp_name, op, tp->tp_name);
            return false;
        }

        m = meta_promote(m, m2);

        if (m == m2) {
            if (is_drjit_array)
                h = tp;
        } else {
            h = nb::handle();
        }
    }

    if (!meta_check(m)) {
        PyErr_Format(PyExc_RuntimeError, "%s.%s(): incompatible arguments!",
                     Py_TYPE(o[0])->tp_name, op);
        return false;
    }

    if (!h.is_valid())
        h = drjit::detail::array_get(m);

    for (size_t i = 0; i < n; ++i) {
        nb::handle h2 = h;

        if (select && i == 0) {
            m.type = (uint16_t) VarType::Bool;
            h2 = drjit::detail::array_get(m);
        }

        if (Py_TYPE(o[i]) == (PyTypeObject *) h2.ptr()) {
            Py_INCREF(o[i]);
        } else {
            PyObject *args[2];
            args[0] = nullptr;
            args[1] = o[i];

            PyObject *res =
                NB_VECTORCALL(h2.ptr(), args + 1,
                              PY_VECTORCALL_ARGUMENTS_OFFSET | 1, nullptr);

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
