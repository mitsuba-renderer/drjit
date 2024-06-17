/*
    meta.cpp -- Logic related to Dr.Jit array metadata and type promotion

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "meta.h"
#include "base.h"
#include "../ext/nanobind/src/buffer.h"
#include <nanobind/ndarray.h>
#include <string>

/// Check if the given metadata record is valid
bool meta_check(ArrayMeta m) noexcept {
    return m.is_valid && m.type > (uint8_t) VarType::Void &&
           m.type < (uint8_t) VarType::Count &&
           m.is_vector + m.is_complex + m.is_quaternion + m.is_matrix +
                   m.is_tensor + m.is_class <= 1;
}

static const char *type_name_lowercase[] = {
    "void", "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "pointer", "float162", "float32", "float64"
};

static const char *type_name[] = {
    "Void", "Bool", "Int8", "UInt8", "Int16", "UInt16", "Int", "UInt",
    "Int64", "UInt64", "Pointer", "Float16", "Float", "Float64"
};

static const char *type_suffix[] = {
    "?", "b", "i8", "u8", "i16", "u16", "i", "u",
    "i64", "u64", "p", "f16", "f", "f64"
};

/// Convert a metadata record into a string representation (for debugging)
dr::string meta_str(ArrayMeta m) {
    dr::string result = "ArrayMeta[";

    if (m.is_valid) {
        result += "\n  type=";
        result += type_name_lowercase[m.type];
        result += ",\n";

        if (m.backend == (uint16_t) JitBackend::None)
            ;
        else if (m.backend == (uint16_t) JitBackend::CUDA)
            result += "  backend=cuda,\n";
        else if (m.backend == (uint16_t) JitBackend::LLVM)
            result += "  backend=llvm,\n";
        else
            result += "  backend=invalid,\n";

        if (m.backend != (uint16_t) JitBackend::None)
            result += "  is_jit=1,\n";
        if (m.is_vector)
            result += "  is_vector=1,\n";
        if (m.is_complex)
            result += "  is_complex=1,\n";
        if (m.is_quaternion)
            result += "  is_quaternion=1,\n";
        if (m.is_matrix)
            result += "  is_matrix=1,\n";
        if (m.is_tensor)
            result += "  is_tensor=1,\n";
        if (m.is_sequence)
            result += "  is_sequence=1,\n";
        if (m.is_diff)
            result += "  is_diff=1,\n";
        if (m.is_class)
            result += "  is_class=1,\n";

        result += "  shape=(";
        for (int i = 0; i < 4; ++i) {
            if (m.shape[i] == 0)
                break;
            if (i > 0)
                result += ", ";
            if (m.shape[i] == DRJIT_DYNAMIC)
                result += "*";
            else
                result += dr::string(m.shape[i]);
        }
        result += ")\n";
    } else {
        result += "invalid";
    }
    result += "]";
    return result;
}

/// Compute the metadata type of an operation combinining 'a' and 'b'
ArrayMeta meta_promote(ArrayMeta a, ArrayMeta b) noexcept {
    ArrayMeta r{};
    r.backend = a.backend > b.backend ? a.backend : b.backend;
    r.type = a.type > b.type ? a.type : b.type;
    r.is_vector = a.is_vector | b.is_vector;
    r.is_complex = a.is_complex | b.is_complex;
    r.is_quaternion = a.is_quaternion | b.is_quaternion;
    r.is_matrix = a.is_matrix | b.is_matrix;
    r.is_tensor = a.is_tensor | b.is_tensor;
    r.is_diff = a.is_diff | b.is_diff;
    r.is_class = a.is_class | b.is_class;
    r.is_valid = a.is_valid && b.is_valid &&
        (a.backend == b.backend || a.backend == 0 || b.backend == 0);

    if (!r.is_tensor && r.is_valid) {
        bool a_is_jit = a.backend != 0,
             b_is_jit = b.backend != 0,
             r_is_jit = r.backend != 0;

        int ndim_a = a.ndim - (a_is_jit || (r_is_jit && a.is_sequence)),
            ndim_b = b.ndim - (b_is_jit || (r_is_jit && b.is_sequence)),
            ndim_max = ndim_a > ndim_b ? ndim_a : ndim_b;

        r.ndim = ndim_max;

        for (int i = 0; i < r.ndim; ++i) {
            int size_a = (i < ndim_a) ? a.shape[i] : 1,
                size_b = (i < ndim_b) ? b.shape[i] : 1;

            if (size_a == size_b)
                r.shape[i] = (uint8_t) size_a;
            else if (size_a == DRJIT_DYNAMIC || size_b == DRJIT_DYNAMIC)
                r.shape[i] = DRJIT_DYNAMIC;
            else if (size_a == 1 || size_b == 1)
                r.shape[i] = (uint8_t) (size_a > size_b ? size_a : size_b);
            else
                r.is_valid = 0;
        }

        if (r_is_jit)
            r.shape[r.ndim++] = DRJIT_DYNAMIC;
    }

    return r;
}

// Infer the metadata from the given Python object
ArrayMeta meta_get(nb::handle h) noexcept {
    nb::handle tp = h.type();

    ArrayMeta m { };
    m.is_valid = true;

    if (is_drjit_type(tp)) {
        m = supp(tp);
    } else if (tp.is(&PyBool_Type)) {
        m.type = (uint8_t) VarType::Bool;
    } else if (tp.is(&PyLong_Type)) {
        int overflow = 0;
        long long result = PyLong_AsLongLongAndOverflow(h.ptr(), &overflow);
        VarType vt;

        if (overflow) {
            vt = overflow > 0 ? VarType::UInt64 : VarType::Int64;
        } else if (result < 0) {
            vt = result < INT_MIN ? VarType::Int64 : VarType::Int32;

            if (result == -1 && PyErr_Occurred()) {
                m.is_valid = false;
                PyErr_Clear();
            }
        } else {
            vt = result > INT_MAX ? VarType::Int64 : VarType::Int32;
        }
        m.type = (uint8_t) vt;
    } else if (tp.is(&PyFloat_Type)) {
        m.type = (uint8_t) VarType::Float32;
    } else if (tp.is(&PyTuple_Type) || tp.is(&PyList_Type)) {
        Py_ssize_t len = PySequence_Size(h.ptr());

        if (len < 0) {
            PyErr_Clear();
            m.is_valid = false;
        } else {
            for (Py_ssize_t i = 0; i < len; ++i) {
                PyObject *o2 = PySequence_GetItem(h.ptr(), i);
                if (!o2) {
                    PyErr_Clear();
                    m.is_valid = false;
                    break;
                }

                ArrayMeta m2 = meta_get(o2);
                Py_DECREF(o2);

                if (m2.ndim >= 3 || !m2.is_valid) {
                    m.is_valid = false;
                    break;
                }

                for (int j = 0; j < m2.ndim; ++j)
                    m2.shape[j + 1] = m2.shape[j];
                m2.shape[0] = len > 4 ? DRJIT_DYNAMIC : (uint8_t) len;
                m2.ndim++;

                if (m != m2)
                    m = meta_promote(m, m2);
            }

            m.is_sequence = true;
        }
    } else if (nb::ndarray_check(h)) {
        try {
            using nb::dlpack::dtype;
            using nb::dlpack::dtype_code;

            nb::ndarray<nb::ro> array = nb::cast<nb::ndarray<nb::ro>>(h);
            dtype dt = array.dtype();
            VarType vt = VarType::Void;
            dtype_code code = (dtype_code)dt.code;

            switch (code) {
                case dtype_code::Bool:
                    vt = VarType::Bool;
                    break;

                case dtype_code::Int:
                    switch (dt.bits) {
                        case 8  : vt = VarType::Int8; break;
                        case 16 : vt = VarType::Int16; break;
                        case 32 : vt = VarType::Int32; break;
                        case 64 : vt = VarType::Int64; break;
                    }
                    break;

                case dtype_code::UInt:
                    switch (dt.bits) {
                        case 8  : vt = VarType::UInt8; break;
                        case 16 : vt = VarType::UInt16; break;
                        case 32 : vt = VarType::UInt32; break;
                        case 64 : vt = VarType::UInt64; break;
                    }
                    break;

                case dtype_code::Float:
                    switch (dt.bits) {
                        case 16 : vt = VarType::Float16; break;
                        case 32 : vt = VarType::Float32; break;
                        case 64 : vt = VarType::Float64; break;
                    }
                    break;

                default:
                    break;
            }

            size_t ndim = array.ndim();
            if (ndim >= 1 && ndim <= 4 &&
                (vt != VarType::Void || code == dtype_code::Complex)) {
                for (size_t i = 0; i < ndim; ++i) {
                    size_t value = array.shape(i);
                    m.shape[i] = (uint8_t) (value > 4 ? DRJIT_DYNAMIC : value);
                }

                m.is_sequence = true;
                m.ndim = ndim;
                m.type = (uint16_t) vt;
                m.is_valid = dt.lanes == 1;
            }

            return m;
        } catch (const std::exception &e) {
            PyErr_WarnFormat(
                PyExc_Warning, 1,
                "meta_get(): could not analyze array type: %s",
                e.what());
        }
    } else if (h.is_none() || nb::type_check(tp)) {
        m.type = (uint8_t) VarType::UInt32;
        m.is_class = true;
    } else {
        m.is_valid = false;
    }

    return m;
}

ArrayMeta meta_get_general(nb::handle h) noexcept {
    ArrayMeta m { };
    m.is_valid = true;

    if (!h.is_type())
        m = meta_get(h);
    else if (is_drjit_type(h))
        m = supp(h);
    else if (h.is(&PyBool_Type))
        m.type = (uint8_t) VarType::Bool;
    else if (h.is(&PyLong_Type))
        m.type = (uint8_t) VarType::Int32;
    else if (h.is(&PyFloat_Type))
        m.type = (uint8_t) VarType::Float32;
    else
        m.is_valid = false;

    return m;
}


/**
 * \brief Given a list of Dr.Jit arrays and scalars, determine the flavor and
 * shape of the result array and broadcast/convert everything into this form.
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
void promote(nb::object *o, size_t n, bool select) {
    ArrayMeta m, mi[3];

    if (n > 3)
        nb::raise("promote(): too many arguments!");

    nb::handle h;
    for (size_t i = 0; i < n; ++i) {
        nb::handle o_i = o[i];
        ArrayMeta m2 = meta_get(o_i);

        if (!m2.is_valid) {
            nb::raise(
                "Encountered an unsupported argument of type '%s' (must be a "
                "Dr.Jit array or a type that can be converted into one)",
                nb::inst_name(o_i).c_str());
        }

        if (i == 0)
            m = m2;
        else
            m = meta_promote(m, m2);

        mi[i] = m2;
    }

    if (!meta_check(m))
        nb::raise("Incompatible arguments.");

    for (size_t i = 0; i < n; ++i) {
        // if this is a compatible Dr.Jit array
        if (m == mi[i] && mi[i].talign)
            h = o[i];
    }

    if (h.is_valid()) {
        h = h.type();
    } else {
        if (!m.is_class) {
            h = meta_get_type(m);
        } else {
            for (size_t i = 0; i < n; ++i) {
                ArrayMeta m2 = meta_get(o[i]);
                if (m2.is_class && m2.ndim == 1) {
                    h = o[i].type();
                    break;
                }
            }

            if (!h.is_valid())
                nb::raise("Incompatible arguments.");
        }
    }

    for (size_t i = 0; i < n; ++i) {
        nb::handle h2 = h;

        if (select && i == 0) {
            m.type = (uint16_t) VarType::Bool;
            m.is_quaternion = 0;
            m.is_matrix = 0;
            h2 = meta_get_type(m);
        }

        if (!o[i].type().is(h2)) {
            PyObject *args[2];
            args[0] = nullptr;
            args[1] = o[i].ptr();

            PyObject *res =
                NB_VECTORCALL(h2.ptr(), args + 1,
                              PY_VECTORCALL_ARGUMENTS_OFFSET | 1, nullptr);

            if (NB_UNLIKELY(!res)) {
                nb::str type_name_i = nb::type_name(o[i].type()),
                        type_name_o = nb::type_name(h2);

                nb::raise("Could not promote type '%s' to '%s'.",
                          type_name_i.c_str(), type_name_o.c_str());
            }

            o[i] = nb::steal(res);
        }
    }
}

/// Look up the nanobind module associated with the given array metadata
nb::handle meta_get_module(ArrayMeta meta) noexcept {
    int index = 0;
    switch ((JitBackend) meta.backend) {
        case JitBackend::CUDA: index = 1; break;
        case JitBackend::LLVM: index = 3; break;
        default: break;
    }
    if ((JitBackend) meta.backend != JitBackend::None)
        index += (int) meta.is_diff;
    return array_submodules[index];
}

static nb::detail::Buffer buffer;

/// Determine the nanobind type name associated with the given array metadata
const char *meta_get_name(ArrayMeta meta) noexcept {
    buffer.clear();

    if (!meta.is_tensor) {
        int ndim = meta.ndim;

        if (meta.backend != (uint16_t) JitBackend::None)
            ndim--;

        const char *suffix = nullptr;

        if (ndim == 0) {
            buffer.put_dstr(type_name[meta.type]);
        } else {
            const char *prefix = "Array";
            if (meta.is_complex) {
                prefix = "Complex";
            } else if (meta.is_quaternion) {
                prefix = "Quaternion";
            } else if (meta.is_matrix) {
                prefix = "Matrix";
            }
            buffer.put_dstr(prefix);
            suffix = type_suffix[meta.type];
        }

        for (int i = 0; i < ndim; ++i) {
            if (meta.is_matrix && i == 1)
                continue;

            if (meta.shape[i] == DRJIT_DYNAMIC)
                buffer.put('X');
            else
                buffer.put_uint32(meta.shape[i]);
        }

        if (suffix)
            buffer.put_dstr(suffix);
    } else {
        buffer.put_dstr("TensorX");
        buffer.put_dstr(type_suffix[meta.type]);
    }

    return buffer.get();
}

/// Look up the nanobind type associated with the given array metadata
nb::handle meta_get_type(ArrayMeta meta, bool fail_if_missing) {
    const char *name = meta_get_name(meta);
    nb::handle result = getattr(meta_get_module(meta), name, nb::handle());
    if (!result.is_valid() && fail_if_missing)
        nb::raise("Operation references type \"%s\", which lacks bindings", name);
    return result;
}
