#include "promote.h"

/// Is a given variable an Enoki variable?
bool var_is_enoki(py::handle h) {
    PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
    return strncmp(type->tp_name, "enoki.", 6) == 0;
}

/// Is a given variable an Enoki variable *or* an Enoki type object?
bool var_is_enoki_type(py::handle h) {
    PyTypeObject *type =
        (PyTypeObject *) (PyType_Check(h.ptr()) ? h.ptr() : h.get_type().ptr());
    return strncmp(type->tp_name, "enoki.", 6) == 0;
}

/// Is a given variable an Enoki variable? (esp. a JIT-related type)
bool var_is_enoki_jit(py::handle h) {
    PyTypeObject *type = (PyTypeObject *) h.get_type().ptr();
    return strncmp(type->tp_name, "enoki.llvm.", 11) == 0 ||
           strncmp(type->tp_name, "enoki.cuda.", 11) == 0;
}

/// Check whether a given Enoki type is dynamic
bool var_is_dynamic(py::handle h) {
    return py::cast<size_t>(h.attr("Size")) == ek::Dynamic;
}

/**
 * Return the VarType of a given Enoki object or plain Python type. Return
 * 'preferred' when there is sufficient room for interpretation (e.g. when
 * given an 'int').
 */
VarType var_type(py::handle h, VarType preferred) {
    PyTypeObject *type =
        (PyTypeObject *) (PyType_Check(h.ptr()) ? h.ptr() : h.get_type().ptr());

    if (strncmp(type->tp_name, "enoki.", 6) == 0) {
        return py::cast<VarType>(h.attr("Type"));
    } else if (type == &PyFloat_Type) {
        return VarType::Float32;
    } else if (type == &PyLong_Type) {
        ssize_t value = py::cast<ssize_t>(h);
        if (preferred != VarType::Invalid) {
            bool ok = false;
            switch (preferred) {
                case VarType::UInt8:
                    ok = value >= 0 && value <= UCHAR_MAX;
                    break;

                case VarType::Int8:
                    ok = value >= SCHAR_MIN && value <= SCHAR_MAX;
                    break;

                case VarType::UInt16:
                    ok = value >= 0 && value <= USHRT_MAX;
                    break;

                case VarType::Int16:
                    ok = value >= SHRT_MIN && value <= SHRT_MAX;
                    break;

                case VarType::Int32:
                    ok = value >= INT_MIN && value <= INT_MAX;
                    break;

                case VarType::UInt32:
                    ok = value >= 0 && value <= UINT_MAX;
                    break;

                case VarType::UInt64:
                    ok = value >= 0;
                    break;

                case VarType::Int64:
                    break;

                default:
                    break;
            }

            if (ok)
                return preferred;
        }
        if (value < 0)
            return (value < INT_MIN)  ? VarType::Int64 : VarType::Int32;
        else
            return (value > UINT_MAX) ? VarType::UInt64 : VarType::UInt32;
    } else {
        return VarType::Invalid;
    }
}

/// Perform promotion of sub-int size variables analogous to C
static VarType var_promote(VarType v) {
    switch (v) {
        case VarType::Int8:
        case VarType::UInt8:
        case VarType::Int16:
        case VarType::UInt16:
            return VarType::Int32;
        case VarType::Float16:
            return VarType::Float32;
        default:
            return v;
    }
}

/// Return the depth of the given Enoki variable (or 0)
size_t var_depth(const py::object &o) {
    if (!var_is_enoki(o))
        return 0;

    return py::cast<size_t>(o.attr("Depth"));
}

/// No-op, accepts all inputs
void test_any(const char *, VarType) { }

/// Throw if 'v' is not a floating point type
void test_float(const char *op, VarType v) {
    if (!jitc_is_floating_point(v))
        ek::enoki_raise("%s(): operation requires floating point operands!", op);
}

/// Throw if 'v' is not an integral type
void test_integral(const char *op, VarType v) {
    if (!jitc_is_integral(v))
        ek::enoki_raise("%s(): operation requires integral operands!", op);
}

/// Throw if 'v' is not an arithmetic type
void test_arithmetic(const char *op, VarType v) {
    if (!jitc_is_arithmetic(v))
        ek::enoki_raise("%s(): operation requires arithmetic operands!", op);
}

/// Throw if 'v' is not a mask type
void test_mask(const char *op, VarType v) {
    if (v != VarType::Bool)
        ek::enoki_raise("%s(): operation requires mask operands!", op);
}

/// Throw if 'v' is not a floating point type (with special error message for 'truediv')
void test_float_truediv(const char *, VarType v) {
    if (!jitc_is_floating_point(v))
        throw std::runtime_error("Use the floor division operator (// and //=) "
                                 "for Enoki integer arrays.");
}

/// Throw if 'v' is not an integral point type (with special error message for 'floordiv')
void test_integral_floordiv(const char *, VarType v) {
    if (!jitc_is_integral(v))
        throw std::runtime_error("Use the true division operator (/ and /=) "
                                 "for Enoki floating point arrays.");
}

/// Perform C-style type promotion for a binary operation
bool var_promote(const char *op, py::object &a0, py::object &a1,
                 bool require_enoki, TypeCheck type_check) {
    VarType v0 = var_promote(var_type(a0)),
            v1 = var_promote(var_type(a1));

    if (v0 != v1) {
        v0 = var_promote(var_type(a0, v1));
        v1 = var_promote(var_type(a1, v0));
    }

    VarType v = (VarType) std::max((uint32_t) v0, (uint32_t) v1);

    size_t d0 = var_depth(a0), d1 = var_depth(a1);
    if (d0 == 0 && d1 == 0) {
        if (require_enoki)
            ek::enoki_raise("%s(): at least one of the inputs must be an Enoki array!", op);
        else
            return false;
    }

    type_check(op, v);
    const py::object &base = d0 >= d1 ? a0 : a1;
    py::object type = base.attr("ReplaceScalar")(v);

    if (!a0.get_type().is(type))
        a0 = type(a0);
    if (!a1.get_type().is(type))
        a1 = type(a1);

    return true;
}

/// Perform C-style type promotion for a binary bit operation (second type may be a mask)
bool var_promote_bitop(const char *op, py::object &a0, py::object &a1,
                       bool require_enoki, TypeCheck type_check) {
    VarType v0 = var_promote(var_type(a0)),
            v1 = var_promote(var_type(a1));

    if (v0 != v1) {
        v0 = var_promote(var_type(a0, v1));
        v1 = var_promote(var_type(a1, v0));
    }

    size_t d0 = var_depth(a0), d1 = var_depth(a1);
    if (d0 == 0 && d1 == 0) {
        if (require_enoki)
            ek::enoki_raise("%s(): at least one of the inputs must be an Enoki array!", op);
        else
            return false;
    }

    if (v1 != VarType::Bool)
        v0 = v1 = (VarType) std::max((uint32_t) v0, (uint32_t) v1);

    type_check(op, v0);

    const py::object &base = d0 >= d1 ? a0 : a1;
    py::object t0 = base.attr("ReplaceScalar")(v0);
    py::object t1 = base.attr("ReplaceScalar")(v1);

    if (!a0.get_type().is(t0))
        a0 = t0(a0);
    if (!a1.get_type().is(t1))
        a1 = t1(a1);

    return true;
}

/// Perform C-style type promotion for a ternary operation
bool var_promote(const char *op, py::object &a0, py::object &a1, py::object &a2,
                        bool require_enoki, TypeCheck type_check) {
    VarType v0 = var_promote(var_type(a0)),
            v1 = var_promote(var_type(a1)),
            v2 = var_promote(var_type(a2));

    VarType v =
        (VarType) std::max({ (uint32_t) v0, (uint32_t) v1, (uint32_t) v2 });

    size_t d0 = var_depth(a0),
           d1 = var_depth(a1),
           d2 = var_depth(a2);

    if (d0 == 0 && d1 == 0 && d2 == 0) {
        if (require_enoki)
            ek::enoki_raise("%s(): at least one of the inputs must be an Enoki array!", op);
        else
            return false;
    }

    type_check(op, v);

    const py::object &base = (d0 >= d1 && d0 >= d2) ? a0 : ((d1 >= d2) ? a1 : a2);
    py::object type = base.attr("ReplaceScalar")(v);

    if (!a0.get_type().is(type))
        a0 = type(a0);
    if (!a1.get_type().is(type))
        a1 = type(a1);
    if (!a2.get_type().is(type))
        a2 = type(a2);

    return true;
}

/// Perform C-style type promotion for a select() operation
bool var_promote_select(py::object &a0, py::object &a1, py::object &a2) {
    VarType v0 = var_promote(var_type(a0)),
            v1 = var_promote(var_type(a1)),
            v2 = var_promote(var_type(a2));

    VarType v = (VarType) std::max((uint32_t) v1, (uint32_t) v2);

    size_t d0 = var_depth(a0),
           d1 = var_depth(a1),
           d2 = var_depth(a2);

    if (d0 == 0 && d1 == 0 && d2 == 0)
        return false;

    const py::object &base = (d0 >= d1 && d0 >= d2) ? a0 : ((d1 >= d2) ? a1 : a2);
    py::object mask_type = base.attr("ReplaceScalar")(VarType::Bool),
               value_type = base.attr("ReplaceScalar")(v);

    if (!a0.get_type().is(mask_type))
        a0 = mask_type(a0);
    if (!a1.get_type().is(value_type))
        a1 = value_type(a1);
    if (!a2.get_type().is(value_type))
        a2 = value_type(a2);

    return true;
}

void array_check(const char *, py::handle a0, size_t s0, size_t &si) {
    si = var_is_dynamic(a0) ? s0 : 0;
}

/**
 * Validate that the inputs of an Enoki array fallback implementation have
 * compatible sizes and types.
 *
 * \param a0
 *    First operand
 *
 * \param a1
 *    Second operand
 *
 * \param s0
 *    Size of the first operand
 *
 * \param s1
 *    Size of the second operand
 *
 * \param si
 *    When the target type is dynamic, this variable specifies the size to
 *    which it should be initialized.
 *
 * \param inplace
 *    Should be set to \c true to enable extra checks in case this is an
 *    in-place operation that overwrites \c a0.
 */
void array_check(const char *op, const py::object &a0, const py::object &a1,
                 size_t s0, size_t s1, size_t &si, bool inplace) {
    size_t sr = std::max(s0, s1);

    if ((sr != s0 && s0 != 1) || (sr != s1 && s1 != 1))
        ek::enoki_raise("%s(): size mismatch (%zu vs %zu)!", op, s0, s1);
    else if (inplace && s0 == 1 && sr > 1)
        ek::enoki_raise("%s(): in-place operation involving a vectorial "
                        "result and a scalar destinaton operand!", op);
    else if (!a0.get_type().is(a1.get_type()))
        ek::enoki_raise("%s(): type mismatch!", op);

    si = var_is_dynamic(a0) ? sr : 0;
}

void array_check_bitop(const char *op, const py::object &a0,
                       const py::object &a1, size_t s0, size_t s1,
                       size_t &si, bool inplace) {
    size_t sr = std::max(s0, s1);

    if ((sr != s0 && s0 != 1) || (sr != s1 && s1 != 1))
        ek::enoki_raise("%s(): size mismatch (%zu and %zu)!", op, s0, s1);
    else if (inplace && s0 == 1 && sr > 1)
        ek::enoki_raise("%s(): in-place operation involving a vectorial "
                        "result and a scalar destinaton operand!", op);
    else if (!a0.get_type().is(a1.get_type()) && var_type(a1) != VarType::Bool)
        ek::enoki_raise("%s(): type mismatch!", op);

    si = var_is_dynamic(a0) ? sr : 0;
}

void array_check(const char *op, const py::object &a0, const py::object &a1,
                 const py::object &a2, size_t s0, size_t s1, size_t s2,
                 size_t &si, bool inplace) {
    size_t sr = std::max(s0, s1);

    if ((sr != s0 && s0 != 1) || (sr != s1 && s1 != 1) || (sr != s2 && s2 != 1))
        ek::enoki_raise("%s(): size mismatch (%zu and %zu)!", op, s0, s1);
    else if (inplace && s0 == 1 && sr > 1)
        ek::enoki_raise("%s(): in-place operation involving a vectorial "
                        "result and a scalar destinaton operand!", op);
    else if (!a0.get_type().is(a1.get_type()) || !a0.get_type().is(a2.get_type()))
        ek::enoki_raise("%s(): type mismatch!", op);

    si = var_is_dynamic(a0) ? sr : 0;
}

void array_check_select(const char *op, const py::object &a0,
                        const py::object &a1, const py::object &a2, size_t s0,
                        size_t s1, size_t s2, size_t &si) {
    size_t sr = std::max({ s0, s1, s2 });

    if ((sr != s0 && s0 != 1) || (sr != s1 && s1 != 1) || (sr != s2 && s2 != 1))
        ek::enoki_raise("%s(): size mismatch (%zu, %zu, and %zu)!", op, s0, s1, s2);
    else if (!a1.get_type().is(a2.get_type()))
        ek::enoki_raise("%s(): type mismatch!", op);

    si = var_is_dynamic(a0) ? sr : 0;
}

