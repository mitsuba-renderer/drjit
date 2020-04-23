#pragma once

#include "common.h"

/// Is a given variable an Enoki variable?
extern bool var_is_enoki(py::handle h);

/// Is a given variable an Enoki variable *or* an Enoki type object?
extern bool var_is_enoki_type(py::handle h);

/// Is a given variable an Enoki variable? (esp. a JIT-related type)
extern bool var_is_enoki_jit(py::handle h);

/// Check whether a given Enoki type is dynamic
extern bool var_is_dynamic(py::handle h);

/**
 * Return the VarType of a given Enoki object or plain Python type. Return
 * 'preferred' when there is sufficient room for interpretation (e.g. when
 * given an 'int').
 */
extern VarType var_type(py::handle h, VarType preferred);

/// Return the depth of the given Enoki variable (or 0)
extern size_t var_depth(const py::object &o) ;

/// Type signature of the test_* functions
using TypeCheck = void (*)(const char *, VarType);

/// No-op, accepts all inputs
extern void test_any(const char *, VarType);

/// Throw if 'v' is not a floating point type
extern void test_float(const char *op, VarType v);

/// Throw if 'v' is not an integral type
extern void test_integral(const char *op, VarType v);

/// Throw if 'v' is not an arithmetic type
extern void test_arithmetic(const char *op, VarType v);

/// Throw if 'v' is not a mask type
extern void test_mask(const char *op, VarType v);

/// Throw if 'v' is not a floating point type (with special error message for 'truediv')
extern void test_float_truediv(const char *, VarType v);

/// Throw if 'v' is not an integral point type (with special error message for 'floordiv')
extern void test_integral_floordiv(const char *, VarType v);

/// Perform C-style type promotion for a binary operation
extern bool var_promote(const char *op, py::object &a0, py::object &a1,
                        bool require_enoki, TypeCheck type_check);

/// Perform C-style type promotion for a binary bit operation (second type may be a mask)
extern bool var_promote_bitop(const char *op, py::object &a0, py::object &a1,
                              bool require_enoki, TypeCheck type_check);

/// Perform C-style type promotion for a ternary operation
extern bool var_promote(const char *op, py::object &a0, py::object &a1,
                        py::object &a2, bool require_enoki,
                        TypeCheck type_check);

/// Perform C-style type promotion for a select() operation
extern bool var_promote_select(py::object &a0, py::object &a1, py::object &a2);

extern void array_check(const char *, py::handle a0, size_t s0, size_t &si);

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
extern void array_check(const char *op, const py::object &a0,
                        const py::object &a1, size_t s0, size_t s1, size_t &si,
                        bool inplace);

extern void array_check_bitop(const char *op, const py::object &a0,
                              const py::object &a1, size_t s0, size_t s1,
                              size_t &si, bool inplace);

extern void array_check(const char *op, const py::object &a0,
                        const py::object &a1, const py::object &a2, size_t s0,
                        size_t s1, size_t s2, size_t &si, bool inplace);

extern void array_check_select(const char *op, const py::object &a0,
                               const py::object &a1, const py::object &a2,
                               size_t s0, size_t s1, size_t s2, size_t &si);
