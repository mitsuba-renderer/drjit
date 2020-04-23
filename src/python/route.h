#pragma once
#include "promote.h"

#define ENOKI_PY_UNARY_OPERATION(name, op)                                     \
    static py::object array_generic_##name(const py::object &a0) {             \
        size_t s0 = py::len(a0), si;                                           \
        array_check(#name, a0, s0, si);                                        \
        py::object ar = a0.attr("empty_")(si);                                 \
        for (size_t i_ = 0; i_ < s0; ++i_) {                                   \
            py::int_ i0(i_);                                                   \
            ar[i0] = op;                                                       \
        }                                                                      \
        return ar;                                                             \
    }

#define ENOKI_PY_BINARY_OPERATION(name, op)                                    \
    static py::object array_generic_##name(const py::object &a0,               \
                                           const py::object &a1) {             \
        size_t s0 = py::len(a0), s1 = py::len(a1), sr = std::max(s0, s1), si;  \
        array_check(#name, a0, a1, s0, s1, si, false);                         \
        py::object ar = a0.attr("empty_")(si);                                 \
        py::int_ z(0);                                                         \
        for (size_t i_ = 0; i_ < sr; ++i_) {                                   \
            py::int_ i(i_);                                                    \
            py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z;               \
            ar[i] = op;                                                        \
        }                                                                      \
        return ar;                                                             \
    }

#define ENOKI_PY_BINARY_OPERATION_BITOP(name, op)                              \
    static py::object array_generic_##name(const py::object &a0,               \
                                           const py::object &a1) {             \
        size_t s0 = py::len(a0), s1 = py::len(a1), sr = std::max(s0, s1), si;  \
        array_check_bitop(#name, a0, a1, s0, s1,  si, false);                  \
        py::object ar = a0.attr("empty_")(si);                                 \
        py::int_ z(0);                                                         \
        for (size_t i_ = 0; i_ < sr; ++i_) {                                   \
            py::int_ i(i_);                                                    \
            py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z;               \
            ar[i] = op;                                                        \
        }                                                                      \
        return ar;                                                             \
    }

#define ENOKI_PY_BINARY_OPERATION_MASK(name, op)                               \
    static py::object array_generic_##name(const py::object &a0,               \
                                           const py::object &a1) {             \
        size_t s0 = py::len(a0), s1 = py::len(a1), sr = std::max(s0, s1), si;  \
        array_check(#name, a0, a1, s0, s1,  si, false);                        \
        py::object ar = var_mask(a0).attr("empty_")(si);                       \
        py::int_ z(0);                                                         \
        for (size_t i_ = 0; i_ < sr; ++i_) {                                   \
            py::int_ i(i_);                                                    \
            py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z;               \
            ar[i] = op;                                                        \
        }                                                                      \
        return ar;                                                             \
    }

#define ENOKI_PY_BINARY_OPERATION_COMPOUND(name, op)                           \
    static py::object array_generic_##name(const py::object &a0,               \
                                           const py::object &a1) {             \
        size_t s0 = py::len(a0), s1 = py::len(a1), sr = std::max(s0, s1), si;  \
        array_check(#name, a0, a1, s0, s1, si, true);                          \
        py::int_ z(0);                                                         \
        for (size_t i_ = 0; i_ < sr; ++i_) {                                   \
            py::int_ i(i_);                                                    \
            py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z;               \
            a0[i] = op;                                                        \
        }                                                                      \
                                                                               \
        return a0;                                                             \
    }

#define ENOKI_PY_BINARY_OPERATION_COMPOUND_BITOP(name, op)                     \
    static py::object array_generic_##name(const py::object &a0,               \
                                           const py::object &a1) {             \
        size_t s0 = py::len(a0), s1 = py::len(a1), sr = std::max(s0, s1), si;  \
        array_check_bitop(#name, a0, a1, s0, s1, si, true);                    \
        py::int_ z(0);                                                         \
        for (size_t i_ = 0; i_ < sr; ++i_) {                                   \
            py::int_ i(i_);                                                    \
            py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z;               \
            a0[i] = op;                                                        \
        }                                                                      \
                                                                               \
        return a0;                                                             \
    }
#define ENOKI_PY_TERNARY_OPERATION(name, op)                                   \
    static py::object array_generic_##name(                                    \
        const py::object &a0, const py::object &a1, const py::object &a2) {    \
        size_t s0 = py::len(a0), s1 = py::len(a1), s2 = py::len(a2),           \
               sr = std::max({ s0, s1, s2 }), si;                              \
        array_check(#name, a0, a1, a2, s0, s1, s2, si, false);                 \
        py::object ar = a0.attr("empty_")(si);                                 \
        py::int_ z(0);                                                         \
        for (size_t i_ = 0; i_ < sr; ++i_) {                                   \
            py::int_ i(i_);                                                    \
            py::handle i0 = s0 > 1 ? i : z, i1 = s1 > 1 ? i : z,               \
                       i2 = s2 > 1 ? i : z;                                    \
            ar[i] = op;                                                        \
        }                                                                      \
        return ar;                                                             \
    }

#define ENOKI_PY_ROUTE_UNARY(name, check)                                      \
    static py::object array_route_##name(const py::object &a) {                \
        check(#name, var_type(a));                                             \
        return a.attr(#name "_")();                                            \
    }

#define ENOKI_PY_ROUTE_BINARY(name, check)                                     \
    static py::object array_route_##name(py::object &a0, py::object &a1) {     \
        var_promote(#name, a0, a1, true, check);                               \
        return a0.attr(#name "_")(a1);                                         \
    }

#define ENOKI_PY_ROUTE_BINARY_BITOP(name, check)                               \
    static py::object array_route_##name(py::object &a0, py::object &a1) {     \
        var_promote_bitop(#name, a0, a1, true, check);                         \
        return a0.attr(#name "_")(a1);                                         \
    }

#define ENOKI_PY_ROUTE_UNARY_FLOAT(name, op)                                   \
    static py::object array_route_##name(const py::object &a0) {               \
        test_float(#name, var_type(a0));                                       \
        if (var_is_enoki(a0)) {                                                \
            return a0.attr(#name "_")();                                       \
        } else {                                                               \
            double a0d = py::cast<double>(a0);                                 \
            return py::cast(op);                                               \
        }                                                                      \
    }

#define ENOKI_PY_ROUTE_BINARY_FLOAT(name, op)                                  \
    static py::object array_route_##name(py::object a0, py::object a1) {       \
        if (var_promote(#name, a0, a1, false, test_float)) {                   \
            return a0.attr(#name "_")(a1);                                     \
        } else {                                                               \
            double a0d = py::cast<double>(a0), a1d = py::cast<double>(a1);     \
            return py::cast(op);                                               \
        }                                                                      \
    }

#define ENOKI_PY_ROUTE_TERNARY_FLOAT(name, op)                                 \
    static py::object array_route_##name(py::object a0, py::object a1,         \
                                         py::object a2) {                      \
        if (var_promote(#name, a0, a1, a2, false, test_float)) {               \
            return a0.attr(#name "_")(a1, a2);                                 \
        } else {                                                               \
            double a0d = py::cast<double>(a0), a1d = py::cast<double>(a1),     \
                   a2d = py::cast<double>(a2);                                 \
            return py::cast(op);                                               \
        }                                                                      \
    }

extern void test_any(const char *, VarType);
extern void test_float(const char *op, VarType v);
extern void test_integral(const char *op, VarType v);
extern void test_arithmetic(const char *op, VarType v);
extern void test_mask(const char *op, VarType v);

extern size_t coeff_evals;
