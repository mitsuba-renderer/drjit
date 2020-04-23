#include "route.h"
#include <enoki/math.h>

extern py::object array_route_zero(py::handle h, size_t size = 1);
extern py::object py_compare(py::handle a, py::handle b, int mode);
extern py::object array_route_select(py::object a0, py::object a1, py::object a2);
extern py::object array_route_abs(const py::object &a);

ENOKI_PY_ROUTE_UNARY_FLOAT(sin, ek::sin(a0d))
ENOKI_PY_ROUTE_UNARY_FLOAT(cos, ek::cos(a0d))
ENOKI_PY_ROUTE_UNARY_FLOAT(sincos, ek::sincos(a0d))

ENOKI_PY_UNARY_OPERATION(sin, array_route_sin(a0[i0]))
ENOKI_PY_UNARY_OPERATION(cos, array_route_cos(a0[i0]))

static py::object array_generic_sincos(const py::object &a0) {
    size_t s0 = py::len(a0), si;
    array_check("sincos", a0, s0, si);
    py::object ar_0 = a0.attr("empty_")(si);
    py::object ar_1 = a0.attr("empty_")(si);
    for (size_t i_ = 0; i_ < s0; ++i_) {
        py::int_ i0(i_);
        py::tuple result = array_route_sincos(a0[i0]);
        ar_0[i0] = result[0];
        ar_1[i0] = result[1];
    }
    return py::make_tuple(ar_0, ar_1);
}

void export_route_math(py::module &m) {
    py::class_<ek::ArrayBase> base = (py::object) m.attr("ArrayBase");

    m.def("sin_", &array_generic_sin)
     .def("cos_", &array_generic_cos);

    m.def("sin", &array_route_sin);
    m.def("cos", &array_route_cos);
    m.def("sincos", &array_route_sincos);

    m.def("sign", [](const py::object &v) {
        py::handle type = v.get_type();
        return array_route_select(
            py_compare(v, array_route_zero(type), Py_GE),
            type(1), type(-1));
    });

    m.def("copysign", [](const py::object &v1, const py::object &v2) {
        py::handle type = v1.get_type();
        py::object v1_a = array_route_abs(v1);
        return array_route_select(
            py_compare(v2, array_route_zero(type), Py_GE),
            v1_a, -v1_a);
    });

    m.def("copysign_neg", [](const py::object &v1, const py::object &v2) {
        py::handle type = v1.get_type();
        py::object v1_a = array_route_abs(v1);
        return array_route_select(
            py_compare(v2, array_route_zero(type), Py_GE),
            -v1_a, v1_a);
    });

    m.def("mulsign", [](const py::object &v1, const py::object &v2) {
        py::handle type = v1.get_type();
        return array_route_select(
            py_compare(v2, array_route_zero(type), Py_GE),
            v1, -v1);
    });

    m.def("mulsign_neg", [](const py::object &v1, const py::object &v2) {
        py::handle type = v1.get_type();
        return array_route_select(
            py_compare(v2, array_route_zero(type), Py_GE),
            -v1, v1);
    });

}
