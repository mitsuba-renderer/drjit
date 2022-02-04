#include "bind.h"
#include "random.h"
#include "tensor.h"
#include "texture.h"

void export_scalar(py::module_ &m) {
    py::module_ scalar = m.def_submodule("scalar");

    scalar.attr("Bool")    = py::handle((PyObject *) &PyBool_Type);
    scalar.attr("Float32") = py::handle((PyObject *) &PyFloat_Type);
    scalar.attr("Float64") = py::handle((PyObject *) &PyFloat_Type);
    scalar.attr("Int32")   = py::handle((PyObject *) &PyLong_Type);
    scalar.attr("Int64")   = py::handle((PyObject *) &PyLong_Type);
    scalar.attr("UInt32")  = py::handle((PyObject *) &PyLong_Type);
    scalar.attr("UInt64")  = py::handle((PyObject *) &PyLong_Type);

    scalar.attr("Float")   = scalar.attr("Float32");
    scalar.attr("Int")     = scalar.attr("Int32");
    scalar.attr("UInt")    = scalar.attr("UInt32");

    DRJIT_BIND_ARRAY_TYPES(scalar, float, true);

    bind_full(d_i32, true); bind_full(d_u32, true); bind_full(d_i64, true);
    bind_full(d_u64, true); bind_full(d_f32, true); bind_full(d_f64, true);
    bind_full(d_b, true);

    bind_pcg32<uint64_t>(scalar);

    struct LoopDummy { LoopDummy(const char*, py::args) { }};
    py::class_<LoopDummy>(scalar, "Loop")
        .def(py::init<const char*, py::args>())
        .def("put", [](LoopDummy&, py::args) {})
        .def("init", [](LoopDummy&) {})
        .def("set_uniform", [](LoopDummy&, bool) { })
        .def("set_max_iterations", [](LoopDummy&, bool) { })
        .def("__call__", [](LoopDummy&, bool value) { return value; });

    bind_texture_all<float>(scalar);

    using Guide = drjit::DynamicArray<float>;
    DRJIT_BIND_TENSOR_TYPES(scalar);
}
