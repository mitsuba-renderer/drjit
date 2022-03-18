#include "bind.h"
#include "random.h"
#include "tensor.h"
#include "texture.h"

void export_scalar(nb::module_ &m) {
    nb::module_ scalar = m.def_submodule("scalar");

    scalar.attr("Bool")    = nb::handle((PyObject *) &PyBool_Type);
    scalar.attr("Float32") = nb::handle((PyObject *) &PyFloat_Type);
    scalar.attr("Float64") = nb::handle((PyObject *) &PyFloat_Type);
    scalar.attr("Int32")   = nb::handle((PyObject *) &PyLong_Type);
    scalar.attr("Int64")   = nb::handle((PyObject *) &PyLong_Type);
    scalar.attr("UInt32")  = nb::handle((PyObject *) &PyLong_Type);
    scalar.attr("UInt64")  = nb::handle((PyObject *) &PyLong_Type);

    scalar.attr("Float")   = scalar.attr("Float32");
    scalar.attr("Int")     = scalar.attr("Int32");
    scalar.attr("UInt")    = scalar.attr("UInt32");

    DRJIT_BIND_ARRAY_TYPES(scalar, float, true);

    bind_full(d_i32, true); bind_full(d_u32, true); bind_full(d_i64, true);
    bind_full(d_u64, true); bind_full(d_f32, true); bind_full(d_f64, true);
    bind_full(d_b, true);

    bind_pcg32<uint64_t>(scalar);

    struct LoopDummy { LoopDummy(const char*, nb::args) { }};
    nb::class_<LoopDummy>(scalar, "Loop")
        .def(nb::init<const char*, nb::args>())
        .def("put", [](LoopDummy&, nb::args) {})
        .def("init", [](LoopDummy&) {})
        .def("set_uniform", [](LoopDummy&, bool) { })
        .def("set_max_iterations", [](LoopDummy&, bool) { })
        .def("__call__", [](LoopDummy&, bool value) { return value; });

    bind_texture_all<float>(scalar);

    using Guide = drjit::DynamicArray<float>;
    DRJIT_BIND_TENSOR_TYPES(scalar);
}
