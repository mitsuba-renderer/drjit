#pragma once

#include <drjit/random.h>
#include "common.h"

template <typename Guide>
void bind_pcg32(py::module_ &m) {
    using UInt64 = dr::uint64_array_t<Guide>;
    using Int64 = dr::int64_array_t<Guide>;
    using PCG32 = dr::PCG32<UInt64>;

    auto pcg32 = py::class_<PCG32>(m, "PCG32")
        .def(py::init<size_t, const UInt64 &, const UInt64 &>(),
             "size"_a = 1,
             "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def(py::init<const PCG32 &>())
        .def("seed", &PCG32::seed, "size"_a = 1,
             "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("next_uint32", py::overload_cast<>(&PCG32::next_uint32))
        .def("next_uint32",
             py::overload_cast<const dr::mask_t<UInt64> &>(&PCG32::next_uint32))
        .def("next_uint32_bounded", &PCG32::next_uint32_bounded, "bound"_a,
             "mask"_a = true)
        .def("next_uint64", py::overload_cast<>(&PCG32::next_uint64))
        .def("next_uint64",
             py::overload_cast<const dr::mask_t<UInt64> &>(&PCG32::next_uint64))
        .def("next_uint64_bounded", &PCG32::next_uint64_bounded, "bound"_a,
             "mask"_a = true)
        .def("next_float32", py::overload_cast<>(&PCG32::next_float32))
        .def("next_float32", py::overload_cast<const dr::mask_t<UInt64> &>(
                                 &PCG32::next_float32))
        .def("next_float64", py::overload_cast<>(&PCG32::next_float64))
        .def("next_float64", py::overload_cast<const dr::mask_t<UInt64> &>(
                                 &PCG32::next_float64))
        .def("__add__", [](const PCG32 &a, const Int64 &x) -> PCG32 { return a + x; }, py::is_operator())
        .def("__iadd__", [](PCG32 *a, const Int64 &x) -> PCG32* { *a += x; return a; }, py::is_operator())
        .def("__sub__", [](const PCG32 &a, const Int64 &x) -> PCG32 { return a - x; }, py::is_operator())
        .def("__isub__", [](PCG32 *a, const Int64 &x) -> PCG32* { *a -= x; return a; }, py::is_operator())
        .def("__sub__", [](const PCG32 &a, const PCG32 &b) -> Int64 { return a - b; }, py::is_operator())
        .def_readwrite("state", &PCG32::state)
        .def_readwrite("inc", &PCG32::inc);

    py::handle u64;
    if constexpr (dr::is_array_v<UInt64>)
        u64 = py::type::of<UInt64>();
    else
        u64 = py::handle((PyObject *) &PyLong_Type);
    py::dict fields;
    fields["state"] = u64;
    fields["inc"] = u64;
    pcg32.attr("DRJIT_STRUCT") = fields;
}
