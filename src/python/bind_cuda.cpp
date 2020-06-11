#if defined(ENOKI_ENABLE_JIT)
#include "bind.h"
#include <enoki/cuda.h>
#include <enoki/random.h>

void export_cuda(py::module &m) {
    py::module cuda = m.def_submodule("cuda");

    using Guide = ek::CUDAArray<float>;
    ENOKI_BIND_ARRAY_BASE_1(cuda, Guide, false);
    ENOKI_BIND_ARRAY_BASE_2(false);
    ENOKI_BIND_ARRAY_TYPES(cuda, Guide, false);

    using UInt64 = ek::CUDAArray<uint64_t>;
    using Int64 = ek::CUDAArray<int64_t>;
    using PCG32 = ek::PCG32<UInt64>;
    py::class_<PCG32>(cuda, "PCG32")
        .def(py::init<size_t, const UInt64 &, const UInt64 &>(), "size"_a,
             "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def(py::init<const PCG32 &>())
        .def("seed", &PCG32::seed, "size"_a,
             "initstate"_a = PCG32_DEFAULT_STATE,
             "initseq"_a = PCG32_DEFAULT_STREAM)
        .def("next_uint32", py::overload_cast<>(&PCG32::next_uint32))
        .def("next_uint32",
             py::overload_cast<const ek::mask_t<UInt64> &>(&PCG32::next_uint32))
        .def("next_uint32_bounded", &PCG32::next_uint32_bounded, "bound"_a,
             "mask"_a = true)
        .def("next_uint64", py::overload_cast<>(&PCG32::next_uint64))
        .def("next_uint64",
             py::overload_cast<const ek::mask_t<UInt64> &>(&PCG32::next_uint64))
        .def("next_uint64_bounded", &PCG32::next_uint64_bounded, "bound"_a,
             "mask"_a = true)
        .def("next_float32", py::overload_cast<>(&PCG32::next_float32))
        .def("next_float32", py::overload_cast<const ek::mask_t<UInt64> &>(
                                 &PCG32::next_float32))
        .def("next_float64", py::overload_cast<>(&PCG32::next_float64))
        .def("next_float64", py::overload_cast<const ek::mask_t<UInt64> &>(
                                 &PCG32::next_float64))
        .def("__add__", [](const PCG32 &a, const Int64 &x) -> PCG32 { return a + x; }, py::is_operator())
        .def("__iadd__", [](PCG32 *a, const Int64 &x) -> PCG32* { *a += x; return a; }, py::is_operator())
        .def("__sub__", [](const PCG32 &a, const Int64 &x) -> PCG32 { return a - x; }, py::is_operator())
        .def("__isub__", [](PCG32 *a, const Int64 &x) -> PCG32* { *a -= x; return a; }, py::is_operator())
        .def("__sub__", [](const PCG32 &a, const PCG32 &b) -> Int64 { return a - b; }, py::is_operator())
        .def_readwrite("state", &PCG32::state)
        .def_readwrite("inc", &PCG32::inc);
}
#endif
