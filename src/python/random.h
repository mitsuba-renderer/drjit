#pragma once

#include <drjit/random.h>
#include "common.h"
#include "base.h"

template <typename Guide>
void bind_pcg32(nb::module_ &m) {
    using UInt64 = dr::uint64_array_t<Guide>;
    using Int64 = dr::int64_array_t<Guide>;
    using PCG32 = dr::PCG32<UInt64>;
    using Mask = dr::mask_t<UInt64>;

    auto pcg32 = nb::class_<PCG32>(m, "PCG32", doc_PCG32)
        .def(nb::init<size_t, const UInt64 &, const UInt64 &>(),
             "size"_a = 1,
             "initstate"_a.sig("UInt64(0x853c49e6748fea9b)") = PCG32_DEFAULT_STATE,
             "initseq"_a.sig("UInt64(0xda3e39cb94b95bdb)") = PCG32_DEFAULT_STREAM, doc_PCG32_PCG32)
        .def(nb::init<const PCG32 &>(), doc_PCG32_PCG32_2)
        .def("seed", &PCG32::seed,
             "initstate"_a.sig("UInt64(0x853c49e6748fea9b)")  = PCG32_DEFAULT_STATE,
             "initseq"_a.sig("UInt64(0xda3e39cb94b95bdb)") = PCG32_DEFAULT_STREAM, doc_PCG32_seed)
        .def("next_uint32",
             nb::overload_cast<>(&PCG32::next_uint32),
             doc_PCG32_next_uint32)
        .def("next_uint32",
             nb::overload_cast<const Mask &>(&PCG32::next_uint32))
        .def("next_uint32_bounded",
             &PCG32::next_uint32_bounded, "bound"_a, "mask"_a.sig("Bool(True)") = true,
             doc_PCG32_next_uint32_bounded)
        .def("next_uint64",
             nb::overload_cast<>(&PCG32::next_uint64),
             doc_PCG32_next_uint64)
        .def("next_uint64",
             nb::overload_cast<const Mask &>(&PCG32::next_uint64))
        .def("next_uint64_bounded", &PCG32::next_uint64_bounded, "bound"_a,
             "mask"_a.sig("Bool(True)") = true,
             doc_PCG32_next_uint64_bounded)
        .def("next_float",
             [](nb::handle self, nb::type_object dtype, nb::handle mask) {
                 const char *key = nullptr;
                 if constexpr (dr::is_jit_v<Guide>) {
                     if (is_drjit_type(dtype)) {
                         const ArraySupplement &s = supp(dtype);
                         if ((VarType) s.type == VarType::Float16)
                             key = "next_float16";
                         else if ((VarType) s.type == VarType::Float32)
                             key = "next_float32";
                         else if ((VarType) s.type == VarType::Float64)
                             key = "next_float64";
                     }
                 } else if (dtype.is(&PyFloat_Type)) {
                     key = "next_float64";
                 }

                 if (!key)
                     nb::raise_type_error("Invalid 'dtype'");

                 auto &&fn = self.attr(key);
                 return !mask.is(Py_True) ? fn(mask) : fn();
             }, "dtype"_a, "mask"_a = true, doc_PCG32_next_float)
        .def("next_float16",
             nb::overload_cast<>(&PCG32::next_float16),
             doc_PCG32_next_float16)
        .def("next_float16",
             nb::overload_cast<const Mask &>(&PCG32::next_float16))
        .def("next_float32",
             nb::overload_cast<>(&PCG32::next_float32),
             doc_PCG32_next_float32)
        .def("next_float32",
             nb::overload_cast<const Mask &>(&PCG32::next_float32))
        .def("next_float64",
             nb::overload_cast<>(&PCG32::next_float64),
             doc_PCG32_next_float64)
        .def("next_float64",
             nb::overload_cast<const Mask &>(&PCG32::next_float64))
        .def("next_float_normal",
             [](nb::handle self, nb::type_object dtype, nb::handle mask) {
                 const char *key = nullptr;
                 if constexpr (dr::is_jit_v<Guide>) {
                     if (is_drjit_type(dtype)) {
                         const ArraySupplement &s = supp(dtype);
                         if ((VarType) s.type == VarType::Float16)
                             key = "next_float16_normal";
                         else if ((VarType) s.type == VarType::Float32)
                             key = "next_float32_normal";
                         else if ((VarType) s.type == VarType::Float64)
                             key = "next_float64_normal";
                     }
                 } else if (dtype.is(&PyFloat_Type)) {
                     key = "next_float64_normal";
                 }

                 if (!key)
                     nb::raise_type_error("Invalid 'dtype'");

                 auto &&fn = self.attr(key);
                 return !mask.is(Py_True) ? fn(mask) : fn();
             }, "dtype"_a, "mask"_a = true, doc_PCG32_next_float_normal)
        .def("next_float16_normal",
             nb::overload_cast<>(&PCG32::next_float16_normal),
             doc_PCG32_next_float16_normal)
        .def("next_float16_normal",
             nb::overload_cast<const Mask &>(&PCG32::next_float16))
        .def("next_float32_normal",
             nb::overload_cast<>(&PCG32::next_float32_normal),
             doc_PCG32_next_float32_normal)
        .def("next_float32_normal",
             nb::overload_cast<const Mask &>(&PCG32::next_float32))
        .def("next_float64_normal",
             nb::overload_cast<>(&PCG32::next_float64_normal),
             doc_PCG32_next_float64_normal)
        .def("next_float64_normal",
             nb::overload_cast<const Mask &>(&PCG32::next_float64_normal))
        .def("__add__", [](const PCG32 &a, const Int64 &x) -> PCG32 { return a + x; }, nb::is_operator(), doc_PCG32_add)
        .def("__iadd__", [](PCG32 *a, const Int64 &x) -> PCG32* { *a += x; return a; }, nb::is_operator(), doc_PCG32_iadd)
        .def("__sub__", [](const PCG32 &a, const Int64 &x) -> PCG32 { return a - x; }, nb::is_operator(), doc_PCG32_sub)
        .def("__isub__", [](PCG32 *a, const Int64 &x) -> PCG32* { *a -= x; return a; }, nb::is_operator(), doc_PCG32_isub)
        .def("__sub__", [](const PCG32 &a, const PCG32 &b) -> Int64 { return a - b; }, nb::is_operator())
        .def_rw("state", &PCG32::state, doc_PCG32_state)
        .def_rw("inc", &PCG32::inc, doc_PCG32_inc);

    nb::handle u64;
    if constexpr (dr::is_array_v<UInt64>)
        u64 = nb::type<UInt64>();
    else
        u64 = nb::handle((PyObject *) &PyLong_Type);

    nb::dict fields;
    fields["state"] = u64;
    fields["inc"] = u64;
    pcg32.attr("DRJIT_STRUCT") = fields;
}
