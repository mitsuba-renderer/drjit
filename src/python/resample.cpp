/*
    resample.cpp -- Python bindings for array resampling operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/resample.h>
#include <nanobind/stl/string.h>
#include "common.h"

void export_resample(nb::module_ &) {
    nb::object detail = nb::module_::import_("drjit").attr("detail");
    using dr::Resampler;

    auto resampler = nb::class_<Resampler>(detail, "Resampler")
        .def("__init__", [](Resampler *self, uint32_t source_res, uint32_t target_res,
                            const char *filter, nb::handle filter_radius) {
                 if (!filter_radius.is_none())
                     nb::raise("drjit.Resampler(): 'filter_radius' must be None when using a filter preset.");
                 new (self) Resampler(source_res, target_res, filter);
             }, "source_res"_a, "target_res"_a, "filter"_a, "filter_radius"_a = nb::none())
        .def("__init__", [](Resampler *self, uint32_t source_res, uint32_t target_res,
                            nb::typed<nb::callable, float, float> filter, double filter_radius) {
                 Resampler::Filter filter_cb = [](double v, const void *ptr) -> double {
                     return nb::cast<double>(nb::handle((PyObject *) ptr)(v));
                 };
                 new (self) Resampler(source_res, target_res, filter_cb,
                                      filter.ptr(), filter_radius);
             }, "source_res"_a, "target_res"_a, "filter"_a, "filter_radius"_a)
#if defined(DRJIT_ENABLE_CUDA)
         .def("resample_fwd",
              (dr::CUDAArray<dr::half>(Resampler::*)(const dr::CUDAArray<dr::half> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_fwd",
              (dr::CUDAArray<float>(Resampler::*)(const dr::CUDAArray<float> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_fwd",
              (dr::CUDAArray<double>(Resampler::*)(const dr::CUDAArray<double> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_bwd",
              (dr::CUDAArray<dr::half>(Resampler::*)(const dr::CUDAArray<dr::half> &, uint32_t) const) &Resampler::resample_bwd,
              "target"_a.noconvert(), "stride"_a)
         .def("resample_bwd",
              (dr::CUDAArray<float>(Resampler::*)(const dr::CUDAArray<float> &, uint32_t) const) &Resampler::resample_bwd,
              "target"_a.noconvert(), "stride"_a)
         .def("resample_bwd",
              (dr::CUDAArray<double>(Resampler::*)(const dr::CUDAArray<double> &, uint32_t) const) &Resampler::resample_bwd,
              "target"_a.noconvert(), "stride"_a)
#endif
#if defined(DRJIT_ENABLE_LLVM)
         .def("resample_fwd",
              (dr::LLVMArray<dr::half>(Resampler::*)(const dr::LLVMArray<dr::half> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_fwd",
              (dr::LLVMArray<float>(Resampler::*)(const dr::LLVMArray<float> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_fwd",
              (dr::LLVMArray<double>(Resampler::*)(const dr::LLVMArray<double> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_bwd",
              (dr::LLVMArray<dr::half>(Resampler::*)(const dr::LLVMArray<dr::half> &, uint32_t) const) &Resampler::resample_bwd,
              "target"_a.noconvert(), "stride"_a)
         .def("resample_bwd",
              (dr::LLVMArray<float>(Resampler::*)(const dr::LLVMArray<float> &, uint32_t) const) &Resampler::resample_bwd,
              "target"_a.noconvert(), "stride"_a)
         .def("resample_bwd",
              (dr::LLVMArray<double>(Resampler::*)(const dr::LLVMArray<double> &, uint32_t) const) &Resampler::resample_bwd,
              "target"_a.noconvert(), "stride"_a)
#endif
         .def("resample_fwd",
              (dr::DynamicArray<dr::half>(Resampler::*)(const dr::DynamicArray<dr::half> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_fwd",
              (dr::DynamicArray<float>(Resampler::*)(const dr::DynamicArray<float> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def("resample_fwd",
              (dr::DynamicArray<double>(Resampler::*)(const dr::DynamicArray<double> &, uint32_t) const) &Resampler::resample_fwd,
              "source"_a.noconvert(), "stride"_a)
         .def_prop_ro("source_res", &Resampler::source_res)
         .def_prop_ro("target_res", &Resampler::target_res)
         .def("__repr__",
              [](const Resampler &r) {
                  return "Resampler[source_res="
                    + std::to_string(r.source_res())
                    + ", target_res=" + std::to_string(r.target_res())
                    + ", taps=" + std::to_string(r.taps())
                    + "]";
              });
}

