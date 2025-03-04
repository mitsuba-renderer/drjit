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
#if defined(DRJIT_ENABLE_CUDA) || defined(DRJIT_ENABLE_LLVM)
         .def("resample_fwd", [](const Resampler &r, nb::handle_t<dr::ArrayBase> &source, uint32_t stride) -> nb::object {
                  const ArraySupplement &s = supp(source.type());
                  if (s.ndim != 1 || (JitBackend) s.backend == JitBackend::None)
                      return nb::steal(NB_NEXT_OVERLOAD);
                  uint32_t index = s.index(inst_ptr(source));
                  VarInfo info = jit_set_backend(index);
                  uint32_t out_index = 0;
                  switch (info.type) {
                      case VarType::Float16:
                          out_index = r.resample_fwd(dr::GenericArray<dr::half>::borrow(index), stride).release();
                          break;
                      case VarType::Float32:
                          out_index = r.resample_fwd(dr::GenericArray<float>::borrow(index), stride).release();
                          break;
                      case VarType::Float64:
                          out_index = r.resample_fwd(dr::GenericArray<double>::borrow(index), stride).release();
                          break;
                      default:
                          nb::raise_type_error("Unsupported input type!");
                  }
                  nb::object result = nb::inst_alloc(source.type());
                  s.init_index(out_index, inst_ptr(result));
                  nb::inst_mark_ready(result);
                  jit_var_dec_ref(out_index);
                  return result;
              },
              "source"_a, "stride"_a,
              nb::sig("def resample_fwd(self, source: ArrayT, stride: int) -> ArrayT"))
         .def("resample_bwd", [](const Resampler &r, nb::handle_t<dr::ArrayBase> &target, uint32_t stride) -> nb::object {
                  const ArraySupplement &s = supp(target.type());
                  if (s.ndim != 1 || (JitBackend) s.backend == JitBackend::None)
                      return nb::steal(NB_NEXT_OVERLOAD);
                  uint32_t index = s.index(inst_ptr(target));
                  VarInfo info = jit_set_backend(index);
                  uint32_t out_index = 0;
                  switch (info.type) {
                      case VarType::Float16:
                          out_index = r.resample_bwd(dr::GenericArray<dr::half>::borrow(index), stride).release();
                          break;
                      case VarType::Float32:
                          out_index = r.resample_bwd(dr::GenericArray<float>::borrow(index), stride).release();
                          break;
                      case VarType::Float64:
                          out_index = r.resample_bwd(dr::GenericArray<double>::borrow(index), stride).release();
                          break;
                      default:
                          nb::raise_type_error("Unsupported input type!");
                  }
                  nb::object result = nb::inst_alloc(target.type());
                  s.init_index(out_index, inst_ptr(result));
                  nb::inst_mark_ready(result);
                  jit_var_dec_ref(out_index);
                  return result;
              },
              "target"_a, "stride"_a,
              nb::sig("def resample_bwd(self, target: ArrayT, stride: int) -> ArrayT"))
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

