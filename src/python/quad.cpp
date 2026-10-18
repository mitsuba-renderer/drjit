/*
    quad.cpp -- Python bindings for numerical quadrature rules

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2025, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/quad.h>
#include <drjit-core/half.h>
#include <memory>
#include "quad.h"
#include "base.h"

using Rule = void (*)(int, double *, double *);

template <typename T>
static void init_converted(const ArraySupplement &s, const double *src,
                           size_t n, nb::handle result) {
    std::unique_ptr<T[]> tmp(new T[n]);
    for (size_t i = 0; i < n; ++i)
        tmp[i] = (T) src[i];
    s.init_data(n, tmp.get(), inst_ptr(result));
}

/// Turn a double precision host buffer into a Dr.Jit array of type 'dtype'
static nb::object load_double(nb::type_object_t<dr::ArrayBase> dtype,
                              const double *src, size_t n, const char *name) {
    const ArraySupplement &s = supp(dtype);

    if (s.is_tensor)
        return dtype(load_double(
            nb::borrow<nb::type_object_t<dr::ArrayBase>>(s.array), src, n, name));

    if (s.ndim != 1 || s.shape[0] != DRJIT_DYNAMIC || !s.init_data)
        nb::raise_type_error("drjit.quad.%s(): unsupported dtype -- must be "
                             "a dynamically sized 1D array.", name);

    nb::object result = nb::inst_alloc(dtype);

    switch ((VarType) s.type) {
        case VarType::Float16: init_converted<dr::half>(s, src, n, result); break;
        case VarType::Float32: init_converted<float>(s, src, n, result); break;
        case VarType::Float64: s.init_data(n, src, inst_ptr(result)); break;
        default:
            nb::raise_type_error("drjit.quad.%s(): unsupported dtype -- must "
                                 "be a floating point type.", name);
    }

    nb::inst_mark_ready(result);
    return result;
}

static nb::object eval_rule(Rule rule, nb::type_object_t<dr::ArrayBase> dtype,
                            int n, const char *name) {
    size_t size = (size_t) (n > 0 ? n : 0);
    std::unique_ptr<double[]> nodes(new double[size]), weights(new double[size]);
    rule(n, nodes.get(), weights.get());
    return nb::make_tuple(load_double(dtype, nodes.get(), size, name),
                          load_double(dtype, weights.get(), size, name));
}

void export_quad(nb::module_ &m) {
    m.def("gauss_legendre",
          [](nb::type_object_t<dr::ArrayBase> dtype, int n) {
              return eval_rule(dr::quad::gauss_legendre, dtype, n, "gauss_legendre");
          }, "dtype"_a, "n"_a, doc_quad_gauss_legendre,
          nb::sig("def gauss_legendre(dtype: type[T], n: int) -> tuple[T, T]"));

    m.def("gauss_lobatto",
          [](nb::type_object_t<dr::ArrayBase> dtype, int n) {
              return eval_rule(dr::quad::gauss_lobatto, dtype, n, "gauss_lobatto");
          }, "dtype"_a, "n"_a, doc_quad_gauss_lobatto,
          nb::sig("def gauss_lobatto(dtype: type[T], n: int) -> tuple[T, T]"));

    m.def("composite_simpson",
          [](nb::type_object_t<dr::ArrayBase> dtype, int n) {
              return eval_rule(dr::quad::composite_simpson, dtype, n, "composite_simpson");
          }, "dtype"_a, "n"_a, doc_quad_composite_simpson,
          nb::sig("def composite_simpson(dtype: type[T], n: int) -> tuple[T, T]"));

    m.def("composite_simpson_38",
          [](nb::type_object_t<dr::ArrayBase> dtype, int n) {
              return eval_rule(dr::quad::composite_simpson_38, dtype, n, "composite_simpson_38");
          }, "dtype"_a, "n"_a, doc_quad_composite_simpson_38,
          nb::sig("def composite_simpson_38(dtype: type[T], n: int) -> tuple[T, T]"));

    m.def("chebyshev",
          [](nb::type_object_t<dr::ArrayBase> dtype, int n) {
              size_t size = (size_t) (n > 0 ? n : 0);
              std::unique_ptr<double[]> nodes(new double[size]);
              dr::quad::chebyshev(n, nodes.get());
              return load_double(dtype, nodes.get(), size, "chebyshev");
          }, "dtype"_a, "n"_a, doc_quad_chebyshev,
          nb::sig("def chebyshev(dtype: type[T], n: int) -> T"));
}
