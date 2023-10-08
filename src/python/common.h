/*
    common.h -- Common definitions used by the Dr.Jit Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <drjit/python.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include "docstr.h"

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

using dr::ArrayMeta;
using dr::ArraySupplement;
using dr::ArrayBinding;
using dr::ArrayOp;
using dr::ArrayBase;
using dr::dr_vector;

inline const ArraySupplement &supp(nb::handle h) {
    return nb::type_supplement<ArraySupplement>(h);
}

inline ArrayBase* inst_ptr(nb::handle h) {
    return nb::inst_ptr<ArrayBase>(h);
}

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T> struct type_caster<dr::dr_vector<T>>
 : list_caster<drjit::dr_vector<T>, T> { };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#define raise_if(expr, ...)                                                    \
    do {                                                                       \
        if (NB_UNLIKELY(expr))                                                 \
            nb::detail::raise(__VA_ARGS__);                                    \
    } while (false)
