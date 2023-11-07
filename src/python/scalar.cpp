/*
    scalar.cpp -- instantiates the drjit.scalar.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "scalar.h"
#include "random.h"
#include "texture.h"

void export_scalar(nb::module_& m) {
    ArrayBinding b;
    dr::bind_all<float>(b);
    bind_pcg32<float>(m);
    bind_texture_all<float>(m);

    m.attr("Bool") = nb::borrow(&PyBool_Type);
    m.attr("Float16") = nb::borrow(&PyFloat_Type);
    m.attr("Float32") = nb::borrow(&PyFloat_Type);
    m.attr("Float64") = nb::borrow(&PyFloat_Type);
    m.attr("Float") = nb::borrow(&PyFloat_Type);
    m.attr("Int16") = nb::borrow(&PyLong_Type);
    m.attr("Int32") = nb::borrow(&PyLong_Type);
    m.attr("Int64") = nb::borrow(&PyLong_Type);
    m.attr("Int") = nb::borrow(&PyLong_Type);
    m.attr("UInt16") = nb::borrow(&PyLong_Type);
    m.attr("UInt32") = nb::borrow(&PyLong_Type);
    m.attr("UInt64") = nb::borrow(&PyLong_Type);
    m.attr("UInt") = nb::borrow(&PyLong_Type);
}
