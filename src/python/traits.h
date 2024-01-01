/*
    traits.cpp -- implementation of Dr.Jit type traits such as
    is_array_v, uint32_array_t, etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_traits(nb::module_ &);
extern nb::object expr_t(nb::handle h0, nb::handle h1);
extern nb::type_object value_t(nb::handle h);
extern bool is_special_v(nb::handle h);
extern bool is_matrix_v(nb::handle h);
extern bool is_complex_v(nb::handle h);
extern bool is_quaternion_v(nb::handle h);
