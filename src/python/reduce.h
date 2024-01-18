/*
    reduce.h -- Bindings for horizontal reduction operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include <optional>

extern void export_reduce(nb::module_ &);

extern nb::object all(nb::handle, std::optional<int>);
extern nb::object any(nb::handle, std::optional<int>);
extern nb::object sum(nb::handle, std::optional<int>);
extern nb::object prod(nb::handle, std::optional<int>);
extern nb::object min(nb::handle, std::optional<int>);
extern nb::object max(nb::handle, std::optional<int>);
extern nb::object compress(nb::handle_t<dr::ArrayBase> h);
