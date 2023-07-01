/*
    meta.h -- Logic related to Dr.Jit array metadata and type promotion

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include <string>

/// Check if the given metadata record is valid
extern bool meta_check(ArrayMeta m) noexcept;

/// Convert a metadata record into a string representation (for debugging)
extern std::string meta_str(ArrayMeta m);

/// Compute the metadata type of an operation combinining 'a' and 'b'
extern ArrayMeta meta_promote(ArrayMeta a, ArrayMeta b) noexcept;

// Infer a metadata record for the given Python object
extern ArrayMeta meta_get(nb::handle h) noexcept;

/**
 * \brief Given a list of Dr.Jit arrays and scalars, determine the flavor and
 * shape of the result array and broadcast/convert everything into this form.
 *
 * \param o
 *    Array of input operands of size 'n'
 *
 * \param n
 *    Number of operands
 *
 * \param select
 *    Should be 'true' if this is a drjit.select() operation, in which case the
 *    first operand will be promoted to a mask array
 */
extern void promote(nb::object *o, size_t n, bool select);

/// Look up the nanobind module name associated with the given array metadata
extern nb::handle meta_get_module(ArrayMeta meta) noexcept;

/// Determine the nanobind type name associated with the given array metadata
extern const char *meta_get_name(ArrayMeta meta) noexcept;

/// Look up the nanobind type associated with the given array metadata
extern nb::handle meta_get_type(ArrayMeta meta);

inline bool operator==(ArrayMeta a, ArrayMeta b) {
    a.talign = a.tsize_rel = b.talign = b.tsize_rel = 0;
    return memcmp(&a, &b, sizeof(ArrayMeta)) == 0;
}

inline bool operator!=(ArrayMeta a, ArrayMeta b) { return !operator==(a, b); }
