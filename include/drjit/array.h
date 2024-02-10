/*
    drjit/array.h -- Main header file for the Dr.Jit array class and
    various template specializations

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if __cplusplus < 201703L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)
#  error Dr.Jit requires compilation in C++17 mode!
#endif

// libc++: only include what is truly requested
#if defined(_LIBCPP_VERSION) && !defined(_LIBCPP_REMOVE_TRANSITIVE_INCLUDES)
#  define _LIBCPP_REMOVE_TRANSITIVE_INCLUDES
#endif

// Core C headers needed by Dr.Jit
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Core C++ headers needed by Dr.Jit
// On libc++, include <utility> without pulling in the entire C/C++ math library
#if defined(_LIBCPP_CMATH) || !defined(_LIBCPP_VERSION)
#  include <utility>
#else
#  define _LIBCPP_CMATH
#    include <utility>
#  undef _LIBCPP_CMATH
#endif

#define _LIBCPP_CMATH_BACKUP _LIBCPP_CMATH
#include <type_traits>

// Tiny self-contained subset of STL-like classes for internal use
#include <drjit-core/nanostl.h>

// Dr.Jit containers and compiler API (relevant when JIT-ted types are used)
#include <drjit-core/jit.h>

// Type traits for the JIT layer
#include <drjit-core/traits.h>

// Forward declarations of this project
#include <drjit/fwd.h>

// Type traits to detect/convert between various array types
#include <drjit/array_traits.h>

// Scalar fallbacks for various mathematical functions
#include <drjit/array_utils.h>

// Central constants ('pi', 'e', etc.)
#include <drjit/array_constants.h>

// Functionality to traverse custom data structures
#include <drjit/array_traverse.h>

// Routing layer that dispatches operations to the right array endpoints
#include <drjit/array_router.h>

// Formatter to convert a Dr.Jit array into a string.
#include <drjit/array_format.h>

// Implementations of drjit::ArrayBase and drjit::ArrayBaseT
#include <drjit/array_base.h>

// Fallback functionality for static arrays
#include <drjit/array_static.h>

// Generic catch-all array implementation
#include <drjit/array_generic.h>
#include <drjit/array_mask.h>
#include <drjit/array_iface.h>
