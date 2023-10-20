/*
    drjit/vcall.h -- Vectorized method call support

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit/autodiff.h>

#define DRJIT_VCALL_BEGIN(Name)                                                \
    namespace drjit {                                                          \
        template <typename Array>                                              \
        struct call_support<Name, Array> {                                     \
            using Class = Name;                                                \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Array &array) : array(array) { }                \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_VCALL_TEMPLATE_BEGIN(Name)                                       \
    namespace drjit {                                                          \
        template <typename Array, typename... Ts>                              \
        struct call_support<Name<Ts...>, Array> {                              \
            using Class = Name<Ts...>;                                         \
            static constexpr const char *Domain = #Name;                       \
            call_support(const Array &array) : array(array) { }                \
            const call_support *operator->() const {                           \
                return this;                                                   \
            }

#define DRJIT_VCALL_END(Name)                                                  \
        private:                                                               \
            const Array &array;                                                \
        };                                                                     \
    }
