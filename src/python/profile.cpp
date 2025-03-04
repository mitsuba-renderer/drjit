/*
    profile.h -- integration for profilers such as NVTX

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "profile.h"

void export_profile(nb::module_ &m) {
    struct profile_enable {
        void __enter__() {
            jit_profile_start();
        }

        void __exit__(nb::handle, nb::handle, nb::handle) {
            jit_profile_stop();
        }
    };

    struct profile_range {
        const char *value;

        void __enter__() {
            jit_profile_range_push(value);
        }

        void __exit__(nb::handle, nb::handle, nb::handle) {
            jit_profile_range_pop();
        }
    };

    nb::class_<profile_range>(m, "profile_range", doc_profile_range)
        .def(nb::init<const char *>())
        .def("__enter__", &profile_range::__enter__)
        .def("__exit__", &profile_range::__exit__, nb::arg().none(),
             nb::arg().none(), nb::arg().none());

    nb::class_<profile_enable>(m, "profile_enable", doc_profile_enable)
        .def(nb::init<>())
        .def("__enter__", &profile_enable::__enter__)
        .def("__exit__", &profile_enable::__exit__, nb::arg().none(),
             nb::arg().none(), nb::arg().none());

    m.def("profile_mark", &jit_profile_mark, doc_profile_mark);
}
