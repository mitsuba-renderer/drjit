/*
    profile.h -- integration for profilers such as NVTX

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "profile.h"

struct profile_event {
    jit_profile_t m_handle = nullptr;

    nb::str __repr__() const {
        if (m_active)
            return nb::str("<drjit.profile: {} captured events>").format(__len__());
        else
            return nb::str("<drjit.profile: capture in progress>");
    }
};

struct profile {
    void __enter__() {
        if (m_active)
            nb::raise("drjit.profile.__exit__(): you must exit the profile "
                      "region before entering it again.");
        jit_profile_begin();
        m_active = true;
        if (m_handle) {
            jit_profile_free(m_handle);
            m_handle = nullptr;
        }
    }

    void __exit__(nb::handle, nb::handle, nb::handle) {
        if (!m_active)
            nb::raise("drjit.profile.__exit__(): you must enter the profile "
                      "region before exiting it.");

        m_handle = jit_profile_end();
        m_active = false;
    }

    ~profile() {
        if (m_active)
            PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                             "drjit.profile: destructing a profile object that has "
                             "not been stopped!");
        if (m_handle)
            jit_profile_free(m_handle);
    }

    size_t __len__() const {
        if (m_active)
            nb::raise("drjit.profile.__len__(): the profile is still being captured!");

        return jit_profile_size(m_handle);
    }

    profile_event __getitem__(size_t index) const {
        if (m_active)
            nb::raise("drjit.profile.__getitem__(): the profile is still being captured!");

        return { jit_profile_event(m_handle, index) };
    }

    nb::str __repr__() const {
        if (m_active)
            return nb::str("<drjit.profile: {} captured events>").format(__len__());
        else
            return nb::str("<drjit.profile: capture in progress>");
    }

    jit_profile_t m_handle = nullptr;
    bool m_active = false;
};

struct profile_range {
    const char *m_value = nullptr;
    bool m_active = false;

    void __enter__() {
        if (m_active)
            nb::raise("drjit.profile_range.__exit__(): you must exit the profile "
                      "range before entering it again.");
        jit_profile_range_begin(m_value);
        m_active = true;
    }

    void __exit__(nb::handle, nb::handle, nb::handle) {
        if (!m_active)
            nb::raise("drjit.profile_range.__exit__(): you must enter the profile "
                      "range before exiting it.");
        jit_profile_range_end();
        m_active = false;
    }

    ~profile() {
        if (m_active)
            PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                             "drjit.profile: destructing a profile range that has "
                             "not been stopped!");
    }
};

void export_profile(nb::module_ &m) {
    auto profile = nb::class_<profile>(m, "profile", doc_profile)
        .def(nb::init<>())
        .def("__enter__", &profile::__enter__)
        .def("__len__", &profile::__len__)
        .def("__getitem__", &profile::__getitem__, nb::rv_policy::reference_internal)
        .def("__repr__", &profile::__repr__)
        .def("__exit__", &profile::__exit__, nb::arg().none(),
             nb::arg().none(), nb::arg().none());

    nb::class_<profile_event>(profile, "event", doc_profile_event)
        .def("__repr__", &profile::__repr__);

    nb::class_<profile_range>(m, "profile_range", doc_profile_range)
        .def(nb::init<const char *>())
        .def("__enter__", &profile_range::__enter__)
        .def("__exit__", &profile_range::__exit__, nb::arg().none(),
             nb::arg().none(), nb::arg().none());

    m.def("profile_mark", &jit_profile_mark, doc_profile_mark);
}
