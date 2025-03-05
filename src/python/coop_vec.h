/*
    src/coop_vec.h -- Python bindings for Cooperative Vectors

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

extern void export_coop_vec(nb::module_ &m);

/// Shared view into a matrix
struct View {
    MatrixDescr descr;
    nb::object buffer;

    View() = default;
    View(const MatrixDescr &descr, const nb::handle &buffer)
        : descr(descr), buffer(nb::borrow(buffer)) { }

    nb::str repr() const;
    View getitem(nb::object arg) const;
    uint32_t index() const;
};
