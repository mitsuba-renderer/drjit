/*
    local.h -- Python bindings for Dr.Jit-Core variable arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "common.h"

class Local {
public:
    /**
     * \brief Localate local memory to store a PyTree of type ``dtype`` with length
     * length ``length``. If desired, a default value can be specified.
     */
    Local(nb::handle dtype, size_t length, nb::handle value = nb::none());

    // Destroy the ``Local`` object, decrease the refcounts of the array variables
    ~Local();

    /// Return the length of the local memory region
    size_t len() const { return m_length; }

    /// Return a human-readable description of the type
    nb::str repr() const;

    /// Perform a masked read of the value at ``mask``
    nb::object read(nb::handle index, nb::handle mask) const;

    /// Perform a masked write
    void write(nb::handle index, nb::handle value, nb::handle mask);

    /// Return a mask type that can be used for \ref read() and \ref write()
    nb::handle mask_type() const { return m_mask_tp; }

    /// Return an index type that can be used for \ref read() and \ref write()
    nb::handle index_type() const { return m_index_tp; }

    const dr::vector<uint32_t> &arrays() const { return m_arrays; }
    dr::vector<uint32_t> &arrays() { return m_arrays; }

protected:
    nb::object m_dtype;
    size_t m_length;
    nb::object m_value;
    dr::vector<uint32_t> m_arrays;
    JitBackend m_backend;
    nb::handle m_index_tp;
    nb::handle m_mask_tp;
};

extern nb::handle local_type;

extern void export_local(nb::module_ &m);

