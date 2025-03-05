/*
    src/coop_vec.h -- Python bindings for Cooperative CoopVecs

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

extern void export_coop_vec(nb::module_ &m);

/// Cooperative vector container data structure
struct CoopVec {
    /// JIT variable ID
    uint64_t m_index = 0;
    /// Number of entries
    uint32_t m_size = 0;
    /// Element type
    nb::handle m_type;

    CoopVec(nb::handle arg);

    /// Steals ownership of 'index'
    CoopVec(uint64_t index, uint32_t size, nb::handle type)
        : m_index(index), m_size(size), m_type(type) { }

    /// Copy constructor
    CoopVec(const CoopVec &vec)
        : m_index(vec.m_index), m_size(vec.m_size), m_type(vec.m_type) {
        ad_var_inc_ref(m_index);
    }
    CoopVec(CoopVec &&vec) noexcept
        : m_index(vec.m_index), m_size(vec.m_size), m_type(vec.m_type) {
        vec.m_index = 0;
        vec.m_size = 0;
        vec.m_type = nb::handle();
    }
    CoopVec& operator=(CoopVec &&x) {
        ad_var_dec_ref(m_index);
        m_index = x.m_index;
        m_size = x.m_size;
        m_type = x.m_type;
        x.m_index  = x.m_size = 0;
        x.m_type = nb::handle();
        return *this;
    }
    ~CoopVec() { ad_var_dec_ref(m_index); }

    /// Expand a cooperative vector into a Python list
    nb::list expand_to_list() const;

    /// Expand a cooperative vector into a Dr.Jit array type (e.g. ArrayXf)
    nb::object expand_to_vector() const;

private:
    void construct(nb::handle arg);
};

/// Shared view into a matrix
struct MatrixView {
    /// Shape, strides, etc.
    MatrixDescr descr{};

    /// Dr.Jit 1D array holding the data
    nb::object buffer;

    /// Should the view be transposed?
    bool transpose = false;

    MatrixView() = default;
    MatrixView(const MatrixDescr &descr, const nb::handle &buffer)
        : descr(descr), buffer(nb::borrow(buffer)), transpose(false) { }

    nb::str repr() const;
    MatrixView getitem(nb::object arg) const;
    uint64_t index() const;
};


extern nb::handle view_type;
extern nb::handle coop_vector_type;
