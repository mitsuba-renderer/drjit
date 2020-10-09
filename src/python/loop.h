#pragma once

#include "common.h"
#include <enoki/loop.h>

template <typename Guide> struct Loop : ek::Loop<Guide> {
    using Base = ek::Loop<Guide>;
    using Base::m_vars;
    using Base::m_vars_phi;
    using Base::m_var_count;
    using Base::m_cuda;
    using Base::m_llvm;

    Loop(py::args args) : m_args(args) {
        py::object detail = py::module_::import("enoki").attr("detail");
        m_read_indices = detail.attr("read_indices");
        m_write_indices = detail.attr("write_indices");
        read_indices();

        m_vars = new uint32_t*[m_var_count];
        m_vars_phi = new uint32_t[m_var_count];
        for (uint32_t i = 0; i< m_var_count; ++i)
            m_vars[i] = &m_vars_val[i];
        m_cuda = Base::IsCUDA;
        m_llvm = Base::IsLLVM;

        Base::init();
        write_indices();
    }

    void read_indices() {
        py::list indices = m_read_indices(*m_args);

        if (m_var_count == 0) {
            m_var_count = indices.size();
            m_vars_val = new uint32_t[m_var_count];
        } else if (indices.size() != m_var_count) {
            ek::enoki_raise("enoki::Loop(): internal error: loop variable index changed!");
        }
        for (uint32_t i = 0; i< m_var_count; ++i)
            m_vars_val[i] = py::cast<uint32_t>(indices[i]);
    }

    void write_indices() {
        py::list list;
        for (size_t i = 0; i < m_var_count; ++i)
            list.append(m_vars_val[i]);
        m_write_indices(list, *m_args);
    }

    bool cond(const ek::mask_t<Guide> &mask) {
        read_indices();
        bool result = Base::cond(mask);
        write_indices();
        return result;
    }

    ~Loop() { delete[] m_vars_val; }

private:
    py::args m_args;
    py::object m_read_indices;
    py::object m_write_indices;
    uint32_t *m_vars_val = nullptr;
};
