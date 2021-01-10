#pragma once

#include "common.h"
#include <enoki/loop.h>

template <typename Mask> struct Loop : ek::Loop<Mask> {
    using Base = ek::Loop<Mask>;
    using Base::m_index_p;
    using Base::m_index_in;
    using Base::m_invariant;

    Loop(const char *name, py::args args) : Base(name) {
        py::object detail = py::module_::import("enoki").attr("detail");
        m_read_indices  = detail.attr("read_indices");
        m_write_indices = detail.attr("write_indices");
        if (args.size() > 0) {
            for (py::handle h : args)
                put(h);
            init();
        }
    }

    void put(py::handle arg) {
        m_args.append(arg);
    }

    void init() {
        py::list indices = m_read_indices(*m_args);
        size_t size = indices.size();

        for (uint32_t i = 0; i < size; ++i)
            m_index_py.push_back(py::cast<uint32_t>(indices[i]));

        for (uint32_t i = 0; i < size; ++i) {
            m_index_p.push_back(&m_index_py[i]);
            m_index_in.push_back(m_index_py[i]);
            m_invariant.push_back(0);
        }

        Base::init();
        write_indices();
    }

    bool operator()(const Mask &mask) {
        read_indices();
        bool result = Base::operator()(mask);
        write_indices();
        return result;
    }

private:
    void read_indices() {
        py::list indices = m_read_indices(*m_args);
        size_t size = indices.size();

        if (size != m_index_py.size())
            ek::enoki_raise("Loop::read_indices(): number of indices changed!");

        for (uint32_t i = 0; i < size; ++i)
            m_index_py[i] = py::cast<uint32_t>(indices[i]);
    }

    void write_indices() {
        py::list list;
        for (size_t i = 0; i < m_index_py.size(); ++i)
            list.append(m_index_py[i]);
        m_write_indices(list, *m_args);
    }

private:
    py::list m_args;
    py::object m_read_indices;
    py::object m_write_indices;
    ek::ek_vector<uint32_t> m_index_py;
};
