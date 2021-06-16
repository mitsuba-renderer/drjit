#pragma once

#include "common.h"
#include <enoki/loop.h>

template <typename Value> struct Loop : ek::Loop<Value> {
    using Base = ek::Loop<Value>;
    using Base::m_index_p;
    using Base::m_index_p_ad;
    using Base::m_index_in;
    using Base::m_invariant;

    Loop(const char *name, py::handle func) : Base(name) {
        py::object detail = py::module_::import("enoki").attr("detail");
        m_read_indices  = detail.attr("read_indices");
        m_write_indices = detail.attr("write_indices");
        if (!func.is_none()) {
            if (!py::isinstance<py::function>(func)) {
                ek::enoki_raise(
                    "Loop(): expected a lambda function as second argument "
                    "that returns the list of loop variables.");
            } else {
                m_args.append(func);
                init();
            }
        }
    }

    void put(py::function func) {
        m_args.append(func);
    }

    void init() {
        py::list indices = m_read_indices(m_args);
        size_t size = indices.size();

        for (uint32_t i = 0; i < size; ++i) {
            py::tuple t = indices[i];
            if (t.size() != 2)
                ek::enoki_raise("Loop::read_indices(): invalid input!");
            m_index_py.push_back(py::cast<uint32_t>(t[0]));
            m_index_py_ad.push_back(py::cast<int32_t>(t[1]));
        }

        for (uint32_t i = 0; i < size; ++i) {
            m_index_p.push_back(&m_index_py[i]);
            m_index_p_ad.push_back(&m_index_py_ad[i]);
            m_index_in.push_back(m_index_py[i]);
            m_invariant.push_back(0);
        }

        Base::init();
        write_indices();
    }

    bool operator()(const ek::mask_t<Value> &mask) {
        read_indices();
        bool result = Base::operator()(mask);
        write_indices();
        return result;
    }

private:
    void read_indices() {
        py::list indices = m_read_indices(m_args);
        size_t size = indices.size();

        if (size != m_index_py.size())
            ek::enoki_raise("Loop::read_indices(): number of indices changed!");

        for (uint32_t i = 0; i < size; ++i) {
            py::tuple t = indices[i];
            if (t.size() != 2)
                ek::enoki_raise("Loop::read_indices(): invalid input!");
            m_index_py[i] = py::cast<uint32_t>(t[0]);
            m_index_py_ad[i] = py::cast<int32_t>(t[1]);
        }
    }

    void write_indices() {
        py::list list;
        for (size_t i = 0; i < m_index_py.size(); ++i)
            list.append(py::make_tuple(m_index_py[i], m_index_py_ad[i]));
        m_write_indices(list, m_args);
    }

private:
    py::list m_args;
    py::object m_read_indices;
    py::object m_write_indices;
    ek::ek_vector<uint32_t> m_index_py;
    ek::ek_vector<int32_t> m_index_py_ad;
};
