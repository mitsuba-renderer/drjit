#pragma once

#include "common.h"
#include <enoki/loop.h>

template <typename Value> struct Loop : ek::Loop<Value> {
    using Base = ek::Loop<Value>;
    using Base::m_indices;
    using Base::m_indices_ad;
    using Base::m_name;

    Loop(const char *name, py::handle func) : Base(name) {
        py::object detail = py::module_::import("enoki").attr("detail");
        m_process_state = detail.attr("loop_process_state");
        if (!func.is_none()) {
            if (!py::isinstance<py::function>(func)) {
                jit_raise("Loop(\"%s\"): expected a lambda function as second "
                          "argument that returns the list of loop variables.",
                          name);
            } else {
                m_funcs.append(func);
                init();
            }
        }
    }

    void put(const py::function &func) { m_funcs.append(func); }

    void init() {
        if (m_indices.size() > 0)
            jit_raise("enoki::Loop(\"%s\"): was already initialized!",
                      m_name.get());

        process_state(false);

        py::int_ i0(0), i1(1);
        for (uint32_t i = 0, size = m_state.size(); i < size; ++i) {
            py::object o = m_state[i];
            if (!py::isinstance<py::tuple>(o))
                continue;
            m_indices_py.push_back(py::cast<uint32_t>(o[i0]));
            m_indices_py_ad.push_back(py::cast<int32_t>(o[i1]));
        }

        for (uint32_t i = 0; i < m_indices_py.size(); ++i) {
            m_indices.push_back(&m_indices_py[i]);
            m_indices_ad.push_back(&m_indices_py_ad[i]);
        }

        Base::init();
        write_state();
    }

    bool operator()(const ek::mask_t<Value> &mask) {
        read_state();
        bool result = Base::operator()(mask);
        write_state();
        return result;
    }

private:
    void read_state() {
        process_state(false);

        py::int_ i0(0), i1(1);
        for (uint32_t i = 0, j = 0, size = m_state.size(); i < size; ++i) {
            py::object o = m_state[i];
            if (!py::isinstance<py::tuple>(o))
                continue;
            m_indices_py[j] = py::cast<uint32_t>(o[i0]);
            m_indices_py_ad[j] = py::cast<int32_t>(o[i1]);
            j++;
        }
    }

    void write_state() {
        py::int_ i0(0), i1(1);
        for (uint32_t i = 0, j = 0, size = m_state.size(); i < size; ++i) {
            py::object o = m_state[i];
            if (!py::isinstance<py::tuple>(o))
                continue;
            m_state[i] = py::make_tuple(m_indices_py[j], m_indices_py_ad[j]);
            j++;
        }

        process_state(true);
    }

    void process_state(bool write) { m_process_state(m_funcs, m_state, write); }

private:
    py::list m_funcs, m_state;
    py::object m_process_state;
    ek::ek_vector<uint32_t> m_indices_py;
    ek::ek_vector<int32_t> m_indices_py_ad;
};
