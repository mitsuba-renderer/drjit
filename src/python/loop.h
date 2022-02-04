#pragma once

#include "common.h"
#include <drjit/loop.h>

template <typename Value> struct Loop : dr::Loop<Value> {
    using Base = dr::Loop<Value>;
    using Base::m_indices;
    using Base::m_indices_ad;
    using Base::m_ad_float_precision;
    using Base::m_name;
    using Base::m_state;

    Loop(const char *name, py::handle func) : Base(name) {
        py::object detail = py::module_::import("drjit").attr("detail");
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
        if (m_state)
            jit_raise("Loop(\"%s\"): was already initialized!",
                      m_name.get());

        process_state(false);

        py::int_ i0(0), i1(1), i2(2);
        for (size_t i = 0, size = m_state_py.size(); i < size; ++i) {
            py::object o = m_state_py[i];
            if (!py::isinstance<py::tuple>(o))
                continue;
            m_indices_py.push_back(py::cast<uint32_t>(o[i0]));
            m_indices_py_ad.push_back(py::cast<int32_t>(o[i1]));
            int ad_float_precision = py::cast<uint32_t>(o[i2]);
            if (ad_float_precision) {
                if (m_ad_float_precision == 0)
                    m_ad_float_precision = ad_float_precision;
                if (m_ad_float_precision != ad_float_precision)
                    jit_raise(
                        "Loop::init(): differentiable loop variables must "
                        "use the same floating point precision! (either "
                        "all single or all double precision)");
            }
        }

        for (size_t i = 0; i < m_indices_py.size(); ++i) {
            m_indices.push_back(&m_indices_py[i]);
            m_indices_ad.push_back(&m_indices_py_ad[i]);
        }

        Base::init();
        write_state();
    }

    bool operator()(const dr::mask_t<Value> &mask) {
        read_state();
        bool result = Base::operator()(mask);
        write_state();
        return result;
    }

private:
    void read_state() {
        process_state(false);

        py::int_ i0(0), i1(1);
        for (size_t i = 0, j = 0, size = m_state_py.size(); i < size; ++i) {
            py::object o = m_state_py[i];
            if (!py::isinstance<py::tuple>(o))
                continue;
            if (j >= m_indices_py.size()) {
                jit_raise("Loop(\"%s\"): must be initialized before "
                          "first loop iteration!", m_name.get());
            } else {
                m_indices_py[j] = py::cast<uint32_t>(o[i0]);
                m_indices_py_ad[j] = py::cast<int32_t>(o[i1]);
            }
            j++;
        }
    }

    void write_state() {
        py::int_ i0(0), i1(1);
        for (size_t i = 0, j = 0, size = m_state_py.size(); i < size; ++i) {
            py::object o = m_state_py[i];
            if (!py::isinstance<py::tuple>(o))
                continue;
            m_state_py[i] = py::make_tuple(m_indices_py[j], m_indices_py_ad[j], 0);
            j++;
        }

        process_state(true);
    }

    void process_state(bool write) {
        m_process_state(this, m_funcs, m_state_py, write);
    }

private:
    py::list m_funcs, m_state_py;
    py::object m_process_state;
    dr::dr_vector<uint32_t> m_indices_py;
    dr::dr_vector<uint32_t> m_indices_py_ad;
};
