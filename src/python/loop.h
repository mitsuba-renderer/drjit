#pragma once

#include "common.h"
#include <enoki/loop.h>

// template <typename Guide> struct Loop : ek::Loop<Guide> {
//     using Base = ek::Loop<Guide>;
//     using Base::m_vars;
//     using Base::m_vars_phi;

//     Loop(py::args args) {
//         py::object detail = py::module_::import("enoki").attr("detail");
//         m_read_indices = detail.attr("read_indices");
//         m_write_indices = detail.attr("write_indices");
//         put(args);
//         init();
//     }

//     void put(py::args args) {
//         for (py::handle h : args)
//             m_args.append(h);
//     }

//     void init() {
//         py::list indices = m_read_indices(*m_args);
//         size_t size = indices.size();

//         for (uint32_t i = 0; i < size; ++i)
//             m_vars_py.push_back(py::cast<uint32_t>(indices[i]));

//         for (uint32_t i = 0; i < size; ++i) {
//             m_vars.push_back(&m_vars_py[i]);
//             m_vars_phi.push_back(0);
//         }

//         Base::init();
//         write_indices();
//     }

//     void read_indices() {
//         py::list indices = m_read_indices(*m_args);
//         size_t size = indices.size();

//         if (size != m_vars_py.size())
//             ek::enoki_raise("Loop::read_indices(): number of indices changed!");

//         for (uint32_t i = 0; i < size; ++i)
//             m_vars_py[i] = py::cast<uint32_t>(indices[i]);
//     }

//     void write_indices() {
//         py::list list;
//         for (size_t i = 0; i < m_vars_py.size(); ++i)
//             list.append(m_vars_py[i]);
//         m_write_indices(list, *m_args);
//     }

//     bool cond(const ek::mask_t<Guide> &mask) {
//         read_indices();
//         bool result = Base::cond(mask);
//         write_indices();
//         return result;
//     }

// private:
//     py::list m_args;
//     py::object m_read_indices;
//     py::object m_write_indices;
//     ek::detail::ek_vector<uint32_t> m_vars_py;
// };
