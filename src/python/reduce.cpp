/*
    reduce.cpp -- Bindings for horizontal reduction operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "reduce.h"
#include "base.h"
#include "meta.h"
#include "shape.h"
#include "init.h"
#include <nanobind/stl/optional.h>

nb::object reduce(const char *name, ArrayOp op_id, nb::handle h,
                  std::optional<int> axis,
                  bool (*reduce_skip)(nb::handle),
                  nb::object (*reduce_init)(),
                  nb::object (*reduce_combine)(nb::handle, nb::handle)) {
    nb::handle tp = h.type();
    try {
        if (reduce_skip(tp))
            return nb::borrow(h);

        const ArraySupplement *s = nullptr;
        if (is_drjit_type(tp))
            s = &supp(tp);

        if (!axis) {
            if (s && s->is_tensor) {
                nb::object arr = nb::steal(s->tensor_array(h.ptr()));
                if (!arr.is_valid())
                    nb::detail::raise_python_error();

                nb::object o = reduce(name, op_id, arr, 0, reduce_skip,
                                      reduce_init, reduce_combine);
                if (!o.is_valid())
                    nb::detail::raise_python_error();

                return tp(o, nb::tuple());
            }

            nb::object o = nb::borrow(h);
            nb::handle tp_prev;
            do {
                tp_prev = tp;
                o = reduce(name, op_id, o, 0, reduce_skip,
                           reduce_init, reduce_combine);

                if (!o.is_valid())
                    nb::detail::raise_python_error();

                tp = o.type();
            } while (!tp_prev.is(tp));

            return o;
        }

        if (axis.value() != 0)
            nb::detail::raise(
                "reductions are currently limited to axis=0 or axis=None!");

        if (s) {
            void *op = s->op[(int) op_id];
            if (op == DRJIT_OP_NOT_IMPLEMENTED)
                nb::detail::raise_type_error(
                    "requires an arithmetic Dr.Jit array "
                    "or Python sequence as input.");

            if (op != DRJIT_OP_DEFAULT) {
                nb::object result = nb::inst_alloc(tp);
                ((ArraySupplement::UnaryOp) op)(inst_ptr(h), inst_ptr(result));
                nb::inst_mark_ready(result);
                return result;
            }

            if (s->is_tensor && s->tensor_shape(inst_ptr(h)).size() <= 1)
                return reduce(name, op_id, h, std::optional<int>(),
                              reduce_skip, reduce_init, reduce_combine);
        }

        nb::object result = reduce_init();
        size_t it = 0;
        for (nb::handle h2 : h) {
            if (it++ == 0)
                result = borrow(h2);
            else
                result = reduce_combine(result, h2);
        }

        return result;
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.%s(<%U>): failed (see above)!",
                        name, tp_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        nb::chain_error(PyExc_RuntimeError, "drjit.%s(<%U>): %s",
                        name, tp_name.ptr(), e.what());
    }

    return nb::object();
}

static nb::object dot(nb::handle h0, nb::handle h1) {
    try {
        size_t l0 = nb::len(h0),
               l1 = nb::len(h1),
               lr = std::max(l0, l1);

        if (l0 != l1 && l0 != 1 && l1 != 1)
            nb::detail::raise("invalid input array sizes (%zu and %zu)", l0, l1);

        bool use_fma = true;

        nb::handle tp0 = h0.type(), tp1 = h1.type();
        if (is_drjit_type(tp0)) {
            const ArraySupplement &s0 = supp(tp0);
            if (s0.ndim == 1 && s0.shape[0] == DRJIT_DYNAMIC)
                use_fma = false;
        }

        if (is_drjit_type(tp1)) {
            const ArraySupplement &s1 = supp(tp1);
            if (s1.ndim == 1 && s1.shape[0] == DRJIT_DYNAMIC)
                use_fma = false;
        }

        if (use_fma) {
            nb::object result = h0[0] * h1[0],
                       fma = array_module.attr("fma");
            for (size_t i = 1; i < lr; ++i)
                result = fma(h0[l0 == 1 ? 0 : i],
                             h1[l1 == 1 ? 0 : i], result);
            return result;
        } else {
            return sum(h0 * h1, 0);
        }
    } catch (nb::python_error &e) {
        nb::str tp0_name = nb::inst_name(h0),
                tp1_name = nb::inst_name(h1);

        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.dot(<%U>, <%U>): failed (see above)!",
                        tp0_name.ptr(), tp1_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp0_name = nb::inst_name(h0),
                tp1_name = nb::inst_name(h1);

        nb::chain_error(PyExc_RuntimeError, "drjit.dot(<%U>, <%U>): %s",
                        tp0_name.ptr(), tp1_name.ptr(), e.what());
    }

    return { };
}

nb::object all(nb::handle h, std::optional<int> axis) {
    return reduce(
        "all", ArrayOp::All, h, axis,
        [](nb::handle tp) { return tp.is(&PyBool_Type); },
        []() { return nb::borrow(Py_True); },
        [](nb::handle h1, nb::handle h2) { return h1 & h2; });
}

nb::object any(nb::handle h, std::optional<int> axis) {
    return reduce(
        "any", ArrayOp::Any, h, axis,
        [](nb::handle tp) { return tp.is(&PyBool_Type); },
        []() { return nb::borrow(Py_False); },
        [](nb::handle h1, nb::handle h2) { return h1 | h2; });
}

nb::object sum(nb::handle h, std::optional<int> axis) {
    return reduce(
        "sum", ArrayOp::Sum, h, axis,
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::int_(0); },
        [](nb::handle h1, nb::handle h2) { return h1 + h2; });
}

nb::object prod(nb::handle h, std::optional<int> axis) {
    return reduce(
        "prod", ArrayOp::Prod, h, axis,
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::int_(1); },
        [](nb::handle h1, nb::handle h2) { return h1 * h2; });
}

nb::object min(nb::handle h, std::optional<int> axis) {
    return reduce(
        "min", ArrayOp::Min, h, axis,
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::float_(INFINITY); },
        [](nb::handle h1, nb::handle h2) {
            return array_module.attr("minimum")(h1, h2);
        });
}

nb::object max(nb::handle h, std::optional<int> axis) {
    return reduce(
        "max", ArrayOp::Max, h, axis,
        [](nb::handle tp) { return tp.is(&PyLong_Type) || tp.is(&PyFloat_Type); },
        []() -> nb::object { return nb::float_(-INFINITY); },
        [](nb::handle h1, nb::handle h2) {
            return array_module.attr("maximum")(h1, h2);
        });
}

nb::object prefix_sum(nb::handle_t<dr::ArrayBase> h, bool exclusive, std::optional<int> axis) {
    nb::handle tp = h.type();
    try {
        const ArraySupplement &s = supp(tp);

        if (!axis)
            nb::detail::raise("the prefix sum reduction is not implemented for the axis=None case!");

        if (axis.value() != 0)
            nb::detail::raise("the prefix sum reduction are currently limited to axis=0!");

        void *op = s.op[(int) ArrayOp::PrefixSum];
        if (op == DRJIT_OP_NOT_IMPLEMENTED)
            nb::detail::raise_type_error(
                "requires an arithmetic Dr.Jit array as input!");

        if (op != DRJIT_OP_DEFAULT) {
            nb::object result = nb::inst_alloc(tp);
            ((ArraySupplement::PrefixSum) op)(inst_ptr(h), exclusive, inst_ptr(result));
            nb::inst_mark_ready(result);
            return result;
        }

        if (s.is_tensor && s.tensor_shape(inst_ptr(h)).size() <= 1) {
            nb::object arr = nb::steal(s.tensor_array(h.ptr()));
            nb::object result = prefix_sum(arr, exclusive, axis);
            return tp(result, shape(h));
        }

        dr_vector<size_t> shape;
        if (!shape_impl(h, shape))
            nb::detail::raise("input array is ragged!");

        nb::object result = full("zeros", tp, nb::int_(0), shape.size(),
                                 shape.data()),
                   accum = nb::int_(0);

        size_t it = 0;
        for (nb::handle h2 : h) {
            if (exclusive) {
                result[it] = accum;
                accum += h2;
            } else {
                accum += h2;
                result[it] = accum;
            }
            it++;
        }

        return result;
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(tp);
        e.restore();
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.prefix_sum(<%U>): failed (see above)!",
                        tp_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(tp);
        nb::chain_error(PyExc_RuntimeError, "drjit.prefix_sum(<%U>): %s",
                        tp_name.ptr(), e.what());
    }

    return nb::object();
}

void export_reduce(nb::module_ & m) {
    m.def("all", &all, "value"_a, "axis"_a.none() = 0, doc_all)
     .def("any", &any, "value"_a, "axis"_a.none() = 0, doc_any)
     .def("sum", &sum, "value"_a, "axis"_a.none() = 0, doc_sum)
     .def("prod", &prod, "value"_a, "axis"_a.none() = 0, doc_prod)
     .def("min", &min, "value"_a, "axis"_a.none() = 0, doc_min)
     .def("max", &max, "value"_a, "axis"_a.none() = 0, doc_max)
     .def("dot", &dot, doc_dot)
     .def("dot", [](double d1, double d2) { return d1 * d2; })
     .def("dot", [](Py_ssize_t d1, Py_ssize_t d2) { return d1 * d2; })
     .def("abs_dot",
          [](nb::handle h0, nb::handle h1) -> nb::object {
              return array_module.attr("abs")(
                  array_module.attr("dot")(h0, h1));
          }, doc_norm)
     .def("norm",
          [](nb::handle h) -> nb::object {
              return array_module.attr("sqrt")(
                  array_module.attr("dot")(h, h));
          }, doc_norm)
     .def("prefix_sum", &prefix_sum,
          "value"_a, "exclusive"_a = true,
          "axis"_a.none() = 0, doc_prefix_sum);
}
