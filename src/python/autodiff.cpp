/*
    autodiff.cpp -- Bindings for autodiff utility functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/autodiff.h>
#include <nanobind/operators.h>
#include "autodiff.h"
#include "apply.h"
#include "meta.h"
#include "base.h"

static void set_grad_enabled(nb::handle h, bool enable_) {
    struct SetGradEnabled : TraverseCallback {
        bool enable;
        SetGradEnabled(bool enable) : enable(enable) { }

        void operator()(nb::handle h) const override {
            nb::handle tp = h.type();
            const ArraySupplement &s = supp(tp);
            if (!s.is_diff || !is_float(s))
                return;

            uint64_t index = s.index(inst_ptr(h));
            bool grad_enabled = ((uint32_t) index) != index;

            if (enable != grad_enabled) {
                uint64_t new_index;
                if (enable)
                    new_index = ad_var_new((uint32_t) index);
                else
                    new_index = (uint32_t) index;
                jit_var_inc_ref((uint32_t) new_index);
                nb::object tmp = nb::inst_alloc(tp);
                s.init_index(new_index, inst_ptr(tmp));
                nb::inst_mark_ready(tmp);
                nb::inst_replace_move(h, tmp);
                ad_var_dec_ref(new_index);
            }
        }
    };

    traverse("drjit.set_grad_enabled", SetGradEnabled{ enable_ }, h);
}

static void enable_grad(nb::handle h) { set_grad_enabled(h, true); }
static void disable_grad(nb::handle h) { set_grad_enabled(h, false); }
static void enable_grad_2(nb::args args) { enable_grad(args); }
static void disable_grad_2(nb::args args) { disable_grad(args); }

static bool grad_enabled(nb::handle h) {
    bool result_ = false;

    struct GradEnabled : TraverseCallback {
        bool &result;
        GradEnabled(bool &result) : result(result) { }

        void operator()(nb::handle h) const override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s)) {
                uint64_t index = s.index(inst_ptr(h));
                result |= ((uint32_t) index) != index;
            }
        }
    };

    traverse("drjit.grad_enabled", GradEnabled{ result_ }, h);
    return result_;
}

static bool grad_enabled_2(nb::args args) { return grad_enabled(args); }

static nb::object detach(nb::handle h, bool preserve_type_) {
    struct Detach : TransformCallback {
        bool preserve_type;
        Detach(bool preserve_type) : preserve_type(preserve_type) { }

        nb::handle transform_type(nb::handle tp) const override {
            ArrayMeta m = supp(tp);

            if (!preserve_type) {
                m.is_diff = false;
                return meta_get_type(m);
            } else {
                return tp;
            }
        }

        void operator()(nb::handle h1, nb::handle h2) const override {
            const ArraySupplement &s1 = supp(h1.type()),
                                  &s2 = supp(h2.type());
            uint64_t index = s1.index(inst_ptr(h1));
            s2.init_index((uint32_t) index, inst_ptr(h2));
        }
    };

    if ((is_drjit_array(h) && !supp(h.type()).is_diff) ||
        (preserve_type_ && !grad_enabled(h)))
        return nb::borrow(h);

    return transform("drjit.detach", Detach{ preserve_type_ }, h);
}

static nb::object grad(nb::handle h, bool preserve_type_) {
    struct Grad : TransformCallback {
        bool preserve_type;
        Grad(bool preserve_type) : preserve_type(preserve_type) { }

        nb::handle transform_type(nb::handle tp) const override {
            ArrayMeta m = supp(tp);

            if (!m.is_diff)
                return nb::handle();

            if (!preserve_type) {
                m.is_diff = false;
                return meta_get_type(m);
            } else {
                return tp;
            }
        }

        void operator()(nb::handle h1, nb::handle h2) const override {
            const ArraySupplement &s1 = supp(h1.type()),
                                  &s2 = supp(h2.type());
            uint64_t index = s1.index(inst_ptr(h1));
            uint32_t grad_index = ad_grad(index);
            s2.init_index(grad_index, inst_ptr(h2));
            jit_var_dec_ref(grad_index);
        }
    };

    return transform("drjit.grad", Grad{ preserve_type_ }, h);
}

static void set_grad(nb::handle dst, nb::handle src_, bool accum_) {
    nb::handle tp = dst.type();

    nb::object src = nb::borrow(src_);
    if (!src.type().is(tp))
        src = tp(src);

    struct SetGrad : TraversePairCallback {
        bool accum;
        SetGrad(bool accum) : accum(accum) { }

        void operator()(nb::handle h1, nb::handle h2) const override {
            const ArraySupplement &s1 = supp(h1.type()),
                                  &s2 = supp(h2.type());

            if (!s1.is_diff || !is_float(s1))
                nb::detail::raise("Input must be a differentiable Dr.Jit type.");

            uint64_t i1 = s1.index(inst_ptr(h1)),
                     i2 = s2.index(inst_ptr(h2));

            ad_set_grad(i1, (uint32_t) i2, accum);
        }
    };

    traverse_pair("drjit.set_grad", SetGrad{ accum_ }, dst, src);
}

static void enqueue(dr::ADMode mode_, nb::handle h_) {
    struct Enqueue : TraverseCallback {
        dr::ADMode mode;
        Enqueue(dr::ADMode mode) : mode(mode) { }

        void operator()(nb::handle h) const override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s))
                ad_enqueue(mode, s.index(inst_ptr(h)));
        }
    };

    traverse("drjit.enqueue", Enqueue { mode_ }, h_);
}

static void check_grad_enabled(const char *name, nb::handle h) {
    nb::handle tp = h.type();

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.is_diff && is_float(s)) {
            if (!grad_enabled(h)) {
                nb::detail::raise_type_error(
                            "%s(): the argument does not depend on the input "
                            "variable(s) being differentiated. Raising an exception "
                            "since this is usually indicative of a bug (for example, "
                            "you may have forgotten to call dr.enable_grad(..)). If "
                            "this is expected behavior, skip the call to %s(..) "
                            "when dr.grad_enabled(..) returns False.", name, name);
            }
        } else {
            nb::str tp_name = nb::type_name(tp);
            nb::detail::raise_type_error(
                "%s(): expected a differentiable array type (got %s)!", name,
                tp_name.c_str());
        }
    } else if (tp.is(&PyTuple_Type)) {
        for (nb::handle v : nb::borrow<nb::tuple>(h))
            check_grad_enabled(name, v);
    } else {
        nb::str tp_name = nb::type_name(tp);
        nb::detail::raise_type_error(
            "%s(): expected a Dr.Jit array type (got %s)!", name,
            tp_name.c_str());
    }
}

static void forward_from(nb::handle h, uint32_t flags) {
    check_grad_enabled("drjit.forward_from", h);
    set_grad(h, nb::float_(1.0), false);
    enqueue(drjit::ADMode::Forward, h);
    ad_traverse(drjit::ADMode::Forward, flags);
}

static void backward_from(nb::handle h, uint32_t flags) {
    check_grad_enabled("drjit.backward_from", h);
    set_grad(h, nb::float_(1.0), false);
    enqueue(drjit::ADMode::Backward, h);
    ad_traverse(drjit::ADMode::Backward, flags);
}

static nb::object forward_to(nb::handle h, uint32_t flags) {
    check_grad_enabled("drjit.forward_to", h);
    enqueue(drjit::ADMode::Backward, h);
    traverse(drjit::ADMode::Forward, flags);
    return grad(h, true);
}

static nb::object backward_to(nb::handle h, uint32_t flags) {
    check_grad_enabled("drjit.backward_to", h);
    enqueue(drjit::ADMode::Forward, h);
    traverse(drjit::ADMode::Backward, flags);
    return grad(h, true);
}

static nb::object forward_to_2(nb::args args, nb::kwargs kwargs) {
    uint32_t flags = (uint32_t) dr::ADFlag::Default;
    size_t nkwargs = nb::len(kwargs);

    if (nkwargs) {
        if (nkwargs != 1 || !kwargs.contains("flags") ||
            !nb::try_cast<uint32_t>(kwargs, flags))
            throw nb::type_error(
                "drjit.forward_to(): incompatible keyword arguments.");
    }

    return forward_to(args, flags);
}

static nb::object backward_to_2(nb::args args, nb::kwargs kwargs) {
    uint32_t flags = (uint32_t) dr::ADFlag::Default;
    size_t nkwargs = nb::len(kwargs);

    if (nkwargs) {
        if (nkwargs != 1 || !kwargs.contains("flags") ||
            !nb::try_cast<uint32_t>(kwargs, flags))
            throw nb::type_error(
                "drjit.backward_to(): incompatible keyword arguments.");
    }

    return backward_to(args, flags);
}

void export_autodiff(nb::module_ &m) {
    nb::enum_<dr::ADMode>(m, "ADMode", doc_ADMode)
        .value("Primal", dr::ADMode::Primal, doc_ADMode_Primal)
        .value("Forward", dr::ADMode::Forward, doc_ADMode_Forward)
        .value("Backward", dr::ADMode::Backward, doc_ADMode_Backward);

    nb::enum_<dr::ADFlag>(m, "ADFlag", nb::is_arithmetic(), doc_ADFlag)
        .value("ClearNone", dr::ADFlag::ClearNone, doc_ADFlag_ClearNone)
        .value("ClearEdges", dr::ADFlag::ClearEdges, doc_ADFlag_ClearEdges)
        .value("ClearInput", dr::ADFlag::ClearInput, doc_ADFlag_ClearInput)
        .value("ClearInterior", dr::ADFlag::ClearInterior, doc_ADFlag_ClearInterior)
        .value("ClearVertices", dr::ADFlag::ClearVertices, doc_ADFlag_ClearVertices)
        .value("Default", dr::ADFlag::Default, doc_ADFlag_Default)
        .def(nb::self == nb::self)
        .def(nb::self | nb::self)
        .def(int() | nb::self)
        .def(nb::self & nb::self)
        .def(int() & nb::self)
        .def(+nb::self)
        .def(~nb::self);

    m.def("set_grad_enabled", &set_grad_enabled, doc_set_grad_enabled)
     .def("enable_grad", &enable_grad, doc_enable_grad)
     .def("enable_grad", &enable_grad_2)
     .def("disable_grad", &disable_grad, doc_disable_grad)
     .def("disable_grad", &disable_grad_2)
     .def("grad_enabled", &grad_enabled, doc_grad_enabled)
     .def("grad_enabled", &grad_enabled_2)
     .def("set_grad",
          [](nb::handle dst, nb::handle src) { set_grad(dst, src, false); },
          "dst"_a, "src"_a, doc_set_grad)
     .def("accum_grad",
          [](nb::handle dst, nb::handle src) { set_grad(dst, src, true); },
          "dst"_a, "src"_a, doc_accum_grad)
     .def("grad", &grad, "arg"_a, "preserve_type"_a = true, doc_grad)
     .def("detach", &detach, "arg"_a, "preserve_type"_a = true, doc_detach)
     .def("enqueue", &enqueue, "mode"_a, "arg"_a, doc_enqueue)
     .def("enqueue", [](dr::ADMode mode, nb::args args) { enqueue(mode, args); }, "mode"_a, "args"_a)
     .def("traverse", &ad_traverse, "mode"_a, "flags"_a = drjit::ADFlag::Default, doc_traverse)
     .def("forward_from", &forward_from,
          "arg"_a, "flags"_a = drjit::ADFlag::Default,
          nb::raw_doc(doc_forward_from))
     .def("backward_from", &backward_from,
          "arg"_a, "flags"_a = drjit::ADFlag::Default,
          nb::raw_doc(doc_backward_from))
     .def("forward_to", &forward_to,
          "arg"_a, "flags"_a = drjit::ADFlag::Default,
          nb::raw_doc(doc_forward_to))
     .def("backward_to", &backward_to,
          "arg"_a, "flags"_a = drjit::ADFlag::Default,
          nb::raw_doc(doc_backward_to))
     .def("forward_to", &forward_to_2, "args"_a, "kwargs"_a)
     .def("backward_to", &backward_to_2, "args"_a, "kwargs"_a);

    m.attr("forward") = m.attr("forward_from");
    m.attr("backward") = m.attr("backward_from");
}
