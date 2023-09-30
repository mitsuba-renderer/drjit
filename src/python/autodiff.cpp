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
                nb::object tmp = nb::inst_alloc(tp);
                uint64_t new_index;

                if (enable) {
                    new_index = ad_var_new((uint32_t) index);
                } else {
                    new_index = (uint32_t) index;
                    ad_var_inc_ref(new_index);
                }

                s.init_index(new_index, inst_ptr(tmp));
                ad_var_dec_ref(new_index);
                nb::inst_mark_ready(tmp);
                nb::inst_replace_move(h, tmp);
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
            if (s.is_diff && is_float(s))
                result |= ad_grad_enabled(s.index(inst_ptr(h)));
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
            s2.init_index((uint32_t) s1.index(inst_ptr(h1)), inst_ptr(h2));
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

static void clear_grad(nb::handle dst) {
    struct ClearGrad : TraverseCallback {
        void operator()(nb::handle h) const override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s))
                ad_clear_grad(s.index(inst_ptr(h)));
        }
    };

    traverse("drjit.clear_grad", ClearGrad{ }, dst);
}

static void accum_grad(nb::handle target, nb::handle source) {
    struct SetGrad : TraversePairCallback {
        void operator()(nb::handle h1, nb::handle h2) const override {
            const ArraySupplement &s1 = supp(h1.type()),
                                  &s2 = supp(h2.type());

            if (s1.is_diff && is_float(s1) && is_float(s2) &&
                s1.backend == s2.backend) {
                uint64_t i1 = s1.index(inst_ptr(h1)),
                         i2 = s2.index(inst_ptr(h2));

                ad_accum_grad(i1, (uint32_t) i2);
            }
        }
    };

    nb::handle tp = target.type();

    nb::object o = nb::borrow(source);
    if (!o.type().is(tp))
        o = tp(o);

    traverse_pair("drjit.accum_grad", SetGrad{ }, target, o);
}

static nb::object replace_grad(nb::handle h0, nb::handle h1) {
    struct ReplaceGrad : TransformPairCallback {
        void operator()(nb::handle h1, nb::handle h2, nb::handle h3) const override {
            const ArraySupplement &s = supp(h1.type());

            if (s.is_diff && is_float(s)) {
                uint64_t i1 = s.index(inst_ptr(h1)),
                         i2 = s.index(inst_ptr(h2)),
                         i3 = ((uint32_t) i1) | ((i2 >> 32) << 32);
                s.init_index(i3, inst_ptr(h3));
            }
        }
    };

    nb::object o[2] = { borrow(h0), borrow(h1) };

    if (!o[0].type().is(o[1].type()))
        promote(o, 2);

    return transform_pair("drjit.replace_grad", ReplaceGrad{ }, o[0], o[1]);
}


static void set_grad(nb::handle target, nb::handle source) {
  ::clear_grad(target);
  ::accum_grad(target, source);
}

static void enqueue_impl(dr::ADMode mode_, nb::handle h_) {
    struct Enqueue : TraverseCallback {
        dr::ADMode mode;
        Enqueue(dr::ADMode mode) : mode(mode) { }

        void operator()(nb::handle h) const override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s))
                ::ad_enqueue(mode, s.index(inst_ptr(h)));
        }
    };

    traverse("drjit.enqueue", Enqueue { mode_ }, h_);
}

static bool check_grad_enabled(const char *name, nb::handle h, uint32_t flags) {
    bool rv = grad_enabled(h);
    if (!rv & !(flags & dr::ADFlag::AllowNoGrad))
        nb::detail::raise(
            "%s(): the argument does not depend on the input variable(s) being "
            "differentiated. Raising an exception since this is usually "
            "indicative of a bug (for example, you may have forgotten to call "
            "dr.enable_grad(..)). If this is expected behavior, provide the "
            "drjit.ADFlag.AllowNoGrad flag to the function (e.g., by "
            "specifying flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad).",
            name);
    return rv;
}

static void forward_from(nb::handle_t<dr::ArrayBase> h, uint32_t flags) {
    if (check_grad_enabled("drjit.forward_from", h, flags)) {
        clear_grad(h);
        ::accum_grad(h, nb::float_(1.0));
        enqueue_impl(dr::ADMode::Forward, h);
        ad_traverse(dr::ADMode::Forward, flags);
    }
}

static void backward_from(nb::handle_t<dr::ArrayBase> h, uint32_t flags) {
    if (check_grad_enabled("drjit.backward_from", h, flags)) {
        clear_grad(h);
        ::accum_grad(h, nb::float_(1.0));
        enqueue_impl(dr::ADMode::Backward, h);
        ad_traverse(dr::ADMode::Backward, flags);
    }
}

static nb::object forward_to(nb::handle h, uint32_t flags) {
    if (check_grad_enabled("drjit.forward_to", h, flags)) {
        enqueue_impl(dr::ADMode::Backward, h);
        ad_traverse(dr::ADMode::Forward, flags);
    }
    return grad(h, true);
}

static nb::object backward_to(nb::handle h, uint32_t flags) {
    if (check_grad_enabled("drjit.backward_to", h, flags)) {
        enqueue_impl(dr::ADMode::Forward, h);
        ad_traverse(dr::ADMode::Backward, flags);
    }
    return grad(h, true);
}

static nb::object strip_tuple(nb::handle h) {
    return nb::len(h) == 1 ? h[0] : nb::borrow(h);
}

static nb::object forward_to_2(nb::args args, nb::kwargs kwargs) {
    uint32_t flags = (uint32_t) dr::ADFlag::Default;
    size_t nkwargs = nb::len(kwargs);

    if (nkwargs) {
        if (nkwargs != 1 || !kwargs.contains("flags") ||
            !nb::try_cast<uint32_t>(kwargs["flags"], flags))
            throw nb::type_error(
                "drjit.forward_to(): incompatible keyword arguments.");
    }

    return strip_tuple(forward_to(args, flags));
}

static nb::object backward_to_2(nb::args args, nb::kwargs kwargs) {
    uint32_t flags = (uint32_t) dr::ADFlag::Default;
    size_t nkwargs = nb::len(kwargs);

    if (nkwargs) {
        if (nkwargs != 1 || !kwargs.contains("flags") ||
            !nb::try_cast<uint32_t>(kwargs["flags"], flags))
            throw nb::type_error(
                "drjit.backward_to(): incompatible keyword arguments.");
    }

    return strip_tuple(backward_to(args, flags));
}

void collect_indices(nb::handle h, nb::list result_) {
    struct CollectIndices : TraverseCallback {
        nb::list &result;
        CollectIndices(nb::list &result) : result(result) { }

        void operator()(nb::handle h) const override {
            nb::handle tp = h.type();
            const ArraySupplement &s = supp(tp);
            if (!s.index)
                return;

            result.append(s.index(inst_ptr(h)));
        }
    };

    traverse("drjit.collect_indices", CollectIndices { result_ }, h);
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
        .value("AllowNoGrad", dr::ADFlag::AllowNoGrad, doc_ADFlag_AllowNoGrad)
        .value("Default", dr::ADFlag::Default, doc_ADFlag_Default);

    m.def("set_grad_enabled", &set_grad_enabled, doc_set_grad_enabled)
     .def("enable_grad", &enable_grad, doc_enable_grad)
     .def("enable_grad", &enable_grad_2)
     .def("disable_grad", &disable_grad, doc_disable_grad)
     .def("disable_grad", &disable_grad_2)
     .def("grad_enabled", &grad_enabled, doc_grad_enabled)
     .def("grad_enabled", &grad_enabled_2)
     .def("set_grad", &::set_grad, "target"_a, "source"_a, doc_set_grad)
     .def("accum_grad", &::accum_grad, "target"_a, "source"_a, doc_accum_grad)
     .def("clear_grad", &::clear_grad, doc_clear_grad)
     .def("replace_grad", &::replace_grad, doc_replace_grad)
     .def("grad", &grad, "arg"_a, "preserve_type"_a = true, doc_grad)
     .def("detach", &detach, "arg"_a, "preserve_type"_a = true, doc_detach)
     .def("enqueue", &enqueue_impl, "mode"_a, "arg"_a, doc_enqueue)
     .def("enqueue", [](dr::ADMode mode, nb::args args) { enqueue_impl(mode, args); }, "mode"_a, "args"_a)
     .def("traverse", &ad_traverse, "mode"_a, "flags"_a = dr::ADFlag::Default, doc_traverse)
     .def("forward_from", &forward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_forward_from))
     .def("forward", &forward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_forward))
     .def("backward_from", &backward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_backward_from))
     .def("backward", &backward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_backward))
     .def("forward_to", &forward_to,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_forward_to))
     .def("backward_to", &backward_to,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_backward_to))
     .def("forward_to", &forward_to_2, "args"_a, "kwargs"_a)
     .def("backward_to", &backward_to_2, "args"_a, "kwargs"_a);

    /// Internal context managers for drjit.isolate_grad(), drjit.suspend_grad(), etc.
    nb::module_ detail = nb::module_::import_("drjit.detail");

    nb::enum_<dr::ADScope>(detail, "ADScope")
        .value("Invalid", dr::ADScope::Invalid)
        .value("Suspend", dr::ADScope::Suspend)
        .value("Resume", dr::ADScope::Resume)
        .value("Isolate", dr::ADScope::Isolate);

    struct NoopContextManager { };
    struct ADContextManager {
        drjit::ADScope scope;
        std::vector<uint64_t> indices;
    };

    nb::class_<NoopContextManager>(detail, "NoopContextManager")
        .def(nb::init<>())
        .def("__enter__", [](NoopContextManager&) { })
        .def("__exit__", [](NoopContextManager&, nb::handle, nb::handle, nb::handle) {
             }, nb::arg().none(), nb::arg().none(), nb::arg().none());

    nb::class_<ADContextManager>(detail, "ADContextManager")
        .def(nb::init<dr::ADScope, std::vector<uint64_t>>())
        .def("__enter__",
             [](ADContextManager &m) {
                 ad_scope_enter(m.scope, m.indices.size(), m.indices.data());
             })
        .def("__exit__",
             [](ADContextManager &, nb::handle exc_type, nb::handle, nb::handle) {
                 ad_scope_leave(exc_type.is(nb::none()));
             }, nb::arg().none(), nb::arg().none(), nb::arg().none());

    detail.def("collect_indices", &collect_indices);
}
