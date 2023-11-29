/*
    autodiff.cpp -- Bindings for autodiff utility functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <drjit/autodiff.h>
#include <drjit/custom.h>
#include <nanobind/trampoline.h>
#include "autodiff.h"
#include "apply.h"
#include "meta.h"
#include "init.h"
#include "base.h"

static void set_grad_enabled(nb::handle h, bool enable_) {
    struct SetGradEnabled : TraverseCallback {
        bool enable;
        SetGradEnabled(bool enable) : enable(enable) { }

        void operator()(nb::handle h) override {
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

    SetGradEnabled sge(enable_);
    traverse("drjit.set_grad_enabled", sge, h);
}

static nb::object new_grad(nb::handle h) {
    struct NewGrad : TransformCallback {
        void operator()(nb::handle h1, nb::handle h2) override {
            nb::handle tp = h1.type();
            const ArraySupplement &s = supp(tp);
            uint32_t index = (uint32_t) s.index(inst_ptr(h1));

            if (s.is_diff && is_float(s)) {
                uint64_t new_index = ad_var_new(index);
                s.init_index(new_index, inst_ptr(h2));
                ad_var_dec_ref(new_index);
            } else {
                s.init_index(index, inst_ptr(h2));
            }
        }
    } ng;

    return transform("drjit.detail.new_grad", ng, h);
}

static void enable_grad(nb::handle h) { set_grad_enabled(h, true); }
static void disable_grad(nb::handle h) { set_grad_enabled(h, false); }
static void enable_grad_2(nb::args args) { enable_grad(args); }
static void disable_grad_2(nb::args args) { disable_grad(args); }

static bool grad_enabled(nb::handle h) {
    struct GradEnabled : TraverseCallback {
        bool result = false;

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s))
                result |= ad_grad_enabled(s.index(inst_ptr(h)));
        }
    };

    GradEnabled ge;
    traverse("drjit.grad_enabled", ge, h);
    return ge.result;
}

static bool grad_enabled_2(nb::args args) { return grad_enabled(args); }

static nb::object detach(nb::handle h, bool preserve_type_ = true) {
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

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s1 = supp(h1.type()),
                                  &s2 = supp(h2.type());
            s2.init_index((uint32_t) s1.index(inst_ptr(h1)), inst_ptr(h2));
        }
    };

    if ((is_drjit_array(h) && !supp(h.type()).is_diff) ||
        (preserve_type_ && !grad_enabled(h)))
        return nb::borrow(h);

    Detach d(preserve_type_);
    return transform("drjit.detach", d, h);
}

static nb::object grad(nb::handle h, bool preserve_type_ = true) {
    struct Grad : TransformCallback {
        bool preserve_type;
        Grad(bool preserve_type) : preserve_type(preserve_type) { }

        nb::handle transform_type(nb::handle tp) const override {
            ArrayMeta m = supp(tp);

            if (m.is_diff && !preserve_type) {
                m.is_diff = false;
                return meta_get_type(m);
            }

            return tp;
        }

        nb::object transform_unknown(nb::handle) const override {
            return nb::float_(0);
        }

        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s1 = supp(h1.type());
            if (!s1.backend) {
                nb::object o2 = full("zeros", h2.type(), nb::int_(0), nb::len(h1));
                nb::inst_replace_move(h2, o2);
                return;
            }

            const ArraySupplement &s2 = supp(h2.type());
            uint64_t index = s1.index(inst_ptr(h1));
            uint32_t grad_index = ad_grad(index);
            if (!grad_index) {
                nb::object o2 = full("zeros", h2.type(), nb::int_(0), 1);
                nb::inst_replace_move(h2, o2);
                return;
            }

            s2.init_index(grad_index, inst_ptr(h2));
            jit_var_dec_ref(grad_index);
        }
    };

    Grad g(preserve_type_);
    return transform("drjit.grad", g, h);
}

static void clear_grad(nb::handle dst) {
    struct ClearGrad : TraverseCallback {
        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s))
                ad_clear_grad(s.index(inst_ptr(h)));
        }
    } cg;

    traverse("drjit.clear_grad", cg, dst);
}

static void accum_grad(nb::handle target, nb::handle source) {
    struct SetGrad : TraversePairCallback {
        void operator()(nb::handle h1, nb::handle h2) override {
            const ArraySupplement &s1 = supp(h1.type()),
                                  &s2 = supp(h2.type());

            if (s1.is_diff && is_float(s1) && is_float(s2) &&
                s1.backend == s2.backend) {
                uint64_t i1 = s1.index(inst_ptr(h1)),
                         i2 = s2.index(inst_ptr(h2));

                ad_accum_grad(i1, (uint32_t) i2);
            }
        }
    } sg;

    nb::handle tp = target.type();

    nb::object o = nb::borrow(source);
    if (!o.type().is(tp))
        o = tp(o);

    traverse_pair("drjit.accum_grad", sg, target, o);
}

static nb::object replace_grad(nb::handle h0, nb::handle h1) {
    struct ReplaceGrad : TransformPairCallback {
        void operator()(nb::handle h1, nb::handle h2, nb::handle h3) override {
            const ArraySupplement &s = supp(h1.type());

            if (s.is_diff && is_float(s)) {
                dr::ArrayBase *p1 = inst_ptr(h1),
                              *p2 = inst_ptr(h2);

                size_t l1 = s.len(p1),
                       l2 = s.len(p2);

                if (l1 != l2)
                    nb::raise("incompatible input sizes (%zu and %zu).", l1, l2);

                uint64_t i1 = s.index(p1),
                         i2 = s.index(p2),
                         i3 = ((uint32_t) i1) | ((i2 >> 32) << 32);

                s.init_index(i3, inst_ptr(h3));
            }
        }
    } rg;

    nb::object o[2] = { borrow(h0), borrow(h1) };

    if (!o[0].type().is(o[1].type()))
        promote(o, 2);

    return transform_pair("drjit.replace_grad", rg, o[0], o[1]);
}


static void set_grad(nb::handle target, nb::handle source) {
  ::clear_grad(target);
  ::accum_grad(target, source);
}

static void enqueue_impl(dr::ADMode mode_, nb::handle h_) {
    struct Enqueue : TraverseCallback {
        dr::ADMode mode;
        Enqueue(dr::ADMode mode) : mode(mode) { }

        void operator()(nb::handle h) override {
            const ArraySupplement &s = supp(h.type());
            if (s.is_diff && is_float(s))
                ::ad_enqueue(mode, s.index(inst_ptr(h)));
        }
    };

    Enqueue e(mode_);
    traverse("drjit.enqueue", e, h_);
}

static bool check_grad_enabled(const char *name, nb::handle h, uint32_t flags) {
    bool rv = grad_enabled(h);
    if (!rv & !(flags & dr::ADFlag::AllowNoGrad))
        nb::raise(
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
        nb::gil_scoped_release r;
        ad_traverse(dr::ADMode::Forward, flags);
    }
}

static void backward_from(nb::handle_t<dr::ArrayBase> h, uint32_t flags) {
    if (check_grad_enabled("drjit.backward_from", h, flags)) {
        clear_grad(h);
        ::accum_grad(h, nb::float_(1.0));
        enqueue_impl(dr::ADMode::Backward, h);
        nb::gil_scoped_release r;
        ad_traverse(dr::ADMode::Backward, flags);
    }
}

static nb::object forward_to(nb::handle h, uint32_t flags) {
    if (check_grad_enabled("drjit.forward_to", h, flags)) {
        enqueue_impl(dr::ADMode::Backward, h);
        nb::gil_scoped_release r;
        ad_traverse(dr::ADMode::Forward, flags);
    }
    return grad(h, true);
}

static nb::object backward_to(nb::handle h, uint32_t flags) {
    if (check_grad_enabled("drjit.backward_to", h, flags)) {
        enqueue_impl(dr::ADMode::Forward, h);
        nb::gil_scoped_release r;
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

class PyCustomOp : public drjit::detail::CustomOpBase {
    NB_TRAMPOLINE(drjit::detail::CustomOpBase, 3);
public:
    PyCustomOp() = default;
    using ticket = nb::detail::ticket;

    nb::str type_name() const { return nb::inst_name(nb_trampoline.base()); }

    void forward() override {
        ticket t(nb_trampoline, "forward", false);
        if (t.key.is_valid())
            nb_trampoline.base().attr(t.key)();
        else
            nb::raise("%s.forward(): not implemented!", type_name().c_str());
    }

    void backward() override {
        ticket t(nb_trampoline, "backward", false);
        if (t.key.is_valid())
            nb_trampoline.base().attr(t.key)();
        else
            nb::raise("%s.backward(): not implemented!", type_name().c_str());
    }

    void eval(nb::args, nb::kwargs) {
        nb::raise("%s.eval(): not implemented!", type_name().c_str());
    }

    const char *name() const override {
        if (!m_name_cache.empty())
            return m_name_cache.c_str();

        ticket t(nb_trampoline, "name", false);
        if (t.key.is_valid())
            m_name_cache = nb::cast<const char *>(nb_trampoline.base().attr(t.key)());
        else
            m_name_cache = type_name().c_str();

        return m_name_cache.c_str();
    }

    nb::object add_generic(const char *name, nb::handle h, bool input_) {
        struct AddInOut : TransformCallback {
            dr::detail::CustomOpBase &op;
            bool input;

            AddInOut(dr::detail::CustomOpBase &op, bool input)
                : op(op), input(input) { }

            void operator()(nb::handle h1, nb::handle h2) override {
                const ArraySupplement &s = supp(h1.type());
                uint32_t ad_index = (uint32_t) (s.index(inst_ptr(h1)) >> 32);
                op.add_index((JitBackend) s.backend, ad_index, input);
                s.init_index(((uint64_t) ad_index) << 32, inst_ptr(h2));
            }
        };

        AddInOut aio(*this, input_);
        return transform(name, aio, h);
    }

    nb::object grad_in(nb::handle key) {
        nb::object result =
            nb::borrow(PyDict_GetItem(m_inputs.ptr(), key.ptr()));
        if (!result.is_valid())
            nb::raise("drjit.CustomOp.grad_in(): could not find an "
                      "input argument named \"%s\"", nb::str(key).c_str());
        return ::grad(result);
    }

    void set_grad_in(nb::handle key, nb::handle value) {
        nb::object result =
            nb::borrow(PyDict_GetItem(m_inputs.ptr(), key.ptr()));
        if (!result.is_valid())
            nb::raise("drjit.CustomOp.set_grad_in(): could not find an "
                      "input argument named \"%s\"", nb::str(key).c_str());
        accum_grad(result, value);
    }

    nb::object grad_out() { return ::grad(m_output); }
    void set_grad_out(nb::handle value) {
        accum_grad(m_output, value);
    }

    nb::object add_input_v(nb::handle h) {
        return add_generic("drjit.CustomOp.add_input", h, true);
    }

    nb::object add_output_v(nb::handle h) {
        return add_generic("drjit.CustomOp.add_output", h, false);
    }

    void add_input(nb::handle h) { (void) add_input_v(h); }
    void add_output(nb::handle h) { (void) add_output_v(h); }
    void set_input(nb::handle h) {
        m_inputs = nb::borrow<nb::dict>(add_input_v(h));
    }
    void set_output(nb::handle h) { m_output = add_output_v(h); }

private:
    mutable std::string m_name_cache;
    nb::dict m_inputs;
    nb::object m_output;
};

nb::object custom(nb::type_object_t<PyCustomOp> cls, nb::args args, nb::kwargs kwargs) {
    try {
        nb::object op = cls();

        auto eval_func = op.attr("eval"),
             eval_code = eval_func.attr("__code__"),
             eval_argc = eval_code.attr("co_argcount"),
             eval_argn = eval_code.attr("co_varnames");

        nb::object output = eval_func(*detach(args), **detach(kwargs));

        // Ensure that the output is registered with the AD layer without depending
        // on previous computation. That dependence is reintroduced later below.
        output = new_grad(output);

        size_t i = 0, argc = nb::cast<size_t>(eval_argc);
        for (nb::handle arg: args) {
            if (i + 1 < argc)
                kwargs[eval_argn[i + 1]] = arg;
            i++;
        }

        PyCustomOp *op_cpp = nb::cast<PyCustomOp *>(op);
        op_cpp->set_input(kwargs);
        op_cpp->set_output(output);

        if (!ad_custom_op(op_cpp))
            ::disable_grad(output);

        return output;
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::type_name(cls);
        nb::raise_from(e, PyExc_RuntimeError,
                       "drjit.custom(<%U>): error while performing a custom "
                       "differentiable operation. (see above).", tp_name.ptr());
    } catch (const std::exception &e) {
        nb::str tp_name = nb::type_name(cls);
        nb::chain_error(PyExc_RuntimeError,
                        "drjit.custom(<%U>): error while performing a custom "
                        "differentiable operation: %s.", tp_name.ptr(), e.what());
        nb::raise_python_error();
    }
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
     .def("enable_grad", &::enable_grad, doc_enable_grad)
     .def("enable_grad", &::enable_grad_2)
     .def("disable_grad", &::disable_grad, doc_disable_grad)
     .def("disable_grad", &disable_grad_2)
     .def("grad_enabled", &::grad_enabled, doc_grad_enabled)
     .def("grad_enabled", &::grad_enabled_2)
     .def("set_grad", &::set_grad, "target"_a, "source"_a, doc_set_grad)
     .def("accum_grad", &::accum_grad, "target"_a, "source"_a, doc_accum_grad)
     .def("clear_grad", &::clear_grad, doc_clear_grad)
     .def("replace_grad", &::replace_grad, doc_replace_grad)
     .def("grad", &::grad, "arg"_a, "preserve_type"_a = true, doc_grad)
     .def("detach", &::detach, "arg"_a, "preserve_type"_a = true, doc_detach)
     .def("enqueue", &enqueue_impl, "mode"_a, "arg"_a, doc_enqueue)
     .def("enqueue",
          [](dr::ADMode mode, nb::args args) {
              enqueue_impl(mode, args);
          }, "mode"_a, "args"_a)
     .def("traverse", &ad_traverse,
          "mode"_a, "flags"_a = dr::ADFlag::Default,
          doc_traverse)
     .def("forward_from", &::forward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_forward_from))
     .def("forward", &::forward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_forward))
     .def("backward_from", &::backward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_backward_from))
     .def("backward", &::backward_from,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_backward))
     .def("forward_to", &::forward_to,
          "arg"_a, "flags"_a = dr::ADFlag::Default,
          nb::raw_doc(doc_forward_to))
     .def("backward_to", &::backward_to,
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

    struct NullContextManager { };
    struct ADContextManager {
        drjit::ADScope scope;
        std::vector<uint64_t> indices;
    };

    nb::class_<NullContextManager>(detail, "NullContextManager")
        .def(nb::init<>())
        .def("__enter__", [](NullContextManager&) { })
        .def("__exit__", [](NullContextManager&, nb::handle, nb::handle, nb::handle) {
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

    detail.def("new_grad", &new_grad);

    nb::class_<PyCustomOp, nb::intrusive_base>(m, "CustomOp", doc_CustomOp)
        .def(nb::init<>())
        .def("forward", &PyCustomOp::forward, doc_CustomOp_forward)
        .def("backward", &PyCustomOp::backward, doc_CustomOp_backward)
        .def("name", &PyCustomOp::name, doc_CustomOp_name)
        .def("eval", &PyCustomOp::eval, doc_CustomOp_eval)
        .def("grad_in", &PyCustomOp::grad_in, doc_CustomOp_grad_in)
        .def("set_grad_in", &PyCustomOp::set_grad_in, doc_CustomOp_set_grad_in)
        .def("grad_out", &PyCustomOp::grad_out, doc_CustomOp_grad_out)
        .def("set_grad_out", &PyCustomOp::set_grad_out, doc_CustomOp_set_grad_out)
        .def("add_input", &PyCustomOp::add_input, doc_CustomOp_add_input)
        .def("add_output", &PyCustomOp::add_output, doc_CustomOp_add_output);

    m.def("custom", &custom, doc_custom);
}
