/*
    autodiff.cpp -- implementation of AD related functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"
#include <nanobind/operators.h>
#include <drjit/jit.h>

static bool is_float(meta m) {
    VarType vt = (VarType) m.type;
    return vt == VarType::Float16 || vt == VarType::Float32 ||
           vt == VarType::Float64;
}

static nb::object detach(nb::handle h, bool preserve_type=true) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff) {
            if (s.meta.ndim == 1) {
                nb::object result;
                if (preserve_type) {
                    result = nb::inst_alloc(h.type());
                    s.op_ad_create(nb::inst_ptr<void>(h), 0, nb::inst_ptr<void>(result));
                } else {
                    meta result_meta = s.meta;
                    result_meta.is_diff = false;
                    result = nb::inst_alloc(drjit::detail::array_get(result_meta));
                    s.op_detach(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                }
                nb::inst_mark_ready(result);
                return result;
            } else {
                meta result_meta = s.meta;
                result_meta.is_diff &= preserve_type;
                nb::object result = nb::inst_alloc(drjit::detail::array_get(result_meta));
                nb::inst_zero(result);
                PySequenceMethods *sm  = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                PySequenceMethods *sm2 = ((PyTypeObject *) result.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid()) {
                        result.clear();
                        break;
                    }

                    nb::object v2 = detach(v, preserve_type);
                    if (!v2.is_valid() || sm2->sq_ass_item(result.ptr(), i, v2.ptr())) {
                        result.clear();
                        break;
                    }
                }
                return result;
            }
        }
    }

    nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
    if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
        if (!preserve_type) {
            PyErr_Format(PyExc_TypeError,
                         "detach(): it is required to preserve the input type when "
                         "detaching a custom struct object.");
            return nb::object();
        }

        nb::object result = h.type()();
        nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
        for (auto [k, v] : dstruct_dict)
            nb::setattr(result, k, detach(nb::getattr(h, k), preserve_type));
        return result;
    }

    return nb::borrow(h);
}

static void set_grad_enabled(nb::handle h, bool value) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                s.op_set_grad_enabled(nb::inst_ptr<void>(h), value);
            } else {
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid())
                        nb::detail::raise_python_error();
                    set_grad_enabled(v, value);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (nb::handle h2 : h)
            set_grad_enabled(h2, value);
    } else if (nb::isinstance<nb::mapping>(h)) {
        set_grad_enabled(nb::borrow<nb::mapping>(h).values(), value);
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
            for (auto [k, v] : dstruct_dict)
                set_grad_enabled(nb::getattr(h, k), value);
        }
    }
}

static void enable_grad(nb::args args) {
    for (nb::handle h : args)
        set_grad_enabled(h, true);
}

static void disable_grad(nb::args args) {
    for (nb::handle h : args)
        set_grad_enabled(h, false);
}

static bool grad_enabled(nb::handle h) {
    bool result = false;
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                return s.op_grad_enabled(nb::inst_ptr<void>(h));
            } else {
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid())
                        nb::detail::raise_python_error();
                    result |= grad_enabled(v);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (nb::handle h2 : h)
            result |= grad_enabled(h2);
    } else if (nb::isinstance<nb::mapping>(h)) {
        result = grad_enabled(nb::borrow<nb::mapping>(h).values());
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
            for (auto [k, v] : dstruct_dict)
                result |= grad_enabled(nb::getattr(h, k));
        }
    }
    return result;
}

static bool grad_enabled(nb::args args) {
    bool result = false;
    for (nb::handle h : args)
        result |= grad_enabled(h);
    return result;
}

static nb::object grad(nb::handle h) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        nb::object result = nb::inst_alloc(h.type());

        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                s.op_grad(nb::inst_ptr<void>(h), nb::inst_ptr<void>(result));
                nb::inst_mark_ready(result);
                return result;
            } else {
                nb::inst_zero(result);
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid()) {
                        result.clear();
                        break;
                    }

                    nb::object vr = grad(v);
                    if (!vr.is_valid() || sm->sq_ass_item(result.ptr(), i, vr.ptr())) {
                        result.clear();
                        break;
                    }
                }
                return result;
            }
        }
    }

    if (nb::isinstance<nb::sequence>(h)) {
        nb::list result;
        for (Py_ssize_t i = 0, l = nb::len(h); i < l; i++)
            result.append(grad(h[i]));
        return std::move(result);
    }


    if (nb::isinstance<nb::mapping>(h)) {
        nb::object result = nb::inst_alloc(h.type());
        nb::mapping m = nb::borrow<nb::mapping>(h);
        for (auto k : m.keys())
            result[k] = grad(m[k]);
        return result;
    }

    nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
    if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
        nb::object result = h.type()();
        nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
        for (auto [k, v] : dstruct_dict) {
            nb::setattr(result, k, grad(nb::getattr(h, k)));
        }
        return result;
    }

    nb::object result = nb::inst_alloc(h.type());
    nb::inst_zero(result);
    return result;
}

template <typename T>
static void grad_setter(nb::handle h, nb::handle value, T op) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                nb::object v2 = nb::borrow(value);
                if (h.type().ptr() != v2.type().ptr())
                    v2 = h.type()(v2);
                op(s, h, v2);
            } else {
                PySequenceMethods *sm  = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                PySequenceMethods *sm2 = ((PyTypeObject *) value.type().ptr())->tp_as_sequence;
                bool value_sq =
                    is_drjit_array(value) &&
                    nb::type_supplement<supp>(value.type()).meta.ndim > 1;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v  = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid())
                        nb::detail::raise_python_error();

                    nb::object v2 = nb::borrow(value);
                    if (value_sq) {
                        v2 = nb::steal(sm2->sq_item(v2.ptr(), i));
                        if (!v2.is_valid())
                            nb::detail::raise_python_error();
                    }
                    grad_setter(v, v2, op);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        if (nb::isinstance<nb::sequence>(value)) {
            if (nb::len(h) != nb::len(value)) {
                PyErr_Format(
                    PyExc_RuntimeError,
                    "grad_setter(): argument sizes are not matching (%i, %i)",
                    nb::len(h), nb::len(value));
                nb::detail::raise_python_error();
            }
            for (Py_ssize_t i = 0, l = nb::len(h); i < l; i++)
                grad_setter(h[i], value[i], op);
        } else {
            for (Py_ssize_t i = 0, l = nb::len(h); i < l; i++)
                grad_setter(h[i], value, op);
        }
    } else if (nb::isinstance<nb::mapping>(h)) {
        nb::mapping m = nb::borrow<nb::mapping>(h);
        for (auto k : m.keys()) {
            if (nb::isinstance<nb::mapping>(value))
                grad_setter(m[k], value[k], op);
            else
                grad_setter(m[k], value, op);
        }
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);

            nb::object dstruct_value = nb::getattr(value.type(), "DRJIT_STRUCT", nb::handle());
            if (dstruct_value.is_valid() && nb::isinstance<nb::dict>(dstruct_value)) {
                for (auto [k, v] : dstruct_dict)
                    grad_setter(nb::getattr(h, k), nb::getattr(value, k), op);
            } else {
                for (auto [k, v] : dstruct_dict)
                    grad_setter(nb::getattr(h, k), value, op);
            }
        }
    }
}


static void set_grad(nb::handle h, nb::handle value) {
    grad_setter(h, value, [](const supp &s, nb::handle h, nb::handle v2) {
        s.op_set_grad(nb::inst_ptr<void>(h), nb::inst_ptr<void>(v2));
    });
}

static void accum_grad(nb::handle h, nb::handle value) {
    grad_setter(h, value, [](const supp &s, nb::handle h, nb::handle v2) {
        s.op_accum_grad(nb::inst_ptr<void>(h), nb::inst_ptr<void>(v2));
    });
}

static nb::object replace_grad(nb::handle h0, nb::handle h1) {
    if (!(is_drjit_array(h0) && is_drjit_array(h1))) {
        PyErr_Format(PyExc_TypeError,
                     "replace_grad(): unsupported input types!");
        return nb::object();
    }

    const supp &s0 = nb::type_supplement<supp>(h0.type());
    const supp &s1 = nb::type_supplement<supp>(h1.type());

    if (!(s1.meta.is_diff && is_float(s0.meta) &&
          s0.meta.is_diff && is_float(s1.meta))) {
        PyErr_Format(PyExc_TypeError,
                     "replace_grad(): unsupported input types!");
        return nb::object();
    }

    meta result_meta = meta_promote(s0.meta, s1.meta);
    auto result_tp = drjit::detail::array_get(result_meta);

    // All arguments must be promoted to the same type first
    nb::object o0 = result_tp(h0);
    nb::object o1 = result_tp(h1);

    // Py_ssize_t l0 = len(o0.ptr()),
    //            l1 = len(o1.ptr());
    // size_t depth = s0.meta.ndim;

    // if (l0 != l1) {
    //     if (l0 == 1 && depth == 1) {
    //         // a = a + zero(ta, lb)
    //     } else if (l1 == 1 && depth == 1) {
    //         // b = b + zero(tb, la)
    //     } else {
    //         PyErr_Format(PyExc_TypeError,
    //                      "replace_grad(): input arguments have "
    //                      "incompatible sizes (%i vs %i)!", l0, l1);
    //         nb::detail::raise_python_error();
    //     }
    // }

    nb::object result = nb::inst_alloc(result_tp);
    if (s0.meta.ndim == 1) {
        s0.op_ad_create(nb::inst_ptr<void>(o0),
                        s1.op_index_ad(nb::inst_ptr<void>(o1)),
                        nb::inst_ptr<void>(result));
        nb::inst_mark_ready(result);
        return result;
    } else {
        nb::inst_zero(result);
        PySequenceMethods *sm = ((PyTypeObject *) result_tp.ptr())->tp_as_sequence;
        for (Py_ssize_t i = 0, l = sm->sq_length(o0.ptr()); i < l; ++i) {
            nb::object v0 = nb::steal(sm->sq_item(o0.ptr(), i));
            if (!v0.is_valid()) {
                result.clear();
                break;
            }

            nb::object v1 = nb::steal(sm->sq_item(o1.ptr(), i));
            if (!v1.is_valid()) {
                result.clear();
                break;
            }

            nb::object vr = replace_grad(v0, v1);
            if (!vr.is_valid() || sm->sq_ass_item(result.ptr(), i, vr.ptr())) {
                result.clear();
                break;
            }
        }
        return result;
    }
}

static void enqueue(drjit::ADMode mode, nb::handle h) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                s.op_enqueue(mode, nb::inst_ptr<void>(h));
            } else {
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0; i < sm->sq_length(h.ptr()); ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid())
                        nb::detail::raise_python_error();
                    ::enqueue(mode, v);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (auto h2 : h)
            ::enqueue(mode, h2);
    } else if (nb::isinstance<nb::mapping>(h)) {
        return ::enqueue(mode, nb::borrow<nb::mapping>(h).values());
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            for (auto [k, v] :  nb::borrow<nb::dict>(dstruct))
                ::enqueue(mode, v);
        }
    }
}

static void enqueue(drjit::ADMode mode, nb::args args) {
    for (nb::handle h : args)
        ::enqueue(mode, h);
}

static void traverse(nb::handle type, drjit::ADMode mode, drjit::ADFlag flags) {
    if (!type.is_type() || !is_drjit_type(type)) {
        PyErr_Format(PyExc_TypeError,
                     "traverse(): first argument should be a Dr.JIT type!");
        nb::detail::raise_python_error();
    }

    // Find leaf array type
    supp *s = (supp *) nb::detail::nb_type_supplement(type.ptr());
    while (s->meta.ndim > 1)
        s = (supp *) nb::detail::nb_type_supplement((PyObject *)s->value);

    if (!s->meta.is_diff) {
        PyErr_Format(PyExc_TypeError,
                     "traverse(): expected a differentiable array type!");
        nb::detail::raise_python_error();
    }

    s->op_traverse(mode, +flags);
}

static nb::object forward_to(nb::handle h, drjit::ADFlag flags) {
    enqueue(drjit::ADMode::Backward, h);
    traverse(h.type(), drjit::ADMode::Forward, flags);
    return grad(h);
}

static void forward_from(nb::handle h, drjit::ADFlag flags) {
    set_grad(h, PyFloat_FromDouble(1.0));
    enqueue(drjit::ADMode::Forward, h);
    traverse(h.type(), drjit::ADMode::Forward, flags);
}

static nb::object backward_to(nb::handle h, drjit::ADFlag flags) {
    enqueue(drjit::ADMode::Forward, h);
    traverse(h.type(), drjit::ADMode::Backward, flags);
    return grad(h);
}

static void backward_from(nb::handle h, drjit::ADFlag flags) {
    // Deduplicate components if `h` is a vector
    if (is_drjit_array(h) && nb::type_supplement<supp>(h).meta.ndim > 1)
        h += nb::handle(PyFloat_FromDouble(0.0));

    set_grad(h, PyFloat_FromDouble(1.0));
    enqueue(drjit::ADMode::Backward, h);
    traverse(h.type(), drjit::ADMode::Backward, flags);
}

static nb::object graphviz_ad(bool as_str = false) {
    nb::str string = nb::str("");

    const char *s = drjit::detail::ad_graphviz<drjit::LLVMArray<float>>();
    if (strlen(s) > 453)
        string = nb::str(string + nb::str(s));

    s = drjit::detail::ad_graphviz<drjit::LLVMArray<double>>();
    if (strlen(s) > 453)
        string = nb::str(string + nb::str(s));

#if defined(DRJIT_ENABLE_CUDA)
    s = drjit::detail::ad_graphviz<drjit::CUDAArray<float>>();
    if (strlen(s) > 453)
        string = nb::str(string + nb::str(s));

    s = drjit::detail::ad_graphviz<drjit::CUDAArray<double>>();
    if (strlen(s) > 453)
        string = nb::str(string + nb::str(s));
#endif

    if (as_str)
        return std::move(string);

    try {
        return nb::module_::import_("graphviz").attr("Source")(string);
    } catch (...) {
        throw nb::type_error(
            "drjit.graphviz_ad(): The 'graphviz' Python package not available! "
            "Install via 'python -m pip install graphviz'. Alternatively, "
            "you can call drjit.graphviz_str() function to obtain a string "
            "representation..");
    }
}

extern void bind_array_autodiff(nb::module_ m) {
    nb::enum_<dr::ADFlag>(m, "ADFlag", nb::is_arithmetic())
        .value("ClearNone", dr::ADFlag::ClearNone)
        .value("ClearEdges", dr::ADFlag::ClearEdges)
        .value("ClearInput", dr::ADFlag::ClearInput)
        .value("ClearInterior", dr::ADFlag::ClearInterior)
        .value("ClearVertices", dr::ADFlag::ClearVertices)
        .value("Default", dr::ADFlag::Default)
        .def(nb::self == nb::self)
        .def(nb::self | nb::self)
        .def(int() | nb::self)
        .def(nb::self & nb::self)
        .def(int() & nb::self)
        .def(+nb::self)
        .def(~nb::self);

    nb::enum_<dr::ADMode>(m, "ADMode")
        .value("Primal", dr::ADMode::Primal)
        .value("Forward", dr::ADMode::Forward)
        .value("Backward", dr::ADMode::Backward);

    m.def("ad_whos_str", &dr::ad_whos);
    m.def("ad_whos", []() { nb::print(dr::ad_whos()); });

    m.def("detach", &detach, "arg"_a, "preserve_type"_a=true, doc_detach);
    m.def("set_grad_enabled", &set_grad_enabled, "arg"_a, "value"_a, doc_set_grad_enabled);
    m.def("enable_grad",  &enable_grad,  "args"_a, doc_enable_grad);
    m.def("disable_grad", &disable_grad, "args"_a, doc_disable_grad);
    m.def("grad_enabled", nb::overload_cast<nb::args>(grad_enabled), doc_grad_enabled);
    m.def("grad", &grad, "arg"_a, doc_grad);
    m.def("set_grad",   &set_grad,   "arg"_a, "value"_a, doc_set_grad);
    m.def("accum_grad", &accum_grad, "arg"_a, "value"_a, doc_accum_grad);
    m.def("replace_grad", &replace_grad, "a"_a, "b"_a, doc_replace_grad);
    m.def("enqueue", nb::overload_cast<drjit::ADMode, nb::args>(enqueue), "mode"_a, "args"_a, doc_enqueue);
    m.def("traverse", &traverse, "type"_a, "mode"_a, "flags"_a=drjit::ADFlag::Default, doc_traverse);
    m.def("forward_to",    &forward_to,    "args"_a, "flags"_a=drjit::ADFlag::Default, doc_forward_to);
    m.def("forward_from",  &forward_from,  "args"_a, "flags"_a=drjit::ADFlag::Default, doc_forward_from);
    m.def("forward",       &forward_from,  "args"_a, "flags"_a=drjit::ADFlag::Default, doc_forward);
    m.def("backward_to",   &backward_to,   "args"_a, "flags"_a=drjit::ADFlag::Default, doc_backward_to);
    m.def("backward_from", &backward_from, "args"_a, "flags"_a=drjit::ADFlag::Default, doc_backward_from);
    m.def("backward",      &backward_from, "args"_a, "flags"_a=drjit::ADFlag::Default, doc_backward);
    m.def("graphviz_ad", &graphviz_ad, "as_str"_a=false, doc_graphviz_ad);
}
