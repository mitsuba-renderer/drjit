/*
    autodiff.cpp -- implementation of AD related functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"
#include <nanobind/operators.h>
#include <nanobind/trampoline.h>
#include <drjit/jit.h>

static nb::object detach(nb::handle h, bool preserve_type=true) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff) {
            if (s.meta.ndim == 1) {
                nb::object result;
                if (preserve_type) {
                    result = nb::inst_alloc(h.type());
                    s.op_ad_create(nb::inst_ptr<void>(h), 0, nb::inst_ptr<void>(result), false);
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
                result_meta.is_diff = preserve_type;
                nb::object result = nb::inst_alloc(drjit::detail::array_get(result_meta));
                nb::inst_zero(result);

                PySequenceMethods *sm  = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                PySequenceMethods *sm2 = ((PyTypeObject *) result.type().ptr())->tp_as_sequence;

                const supp &s2 = nb::type_supplement<supp>(result.type());
                if (s2.meta.shape[0] == DRJIT_DYNAMIC)
                    s2.init(nb::inst_ptr<void>(result), sm->sq_length(h.ptr()));

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

        return nb::borrow(h);
    }

    if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        nb::list result;
        for (size_t i = 0, l = nb::len(h); i < l; i++)
            result.append(detach(h[i], preserve_type));
        return std::move(result);
    }

    if (nb::isinstance<nb::dict>(h)) {
        nb::dict result;
        for (auto [k, v] : nb::borrow<nb::dict>(h))
            result[k] = detach(v, preserve_type);
        return result;
    }

    nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
    if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
        if (!preserve_type) {
          PyErr_Format(PyExc_TypeError,
                       "detach(): preserve_type=True is required when "
                       "detaching custom data structures.");
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
        if (s.meta.is_diff && is_float_v(h.type())) {
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
    } else if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        for (nb::handle h2 : h)
            set_grad_enabled(h2, value);
    } else if (nb::isinstance<nb::dict>(h)) {
        set_grad_enabled(nb::borrow<nb::dict>(h).values(), value);
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
        if (s.meta.is_diff && is_float_v(h.type())) {
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
    } else if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        for (nb::handle h2 : h)
            result |= grad_enabled(h2);
    } else if (nb::isinstance<nb::dict>(h)) {
        result = grad_enabled(nb::borrow<nb::dict>(h).values());
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

static nb::object grad(nb::handle h, bool preserve_type=true) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float_v(h.type())) {
            if (s.meta.ndim == 1) {
                meta grad_meta = s.meta;
                grad_meta.is_diff = false;
                nb::object gradient = nb::inst_alloc(drjit::detail::array_get(grad_meta));
                s.op_grad(nb::inst_ptr<void>(h), nb::inst_ptr<void>(gradient));
                nb::inst_mark_ready(gradient);

                if (preserve_type) {
                    nb::object result = nb::inst_alloc(h.type());
                    s.op_ad_create(nb::inst_ptr<void>(gradient), 0, nb::inst_ptr<void>(result), true);
                    nb::inst_mark_ready(result);
                    return result;
                } else {
                    return gradient;
                }
            } else {
                meta result_meta = s.meta;
                result_meta.is_diff = preserve_type;
                nb::object result = nb::inst_alloc(drjit::detail::array_get(result_meta));
                nb::inst_zero(result);

                PySequenceMethods *sm  = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                PySequenceMethods *sm2 = ((PyTypeObject *) result.type().ptr())->tp_as_sequence;

                const supp &s2 = nb::type_supplement<supp>(result.type());
                if (s2.meta.shape[0] == DRJIT_DYNAMIC)
                    s2.init(nb::inst_ptr<void>(result), sm->sq_length(h.ptr()));

                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid()) {
                        result.clear();
                        break;
                    }

                    nb::object vr = grad(v, preserve_type);
                    if (!vr.is_valid() || sm2->sq_ass_item(result.ptr(), i, vr.ptr())) {
                        result.clear();
                        break;
                    }
                }
                return result;
            }
        }
    }

    if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        nb::list result;
        for (size_t i = 0, l = nb::len(h); i < l; i++)
            result.append(grad(h[i], preserve_type));
        return std::move(result);
    }

    if (nb::isinstance<nb::dict>(h)) {
        nb::dict result;
        for (auto [k, v] : nb::borrow<nb::dict>(h))
            result[k] = grad(v, preserve_type);
        return std::move(result);
    }

    nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
    if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
        if (!preserve_type) {
            PyErr_Format(PyExc_TypeError,
                         "grad(): preserve_type=True is required when getting "
                         "the gradient of a custom data structures.");
            return nb::object();
        }
        nb::object result = h.type()();
        nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
        for (auto [k, v] : dstruct_dict)
            nb::setattr(result, k, grad(nb::getattr(h, k), preserve_type));
        return result;
    }

    return h.type()();
}

template <typename T>
static void grad_setter(nb::handle h, nb::handle value, T op) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float_v(h.type())) {
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
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
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
    } else if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        if (nb::isinstance<nb::sequence>(value)) {
            if (nb::len(h) != nb::len(value)) {
                PyErr_Format(
                    PyExc_RuntimeError,
                    "grad_setter(): argument sizes are not matching (%i, %i)",
                    nb::len(h), nb::len(value));
                nb::detail::raise_python_error();
            }
            for (size_t i = 0, l = nb::len(h); i < l; i++)
                grad_setter(h[i], value[i], op);
        } else {
            for (size_t i = 0, l = nb::len(h); i < l; i++)
                grad_setter(h[i], value, op);
        }
    } else if (nb::isinstance<nb::dict>(h)) {
        if (nb::isinstance<nb::dict>(value)) {
            for (auto [k, v] : nb::borrow<nb::dict>(h))
                grad_setter(v, value[k], op);
        } else {
            for (auto [k, v] : nb::borrow<nb::dict>(h))
                grad_setter(v, value, op);
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
    if (!is_drjit_array(h1)) {
        PyErr_Format(PyExc_TypeError,
                     "replace_grad(): unsupported input types!");
        return nb::object();
    }

    const supp &s1 = nb::type_supplement<supp>(h1.type());
    if (!(s1.meta.is_diff && is_float_v(h1.type()))) {
        PyErr_Format(PyExc_TypeError,
                     "replace_grad(): unsupported input types!");
        return nb::object();
    }

    nb::object o0, o1;

    // All arguments must be promoted to the same type first
    if (Py_TYPE(h0.ptr()) == Py_TYPE(h1.ptr())) {
        o0 = nb::borrow(h0);
        o1 = nb::borrow(h1);
    } else {
        PyObject *o[2] = { h0.ptr(), h1.ptr() };
        if (!promote("replace_grad", o, 2))
            return nb::object();
        o0 = nb::steal(o[0]);
        o1 = nb::steal(o[1]);
    }

    const supp &s = nb::type_supplement<supp>(o0.type());

    nb::object result = nb::inst_alloc(o0.type());
    if (s.meta.ndim == 1) {
        s.op_ad_create(nb::inst_ptr<void>(o0),
                       s.op_index_ad(nb::inst_ptr<void>(o1)),
                       nb::inst_ptr<void>(result),
                       false);
        nb::inst_mark_ready(result);
        return result;
    } else {
        nb::inst_zero(result);
        PySequenceMethods *sm = ((PyTypeObject *) result.type().ptr())->tp_as_sequence;

        if (s.meta.shape[0] == DRJIT_DYNAMIC)
            s.init(nb::inst_ptr<void>(result), sm->sq_length(o0.ptr()));

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
        if (s.meta.is_diff && is_float_v(h.type())) {
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
    } else if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        for (auto h2 : h)
            ::enqueue(mode, h2);
    } else if (nb::isinstance<nb::dict>(h)) {
        return ::enqueue(mode, nb::borrow<nb::dict>(h).values());
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

    type = leaf_array_t(type);
    const supp &s = nb::type_supplement<supp>(type);

    if (!s.meta.is_diff) {
        PyErr_Format(PyExc_TypeError,
                     "traverse(): expected a differentiable array type!");
        nb::detail::raise_python_error();
    }

    s.op_traverse(mode, +flags);
}

static void _check_grad_enabled(const char *name, nb::handle tp, nb::handle a) {
    if (is_drjit_type(tp)) {
        const supp &s = nb::type_supplement<supp>(tp);
        if (s.meta.is_diff && is_float_v(tp)) {
            if (!::grad_enabled(a)) {
                PyErr_Format(PyExc_TypeError,
                            "%s(): the argument does not depend on the input "
                            "variable(s) being differentiated. Raising an exception "
                            "since this is usually indicative of a bug (for example, "
                            "you may have forgotten to call ek.enable_grad(..)). If "
                            "this is expected behavior, skip the call to %s(..) "
                            "if ek.grad_enabled(..) returns False.", name, name);
                nb::detail::raise_python_error();
            }
        } else {
            PyErr_Format(PyExc_TypeError,
                            "%s(): expected a differentiable array type!", name);
            nb::detail::raise_python_error();
        }
    } else {
        PyErr_Format(PyExc_TypeError,
                        "%s(): expected a Dr.JIT array type!", name);
        nb::detail::raise_python_error();
    }
}

static drjit::ADFlag _ad_flags_from_kwargs(const char *name, nb::args args, nb::kwargs kwargs) {
    drjit::ADFlag flags = drjit::ADFlag::Default;
    if (nb::len(kwargs) > 1) {
        PyErr_Format(PyExc_TypeError,
                     "%s(): only AD flags should be passed via the "
                     "'flags=..' keyword argument!", name);
        nb::detail::raise_python_error();
    }

    if (nb::len(kwargs) == 1) {
        try {
            flags = nb::cast<drjit::ADFlag>(kwargs["flags"]);
        } catch (...) {
            PyErr_Format(PyExc_TypeError,
                         "%s(): only AD flags should be passed via the "
                         "'flags=..' keyword argument!", name);
            nb::detail::raise_python_error();
        }
    }

    for (auto h : args) {
        if (nb::isinstance<drjit::ADFlag>(h)) {
            PyErr_Format(PyExc_TypeError,
                         "%s(): AD flags should be passed via the "
                         "'flags=..' keyword argument!", name);
            nb::detail::raise_python_error();
        }
    }

    return flags;
}

static nb::object forward_to(nb::args args, nb::kwargs kwargs) {
    drjit::ADFlag flags = _ad_flags_from_kwargs("forward_to", args, kwargs);
    nb::handle tp = leaf_array_t(args);
    _check_grad_enabled("forward_to", tp, args);
    enqueue(drjit::ADMode::Backward, args);
    traverse(tp, drjit::ADMode::Forward, flags);

    if (nb::len(args) == 1)
        return grad(args[0]);
    else
        return grad(args);
}

static void forward_from(nb::handle h, drjit::ADFlag flags) {
    _check_grad_enabled("forward_from", leaf_array_t(h), h);
    set_grad(h, PyFloat_FromDouble(1.0));
    enqueue(drjit::ADMode::Forward, h);
    traverse(h.type(), drjit::ADMode::Forward, flags);
}

static nb::object backward_to(nb::args args, nb::kwargs kwargs) {
    drjit::ADFlag flags = _ad_flags_from_kwargs("backward_to", args, kwargs);
    nb::handle tp = leaf_array_t(args);
    _check_grad_enabled("backward_to", tp, args);
    enqueue(drjit::ADMode::Forward, args);
    traverse(tp, drjit::ADMode::Backward, flags);
    return grad(args);
}

static void backward_from(nb::handle h, drjit::ADFlag flags) {
    _check_grad_enabled("backward_from", leaf_array_t(h), h);

    // Deduplicate components if `h` is a vector
    const supp &s = nb::type_supplement<supp>(h.type());
    if (s.meta.ndim > 1) {
        PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
        for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
            nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
            if (!v.is_valid())
                nb::detail::raise_python_error();
            v = v + nb::handle(PyFloat_FromDouble(0.0));
            if (sm->sq_ass_item(h.ptr(), i, v.ptr()))
                nb::detail::raise_python_error();
        }
    }

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

// -----------------------------------------------------------------------------

// Extract indices of differentiable variables, returns
// the type of the underlying differentiable array
nb::handle diff_vars(nb::handle h, std::vector<uint32_t> &indices) {
    nb::handle result = nb::none();
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float_v(h.type())) {
            if (s.meta.ndim == 1) {
                if (s.op_grad_enabled(nb::inst_ptr<void>(h))) {
                    indices.push_back(s.op_index_ad(nb::inst_ptr<void>(h)));
                    result = h.type();
                }
            } else {
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid())
                        nb::detail::raise_python_error();
                    nb::handle result2 = diff_vars(v, indices);
                    if (!result2.is_none())
                        result = result2;
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (nb::handle h2 : h) {
            nb::handle result2 = diff_vars(h2, indices);
            if (!result2.is_none())
                result = result2;
        }
    } else if (nb::isinstance<nb::mapping>(h)) {
        nb::handle result2 = diff_vars(nb::borrow<nb::mapping>(h).values(), indices);
        if (!result2.is_none())
            result = result2;
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
            for (auto [k, v] : dstruct_dict) {
                nb::handle result2 = diff_vars(nb::getattr(h, k), indices);
                if (!result2.is_none())
                    result = result2;
            }
        }
    }
    return result;
}

nb::handle diff_vars(nb::args args, std::vector<uint32_t> indices) {
    nb::handle result = nb::none();
    for (nb::handle h : args) {
        nb::handle result2 = diff_vars(h, indices);
        if (!result2.is_none())
            result = result2;
    }
    return result;
}

// Helper function to clear (in-place) the AD part of a Dr.Jit array, sequence, struct, ...
void clear_ad(nb::handle h) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float_v(h.type())) {
            if (s.meta.ndim == 1) {
                s.op_set_index_ad(nb::inst_ptr<void>(h), 0);
            } else {
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid())
                        nb::detail::raise_python_error();
                    clear_ad(v);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (nb::handle h2 : h)
            clear_ad(h2);
    } else if (nb::isinstance<nb::mapping>(h)) {
        clear_ad(nb::borrow<nb::mapping>(h).values());
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
            for (auto [k, v] : dstruct_dict)
                clear_ad(nb::getattr(h, k));
        }
    }
}

// Helper function to clear primal part of a Dr.Jit array, sequence, struct, ...
// and return it.
nb::object clear_primal(nb::handle h, bool dec_ref) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float_v(h.type())) {
            if (s.meta.ndim == 1) {
                nb::object result = nb::inst_alloc(h.type());
                nb::object zero = full(nb::borrow<nb::type_object>(h.type()),
                                       PyFloat_FromDouble(0.0), {1});
                s.op_ad_create(nb::inst_ptr<void>(zero),
                               s.op_index_ad(nb::inst_ptr<void>(h)),
                               nb::inst_ptr<void>(result),
                               false);

                if (dec_ref)
                    s.op_ad_dec_ref(nb::inst_ptr<void>(result));

                nb::inst_mark_ready(result);
                return result;
            } else {
                nb::object result = nb::inst_alloc(h.type());
                nb::inst_zero(result);

                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                if (s.meta.shape[0] == DRJIT_DYNAMIC)
                    s.init(nb::inst_ptr<void>(result), sm->sq_length(h.ptr()));

                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid()) {
                        result.clear();
                        break;
                    }

                    nb::object vr = clear_primal(v, dec_ref);
                    if (!vr.is_valid() || sm->sq_ass_item(result.ptr(), i, vr.ptr())) {
                        result.clear();
                        break;
                    }
                }
                return result;
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        nb::list result;
        for (nb::handle h2 : h)
            result.append(clear_primal(h2, dec_ref));
        return std::move(result);
    } else if (nb::isinstance<nb::mapping>(h)) {
        nb::object result = h.type()();
        for (nb::handle p : nb::borrow<nb::mapping>(h).items()) {
            nb::handle k = p[0], v = p[1];
            result[k] = clear_primal(v, dec_ref);
        }
        return result;
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::object result = h.type()();
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
            for (auto [k, v] : dstruct_dict)
                nb::setattr(result, k, clear_primal(nb::getattr(h, k), dec_ref));
            return result;
        }
    }

    return nb::borrow(h);
}

// Helper function to convert all input instances into their differentiable version
nb::object diff_array(nb::handle h) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff) {
            return nb::borrow(h);
        } else {
            meta result_meta = s.meta;
            result_meta.is_diff = true;
            nb::object result = nb::inst_alloc(drjit::detail::array_get(result_meta));

            if (s.meta.ndim == 1) {
                s.op_ad_create(nb::inst_ptr<void>(h), 0, nb::inst_ptr<void>(result), true);
                nb::inst_mark_ready(result);
                return result;
            } else {
                nb::inst_zero(result);

                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                PySequenceMethods *sm2 = ((PyTypeObject *) result.type().ptr())->tp_as_sequence;

                if (s.meta.shape[0] == DRJIT_DYNAMIC)
                    s.init(nb::inst_ptr<void>(result), sm->sq_length(h.ptr()));

                for (Py_ssize_t i = 0, l = sm->sq_length(h.ptr()); i < l; ++i) {
                    nb::object v = nb::steal(sm->sq_item(h.ptr(), i));
                    if (!v.is_valid()) {
                        result.clear();
                        break;
                    }

                    nb::object vr = diff_array(v);
                    if (!vr.is_valid() || sm2->sq_ass_item(result.ptr(), i, vr.ptr())) {
                        result.clear();
                        break;
                    }
                }
                return result;
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        nb::list result;
        for (nb::handle h2 : h)
            result.append(diff_array(h2));
        return std::move(result);
    } else if (nb::isinstance<nb::mapping>(h)) {
        nb::object result = h.type()();
        for (nb::handle p : nb::borrow<nb::mapping>(h).items()) {
            nb::handle k = p[0], v = p[1];
            result[k] = diff_array(v);
        }
        return result;
    }

    return nb::borrow(h);
}

void ad_scope_enter(nb::handle tp, drjit::detail::ADScope scope_type,
                    const std::vector<uint32_t> &indices) {
    if (tp.is_none()) {
#if defined(DRJIT_ENABLE_CUDA)
        drjit::detail::ad_scope_enter<drjit::CUDAArray<float>>(scope_type,  indices.size(), indices.data());
        drjit::detail::ad_scope_enter<drjit::CUDAArray<double>>(scope_type, indices.size(), indices.data());
#endif
        drjit::detail::ad_scope_enter<drjit::LLVMArray<float>>(scope_type,  indices.size(), indices.data());
        drjit::detail::ad_scope_enter<drjit::LLVMArray<double>>(scope_type, indices.size(), indices.data());
        return;
    }

    if (!is_drjit_type(tp)) {
        PyErr_Format(PyExc_TypeError,
                     "ad_scope_enter(): expected a Dr.JIT array type!");
        nb::detail::raise_python_error();
    }

    const supp &s = nb::type_supplement<supp>(tp);

    if (!s.meta.is_diff) {
        PyErr_Format(PyExc_TypeError,
                     "ad_scope_enter(): expected a differentiable Dr.JIT array type!");
        nb::detail::raise_python_error();
    }

    if (s.meta.is_cuda) {
        if ((VarType) s.meta.type == VarType::Float32) {
            drjit::detail::ad_scope_enter<drjit::CUDAArray<float>>(scope_type, indices.size(), indices.data());
        } else if ((VarType) s.meta.type == VarType::Float64) {
            drjit::detail::ad_scope_enter<drjit::CUDAArray<double>>(scope_type, indices.size(), indices.data());
        } else  {
            PyErr_Format(PyExc_TypeError,
                         "ad_scope_enter(): expected a floating point Dr.JIT array type!");
            nb::detail::raise_python_error();
        }
    } else if (s.meta.is_llvm) {
        if ((VarType) s.meta.type == VarType::Float32) {
            drjit::detail::ad_scope_enter<drjit::LLVMArray<float>>(scope_type, indices.size(), indices.data());
        } else if ((VarType) s.meta.type == VarType::Float64) {
            drjit::detail::ad_scope_enter<drjit::LLVMArray<double>>(scope_type, indices.size(), indices.data());
        } else  {
            PyErr_Format(PyExc_TypeError,
                         "ad_scope_enter(): expected a floating point Dr.JIT array type!");
            nb::detail::raise_python_error();
        }
    }
}

void ad_scope_leave(nb::handle tp, bool process_postoned) {
    if (tp.is_none()) {
#if defined(DRJIT_ENABLE_CUDA)
        drjit::detail::ad_scope_leave<drjit::CUDAArray<float>>(process_postoned);
        drjit::detail::ad_scope_leave<drjit::CUDAArray<double>>(process_postoned);
#endif
        drjit::detail::ad_scope_leave<drjit::LLVMArray<float>>(process_postoned);
        drjit::detail::ad_scope_leave<drjit::LLVMArray<double>>(process_postoned);
        return;
    }

    if (!is_drjit_type(tp)) {
        PyErr_Format(PyExc_TypeError,
                     "ad_scope_leave(): expected a Dr.JIT array type!");
        nb::detail::raise_python_error();
    }

    const supp &s = nb::type_supplement<supp>(tp);

    if (!s.meta.is_diff) {
        PyErr_Format(PyExc_TypeError,
                     "ad_scope_leave(): expected a differentiable Dr.JIT array type!");
        nb::detail::raise_python_error();
    }

    if (s.meta.is_cuda) {
        if ((VarType) s.meta.type == VarType::Float32) {
            drjit::detail::ad_scope_leave<drjit::CUDAArray<float>>(process_postoned);
        } else if ((VarType) s.meta.type == VarType::Float64) {
            drjit::detail::ad_scope_leave<drjit::CUDAArray<double>>(process_postoned);
        } else  {
            PyErr_Format(PyExc_TypeError,
                         "ad_scope_leave(): expected a floating point Dr.JIT array type!");
            nb::detail::raise_python_error();
        }
    } else if (s.meta.is_llvm) {
        if ((VarType) s.meta.type == VarType::Float32) {
            drjit::detail::ad_scope_leave<drjit::LLVMArray<float>>(process_postoned);
        } else if ((VarType) s.meta.type == VarType::Float64) {
            drjit::detail::ad_scope_leave<drjit::LLVMArray<double>>(process_postoned);
        } else  {
            PyErr_Format(PyExc_TypeError,
                         "ad_scope_leave(): expected a floating point Dr.JIT array type!");
            nb::detail::raise_python_error();
        }
    }
}

class ADContextManager {
    drjit::detail::ADScope m_scope_type;
    nb::handle m_array_type;
    std::vector<uint32_t> m_indices;
public:
    ADContextManager() {
        m_scope_type = drjit::detail::ADScope::Invalid;
    }

    ADContextManager(drjit::detail::ADScope scope_type, nb::handle array_type, const std::vector<uint32_t> &indices) {
        m_scope_type = scope_type;
        m_array_type = array_type;
        m_indices = indices;
    }

    void __enter__() {
        if (m_scope_type != drjit::detail::ADScope::Invalid)
            ad_scope_enter(m_array_type, m_scope_type, m_indices);
    }

    void __exit__(nb::handle exc_type, nb::handle /*exc_val*/, nb::handle /*exc_tb*/) {
        if (m_scope_type != drjit::detail::ADScope::Invalid)
            ad_scope_leave(m_array_type, exc_type.is_none());
    }
};

bool __check_when_kwargs(const char *name, nb::kwargs kwargs) {
    bool when = true;
    if (nb::len(kwargs) > 1) {
        PyErr_Format(PyExc_TypeError,
                    "%s(): only boolean 'when' should be passed via keyword argument!", name);
        nb::detail::raise_python_error();
    }

    if (nb::len(kwargs) == 1) {
        try {
            when = nb::cast<bool>(kwargs["when"]);
        } catch (...) {
            PyErr_Format(PyExc_TypeError,
                            "%s(): only boolean 'when' should be passed via keyword argument!", name);
            nb::detail::raise_python_error();
        }
    }
    return when;
}

ADContextManager suspend_grad(nb::args args, nb::kwargs kwargs) {
    bool when = __check_when_kwargs("suspend_grad", kwargs);
    if (!when)
        return ADContextManager();

    std::vector<uint32_t> indices;
    nb::handle array_type = diff_vars(args, indices);
    if (nb::len(args) > 0 && indices.empty())
        indices = { 0 };

    return ADContextManager(drjit::detail::ADScope::Suspend, array_type, indices);
}

ADContextManager resume_grad(nb::args args, nb::kwargs kwargs) {
    bool when = __check_when_kwargs("resume_grad", kwargs);
    if (!when)
        return ADContextManager();

    std::vector<uint32_t> indices;
    nb::handle array_type = diff_vars(args, indices);
    if (nb::len(args) > 0 && indices.empty())
        indices = { 0 };

    return ADContextManager(drjit::detail::ADScope::Resume, array_type, indices);
}

ADContextManager isolate_grad(bool when) {
    if (!when)
        return ADContextManager();

    return ADContextManager(drjit::detail::ADScope::Isolate, nb::none(), {});
}

// -----------------------------------------------------------------------------

struct CustomOp {
    friend nb::object custom(nb::handle cls, nb::args args, nb::kwargs kwargs);

public:
    CustomOp() { }
    virtual ~CustomOp() {
        if (m_output.is_valid())
            clear_ad(m_output);
    }

    virtual nb::object eval(nb::handle /*inputs*/) {
        PyErr_Format(PyExc_RuntimeError, "CustomOp.eval(): not implemented!");
        return nb::object();
    };
    virtual void forward() {
        PyErr_Format(PyExc_RuntimeError, "CustomOp.forward(): not implemented!!");
        nb::detail::raise_python_error();
    };
    virtual void backward() {
        PyErr_Format(PyExc_RuntimeError, "CustomOp.backward(): not implemented!");
        nb::detail::raise_python_error();
    };
    virtual const char *name() const { return "CustomOp[unnamed]"; };

    nb::object grad_out() {
        return ::grad(m_output);
    }

    void set_grad_out(nb::handle value) {
        ::accum_grad(m_output, value);
    }

    nb::object grad_in(nb::str name) const {
        nb::handle input;
        try {
            input = m_inputs[name];
        } catch (...) {
            PyErr_Format(PyExc_RuntimeError,
                         "CustomOp.grad_in(): Could not find "
                         "input argument named \"%s\"!", name.c_str());
            return nb::object();
        }
        return ::grad(input);
    }

    void set_grad_in(nb::str name, nb::handle value) {
        nb::handle input;
        try {
            input = m_inputs[name];
        } catch (...) {
            PyErr_Format(PyExc_RuntimeError,
                         "CustomOp.set_grad_in(): Could not find "
                         "input argument named \"%s\"!", name.c_str());
            nb::detail::raise_python_error();
        }
        ::accum_grad(input, value);
    }

    void add_input(nb::handle value) {
        m_implicit_in.append(value);
    }

    void add_output(nb::handle value) {
        m_implicit_out.append(value);
    }

protected:
    nb::list m_implicit_in;
    nb::list m_implicit_out;
    nb::dict m_inputs;
    nb::object m_output;
};

class PyCustomOp : public CustomOp {
    NB_TRAMPOLINE(CustomOp, 4);
    virtual nb::object eval(nb::handle inputs) override {
        NB_OVERRIDE(nb::object, CustomOp, eval, inputs);
    };
    virtual void forward() override {
        NB_OVERRIDE(void, CustomOp, forward);
    };
    virtual void backward() override {
        NB_OVERRIDE(void, CustomOp, backward);
    };
    virtual const char *name() const override {
        NB_OVERRIDE(const char *, CustomOp, name);
    };
};

struct CppCustomOp : drjit::detail::DiffCallback {
    CppCustomOp(nb::handle handle) : m_handle(handle) {
        m_handle.inc_ref();
    }

    virtual void forward() override {
        nb::gil_scoped_acquire gsa;
        m_handle.attr("forward")();
    }

    virtual void backward() override {
        nb::gil_scoped_acquire gsa;
        m_handle.attr("backward")();
    }

    ~CppCustomOp() {
        nb::gil_scoped_acquire gsa;
        m_handle.dec_ref();
    }

    nb::handle m_handle;
};

nb::object custom(nb::handle cls, nb::args args, nb::kwargs kwargs) {
    nb::object inst = cls();
    CustomOp *inst_ptr = nb::inst_ptr<CustomOp>(inst);

    // Convert args to kwargs
    auto varnames = inst.attr("eval").attr("__code__").attr("co_varnames");
    for (Py_ssize_t i = 0, l = nb::len(args); i < l; i++)
        kwargs[varnames[i+1]] = args[i];

    // Detach all kwargs
    nb::kwargs eval_kwargs;
    for (auto [k, v] : kwargs)
        eval_kwargs[k] = ::detach(v);

    nb::object output = inst.attr("eval")(**eval_kwargs);

    if (::grad_enabled(output)) {
        PyErr_Format(PyExc_RuntimeError,
                     "drjit.custom(): the return value of CustomOp.eval() "
                     "should not be attached to the AD graph!");
        nb::detail::raise_python_error();
    }

    std::vector<uint32_t> diff_vars_in;
    ::diff_vars(kwargs, diff_vars_in);
    ::diff_vars(inst_ptr->m_implicit_in, diff_vars_in);

    if (!diff_vars_in.empty()) {
        output = diff_array(output);
        nb::handle type = leaf_array_t(output);
        const supp &s = nb::type_supplement<supp>(type);

        nb::object tmp_in = nb::inst_alloc(type);
        nb::object tmp_out = nb::inst_alloc(type);
        nb::inst_zero(tmp_in);
        nb::inst_zero(tmp_out);

        ::set_grad_enabled(tmp_in,  true);
        ::set_grad_enabled(tmp_out, true);
        ::set_grad_enabled(output,  true);

        inst_ptr->m_inputs = nb::cast<nb::dict>(clear_primal(kwargs, false));
        inst_ptr->m_output = clear_primal(output, true);

        std::vector<uint32_t> diff_vars_out;
        ::diff_vars(inst_ptr->m_output, diff_vars_out);
        ::diff_vars(inst_ptr->m_implicit_out, diff_vars_out);

        if (diff_vars_out.empty())
            return output; // Not relevant for AD after all..

        if (diff_vars_in.size() > 1) {
            set_label(tmp_in, nb::str(inst_ptr->name()) + nb::str("_in"));
            for (uint32_t index : diff_vars_in)
                s.op_ad_add_edge(index, s.op_index_ad(nb::inst_ptr<void>(tmp_in)), nullptr);
        }

        if (diff_vars_out.size() > 1) {
            set_label(tmp_out, nb::str(inst_ptr->name()) + nb::str("_out"));
            for (uint32_t index : diff_vars_out)
                s.op_ad_add_edge(s.op_index_ad(nb::inst_ptr<void>(tmp_out)), index, nullptr);
        }

        s.op_ad_add_edge(
            diff_vars_in.size() == 1  ? diff_vars_in[0]  : s.op_index_ad(nb::inst_ptr<void>(tmp_in)),
            diff_vars_out.size() == 1 ? diff_vars_out[0] : s.op_index_ad(nb::inst_ptr<void>(tmp_out)),
            new CppCustomOp(inst)
        );

        inst_ptr->m_implicit_in = {};
        inst_ptr->m_implicit_out = {};
    }

    return output;
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
    m.def("grad", &grad, "arg"_a, "preserve_type"_a=true, doc_grad);
    m.def("set_grad",     &set_grad,     "dst"_a, "src"_a, doc_set_grad);
    m.def("accum_grad",   &accum_grad,   "dst"_a, "src"_a, doc_accum_grad);
    m.def("replace_grad", &replace_grad, "dst"_a, "src"_a, doc_replace_grad);
    m.def("enqueue", nb::overload_cast<drjit::ADMode, nb::args>(enqueue), "mode"_a, "args"_a, doc_enqueue);
    m.def("traverse", &traverse, "type"_a, "mode"_a, "flags"_a=drjit::ADFlag::Default, doc_traverse);
    m.def("forward_to",    &forward_to,    "args"_a, "flags"_a, doc_forward_to);
    m.def("forward_from",  &forward_from,  "args"_a, "flags"_a=drjit::ADFlag::Default, doc_forward_from);
    m.def("forward",       &forward_from,  "args"_a, "flags"_a=drjit::ADFlag::Default, doc_forward);
    m.def("backward_to",   &backward_to,   "args"_a, "flags"_a, doc_backward_to);
    m.def("backward_from", &backward_from, "args"_a, "flags"_a=drjit::ADFlag::Default, doc_backward_from);
    m.def("backward",      &backward_from, "args"_a, "flags"_a=drjit::ADFlag::Default, doc_backward);
    m.def("graphviz_ad", &graphviz_ad, "as_str"_a=false, doc_graphviz_ad);

    // Enabling/disabling AD

    nb::class_<ADContextManager>(m, "__ADContextManager")
        .def("__enter__", &ADContextManager::__enter__)
        .def("__exit__", &ADContextManager::__exit__);

    m.def("ad_scope_enter", [](drjit::detail::ADScope scope_type, nb::args args) {
        std::vector<uint32_t> indices;
        nb::handle array_type = diff_vars(args, indices);
        ad_scope_enter(array_type, scope_type, indices);
    }, doc_ad_scope_enter);
    m.def("ad_scope_leave", &ad_scope_leave, doc_ad_scope_leave);

    m.def("suspend_grad", &suspend_grad); // TODO docstring
    m.def("resume_grad",  &resume_grad); // TODO docstring
    m.def("isolate_grad", &isolate_grad); // TODO docstring

    // CustomOp

    nb::class_<CustomOp, PyCustomOp>(m, "CustomOp") // TODO docstring
        .def(nb::init<>())
        .def("eval",     &CustomOp::eval)
        .def("forward",  &CustomOp::forward)
        .def("backward", &CustomOp::backward)
        .def("name",     &CustomOp::name)
        .def("grad_out",     &CustomOp::grad_out)
        .def("set_grad_out", &CustomOp::set_grad_out)
        .def("grad_in",      &CustomOp::grad_in)
        .def("set_grad_in",  &CustomOp::set_grad_in)
        .def("add_input",    &CustomOp::add_input)
        .def("add_output",   &CustomOp::add_output);

    m.def("custom", &custom); // TODO docstring
}
