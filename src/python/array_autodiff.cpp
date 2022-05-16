/*
    autodiff.cpp -- implementation of AD related functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"

static bool is_float(meta m) {
    VarType vt = (VarType) m.type;
    return vt == VarType::Float16 || vt == VarType::Float32 ||
           vt == VarType::Float64;
}

nb::handle detach_t(nb::handle h) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        meta detached_meta = s.meta;
        detached_meta.is_diff = false;
        return drjit::detail::array_get(detached_meta);
    }
    return h;
}

nb::object detach(nb::handle h, bool preserve_type=true) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff) {
            if (s.meta.ndim == 1) {
                meta detached_meta = s.meta;
                detached_meta.is_diff = false;
                nb::handle detached_tp = drjit::detail::array_get(detached_meta);
                nb::object detached = nb::inst_alloc(detached_tp);
                s.op_detach(nb::inst_ptr<void>(h), nb::inst_ptr<void>(detached));
                nb::inst_mark_ready(detached);
                if (preserve_type)
                    return h.type()(detached);
                else
                    return detached;
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
            PyErr_Format(PyExc_RuntimeError,
                         "dr.detach(): it is required to preserve the input type when "
                         "detaching a custom struct object.");
            return nb::object();
        }

        nb::object result = h.type()();
        nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
        for (auto [k, v] : dstruct_dict)
            nb::setattr(result, k, detach(v, preserve_type));
        return result;
    }

    return nb::borrow(h);
}

void set_grad_enabled(nb::handle h, bool value) {
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

void enable_grad(nb::handle h) { set_grad_enabled(h, true); }
void disable_grad(nb::handle h) { set_grad_enabled(h, false); }

bool grad_enabled(nb::handle h) {
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

bool grad_enabled(nb::args args) {
    bool result = false;
    for (nb::handle h : args)
        result |= grad_enabled(h);
    return result;
}

nb::object grad(nb::handle h) {
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

    nb::object result = nb::inst_alloc(h.type());

    if (nb::isinstance<nb::sequence>(h)) {
        for (Py_ssize_t i = 0; i < PySequence_Length((PyObject *)h.ptr()); i++)
            result[i] = grad(h[i]);
        return result;
    }

    if (nb::isinstance<nb::mapping>(h)) {
        nb::mapping m = nb::borrow<nb::mapping>(h);
        for (auto k : m.keys())
            result[k] = grad(m[k]);
        return result;
    }

    nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
    if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
        nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);
        for (auto [k, v] : dstruct_dict)
            nb::setattr(result, k, grad(v));
        return result;
    }

    nb::inst_zero(result);
    return result;
}

void set_grad(nb::handle h, nb::handle value) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                nb::object v2 = nb::borrow(value);
                if (h.type().ptr() != v2.type().ptr())
                    v2 = h.type()(v2);
                s.op_set_grad(nb::inst_ptr<void>(h), nb::inst_ptr<void>(v2));
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
                    set_grad(v, v2);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (Py_ssize_t i = 0; i < PySequence_Length((PyObject *)h.ptr()); i++) {
            if (nb::isinstance<nb::sequence>(value))
                set_grad(h[i], value[i]);
            else
                set_grad(h[i], value);
        }
    } else if (nb::isinstance<nb::mapping>(h)) {
        nb::mapping m = nb::borrow<nb::mapping>(h);
        for (auto k : m.keys()) {
            if (nb::isinstance<nb::mapping>(value))
                set_grad(m[k], value[k]);
            else
                set_grad(m[k], value);
        }
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);

            nb::object dstruct_value = nb::getattr(value.type(), "DRJIT_STRUCT", nb::handle());
            if (dstruct_value.is_valid() && nb::isinstance<nb::dict>(dstruct_value)) {
                nb::dict value_dstruct_dict = nb::borrow<nb::dict>(dstruct);
                for (auto [k, v] : dstruct_dict)
                    set_grad(v, value_dstruct_dict[k]);
            } else {
                for (auto [k, v] : dstruct_dict)
                    set_grad(v, value);
            }
        }
    }
}

void accum_grad(nb::handle h, nb::handle value) {
    if (is_drjit_array(h)) {
        const supp &s = nb::type_supplement<supp>(h.type());
        const supp &s_value = nb::type_supplement<supp>(value.type());
        if (s.meta.is_diff && is_float(s.meta)) {
            if (s.meta.ndim == 1) {
                nb::object v2 = nb::borrow(value);
                if (h.type().ptr() != v2.type().ptr())
                    v2 = h.type()(v2);
                s.op_accum_grad(nb::inst_ptr<void>(h), nb::inst_ptr<void>(v2));
            } else {
                PySequenceMethods *sm = ((PyTypeObject *) h.type().ptr())->tp_as_sequence;
                PySequenceMethods *sm2 = ((PyTypeObject *) value.type().ptr())->tp_as_sequence;
                bool value_sq = is_drjit_array(value) && s_value.meta.ndim > 1;
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
                    accum_grad(v, v2);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (Py_ssize_t i = 0; i < PySequence_Length((PyObject *)h.ptr()); i++) {
            if (nb::isinstance<nb::sequence>(value))
                accum_grad(h[i], value[i]);
            else
                accum_grad(h[i], value);
        }
    } else if (nb::isinstance<nb::mapping>(h)) {
        nb::mapping m = nb::borrow<nb::mapping>(h);
        for (auto k : m.keys()) {
            if (nb::isinstance<nb::mapping>(value))
                accum_grad(m[k], value[k]);
            else
                accum_grad(m[k], value);
        }
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            nb::dict dstruct_dict = nb::borrow<nb::dict>(dstruct);

            nb::object dstruct_value = nb::getattr(value.type(), "DRJIT_STRUCT", nb::handle());
            if (dstruct_value.is_valid() && nb::isinstance<nb::dict>(dstruct_value)) {
                nb::dict value_dstruct_dict = nb::borrow<nb::dict>(dstruct);
                for (auto [k, v] : dstruct_dict)
                    accum_grad(v, value_dstruct_dict[k]);
            } else {
                for (auto [k, v] : dstruct_dict)
                    accum_grad(v, value);
            }
        }
    }
}

void enqueue(drjit::ADMode mode, nb::handle h) {
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
                    enqueue(mode, v);
                }
            }
        }
    } else if (nb::isinstance<nb::sequence>(h)) {
        for (auto h2 : h)
            enqueue(mode, h2);
    } else if (nb::isinstance<nb::mapping>(h)) {
        return enqueue(mode, nb::borrow<nb::mapping>(h).values());
    } else {
        nb::object dstruct = nb::getattr(h.type(), "DRJIT_STRUCT", nb::handle());
        if (dstruct.is_valid() && nb::isinstance<nb::dict>(dstruct)) {
            for (auto [k, v] :  nb::borrow<nb::dict>(dstruct))
                enqueue(mode, v);
        }
    }
}

void enqueue(drjit::ADMode mode, nb::args args) {
    for (nb::handle h : args)
        enqueue(mode, h);
}

extern void bind_array_autodiff(nb::module_ m) {
    m.def("detach", &detach, "arg"_a, "preserve_type"_a=true, doc_detach);
    m.def("set_grad_enabled", &set_grad_enabled, "arg"_a, "value"_a, doc_set_grad_enabled);
    m.def("enable_grad", &enable_grad, "arg"_a, doc_enable_grad);
    m.def("disable_grad", &disable_grad, "arg"_a, doc_disable_grad);
    m.def("grad_enabled", nb::overload_cast<nb::args>(grad_enabled), doc_grad_enabled);
    m.def("grad", &grad, "arg"_a, doc_grad);
    m.def("set_grad", &set_grad, "arg"_a, "value"_a, doc_set_grad);
    m.def("accum_grad", &accum_grad, "arg"_a, "value"_a, doc_accum_grad);
    m.def("enqueue", nb::overload_cast<drjit::ADMode, nb::args>(enqueue), "mode"_a, "args"_a, doc_enqueue);
}

// TODO: replace_grad
// TODO: traverse
// TODO: forward_from
// TODO: forward_to
// TODO: forward
// TODO: backward_from
// TODO: backward_to
// TODO: backward